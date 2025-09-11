import json
import re
import time
import warnings

from typing import Any

import litellm

from pydantic import BaseModel, Field, field_validator

from .constants import (
    DEFAULT_MAX_TOKENS,
    MAX_RETRY_ATTEMPTS,
    RETRY_BASE_DELAY,
    TOPIC_TREE_DEFAULT_DEGREE,
    TOPIC_TREE_DEFAULT_DEPTH,
    TOPIC_TREE_DEFAULT_MODEL,
    TOPIC_TREE_DEFAULT_TEMPERATURE,
)
from .exceptions import TopicTreeError
from .prompts import TREE_GENERATION_PROMPT, TREE_JSON_INSTRUCTIONS
from .topic_model import TopicModel
from .utils import extract_list

warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings:.*")


UPPER_TREE_DEGREE = 50
UPPER_TREE_DEPTH = 10


def validate_and_clean_response(response_text: str) -> str | list[str] | None:
    """Clean and validate the response from the LLM."""
    try:
        # First try to extract a JSON array if present
        json_match = re.search(r"[.*]", response_text, re.DOTALL)
        if json_match:
            cleaned_json = json_match.group(0)
            # Remove any markdown code block markers
            cleaned_json = re.sub(r"```json\s*|\s*```", "", cleaned_json)
            return json.loads(cleaned_json)

        # If no JSON array found, fall back to extract_list
        topics = extract_list(response_text)
        if topics:
            return [topic.strip() for topic in topics if topic.strip()]
        return None  # noqa: TRY300
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing response: {str(e)}")
        return None


class TopicTreeArguments(BaseModel):
    """Arguments for constructing a topic tree."""

    root_prompt: str = Field(
        ..., min_length=1, description="The initial prompt to start the topic tree"
    )
    model_system_prompt: str = Field("", description="The system prompt for the model")
    tree_degree: int = Field(
        TOPIC_TREE_DEFAULT_DEGREE,
        ge=1,
        le=50,
        description="The branching factor of the tree",
    )
    tree_depth: int = Field(
        TOPIC_TREE_DEFAULT_DEPTH, ge=1, le=10, description="The depth of the tree"
    )
    model_name: str = Field(
        TOPIC_TREE_DEFAULT_MODEL,
        min_length=1,
        description="The name of the model to be used",
    )
    temperature: float = Field(
        TOPIC_TREE_DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Temperature for model generation",
    )

    @field_validator("tree_degree")
    @classmethod
    def validate_tree_degree(cls, v):
        if v <= 0:
            raise ValueError("positive")
        if v > UPPER_TREE_DEGREE:
            raise ValueError("max")
        return v

    @field_validator("tree_depth")
    @classmethod
    def validate_tree_depth(cls, v):
        if v <= 0:
            raise ValueError("positive")
        if v > UPPER_TREE_DEPTH:
            raise ValueError("max")
        return v

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v):
        if not v or not v.strip():
            raise ValueError("required")
        return v.strip()


class TopicTreeValidator:
    """
    TopicTreeValidator validates and calculates unique paths in a tree structure.
    """

    def __init__(self, tree_degree: int, tree_depth: int):
        self.tree_degree = tree_degree
        self.tree_depth = tree_depth

    def calculate_paths(self) -> int:
        """Calculate total number of paths in the tree."""
        return self.tree_degree**self.tree_depth

    def validate_configuration(self, num_steps: int, batch_size: int) -> dict[str, Any]:
        """Validates tree configuration and provides recommendations if invalid."""
        total_requested_paths = num_steps * batch_size
        total_tree_paths = self.calculate_paths()

        print(f"Total tree paths available: {total_tree_paths}")
        print(f"Total requested paths: {total_requested_paths}")

        if total_requested_paths > total_tree_paths:
            print("Warning: The requested paths exceed the available tree paths.")
            recommendation = {
                "valid": False,
                "suggested_num_steps": total_tree_paths // batch_size,
                "suggested_batch_size": total_tree_paths // num_steps,
                "total_tree_paths": total_tree_paths,
                "total_requested_paths": total_requested_paths,
            }
            print("Recommended configurations to fit within the tree paths:")
            print(f" - Reduce num_steps to: {recommendation['suggested_num_steps']} or")
            print(f" - Reduce batch_size to: {recommendation['suggested_batch_size']} or")
            print(" - Increase tree_depth or tree_degree to provide more paths.")
            return recommendation

        return {
            "valid": True,
            "total_tree_paths": total_tree_paths,
            "total_requested_paths": total_requested_paths,
        }


class TopicTree(TopicModel):
    """A class to represent and build a hierarchical topic tree."""

    def __init__(self, args: TopicTreeArguments):
        """Initialize the TopicTree with the given arguments."""
        if not isinstance(args, TopicTreeArguments):
            raise TopicTreeError("invalid")

        try:
            # Validate args if it's a dict (for backward compatibility)
            if isinstance(args, dict):
                args = TopicTreeArguments(**args)
        except Exception as e:
            raise TopicTreeError("invalid args") from e  # noqa: TRY003

        json_instructions = TREE_JSON_INSTRUCTIONS

        self.args = args
        self.system_prompt = json_instructions + args.model_system_prompt
        self.temperature = args.temperature
        self.model_name = args.model_name
        self.tree_degree = args.tree_degree
        self.tree_depth = args.tree_depth
        self.tree_paths: list[list[str]] = []
        self.failed_generations: list[dict[str, Any]] = []

    def build(self, model_name: str | None = None) -> None:
        """
        Build the complete topic tree.

        Args:
            model_name: Optional model name to override the configured model
        """

        if model_name:
            self.model_name = model_name

        print(f"Building the topic tree with model: {self.model_name}")

        try:
            self.tree_paths = self.build_subtree(
                [self.args.root_prompt],
                self.system_prompt,
                self.args.tree_degree,
                self.args.tree_depth,
                model_name=self.model_name,
            )

            print(f"Tree building complete. Generated {len(self.tree_paths)} paths.")
            if self.failed_generations:
                print(f"Warning: {len(self.failed_generations)} subtopic generations failed.")

        except Exception as e:
            print(f"Error building tree: {str(e)}")
            if self.tree_paths:
                print("Saving partial tree...")
                self.save("partial_tree.jsonl")
            raise

    def get_all_paths(self) -> list[list[str]]:
        """Returns all the paths in the topic model."""
        return self.tree_paths

    def get_subtopics(  # noqa: PLR0912
        self, system_prompt: str, node_path: list[str], num_subtopics: int
    ) -> list[str]:
        """Generate subtopics with improved error handling and validation."""
        print(f"Generating {num_subtopics} subtopics for: {' -> '.join(node_path)}")

        prompt = TREE_GENERATION_PROMPT
        prompt = prompt.replace("{{{{system_prompt}}}}", system_prompt if system_prompt else "")
        prompt = prompt.replace("{{{{subtopics_list}}}}", " -> ".join(node_path))
        prompt = prompt.replace("{{{{num_subtopics}}}}", str(num_subtopics))

        max_retries = MAX_RETRY_ATTEMPTS
        retries = 0
        last_error = "No error recorded"

        while retries < max_retries:
            try:
                # Prepare completion arguments
                completion_args = {
                    "model": self.model_name,
                    "max_tokens": DEFAULT_MAX_TOKENS,
                    "temperature": self.temperature,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,  # Ensure we get a complete response
                }

                response = litellm.completion(**completion_args)

                # Extract content from the response - use getattr to safely access attributes
                response_content = None

                # Try standard OpenAI format first
                choices = getattr(response, "choices", None)
                if choices and len(choices) > 0:
                    choice = choices[0]
                    message = getattr(choice, "message", None)
                    if message:
                        response_content = getattr(message, "content", None)
                    if not response_content:
                        response_content = getattr(choice, "text", None)

                # Try direct content access if no content found yet
                if not response_content:
                    response_content = getattr(response, "content", None)

                if not response_content:
                    response_content = getattr(response, "text", None)

                # Try getting content from response as dict if still no content
                if not response_content and isinstance(response, dict):
                    try:
                        if "choices" in response and response["choices"]:
                            choice = response["choices"][0]
                            if isinstance(choice, dict):
                                if "message" in choice and "content" in choice["message"]:
                                    response_content = choice["message"]["content"]
                                elif "text" in choice:
                                    response_content = choice["text"]
                        elif "content" in response:
                            response_content = response["content"]
                    except (KeyError, IndexError, TypeError):
                        pass

                if not response_content:
                    raise ValueError("No content in response")  # noqa: TRY003, TRY301

                subtopics = validate_and_clean_response(response_content)

                if subtopics and len(subtopics) > 0:
                    # Validate and clean each subtopic
                    cleaned_subtopics = []
                    for topic in subtopics:
                        if isinstance(topic, str):
                            # Keep more special characters but ensure JSON safety
                            cleaned_topic = topic.strip()
                            if cleaned_topic:
                                cleaned_subtopics.append(cleaned_topic)

                    if len(cleaned_subtopics) >= num_subtopics:
                        return cleaned_subtopics[:num_subtopics]

                last_error = "Insufficient valid subtopics generated"
                print(f"Attempt {retries + 1}: {last_error}. Retrying...")

            except Exception as e:
                last_error = str(e)
                print(
                    f"Error generating subtopics (attempt {retries + 1}/{max_retries}): {last_error}"
                )

            retries += 1
            if retries < max_retries:
                time.sleep(RETRY_BASE_DELAY**retries)  # Exponential backoff

        # If all retries failed, generate default subtopics and log the failure
        default_subtopics = [f"subtopic_{i + 1}_for_{node_path[-1]}" for i in range(num_subtopics)]
        self.failed_generations.append(
            {"path": node_path, "attempts": retries, "last_error": last_error}
        )
        print(f"Failed to generate valid subtopics after {max_retries} attempts. Using defaults.")
        return default_subtopics

    def build_subtree(
        self,
        node_path: list[str],
        system_prompt: str,
        tree_degree: int,
        subtree_depth: int,
        model_name: str,
    ) -> list[list[str]]:
        """Build a subtree with improved error handling and validation."""
        # Convert any non-string elements to strings
        node_path = [str(node) if not isinstance(node, str) else node for node in node_path]
        print(f"Building topic subtree: {' -> '.join(node_path)}")

        if subtree_depth == 0:
            return [node_path]

        subnodes = self.get_subtopics(system_prompt, node_path, tree_degree)

        # Clean and validate subnodes
        cleaned_subnodes = []
        for subnode in subnodes:
            try:
                if isinstance(subnode, dict | list):
                    cleaned_subnodes.append(json.dumps(subnode))
                else:
                    cleaned_subnodes.append(str(subnode))
            except Exception as e:
                print(f"Error cleaning subnode: {str(e)}")
                continue

        result = []
        for subnode in cleaned_subnodes:
            try:
                new_path = node_path + [subnode]
                result.extend(
                    self.build_subtree(
                        new_path,
                        system_prompt,
                        tree_degree,
                        subtree_depth - 1,
                        model_name,
                    )
                )
            except Exception as e:
                print(f"Error building subtree for {subnode}: {str(e)}")
                continue

        return result

    def save(self, save_path: str) -> None:
        """Save the topic tree to a file."""
        try:
            with open(save_path, "w") as f:
                for path in self.tree_paths:
                    f.write(json.dumps({"path": path}) + "\n")

            # Save failed generations if any
            if self.failed_generations:
                failed_path = save_path.replace(".jsonl", "_failed.jsonl")
                with open(failed_path, "w") as f:
                    for failure in self.failed_generations:
                        f.write(json.dumps(failure) + "\n")
                print(f"Failed generations saved to {failed_path}")

            print(f"Topic tree saved to {save_path}")
            print(f"Total paths: {len(self.tree_paths)}")

        except Exception as e:
            print(f"Error saving topic tree: {str(e)}")
            raise

    def print_tree(self) -> None:
        """Print the topic tree in a readable format."""
        print("Topic Tree Structure:")
        for path in self.tree_paths:
            print(" -> ".join(path))

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the topic tree to a dictionary representation.

        Returns:
            dict: Dictionary containing the tree structure and metadata
        """
        return {
            "root_prompt": self.args.root_prompt,
            "tree_degree": self.tree_degree,
            "tree_depth": self.tree_depth,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "paths": self.tree_paths,
            "failed_generations": self.failed_generations,
            "total_paths": len(self.tree_paths),
            "args": {
                "root_prompt": self.args.root_prompt,
                "model_system_prompt": self.args.model_system_prompt,
                "tree_degree": self.args.tree_degree,
                "tree_depth": self.args.tree_depth,
                "model_name": self.args.model_name,
                "temperature": self.args.temperature,
            },
        }

    def from_dict_list(self, dict_list: list[dict[str, Any]]) -> None:
        """
        Construct the topic tree from a list of dictionaries.

        Args:
            dict_list (list[dict]): The list of dictionaries representing the topic tree.
        """
        self.tree_paths = []
        self.failed_generations = []

        for d in dict_list:
            if "path" in d:
                self.tree_paths.append(d["path"])
            if "failed_generation" in d:
                self.failed_generations.append(d["failed_generation"])

        print(
            f"Loaded {len(self.tree_paths)} paths and {len(self.failed_generations)} failed generations from JSONL file"
        )
