"""High-level pipeline API for end-to-end workflows with DeepFabric.

This module provides a simplified API for complete workflows from topic generation
through dataset creation to model training and deployment.
"""

import asyncio
import logging

from typing import Any, Literal

from .dataset import Dataset
from .exceptions import DeepFabricError
from .generator import DataSetGenerator
from .tree import Tree
from .utils import ensure_not_running_loop

logger = logging.getLogger(__name__)


class DeepFabricPipeline:
    """High-level pipeline for end-to-end synthetic data generation and training.

    This class provides a simplified interface for complete workflows:
    1. Generate topic tree/graph
    2. Create synthetic dataset
    3. Fine-tune model (optional)
    4. Save and deploy

    Example:
        ```python
        from deepfabric.pipeline import DeepFabricPipeline
        from deepfabric.training import SFTTrainingConfig, LoRAConfig

        # Create pipeline with local model
        pipeline = DeepFabricPipeline(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            provider="transformers",
            device="cuda",
            dtype="bfloat16"
        )

        # Generate dataset
        dataset = pipeline.generate_dataset(
            topic_prompt="Advanced Python Programming",
            num_samples=1000,
            conversation_type="cot_structured"
        )

        # Train model
        training_config = SFTTrainingConfig(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            output_dir="./output",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            lora=LoRAConfig(enabled=True, r=16)
        )

        metrics = pipeline.train(training_config)

        # Save and upload
        pipeline.save_and_upload(
            output_dir="./final-model",
            hf_repo="username/my-model"
        )
        ```
    """

    def __init__(
        self,
        model_name: str,
        provider: str = "openai",
        **provider_kwargs,
    ):
        """Initialize the pipeline with a model.

        Args:
            model_name: Model identifier (HuggingFace or API model)
            provider: Provider name (openai, anthropic, gemini, ollama, transformers)
            **provider_kwargs: Additional provider-specific configuration
                (e.g., device, dtype for transformers)

        Raises:
            DeepFabricError: If initialization fails
        """
        self.model_name = model_name
        self.provider = provider
        self.provider_kwargs = provider_kwargs

        self.tree = None
        self.dataset = None
        self.trainer = None

        logger.info(
            "Initialized DeepFabricPipeline with model: %s, provider: %s",
            model_name,
            provider,
        )

    def generate_dataset(
        self,
        topic_prompt: str,
        num_samples: int = 100,
        batch_size: int = 5,
        tree_depth: int = 3,
        tree_degree: int = 10,
        conversation_type: Literal[
            "basic",
            "structured",
            "tool_calling",
            "cot_freetext",
            "cot_structured",
            "cot_hybrid",
            "agent_cot_tools",
            "agent_cot_hybrid",
            "agent_cot_multi_turn",
            "xlam_multi_turn",
        ] = "basic",
        generation_system_prompt: str | None = None,
        tree_model_name: str | None = None,
        generation_model_name: str | None = None,
        **kwargs,
    ) -> Dataset:
        """Generate a complete synthetic dataset.

        This is a synchronous wrapper around generate_dataset_async.

        Args:
            topic_prompt: Initial topic for tree generation
            num_samples: Number of samples to generate
            batch_size: Batch size for generation
            tree_depth: Depth of topic tree
            tree_degree: Branching factor of topic tree
            conversation_type: Type of conversations to generate
            generation_system_prompt: System prompt for dataset generation
            tree_model_name: Model for tree generation (defaults to pipeline model)
            generation_model_name: Model for data generation (defaults to pipeline model)
            **kwargs: Additional arguments passed to DataSetGenerator

        Returns:
            Generated Dataset

        Raises:
            DeepFabricError: If generation fails
        """
        ensure_not_running_loop("DeepFabricPipeline.generate_dataset")
        return asyncio.run(
            self.generate_dataset_async(
                topic_prompt=topic_prompt,
                num_samples=num_samples,
                batch_size=batch_size,
                tree_depth=tree_depth,
                tree_degree=tree_degree,
                conversation_type=conversation_type,
                generation_system_prompt=generation_system_prompt,
                tree_model_name=tree_model_name,
                generation_model_name=generation_model_name,
                **kwargs,
            )
        )

    async def generate_dataset_async(
        self,
        topic_prompt: str,
        num_samples: int = 100,
        batch_size: int = 5,
        tree_depth: int = 3,
        tree_degree: int = 10,
        conversation_type: Literal[
            "basic",
            "structured",
            "tool_calling",
            "cot_freetext",
            "cot_structured",
            "cot_hybrid",
            "agent_cot_tools",
            "agent_cot_hybrid",
            "agent_cot_multi_turn",
            "xlam_multi_turn",
        ] = "basic",
        generation_system_prompt: str | None = None,
        tree_model_name: str | None = None,
        generation_model_name: str | None = None,
        **kwargs,
    ) -> Dataset:
        """Generate a complete synthetic dataset (async).

        Args:
            topic_prompt: Initial topic for tree generation
            num_samples: Number of samples to generate
            batch_size: Batch size for generation
            tree_depth: Depth of topic tree
            tree_degree: Branching factor of topic tree
            conversation_type: Type of conversations to generate
            generation_system_prompt: System prompt for dataset generation
            tree_model_name: Model for tree generation (defaults to pipeline model)
            generation_model_name: Model for data generation (defaults to pipeline model)
            **kwargs: Additional arguments passed to DataSetGenerator

        Returns:
            Generated Dataset

        Raises:
            DeepFabricError: If generation fails
        """
        try:
            # Step 1: Generate topic tree
            logger.info("Step 1/2: Generating topic tree...")
            self.tree = Tree(
                topic_prompt=topic_prompt,
                provider=self.provider,
                model_name=tree_model_name or self.model_name,
                depth=tree_depth,
                degree=tree_degree,
                **self.provider_kwargs,
            )

            # Build tree asynchronously
            async for event in self.tree.build_async():
                if event.get("event") == "build_complete":
                    total_paths = event.get("total_paths", 0)
                    logger.info(
                        "Topic tree built successfully: %d paths generated",
                        total_paths,
                    )

            # Step 2: Generate dataset
            logger.info("Step 2/2: Generating synthetic dataset...")

            if generation_system_prompt is None:
                generation_system_prompt = (
                    f"Generate high-quality training data related to {topic_prompt}"
                )

            generator = DataSetGenerator(
                generation_system_prompt=generation_system_prompt,
                provider=self.provider,
                model_name=generation_model_name or self.model_name,
                conversation_type=conversation_type,
                **self.provider_kwargs,
                **kwargs,
            )

            # Calculate num_steps from num_samples and batch_size
            num_steps = (num_samples + batch_size - 1) // batch_size

            self.dataset = await generator.create_data_async(
                num_steps=num_steps,
                batch_size=batch_size,
                topic_model=self.tree,
            )

            logger.info(
                "Dataset generation complete: %d samples created",
                len(self.dataset),
            )

            return self.dataset  # noqa: TRY300

        except Exception as e:
            msg = f"Dataset generation failed: {str(e)}"
            raise DeepFabricError(msg) from e

    def train(
        self, training_config: Any, formatting_func: str | None = "trl_sft_tools"
    ) -> dict[str, Any]:
        """Train a model using the generated dataset.

        Args:
            training_config: SFTTrainingConfig instance
            formatting_func: Formatting function for dataset
                ("default", "trl_sft_tools", or custom callable)

        Returns:
            Training metrics dictionary

        Raises:
            DeepFabricError: If training fails or dataset not generated
        """
        if self.dataset is None:
            msg = "No dataset available. Call generate_dataset() first."
            raise DeepFabricError(msg)

        try:
            # Import training dependencies
            try:
                from .training import DeepFabricSFTTrainer  # noqa: PLC0415
            except ImportError as e:
                msg = (
                    "Training dependencies not installed. "
                    "Install with: pip install 'deepfabric[training]'"
                )
                raise DeepFabricError(msg) from e

            logger.info("Initializing trainer...")
            self.trainer = DeepFabricSFTTrainer(
                config=training_config,
                train_dataset=self.dataset,
                formatting_func=formatting_func,
            )

            logger.info("Starting training...")
            metrics = self.trainer.train()

        except DeepFabricError:
            raise
        except Exception as e:
            msg = f"Training failed: {str(e)}"
            raise DeepFabricError(msg) from e
        else:
            logger.info("Training complete")
            return metrics

    def save_and_upload(
        self,
        output_dir: str | None = None,
        dataset_path: str | None = None,
        hf_repo: str | None = None,
    ):
        """Save model and dataset, optionally upload to HuggingFace Hub.

        Args:
            output_dir: Directory to save trained model (if training was done)
            dataset_path: Path to save dataset
            hf_repo: HuggingFace repository ID for model upload

        Raises:
            DeepFabricError: If saving or upload fails
        """
        try:
            # Save dataset if available
            if self.dataset and dataset_path:
                logger.info("Saving dataset to: %s", dataset_path)
                self.dataset.save(dataset_path)
                logger.info("Dataset saved successfully")

            # Save trained model if available
            if self.trainer and output_dir:
                logger.info("Saving trained model...")
                self.trainer.save_model(output_dir)
                logger.info("Model saved successfully")

            # Upload to Hub if requested
            if self.trainer and hf_repo:
                logger.info("Uploading model to HuggingFace Hub: %s", hf_repo)
                self.trainer.push_to_hub(hf_repo)
                logger.info("Model uploaded successfully")

        except Exception as e:
            msg = f"Save/upload failed: {str(e)}"
            raise DeepFabricError(msg) from e

    def __repr__(self) -> str:
        """Return string representation of the pipeline."""
        return (
            f"DeepFabricPipeline("
            f"model={self.model_name}, "
            f"provider={self.provider}, "
            f"dataset_samples={len(self.dataset) if self.dataset else 0}, "
            f"trained={'yes' if self.trainer else 'no'})"
        )
