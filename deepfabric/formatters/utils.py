from typing import Any

from .models import ConversationSample, GenericSample, InstructionSample, QASample


def extract_messages(sample: Any) -> list[dict[str, str]]:  # noqa: PLR0911
    """
    Extract messages from various sample types.

    This function handles multiple input formats:
    - ConversationSample: Direct message extraction
    - QASample: Question/answer to user/assistant messages
    - InstructionSample: Instruction/output to user/assistant messages
    - GenericSample: Extract from data dict
    - Dict: Direct extraction with format detection

    Args:
        sample: Sample to extract messages from

    Returns:
        List of message dictionaries with 'role' and 'content' keys

    Raises:
        ValueError: If messages cannot be extracted from the sample
    """
    # Handle ConversationSample
    if isinstance(sample, ConversationSample):
        return [{"role": msg.role, "content": msg.content} for msg in sample.messages]

    # Handle QASample
    if isinstance(sample, QASample):
        messages = []
        if hasattr(sample, "question") and sample.question:
            messages.append({"role": "user", "content": sample.question})
        if hasattr(sample, "answer") and sample.answer:
            messages.append({"role": "assistant", "content": sample.answer})
        return messages

    # Handle InstructionSample
    if isinstance(sample, InstructionSample):
        messages = []
        if hasattr(sample, "instruction") and sample.instruction:
            content = sample.instruction
            if hasattr(sample, "input") and sample.input:
                content = f"{content}\n\nInput: {sample.input}"
            messages.append({"role": "user", "content": content})
        if hasattr(sample, "output") and sample.output:
            messages.append({"role": "assistant", "content": sample.output})
        return messages

    # Handle GenericSample or dict
    data = sample.data if isinstance(sample, GenericSample) else sample

    # Convert Pydantic objects to dict
    if hasattr(data, "model_dump") and callable(getattr(data, "model_dump", None)):
        data = data.model_dump(exclude_none=True)  # type: ignore[union-attr]

    # Try to extract messages from common formats
    if isinstance(data, dict):
        # Check for messages field
        if "messages" in data:
            return data["messages"]

        # Check for question/answer format
        if "question" in data and "answer" in data:
            messages = []
            messages.append({"role": "user", "content": data["question"]})
            messages.append({"role": "assistant", "content": data["answer"]})
            return messages

        # Check for instruction format
        if "instruction" in data:
            messages = []
            content = data["instruction"]
            if "input" in data and data["input"]:
                content = f"{content}\n\nInput: {data['input']}"
            messages.append({"role": "user", "content": content})
            if "output" in data:
                messages.append({"role": "assistant", "content": data["output"]})
            return messages

        # Check for user/assistant fields directly
        if "user" in data and "assistant" in data:
            messages = []
            messages.append({"role": "user", "content": data["user"]})
            messages.append({"role": "assistant", "content": data["assistant"]})
            return messages

    raise ValueError(f"Cannot extract messages from sample type: {type(sample)}")


def extract_data(sample: Any) -> dict:
    """
    Extract data dictionary from various sample types.

    Args:
        sample: Sample to extract data from

    Returns:
        Data dictionary
    """
    if isinstance(sample, GenericSample):
        return sample.data
    if hasattr(sample, "model_dump") and callable(getattr(sample, "model_dump", None)):
        return sample.model_dump(exclude_none=True)  # type: ignore[union-attr]
    if isinstance(sample, dict):
        return sample
    raise ValueError(f"Cannot extract data from sample type: {type(sample)}")
