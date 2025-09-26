"""
DeepFabric Chain of Thought Example - Free-text Format

Generates GSM8K-style reasoning datasets with question/chain_of_thought/final_answer format.
"""

import logging
import os
import sys

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('cot_freetext_debug.log', mode='w')
    ]
)

# Enable debug logging for deepfabric modules
logging.getLogger('deepfabric').setLevel(logging.INFO)
logging.getLogger('deepfabric.generator').setLevel(logging.INFO)
logging.getLogger('deepfabric.llm').setLevel(logging.INFO)

logger = logging.getLogger(__name__)

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepfabric import DataSetGenerator  # noqa: E402
from deepfabric.dataset import Dataset  # noqa: E402
from deepfabric.tree import Tree  # noqa: E402


def main():  # noqa: PLR0912
    """Generate a free-text Chain of Thought reasoning dataset."""

    logger.info("🚀 Starting CoT Free-text Dataset Generation")

    # Step 1: Create a topic tree for math problems
    logger.info("📝 Creating topic tree configuration")
    tree = Tree(
        topic_prompt="Mathematical word problems and logical reasoning challenges",
        provider="openai",
        model_name="gpt-4o-mini",
        degree=2,
        depth=2,
        temperature=0.7,
    )
    logger.debug(f"Tree config: provider={tree.provider}, model={tree.model_name}, temp={tree.temperature}")

    # Build the tree
    print("Building math reasoning topic tree...")
    logger.info("🌳 Building topic tree...")
    for event in tree.build():
        logger.debug(f"Tree build event: {event}")
        if event['event'] == 'build_complete':
            print(f"✅ Tree built with {event['total_paths']} topic paths")
            logger.info(f"Tree building complete: {event['total_paths']} paths generated")

    # Step 2: Create CoT dataset generator
    logger.info("🛠️ Creating CoT dataset generator")
    engine = DataSetGenerator(
        instructions="Create clear mathematical and logical reasoning problems that require step-by-step thinking to solve.",
        generation_system_prompt="You are a mathematics tutor creating educational problems that require step-by-step reasoning.",
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.3,
        conversation_type="cot_freetext",  # Use free-text CoT format
        reasoning_style="mathematical",
    )
    logger.debug(f"Generator config: conversation_type={engine.config.conversation_type}, reasoning_style={engine.config.reasoning_style}")
    logger.debug(f"Generator provider: {engine.config.provider}, model: {engine.config.model_name}")

    # Step 3: Generate the CoT dataset
    print("Generating free-text CoT dataset...")
    logger.info("🔄 Starting dataset generation with events...")

    # Use the event-based generation for detailed logging
    generation_events = engine.create_data_with_events(
        num_steps=4,      # Generate 4 examples
        batch_size=1,     # One at a time
        topic_model=tree, # Use our topic tree
        sys_msg=False,    # Free-text CoT doesn't use system messages
    )

    dataset = None
    for event in generation_events:
        if isinstance(event, dict):
            logger.debug(f"Generation event: {event}")
            if event.get('event') == 'generation_start':
                logger.info(f"Generation started: {event.get('total_samples')} total samples")
            elif event.get('event') == 'step_start':
                logger.info(f"Starting step {event.get('step')}/{event.get('total_steps')}")
            elif event.get('event') == 'step_complete':
                logger.info(f"Step {event.get('step')} complete: {event.get('samples_generated')} samples generated")
            elif event.get('event') == 'step_failed':
                logger.error(f"❌ Step {event.get('step')} failed: {event.get('message')}")
            elif event.get('event') == 'generation_complete':
                logger.info(f"🎉 Generation complete: {event.get('total_samples')} total samples")
        else:
            # Final dataset result
            dataset = event
            logger.info(f"📦 Received final dataset with {len(dataset.samples) if dataset and hasattr(dataset, 'samples') else 0} samples")

    # Validate dataset was created
    if dataset is None:
        msg = "Dataset generation failed"
        raise RuntimeError(msg)
    if not isinstance(dataset, Dataset):
        msg = f"Expected Dataset object, got {type(dataset)}"
        raise TypeError(msg)

    # Step 4: Save the dataset
    dataset.save("math_reasoning_freetext_cot.jsonl")

    print(f"✅ Generated {len(dataset.samples)} CoT training examples")
    print("📁 Saved to math_reasoning_freetext_cot.jsonl")

    # Show a sample
    if dataset.samples:
        print("\n📖 Sample CoT entry:")
        sample = dataset.samples[0]
        print(f"Question: {sample.get('question', 'N/A')}")
        print(f"Reasoning: {sample.get('chain_of_thought', 'N/A')[:100]}...")
        print(f"Answer: {sample.get('final_answer', 'N/A')}")


if __name__ == "__main__":
    main()
