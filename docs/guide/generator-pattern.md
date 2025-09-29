# Generator Pattern

DeepFabric uses a **Generator Pattern** to provide clean separation between core logic and user interface concerns, enabling flexible integration into different applications and workflows.

## How Generators Work

The generator pattern allows DeepFabric's core components to yield events during processing, letting you handle progress monitoring, logging, and user interface updates as needed:

```python
import asyncio

async def build_tree(tree):
    async for event in tree.build_async():  # Core yields events, caller handles UI
        if event['event'] == 'build_complete':
            print(f"Done! {event['total_paths']} paths")

asyncio.run(build_tree(tree))
```

This approach enables using DeepFabric as a library without any UI dependencies:

```python
import asyncio
from deepfabric import Tree, Graph, DataSetGenerator

# Silent usage - just consume the async generator
tree = Tree(topic_prompt="AI Ethics", provider="ollama", model_name="qwen3:8b")

async def build_silently() -> None:
    async for _ in tree.build_async():
        pass

asyncio.run(build_silently())
tree.save("ai_ethics.jsonl")
```

### Custom Progress Monitoring

Create your own progress handling:

```python
import asyncio
import logging

async def build_with_logging(tree):
    """Build tree with custom logging."""
    logger = logging.getLogger(__name__)

    async for event in tree.build_async():
        if event['event'] == 'subtopics_generated':
            logger.info(f"Generated {event['count']} subtopics")
        elif event['event'] == 'build_complete':
            logger.info(f"Build complete: {event['total_paths']} paths")

async def build_with_metrics(graph):
    """Build graph with metrics collection."""
    metrics = {'nodes_created': 0, 'failures': 0}

    async for event in graph.build_async():
        if event['event'] == 'node_expanded':
            metrics['nodes_created'] += event['subtopics_added']
        elif event['event'] == 'build_complete':
            metrics['failures'] = event.get('failed_generations', 0)

    return metrics

# Execute helpers (assuming `tree` and `graph` have been created earlier)
asyncio.run(build_with_logging(tree))
metrics = asyncio.run(build_with_metrics(graph))
print(metrics)
```

### Easy Testing

Test core logic without mocking UI:

```python
import asyncio

async def collect_events(tree):
    return [event async for event in tree.build_async()]

async def test_tree_generation():
    tree = Tree(topic_prompt="Test", provider="ollama", model_name="test")

    # Collect all events
    events = await collect_events(tree)

    # Assert on specific events
    start_events = [e for e in events if e['event'] == 'build_start']
    assert len(start_events) == 1

    complete_events = [e for e in events if e['event'] == 'build_complete']
    assert len(complete_events) == 1
    assert complete_events[0]['total_paths'] > 0

asyncio.run(test_tree_generation())
```

## Event Types

### Tree Events

| Event Type | Description | Key Fields |
|------------|-------------|------------|
| `build_start` | Build initialization | `model_name`, `depth`, `degree` |
| `subtree_start` | Beginning subtree generation | `node_path`, `depth` |
| `subtopics_generated` | Subtopic generation result | `parent_path`, `count`, `success` |
| `leaf_reached` | Path reached maximum depth | `path` |
| `build_complete` | Build finished | `total_paths`, `failed_generations` |
| `error` | Build error occurred | `error` |

### Graph Events

| Event Type | Description | Key Fields |
|------------|-------------|------------|
| `depth_start` | Beginning depth level | `depth`, `leaf_count` |
| `node_expanded` | Node expansion completed | `node_topic`, `subtopics_added`, `connections_added` |
| `depth_complete` | Depth level finished | `depth` |
| `build_complete` | Graph construction finished | `nodes_count`, `failed_generations` |
| `error` | Build error occurred | `error` |

## Usage Patterns

### Pattern 1: Silent Consumption

```python
import asyncio

async def build_all() -> None:
    async for _ in tree.build_async():
        pass
    async for _ in graph.build_async():
        pass

asyncio.run(build_all())
```

### Pattern 2: Progress Monitoring

```python
import asyncio

async def monitor_tree() -> None:
    async for event in tree.build_async():
        if event['event'] == 'build_complete':
            print(f"✅ Complete: {event['total_paths']} paths")

asyncio.run(monitor_tree())
```

### Pattern 3: Event Collection

```python
import asyncio

async def collect_graph_events():
    return [event async for event in graph.build_async()]

async def analyze_graph():
    events = await collect_graph_events()
    failed_count = sum(1 for e in events if e['event'] == 'error')
    node_expansions = [e for e in events if e['event'] == 'node_expanded']
    return events, failed_count, node_expansions

EVENTS, FAILED_COUNT, NODE_EXPANSIONS = asyncio.run(analyze_graph())
```

### Pattern 4: Real-time Streaming

```python
import asyncio

async def process_build_events(generator):
    async for event in generator:
        # Send to monitoring system
        metrics_client.send_event(event)

        # Log important events
        if event['event'] in ['error', 'build_complete']:
            logger.info(f"Build event: {event}")

asyncio.run(process_build_events(tree.build_async()))
```

## CLI Integration

The CLI uses adapter functions to bridge generators to TUI components:

```python
# cli.py - Adapts generator events to TUI
import asyncio

async def handle_tree_events_async(tree, show_progress=True):
    if show_progress:
        tui = get_tree_tui()

    async for event in tree.build_async():
        if show_progress:
            if event['event'] == 'build_start':
                tui.start_building(event['model_name'], event['depth'], event['degree'])
            elif event['event'] == 'build_complete':
                tui.finish_building(event['total_paths'], event['failed_generations'])

    return event  # Return final event

# Synchronous entry point for CLI commands

def handle_tree_events(tree, show_progress=True):
    return asyncio.run(handle_tree_events_async(tree, show_progress=show_progress))
```

This approach maintains clean separation between core logic and user interface concerns while providing rich interactive experiences when needed.