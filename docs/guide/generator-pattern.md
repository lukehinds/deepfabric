# Generator Pattern

DeepFabric uses a **Generator Pattern** to provide clean separation between core logic and user interface concerns, enabling flexible integration into different applications and workflows.

## How Generators Work

The generator pattern allows DeepFabric's core components to yield events during processing, letting you handle progress monitoring, logging, and user interface updates as needed:

```python
for event in tree.build():  # Core yields events, caller handles UI
    if event['event'] == 'build_complete':
        print(f"Done! {event['total_paths']} paths")
```

This approach enables using DeepFabric as a library without any UI dependencies:

```python
from deepfabric import Tree, Graph, DataSetGenerator

# Silent usage - just consume the generator
tree = Tree(topic_prompt="AI Ethics", provider="ollama", model_name="qwen3:8b")
list(tree.build())  # Build complete, no UI
tree.save("ai_ethics.jsonl")
```

### Custom Progress Monitoring

Create your own progress handling:

```python
def build_with_logging(tree):
    """Build tree with custom logging."""
    import logging
    logger = logging.getLogger(__name__)

    for event in tree.build():
        if event['event'] == 'subtopics_generated':
            logger.info(f"Generated {event['count']} subtopics")
        elif event['event'] == 'build_complete':
            logger.info(f"Build complete: {event['total_paths']} paths")

def build_with_metrics(graph):
    """Build graph with metrics collection."""
    metrics = {'nodes_created': 0, 'failures': 0}

    for event in graph.build():
        if event['event'] == 'node_expanded':
            metrics['nodes_created'] += event['subtopics_added']
        elif event['event'] == 'build_complete':
            metrics['failures'] = event.get('failed_generations', 0)

    return metrics
```

### Easy Testing

Test core logic without mocking UI:

```python
def test_tree_generation():
    tree = Tree(topic_prompt="Test", provider="ollama", model_name="test")

    # Collect all events
    events = list(tree.build())

    # Assert on specific events
    start_events = [e for e in events if e['event'] == 'build_start']
    assert len(start_events) == 1

    complete_events = [e for e in events if e['event'] == 'build_complete']
    assert len(complete_events) == 1
    assert complete_events[0]['total_paths'] > 0
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
# Just run it, ignore progress
list(tree.build())
list(graph.build())
```

### Pattern 2: Progress Monitoring

```python
# Monitor specific events
for event in tree.build():
    if event['event'] == 'build_complete':
        print(f"✅ Complete: {event['total_paths']} paths")
```

### Pattern 3: Event Collection

```python
# Collect all events for analysis
events = list(graph.build())
failed_count = sum(1 for e in events if e['event'] == 'error')
node_expansions = [e for e in events if e['event'] == 'node_expanded']
```

### Pattern 4: Real-time Streaming

```python
# Process events as they occur
def process_build_events(generator):
    for event in generator:
        # Send to monitoring system
        metrics_client.send_event(event)

        # Log important events
        if event['event'] in ['error', 'build_complete']:
            logger.info(f"Build event: {event}")

process_build_events(tree.build())
```

## CLI Integration

The CLI uses adapter functions to bridge generators to TUI components:

```python
# cli.py - Adapts generator events to TUI
def handle_tree_events(tree, show_progress=True):
    if show_progress:
        tui = get_tree_tui()

    for event in tree.build():
        if show_progress:
            if event['event'] == 'build_start':
                tui.start_building(event['model_name'], event['depth'], event['degree'])
            elif event['event'] == 'build_complete':
                tui.finish_building(event['total_paths'], event['failed_generations'])

    return event  # Return final event
```

This approach maintains clean separation between core logic and user interface concerns while providing rich interactive experiences when needed.