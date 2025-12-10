import json

from transformers.utils import get_json_schema


def control_light(room: str, state: str) -> str:
    """
    Controls the lights in a room.

    Args:
        room: The name of the room.
        state: The desired state of the light ("on" or "off").

    Returns:
        str: A message indicating the new state of the lights.
    """
    return f"The lights in {room} are now {state}."

json_schema = get_json_schema(control_light)
print(json.dumps(json_schema, indent=4))
