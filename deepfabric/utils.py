import ast
import asyncio
import json
import re

# ANSI escape codes for TUI updates
# ESC[F moves cursor up one line
# ESC[K clears from cursor to end of line
CLEAR_LINE = "\033[K"
MOVE_CURSOR_UP = "\033[F"

# Global to keep track of the number of lines printed by the TUI status.
# This allows for clearing the previous TUI block before printing a new one,
# enabling in-place updates rather than endless scrolling.
_tui_last_printed_lines: int = 0


def ensure_not_running_loop(method_name: str) -> None:
    """Raise when invoked inside an active asyncio event loop."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    if loop.is_running():
        msg = (
            f"{method_name} cannot be called while an event loop is running. "
            "Use the async variant instead."
        )
        raise RuntimeError(msg)


def extract_list(input_string: str):
    """
    Extracts a Python list from a given input string.

    This function attempts to parse the input string as JSON. If that fails,
    it searches for the first Python list within the string by identifying
    the opening and closing brackets. If a list is found, it is evaluated
    safely to ensure it is a valid Python list.

    Args:
        input_string (str): The input string potentially containing a Python list.

    Returns:
        list: The extracted Python list if found and valid, otherwise an empty list.

    Raises:
        None: This function handles its own exceptions and does not raise any.
    """
    try:
        return json.loads(input_string)
    except json.JSONDecodeError:
        print("Failed to parse the input string as JSON.")

    start = input_string.find("[")
    if start == -1:
        print("No Python list found in the input string.")
        return []

    count = 0
    for i, char in enumerate(input_string[start:]):
        if char == "[":
            count += 1
        elif char == "]":
            count -= 1
        if count == 0:
            end = i + start + 1
            break
    else:
        print("No matching closing bracket found.")
        return []

    found_list_str = input_string[start:end]
    found_list = safe_literal_eval(found_list_str)
    if found_list is None:
        print("Failed to parse the list due to syntax issues.")
        return []

    return found_list


def remove_linebreaks_and_spaces(input_string):
    """
    Remove line breaks and extra spaces from the input string.

    This function replaces all whitespace characters (including line breaks)
    with a single space and then ensures that there are no consecutive spaces
    in the resulting string.

    Args:
        input_string (str): The string from which to remove line breaks and extra spaces.

    Returns:
        str: The processed string with line breaks and extra spaces removed.
    """
    no_linebreaks = re.sub(r"\s+", " ", input_string)
    return " ".join(no_linebreaks.split())


def safe_literal_eval(list_string: str):
    """
    Safely evaluate a string containing a Python literal expression.

    This function attempts to evaluate a string containing a Python literal
    expression using `ast.literal_eval`. If a `SyntaxError` or `ValueError`
    occurs, it tries to sanitize the string by replacing problematic apostrophes
    with the actual right single quote character and attempts the evaluation again.

    Args:
        list_string (str): The string to be evaluated.

    Returns:
        The result of the evaluated string if successful, otherwise `None`.
    """
    try:
        return ast.literal_eval(list_string)
    except (SyntaxError, ValueError):
        # Replace problematic apostrophes with the actual right single quote character
        sanitized_string = re.sub(r"(\w)'(\w)", r"\1\u2019\2", list_string)
        try:
            return ast.literal_eval(sanitized_string)
        except (SyntaxError, ValueError):
            print("Failed to parse the list due to syntax issues.")
            return None


def read_topic_tree_from_jsonl(file_path: str) -> list[dict]:
    """
    Read the topic tree from a JSONL file.

    Args:
        file_path (str): The path to the JSONL file.

    Returns:
        list[dict]: The topic tree.
    """
    topic_tree = []
    with open(file_path) as file:
        for line in file:
            topic_tree.append(json.loads(line.strip()))
    return topic_tree


def render_tui_status(
    dataset_status: str,
    topic_status: str,
    ticker_text: str,
    width: int = 80,
    header: str = "DeepFabric Status",
) -> None:
    """
    Renders a well-formatted Terminal User Interface (TUI) status block in-place.

    This function updates the TUI in-place using ANSI escape codes, preventing
    endless scrolling. It ensures consistent alignment, padding, and includes
    simple ASCII borders for improved readability and visual separation.

    Args:
        dataset_status (str): The current status of the dataset manager.
        topic_status (str): The current status of the topic manager.
        ticker_text (str): A dynamic, scrolling or updating ticker message.
        width (int): The total fixed width of the TUI block.
        header (str): The header text for the TUI block.
    """
    global _tui_last_printed_lines

    # Clear previous output by moving the cursor up and clearing each line.
    if _tui_last_printed_lines > 0:
        for _ in range(_tui_last_printed_lines):
            print(MOVE_CURSOR_UP + CLEAR_LINE, end="", flush=True)

    # --- TUI components ---
    border_char = "="
    spacer_char = "-"
    side_border = "|"
    content_width = width - 2  # Width between the two side borders

    # --- Header and Footer ---
    header_padded_text = f" {header} "
    header_line = header_padded_text.center(width, border_char)
    bottom_line = border_char * width
    spacer_line = f"{side_border}{spacer_char * content_width}{side_border}"
    empty_line = f"{side_border}{' ' * content_width}{side_border}"

    # --- Content Lines (with programmatic alignment) ---
    labels = ["Dataset", "Topics", "Ticker"]
    max_label_len = max(len(label) for label in labels)

    def create_line(label: str, value: str) -> str:
        padded_label = label.ljust(max_label_len)
        prefix = f"  {padded_label}: "
        content_len = content_width - len(prefix)
        display_value = value[:content_len].ljust(content_len)
        return f"{side_border}{prefix}{display_value}{side_border}"

    dataset_line = create_line("Dataset", dataset_status)
    topic_line = create_line("Topics", topic_status)
    ticker_line = create_line("Ticker", ticker_text)

    # --- Assemble and Print ---
    tui_lines = [
        header_line,
        empty_line,
        dataset_line,
        topic_line,
        spacer_line,
        ticker_line,
        empty_line,
        bottom_line,
    ]

    output = "\n".join(tui_lines)
    print(output, flush=True)
    _tui_last_printed_lines = len(tui_lines)