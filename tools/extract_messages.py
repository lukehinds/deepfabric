#!/usr/bin/env python3
"""
Script to extract messages from JSONL file, remove system role, and pretty print.
"""

import json
import sys


def process_jsonl(input_file: str) -> None:
    """
    Read a JSONL file, extract messages, remove system role, and pretty print.

    Args:
        input_file: Path to the input JSONL file.
    """
    with open(input_file) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()  # noqa: PLW2901
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}", file=sys.stderr)
                continue

            if "messages" not in data:
                print(f"Line {line_num}: No 'messages' field found", file=sys.stderr)
                continue

            messages = data["messages"]
            filtered_messages = [
                msg for msg in messages if msg.get("role") != "system"
            ]

            print(f"--- Entry {line_num} ---")
            print(json.dumps(filtered_messages, indent=2))
            print()


if __name__ == "__main__":
    if len(sys.argv) != 2:  # noqa: PLR2004
        print(f"Usage: {sys.argv[0]} <input.jsonl>", file=sys.stderr)
        sys.exit(1)

    process_jsonl(sys.argv[1])
