#!/usr/bin/env python3
"""
Convert YAML tool definitions to OpenAI function calling format.

Usage:
    python yaml_to_openai_tools.py input.yaml output.json
    python yaml_to_openai_tools.py input.yaml  # prints to stdout
"""

import argparse
import json
import sys

from pathlib import Path

import yaml

# Type mapping from YAML types to OpenAI/JSON Schema types
_TYPE_MAPPING = {
    'str': 'string',
    'int': 'integer',
    'float': 'number',
    'bool': 'boolean',
    'list': 'array',
    'dict': 'object'
}


def yaml_tools_to_openai_format(yaml_tools):
    """
    Convert YAML tool definitions to OpenAI function calling format.

    Args:
        yaml_tools: List of tools from YAML (either as dicts or YAML string)

    Returns:
        List of tools in OpenAI format for apply_chat_template
    """
    tools_list = yaml.safe_load(yaml_tools)['tools'] if isinstance(yaml_tools, str) else yaml_tools

    openai_tools = []

    for tool in tools_list:
        properties = {}
        required = []

        for param in tool.get('parameters', []):
            param_name = param['name']
            param_type = _TYPE_MAPPING.get(param['type'], param['type'])

            properties[param_name] = {
                'type': param_type,
                'description': param['description']
            }

            if param.get('required', False):
                required.append(param_name)

        openai_tool = {
            'type': 'function',
            'function': {
                'name': tool['name'],
                'description': tool['description'],
                'parameters': {
                    'type': 'object',
                    'properties': properties,
                    'required': required
                }
            }
        }

        openai_tools.append(openai_tool)

    return openai_tools


def main():
    parser = argparse.ArgumentParser(
        description='Convert YAML tool definitions to OpenAI function calling format'
    )
    parser.add_argument(
        'input_yaml',
        type=str,
        help='Path to input YAML file'
    )
    parser.add_argument(
        'output_json',
        type=str,
        nargs='?',
        help='Path to output JSON file (optional, prints to stdout if not provided)'
    )
    parser.add_argument(
        '--indent',
        type=int,
        default=2,
        help='JSON indentation level (default: 2)'
    )

    args = parser.parse_args()

    input_path = Path(args.input_yaml)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_yaml}' not found", file=sys.stderr)
        sys.exit(1)

    with open(input_path) as f:
        yaml_data = yaml.safe_load(f)

    if 'tools' not in yaml_data:
        print("Error: YAML file must contain a 'tools' key", file=sys.stderr)
        sys.exit(1)

    openai_tools = yaml_tools_to_openai_format(yaml_data['tools'])

    json_output = json.dumps(openai_tools, indent=args.indent)

    if args.output_json:
        output_path = Path(args.output_json)
        with open(output_path, 'w') as f:
            f.write(json_output)
        print(f"Converted {len(openai_tools)} tools to {output_path}")
    else:
        print(json_output)


if __name__ == '__main__':
    main()
