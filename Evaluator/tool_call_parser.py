"""
Parser for Mistral's native [TOOL_CALLS] format.

Converts Mistral's text-based tool calls to OpenAI-compatible structured format.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple


def parse_mistral_tool_calls(content: str) -> Optional[List[Dict[str, Any]]]:
    """
    Parse Mistral's [TOOL_CALLS] format and convert to OpenAI tool_calls format.

    Mistral format (in content string):
        [TOOL_CALLS] [{"name": "tool_name", "arguments": "{...}", "id": "123"}]

    OpenAI format (structured):
        [{"id": "123", "type": "function", "function": {"name": "tool_name", "arguments": "{...}"}}]

    Args:
        content: The content string from the model response

    Returns:
        List of OpenAI-format tool calls, or None if no [TOOL_CALLS] found
    """
    if not content or "[TOOL_CALLS]" not in content:
        return None

    # Extract the JSON array after [TOOL_CALLS]
    # Pattern matches [TOOL_CALLS] followed by a JSON array
    match = re.search(r'\[TOOL_CALLS\]\s*(\[[\s\S]*\])', content)
    if not match:
        return None

    try:
        tool_calls_json = json.loads(match.group(1))
    except json.JSONDecodeError:
        # Try to extract just the first complete JSON array
        # Sometimes there's trailing text after the array
        json_str = match.group(1)
        bracket_count = 0
        end_idx = 0
        for i, char in enumerate(json_str):
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    end_idx = i + 1
                    break

        if end_idx > 0:
            try:
                tool_calls_json = json.loads(json_str[:end_idx])
            except json.JSONDecodeError:
                return None
        else:
            return None

    if not isinstance(tool_calls_json, list):
        return None

    # Convert to OpenAI format
    openai_tool_calls = []
    for i, tc in enumerate(tool_calls_json):
        if not isinstance(tc, dict):
            continue

        name = tc.get("name", "")
        arguments = tc.get("arguments", "{}")
        call_id = tc.get("id", f"call_{i}")

        # Ensure arguments is a string (OpenAI format requirement)
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments)
        elif not isinstance(arguments, str):
            arguments = str(arguments)

        openai_tool_calls.append({
            "id": str(call_id),
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments
            }
        })

    return openai_tool_calls if openai_tool_calls else None


def convert_to_openai_message(content: str) -> Dict[str, Any]:
    """
    Convert a Mistral response to OpenAI message format.

    If [TOOL_CALLS] is present, returns dict with tool_calls.
    Otherwise returns dict with content.

    Args:
        content: The content string from the model response

    Returns:
        OpenAI-format message dict
    """
    tool_calls = parse_mistral_tool_calls(content)

    if tool_calls:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls
        }
    else:
        return {
            "role": "assistant",
            "content": content
        }


def extract_tool_info(content: str) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Extract tool names and parsed arguments from Mistral format.

    Useful for validation - returns (name, parsed_arguments) tuples.

    Args:
        content: The content string containing [TOOL_CALLS]

    Returns:
        List of (tool_name, arguments_dict) tuples
    """
    tool_calls = parse_mistral_tool_calls(content)
    if not tool_calls:
        return []

    result = []
    for tc in tool_calls:
        name = tc["function"]["name"]
        args_str = tc["function"]["arguments"]
        try:
            args = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            args = {}
        result.append((name, args))

    return result


def is_mistral_tool_call(content: str) -> bool:
    """Check if content contains Mistral-format tool calls."""
    return bool(content and "[TOOL_CALLS]" in content)
