"""
Parser for native tool call formats (Mistral, Qwen).

Converts text-based tool calls to OpenAI-compatible structured format.

Supported formats:
- Mistral: [TOOL_CALLS] [{"name": "...", "arguments": {...}}]
- Qwen: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
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


# ---------------------------------------------------------------------------
# JSON Fixing Utilities
# ---------------------------------------------------------------------------

def _fix_json_newlines(json_str: str) -> str:
    """Fix literal newlines inside JSON string values.

    Models sometimes output literal newlines inside JSON strings instead of
    escaped \\n sequences. This makes the JSON invalid.

    Args:
        json_str: Potentially malformed JSON string

    Returns:
        Fixed JSON string with escaped newlines
    """
    result = []
    in_string = False
    escape_next = False

    for char in json_str:
        if escape_next:
            result.append(char)
            escape_next = False
            continue

        if char == '\\':
            escape_next = True
            result.append(char)
            continue

        if char == '"':
            in_string = not in_string
            result.append(char)
            continue

        if in_string:
            if char == '\n':
                result.append('\\n')
            elif char == '\r':
                result.append('\\r')
            elif char == '\t':
                result.append('\\t')
            else:
                result.append(char)
        else:
            result.append(char)

    return ''.join(result)


# ---------------------------------------------------------------------------
# Qwen Format Parsing
# ---------------------------------------------------------------------------

def parse_qwen_tool_calls(content: str) -> Optional[List[Dict[str, Any]]]:
    """
    Parse Qwen's <tool_call> format and convert to OpenAI tool_calls format.

    Qwen format (in content string):
        <tool_call>
        {"name": "tool_name", "arguments": {...}}
        </tool_call>

    OpenAI format (structured):
        [{"id": "call_0", "type": "function", "function": {"name": "tool_name", "arguments": "{...}"}}]

    Args:
        content: The content string from the model response

    Returns:
        List of OpenAI-format tool calls, or None if no <tool_call> found
    """
    if not content or "<tool_call>" not in content:
        return None

    # Find all <tool_call>...</tool_call> blocks
    tool_call_pattern = re.compile(
        r'<tool_call>\s*([\s\S]*?)\s*</tool_call>',
        re.IGNORECASE
    )

    openai_tool_calls = []

    for i, match in enumerate(tool_call_pattern.finditer(content)):
        json_content = match.group(1).strip()
        tool_obj = None

        # Try to parse JSON, with fallback for malformed output
        try:
            tool_obj = json.loads(json_content)
        except json.JSONDecodeError as e:
            # If it's a control character error, try fixing newlines
            if "control character" in str(e).lower():
                try:
                    fixed_json = _fix_json_newlines(json_content)
                    tool_obj = json.loads(fixed_json)
                except json.JSONDecodeError:
                    pass

        if not tool_obj:
            # Fallback: Try to extract name and arguments separately
            name_match = re.search(r'"name"\s*:\s*"([^"]+)"', json_content)
            args_match = re.search(r'"arguments"\s*:\s*(\{[\s\S]*\})', json_content)

            if name_match:
                name = name_match.group(1)
                arguments = "{}"
                if args_match:
                    args_str = args_match.group(1)
                    try:
                        args_obj = json.loads(args_str)
                        arguments = json.dumps(args_obj)
                    except json.JSONDecodeError:
                        # Try fixing newlines in arguments
                        try:
                            fixed_args = _fix_json_newlines(args_str)
                            args_obj = json.loads(fixed_args)
                            arguments = json.dumps(args_obj)
                        except json.JSONDecodeError:
                            arguments = args_str

                openai_tool_calls.append({
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments
                    }
                })
            continue

        if not isinstance(tool_obj, dict):
            continue

        name = tool_obj.get("name", "")
        arguments = tool_obj.get("arguments", {})

        # Ensure arguments is a string (OpenAI format requirement)
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments)
        elif not isinstance(arguments, str):
            arguments = str(arguments)

        openai_tool_calls.append({
            "id": f"call_{i}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments
            }
        })

    return openai_tool_calls if openai_tool_calls else None


def is_qwen_tool_call(content: str) -> bool:
    """Check if content contains Qwen-format tool calls."""
    return bool(content and "<tool_call>" in content)


# ---------------------------------------------------------------------------
# Unified Parsing
# ---------------------------------------------------------------------------

def parse_tool_calls(content: str) -> Optional[List[Dict[str, Any]]]:
    """
    Parse tool calls from any supported format.

    Tries Qwen format first (more specific), then Mistral format.

    Args:
        content: The content string from the model response

    Returns:
        List of OpenAI-format tool calls, or None if no tool calls found
    """
    if not content:
        return None

    # Try Qwen format first (<tool_call>...</tool_call>)
    if is_qwen_tool_call(content):
        return parse_qwen_tool_calls(content)

    # Try Mistral format ([TOOL_CALLS] [...])
    if is_mistral_tool_call(content):
        return parse_mistral_tool_calls(content)

    return None


def has_tool_call(content: str) -> bool:
    """Check if content contains any supported tool call format."""
    return is_qwen_tool_call(content) or is_mistral_tool_call(content)
