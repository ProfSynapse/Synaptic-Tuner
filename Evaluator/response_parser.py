"""Unified response parsing for multiple tool call formats.

This module provides a single, centralized implementation for parsing
model responses in different formats (ChatML, OpenAI, Mistral). This
eliminates code duplication across behavior_validator.py and other modules.

Supported formats:
- ChatML: 'tool_call: toolName\\narguments: {...}'
- OpenAI: dict with 'tool_calls' array containing function objects
- Mistral: '[TOOL_CALLS] [{\"name\": \"...\", \"arguments\": {...}}]'
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .enums import ResponseType, ToolCallFormat


@dataclass
class ParsedToolCall:
    """A single parsed tool call.

    Attributes:
        name: The tool/function name
        arguments: The arguments dict (parsed from JSON)
        raw: The raw arguments string (before JSON parsing)
    """

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    raw: Optional[str] = None


@dataclass
class ParsedResponse:
    """Parsed model response with extracted components.

    This provides a normalized view of the response regardless of the
    original format (ChatML, OpenAI, or Mistral).

    Attributes:
        text_content: Text content before/outside tool calls
        tool_calls: List of parsed tool calls
        format_detected: The format that was detected
        response_type: Classification of the response
        raw_response: The original raw response
    """

    text_content: str = ""
    tool_calls: List[ParsedToolCall] = field(default_factory=list)
    format_detected: ToolCallFormat = ToolCallFormat.NONE
    response_type: ResponseType = ResponseType.EMPTY
    raw_response: Union[str, Dict[str, Any], None] = None

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0

    @property
    def has_text(self) -> bool:
        """Check if response has meaningful text (> 20 chars)."""
        return len(self.text_content.strip()) > 20

    @property
    def first_tool_call(self) -> Optional[ParsedToolCall]:
        """Get the first tool call if present."""
        return self.tool_calls[0] if self.tool_calls else None

    @property
    def context(self) -> Optional[Dict[str, Any]]:
        """Extract context object from first tool call's arguments."""
        if self.first_tool_call:
            return self.first_tool_call.arguments.get("context")
        return None

    def get_argument(self, key: str) -> Optional[Any]:
        """Get an argument value from the first tool call."""
        if self.first_tool_call:
            return self.first_tool_call.arguments.get(key)
        return None


def parse_response(response: Union[str, Dict[str, Any]]) -> ParsedResponse:
    """Parse a model response into normalized components.

    This is the main entry point for response parsing. It automatically
    detects the format and extracts tool calls and text content.

    Args:
        response: The raw model response (string or dict)

    Returns:
        ParsedResponse with all extracted components

    Example:
        parsed = parse_response(model_output)
        if parsed.has_tool_calls:
            tool_name = parsed.first_tool_call.name
            context = parsed.context
    """
    result = ParsedResponse(raw_response=response)

    if isinstance(response, dict):
        _parse_openai_format(response, result)
    elif isinstance(response, str):
        _parse_string_format(response, result)

    # Determine response type based on content
    result.response_type = _classify_response_type(result)

    return result


def _parse_openai_format(response: Dict[str, Any], result: ParsedResponse) -> None:
    """Parse OpenAI format response with tool_calls array."""
    result.text_content = response.get("content") or ""

    tool_calls_data = response.get("tool_calls") or []
    if tool_calls_data and isinstance(tool_calls_data, list):
        result.format_detected = ToolCallFormat.OPENAI

        for tc in tool_calls_data:
            if not isinstance(tc, dict):
                continue

            function_data = tc.get("function", {})
            name = function_data.get("name", "")
            args_raw = function_data.get("arguments", "{}")

            # Parse arguments JSON
            args: Dict[str, Any] = {}
            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw)
                except json.JSONDecodeError:
                    pass
            elif isinstance(args_raw, dict):
                args = args_raw

            result.tool_calls.append(ParsedToolCall(
                name=name,
                arguments=args if isinstance(args, dict) else {},
                raw=args_raw if isinstance(args_raw, str) else None,
            ))


def _parse_string_format(response: str, result: ParsedResponse) -> None:
    """Parse string format response (Qwen, ChatML, or Mistral)."""
    if "<tool_call>" in response:
        _parse_qwen_format(response, result)
    elif "[TOOL_CALLS]" in response:
        _parse_mistral_format(response, result)
    elif "tool_call:" in response:
        _parse_chatml_format(response, result)
    else:
        # No tool calls, just text
        result.text_content = response.strip()
        result.format_detected = ToolCallFormat.NONE


def _parse_qwen_format(response: str, result: ParsedResponse) -> None:
    """Parse Qwen format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>

    This format is used by Qwen models trained with ChatML templates.
    The tool call is wrapped in <tool_call>...</tool_call> tags and contains
    a JSON object with "name" and "arguments" fields.
    """
    result.format_detected = ToolCallFormat.QWEN

    # Extract text before first <tool_call>
    parts = response.split("<tool_call>", 1)
    result.text_content = parts[0].strip()

    # Find all <tool_call>...</tool_call> blocks
    tool_call_pattern = re.compile(
        r'<tool_call>\s*([\s\S]*?)\s*</tool_call>',
        re.IGNORECASE
    )

    for match in tool_call_pattern.finditer(response):
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

        if tool_obj and isinstance(tool_obj, dict):
            name = tool_obj.get("name", "")
            args = tool_obj.get("arguments", {})

            # Arguments might be a string that needs parsing
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            result.tool_calls.append(ParsedToolCall(
                name=name,
                arguments=args if isinstance(args, dict) else {},
                raw=json_content,
            ))
        else:
            # Fallback: Try to extract name and arguments separately
            name_match = re.search(r'"name"\s*:\s*"([^"]+)"', json_content)
            if name_match:
                name = name_match.group(1)

                # Try to find arguments block and fix it
                args_match = re.search(r'"arguments"\s*:\s*(\{[\s\S]*\})', json_content)
                args = {}
                raw = json_content

                if args_match:
                    args_str = args_match.group(1)
                    try:
                        args = json.loads(args_str)
                    except json.JSONDecodeError:
                        # Try fixing newlines in arguments
                        try:
                            fixed_args = _fix_json_newlines(args_str)
                            args = json.loads(fixed_args)
                        except json.JSONDecodeError:
                            pass

                result.tool_calls.append(ParsedToolCall(
                    name=name,
                    arguments=args if isinstance(args, dict) else {},
                    raw=raw,
                ))


def _parse_mistral_format(response: str, result: ParsedResponse) -> None:
    """Parse Mistral format: [TOOL_CALLS] [{...}]"""
    result.format_detected = ToolCallFormat.MISTRAL

    parts = response.split("[TOOL_CALLS]", 1)
    result.text_content = parts[0].strip()

    if len(parts) < 2:
        return

    json_part = parts[1].strip()
    if not json_part.startswith("["):
        return

    # Find the matching closing bracket
    try:
        tool_calls_array = _extract_json_array(json_part)
        for tc in tool_calls_array:
            if not isinstance(tc, dict):
                continue

            name = tc.get("name", "")
            args = tc.get("arguments", {})

            # Arguments might be a string that needs parsing
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            result.tool_calls.append(ParsedToolCall(
                name=name,
                arguments=args if isinstance(args, dict) else {},
            ))
    except (json.JSONDecodeError, ValueError):
        pass


def _parse_chatml_format(response: str, result: ParsedResponse) -> None:
    """Parse ChatML format: tool_call: toolName\\narguments: {...}"""
    result.format_detected = ToolCallFormat.CHATML

    # Extract text before first tool_call
    parts = response.split("tool_call:", 1)
    result.text_content = parts[0].strip()

    if len(parts) < 2:
        return

    # Find all tool calls
    tool_call_pattern = re.compile(
        r'tool_call:\s*(\w+)\s*\narguments:\s*(\{.*?\})\s*(?:\n\n|Result:|$)',
        re.DOTALL
    )

    for match in tool_call_pattern.finditer(response):
        name = match.group(1)
        args_raw = match.group(2)

        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            args = {}

        result.tool_calls.append(ParsedToolCall(
            name=name,
            arguments=args if isinstance(args, dict) else {},
            raw=args_raw,
        ))

    # If regex didn't find any, try simpler extraction
    if not result.tool_calls:
        name_match = re.search(r'tool_call:\s*(\w+)', response)
        args_match = re.search(r'arguments:\s*(\{.*?\})', response, re.DOTALL)

        if name_match:
            name = name_match.group(1)
            args = {}
            args_raw = None

            if args_match:
                args_raw = args_match.group(1)
                try:
                    args = json.loads(args_raw)
                except json.JSONDecodeError:
                    pass

            result.tool_calls.append(ParsedToolCall(
                name=name,
                arguments=args if isinstance(args, dict) else {},
                raw=args_raw,
            ))


def _fix_json_newlines(json_str: str) -> str:
    """Fix literal newlines inside JSON string values.

    Models sometimes output literal newlines inside JSON strings instead of
    escaped \\n sequences. This makes the JSON invalid.

    This function attempts to fix this by escaping newlines that appear
    inside string values (between quotes).

    Args:
        json_str: Potentially malformed JSON string

    Returns:
        Fixed JSON string with escaped newlines
    """
    # Strategy: find all string values and escape newlines within them
    result = []
    in_string = False
    escape_next = False
    i = 0

    while i < len(json_str):
        char = json_str[i]

        if escape_next:
            result.append(char)
            escape_next = False
            i += 1
            continue

        if char == '\\':
            escape_next = True
            result.append(char)
            i += 1
            continue

        if char == '"':
            in_string = not in_string
            result.append(char)
            i += 1
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

        i += 1

    return ''.join(result)


def _extract_json_array(json_str: str) -> List[Any]:
    """Extract a JSON array from a string, handling nested brackets.

    Args:
        json_str: String starting with '['

    Returns:
        Parsed JSON array

    Raises:
        ValueError: If no valid JSON array found
    """
    bracket_count = 0
    for i, char in enumerate(json_str):
        if char == "[":
            bracket_count += 1
        elif char == "]":
            bracket_count -= 1
            if bracket_count == 0:
                return json.loads(json_str[:i + 1])

    raise ValueError("No matching closing bracket found")


def _classify_response_type(parsed: ParsedResponse) -> ResponseType:
    """Classify the response type based on content.

    Args:
        parsed: The parsed response

    Returns:
        ResponseType classification
    """
    has_text = parsed.has_text
    has_tool = parsed.has_tool_calls

    if has_text and has_tool:
        return ResponseType.TOOL_TEXT
    elif has_tool and not has_text:
        return ResponseType.TOOL_ONLY
    elif has_text and not has_tool:
        return ResponseType.TEXT_ONLY
    else:
        return ResponseType.EMPTY


# ---------------------------------------------------------------------------
# Convenience Functions (for backwards compatibility)
# ---------------------------------------------------------------------------

def detect_response_type(response: Union[str, Dict[str, Any]]) -> str:
    """Detect the response type from model output.

    This is a convenience wrapper that returns the string value
    for backwards compatibility.

    Args:
        response: Model response

    Returns:
        String response type: "tool_only", "text_only", "tool_text", or "empty"
    """
    parsed = parse_response(response)
    return str(parsed.response_type)


def extract_arguments_from_response(
    response: Union[str, Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Extract full arguments dict from first tool call.

    Args:
        response: Model response

    Returns:
        Arguments dict or None if no tool call found
    """
    parsed = parse_response(response)
    if parsed.first_tool_call:
        return parsed.first_tool_call.arguments
    return None


def extract_context_from_response(
    response: Union[str, Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Extract context object from first tool call's arguments.

    Args:
        response: Model response

    Returns:
        Context dict or None if not found
    """
    parsed = parse_response(response)
    return parsed.context


def extract_tool_name_from_response(
    response: Union[str, Dict[str, Any]]
) -> Optional[str]:
    """Extract the tool name from first tool call.

    Args:
        response: Model response

    Returns:
        Tool name or None if no tool call found
    """
    parsed = parse_response(response)
    if parsed.first_tool_call:
        return parsed.first_tool_call.name
    return None


def extract_limit_from_response(
    response: Union[str, Dict[str, Any]]
) -> Optional[int]:
    """Extract the 'limit' parameter from tool call arguments.

    Args:
        response: Model response

    Returns:
        Limit value as int, or None if not found
    """
    parsed = parse_response(response)
    limit = parsed.get_argument("limit")
    if limit is not None:
        try:
            return int(limit)
        except (ValueError, TypeError):
            pass
    return None


def get_text_content(response: Union[str, Dict[str, Any]]) -> str:
    """Extract text content from response (excluding tool call markers).

    Args:
        response: Model response

    Returns:
        Text content string
    """
    parsed = parse_response(response)
    return parsed.text_content


def get_text_length(response: Union[str, Dict[str, Any]]) -> int:
    """Get length of text content in response.

    Args:
        response: Model response

    Returns:
        Length of text content
    """
    return len(get_text_content(response))
