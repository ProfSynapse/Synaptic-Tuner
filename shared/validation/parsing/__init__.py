"""
Response parsing module - format-agnostic tool call extraction.

This module handles automatic detection and parsing of different
tool call formats:
- Qwen: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
- Mistral: [TOOL_CALLS] [{"name": "...", "arguments": {...}}]
- ChatML: tool_call: toolName\narguments: {...}
- OpenAI: dict with 'tool_calls' array

All formats are normalized to a common ParsedResponse structure.
"""

from .response_parser import (
    parse_response,
    ParsedResponse,
    ParsedToolCall,
    detect_response_type,
    extract_arguments_from_response,
    extract_context_from_response,
    extract_tool_name_from_response,
    extract_limit_from_response,
    get_text_content,
    get_text_length,
)
from .tool_call_parser import (
    parse_tool_calls,
    parse_mistral_tool_calls,
    parse_qwen_tool_calls,
    is_mistral_tool_call,
    is_qwen_tool_call,
    has_tool_call,
    convert_to_openai_message,
    extract_tool_info,
)
from .utilities import fix_json_newlines
from .enums import ResponseType, ToolCallFormat

__all__ = [
    # Main parsing interface
    "parse_response",
    "ParsedResponse",
    "ParsedToolCall",
    # Tool call parsing
    "parse_tool_calls",
    "parse_mistral_tool_calls",
    "parse_qwen_tool_calls",
    "is_mistral_tool_call",
    "is_qwen_tool_call",
    "has_tool_call",
    "convert_to_openai_message",
    "extract_tool_info",
    # Convenience extractors
    "detect_response_type",
    "extract_arguments_from_response",
    "extract_context_from_response",
    "extract_tool_name_from_response",
    "extract_limit_from_response",
    "get_text_content",
    "get_text_length",
    # Utilities
    "fix_json_newlines",
    # Enums
    "ResponseType",
    "ToolCallFormat",
]
