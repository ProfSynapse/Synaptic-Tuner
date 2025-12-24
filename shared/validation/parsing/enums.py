"""Enumeration types for response parsing.

These enums are used by the parsing module to classify responses
and identify tool call formats.
"""
from __future__ import annotations

from enum import Enum


class ResponseType(str, Enum):
    """Classification of model response types.

    Used to categorize responses based on their content structure,
    enabling behavior validation to check if the response matches expectations.
    """

    TOOL_ONLY = "tool_only"
    """Response contains tool call(s) but no meaningful text (<= 20 chars)."""

    TEXT_ONLY = "text_only"
    """Response contains meaningful text (> 20 chars) but no tool calls."""

    TOOL_TEXT = "tool_text"
    """Response contains both meaningful text and tool call(s)."""

    EMPTY = "empty"
    """Response contains neither meaningful text nor tool calls."""

    def __str__(self) -> str:
        """Return the string value for serialization compatibility."""
        return self.value


class ToolCallFormat(str, Enum):
    """Formats for tool calls in model responses.

    Different models/backends use different formats for expressing tool calls.
    This enum helps identify and handle each format appropriately.
    """

    CHATML = "chatml"
    """ChatML format: 'tool_call: toolName\\narguments: {...}'"""

    OPENAI = "openai"
    """OpenAI format: dict with 'tool_calls' array containing function objects."""

    MISTRAL = "mistral"
    """Mistral format: '[TOOL_CALLS] [{\"name\": \"...\", \"arguments\": {...}}]'"""

    QWEN = "qwen"
    """Qwen format: '<tool_call>\\n{\"name\": \"...\", \"arguments\": {...}}\\n</tool_call>'"""

    NONE = "none"
    """No tool call detected in response."""

    def __str__(self) -> str:
        return self.value
