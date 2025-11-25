"""Enumeration types for the Evaluator module.

Using enums instead of magic strings provides:
- Type safety and IDE autocompletion
- Self-documenting code
- Compile-time error detection for typos
- Centralized value definitions
"""
from __future__ import annotations

from enum import Enum, auto


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


class BackendType(str, Enum):
    """Supported backend types for model inference.

    Each backend type corresponds to a specific client implementation
    and API format.
    """

    OLLAMA = "ollama"
    """Ollama local inference server."""

    LMSTUDIO = "lmstudio"
    """LM Studio local inference server (OpenAI-compatible API)."""

    def __str__(self) -> str:
        return self.value


class ValidationLevel(str, Enum):
    """Severity levels for validation issues.

    Used by both schema validation and behavior validation
    to indicate the importance of detected issues.
    """

    ERROR = "error"
    """Critical issue that causes validation to fail."""

    WARNING = "warning"
    """Non-critical issue that should be reviewed but doesn't fail validation."""

    INFO = "info"
    """Informational message, does not affect validation result."""

    def __str__(self) -> str:
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

    NONE = "none"
    """No tool call detected in response."""

    def __str__(self) -> str:
        return self.value
