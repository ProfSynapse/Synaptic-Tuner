"""Enumeration types for the Evaluator module.

Using enums instead of magic strings provides:
- Type safety and IDE autocompletion
- Self-documenting code
- Compile-time error detection for typos
- Centralized value definitions

ResponseType and ToolCallFormat are re-exported from shared.validation.parsing
for backward compatibility. BackendType and ValidationLevel remain Evaluator-specific.
"""
from __future__ import annotations

from enum import Enum

# Re-export from shared for backward compatibility
from shared.validation.parsing.enums import ResponseType, ToolCallFormat


class BackendType(str, Enum):
    """Supported backend types for model inference.

    Each backend type corresponds to a specific client implementation
    and API format.
    """

    OLLAMA = "ollama"
    """Ollama local inference server."""

    LMSTUDIO = "lmstudio"
    """LM Studio local inference server (OpenAI-compatible API)."""

    VLLM = "vllm"
    """vLLM high-performance inference server (OpenAI-compatible API)."""

    LLAMACPP = "llamacpp"
    """llama.cpp local inference via llama-cli (direct GGUF execution)."""

    UNSLOTH = "unsloth"
    """Direct Unsloth/LoRA inference (loads adapter on base model)."""

    OPENROUTER = "openrouter"
    """OpenRouter API (cloud inference with multiple model providers)."""

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
