"""
Shared LLM client system for Toolset-Training.

Provides unified interface to multiple LLM providers:
- OpenRouter (cloud)
- LM Studio (local)
- Ollama (local)

Easy to add new providers - just implement BaseLLMClient interface.

Usage:
    from shared.llm import create_client

    # Auto-detect from environment variables
    client = create_client()

    # Or specify explicitly
    client = create_client(provider="lmstudio", model="local-model")

    # Chat completion
    response = client.chat([
        {"role": "user", "content": "Hello!"}
    ])

    # Structured output with JSON schema
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number"}
        },
        "required": ["answer", "confidence"]
    }
    result = client.structured_output(messages, schema)

Environment variables:
    IMPROVEMENT_BACKEND=openrouter  # or lmstudio, ollama
    IMPROVEMENT_MODEL=openai/gpt-5-mini
    OPENROUTER_API_KEY=sk-or-v1-...
    LMSTUDIO_HOST=localhost
    LMSTUDIO_PORT=1234
    OLLAMA_HOST=localhost
    OLLAMA_PORT=11434
"""

from .base import BaseLLMClient
from .config import LLMConfig
from .factory import create_client, list_providers
from .exceptions import (
    LLMError,
    LLMConnectionError,
    LLMResponseError,
    LLMConfigError
)

__all__ = [
    # Main API
    "create_client",
    "list_providers",

    # Base classes
    "BaseLLMClient",
    "LLMConfig",

    # Exceptions
    "LLMError",
    "LLMConnectionError",
    "LLMResponseError",
    "LLMConfigError",
]
