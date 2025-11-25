"""HTTP client for interacting with vLLM server.

This module provides the VLLMClient for sending chat requests to
vLLM's OpenAI-compatible API. It inherits from OpenAICompatClient
for shared OpenAI API handling.
"""
from __future__ import annotations

from .config import VLLMSettings
from .openai_compat_client import OpenAICompatClient, OpenAICompatError
from .protocols import BackendError, BackendResponse


class VLLMError(OpenAICompatError):
    """Raised when the vLLM API returns an error or malformatted payload."""
    pass


# Backwards compatibility alias
VLLMResponse = BackendResponse


class VLLMClient(OpenAICompatClient):
    """Chat client for vLLM's OpenAI-compatible /v1/chat/completions endpoint.

    vLLM implements the OpenAI API format, so this client inherits
    all functionality from OpenAICompatClient.

    vLLM-specific features:
    - High-performance inference with PagedAttention
    - LoRA adapter support via model naming
    - Dynamic adapter loading at runtime

    Example usage:
        settings = VLLMSettings(model="mistral-7b")
        client = VLLMClient(settings=settings)
        response = client.chat([{"role": "user", "content": "Hello"}])

        # List available models (including LoRA adapters)
        models = client.list_models()

        # Check if server is running
        if client.is_server_running():
            print("vLLM server is ready")
    """

    settings: VLLMSettings  # Type narrowing for IDE support

    @property
    def _client_name(self) -> str:
        return "vLLM"

    def _create_error(self, message: str) -> BackendError:
        """Create a VLLMError."""
        return VLLMError(message)
