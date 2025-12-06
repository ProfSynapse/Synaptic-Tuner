"""HTTP client for interacting with LM Studio server.

**DEPRECATED**: This module is deprecated in favor of SharedLMStudioAdapter
which uses the shared LLM client system (shared/llm/). The legacy client
is maintained for backward compatibility but will be removed in a future version.

Use SharedLMStudioAdapter from shared_llm_adapters.py instead.

This module provides the LMStudioClient for sending chat requests to
LM Studio's OpenAI-compatible API. It inherits from OpenAICompatClient
for shared OpenAI API handling.
"""
from __future__ import annotations

import warnings

from .config import LMStudioSettings
from .openai_compat_client import OpenAICompatClient, OpenAICompatError
from .protocols import BackendError, BackendResponse

# Deprecation warning
warnings.warn(
    "LMStudioClient is deprecated. Use SharedLMStudioAdapter from "
    "shared_llm_adapters.py instead, which uses the shared LLM client system.",
    DeprecationWarning,
    stacklevel=2
)


class LMStudioError(OpenAICompatError):
    """Raised when the LM Studio API returns an error or malformatted payload."""
    pass


# Backwards compatibility alias
LMStudioResponse = BackendResponse


# WSL connection troubleshooting instructions
WSL_TROUBLESHOOTING = """
WSL Users: If connecting to LM Studio on Windows fails, try:

1. In LM Studio (Windows):
   - Click "Developer" in the left sidebar
   - Go to "Server" settings
   - Toggle ON "Serve on Local Network"
   - Note the IP address shown (e.g., 192.168.1.104)

2. Set the environment variable:
   export LMSTUDIO_HOST=<your-ip-address>

3. Re-run the evaluation:
   ./run.sh eval

Note: The IP may change when your network changes.
"""


class LMStudioClient(OpenAICompatClient):
    """Chat client for LM Studio's OpenAI-compatible /v1/chat/completions endpoint.

    LM Studio implements the OpenAI API format, so this client inherits
    all functionality from OpenAICompatClient.

    Example usage:
        settings = LMStudioSettings(model="local-model")
        client = LMStudioClient(settings=settings)
        response = client.chat([{"role": "user", "content": "Hello"}])

        # List available models
        models = client.list_models()
    """

    settings: LMStudioSettings  # Type narrowing for IDE support

    @property
    def _client_name(self) -> str:
        return "LM Studio"

    def _create_error(self, message: str) -> BackendError:
        """Create an LMStudioError with helpful troubleshooting info."""
        # Check if this is a connection error
        if "connect" in message.lower() or "connection" in message.lower():
            message = f"{message}\n{WSL_TROUBLESHOOTING}"
        return LMStudioError(message)

    def is_server_running(self) -> bool:
        """Check if the server is running and accessible.

        Returns:
            True if server responds to health check, False otherwise
        """
        is_running = super().is_server_running()
        if not is_running:
            print(f"\n⚠ Cannot connect to LM Studio at {self.settings.base_url()}")
            print(WSL_TROUBLESHOOTING)
        return is_running
