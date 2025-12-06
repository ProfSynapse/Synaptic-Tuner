"""Adapters that bridge shared LLM client with Evaluator's BackendClient interface.

This module provides adapter classes that wrap the shared LLM client (shared/llm/)
to provide the Evaluator's specialized BackendClient interface with:
- BackendResponse objects (latency, raw data, tool calls)
- Settings-based configuration
- Health checks and connection testing
- Backward compatibility with existing Evaluator code
"""
from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Sequence

from shared.llm import create_client, LLMError
from shared.llm.base import BaseLLMClient

from .config import LMStudioSettings, OllamaSettings
from .protocols import BackendError, BackendResponse
from .base_client import extract_message_content


class SharedLLMAdapter:
    """Base adapter that wraps shared LLM client for Evaluator use.

    Provides:
    - BackendResponse format (latency tracking, raw data)
    - Settings-based configuration
    - Evaluator-compatible error handling
    """

    def __init__(
        self,
        provider: str,
        settings: LMStudioSettings | OllamaSettings,
        timeout: float = 60.0,
        retries: int = 2,
    ):
        """Initialize adapter with shared LLM client.

        Args:
            provider: Provider name for shared client (lmstudio, ollama)
            settings: Evaluator settings object
            timeout: HTTP request timeout (passed to shared client)
            retries: Number of retries (shared client handles internally)
        """
        self.settings = settings
        self.timeout = timeout
        self.retries = retries

        # Create shared LLM client
        # Note: API keys/hosts come from environment
        try:
            self.client: BaseLLMClient = create_client(
                provider=provider,
                model=settings.model
            )
        except LLMError as e:
            raise self._create_error(f"Failed to create {provider} client: {e}")

    def chat(self, messages: Sequence[Mapping[str, str]]) -> BackendResponse:
        """Send chat request and return BackendResponse.

        Args:
            messages: Chat messages

        Returns:
            BackendResponse with latency and raw data

        Raises:
            BackendError: If request fails
        """
        try:
            start = time.perf_counter()

            # Use shared client's chat method
            response_text = self.client.chat(
                messages=list(messages),
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens
            )

            latency_s = time.perf_counter() - start

            # Wrap in BackendResponse format
            return BackendResponse(
                message=response_text,
                raw={"content": response_text},  # Simplified raw format
                latency_s=latency_s
            )

        except LLMError as e:
            raise self._create_error(f"Chat request failed: {e}")

    def is_server_running(self) -> bool:
        """Check if server is accessible.

        Returns:
            True if server responds, False otherwise
        """
        try:
            return self.client.test_connection()
        except Exception:
            return False

    @property
    def _client_name(self) -> str:
        """Return client name for error messages."""
        return self.client.provider_name

    def _create_error(self, message: str) -> BackendError:
        """Create backend-specific error."""
        return BackendError(message)


class SharedLMStudioAdapter(SharedLLMAdapter):
    """LM Studio adapter using shared LLM client.

    Provides backward compatibility with Evaluator's LMStudioClient interface
    while using the shared LLM client internally.
    """

    settings: LMStudioSettings  # Type narrowing

    def __init__(
        self,
        settings: LMStudioSettings,
        timeout: float = 60.0,
        retries: int = 2,
    ):
        """Initialize LM Studio adapter.

        Args:
            settings: LM Studio settings
            timeout: Request timeout
            retries: Retry attempts
        """
        super().__init__(
            provider="lmstudio",
            settings=settings,
            timeout=timeout,
            retries=retries
        )

    @property
    def _client_name(self) -> str:
        return "LM Studio (Shared)"

    def is_server_running(self) -> bool:
        """Check if LM Studio server is running with helpful error message."""
        is_running = super().is_server_running()
        if not is_running:
            print(f"\n⚠ Cannot connect to LM Studio at {self.settings.base_url()}")
            print("\nWSL Users: If connecting to LM Studio on Windows fails, ensure:")
            print("1. LM Studio > Developer > Server > 'Serve on Local Network' is ON")
            print("2. Set LMSTUDIO_HOST=<your-ip> in .env file")
            print("3. The IP matches what's shown in LM Studio")
        return is_running


class SharedOllamaAdapter(SharedLLMAdapter):
    """Ollama adapter using shared LLM client.

    Provides backward compatibility with Evaluator's OllamaClient interface
    while using the shared LLM client internally.
    """

    settings: OllamaSettings  # Type narrowing

    def __init__(
        self,
        settings: OllamaSettings,
        timeout: float = 60.0,
        retries: int = 2,
    ):
        """Initialize Ollama adapter.

        Args:
            settings: Ollama settings
            timeout: Request timeout
            retries: Retry attempts
        """
        super().__init__(
            provider="ollama",
            settings=settings,
            timeout=timeout,
            retries=retries
        )

    @property
    def _client_name(self) -> str:
        return "Ollama (Shared)"
