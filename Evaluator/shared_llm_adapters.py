"""Adapters that bridge shared LLM client with Evaluator's BackendClient interface.

This module provides adapter classes that wrap the shared LLM client (shared/llm/)
to provide the Evaluator's specialized BackendClient interface with:
- BackendResponse objects (latency, raw data, tool calls)
- Settings-based configuration
- Health checks and connection testing
- Backward compatibility with existing Evaluator code
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, Mapping, Optional, Sequence

from shared.llm import create_client, LLMError
from shared.llm.base import BaseLLMClient
from shared.cloud_stage_logging import stage_logger_from_env

from .config import LMStudioSettings, OllamaSettings, OpenRouterSettings, OpenAIResponsesSettings, UnslothSettings
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
        settings: LMStudioSettings | OllamaSettings | OpenRouterSettings | OpenAIResponsesSettings,
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
        config_defaults = {"timeout_seconds": timeout}
        thinking_effort = getattr(settings, "thinking_effort", None)
        if thinking_effort is not None:
            config_defaults["thinking_effort"] = thinking_effort

        try:
            self.client: BaseLLMClient = create_client(
                provider=provider,
                model=settings.model,
                config_defaults=config_defaults,
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

            # Use shared client's chat method. Reasoning effort is carried as
            # the client's instance thinking_effort (set at construction from
            # settings.thinking_effort, the upstream #98 path), so no per-call
            # effort argument is needed here.
            response_text = self.client.chat(
                messages=list(messages),
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
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

    def list_models(self) -> list:
        """Return available models if provider supports listing."""
        try:
            return self.client.list_models()
        except Exception:
            return []

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


class SharedOpenRouterAdapter(SharedLLMAdapter):
    """OpenRouter adapter using shared LLM client.

    Provides access to OpenRouter's cloud inference API with multiple
    model providers available.
    """

    settings: OpenRouterSettings  # Type narrowing

    def __init__(
        self,
        settings: OpenRouterSettings,
        timeout: float = 60.0,
        retries: int = 2,
    ):
        """Initialize OpenRouter adapter.

        Args:
            settings: OpenRouter settings
            timeout: Request timeout
            retries: Retry attempts
        """
        super().__init__(
            provider="openrouter",
            settings=settings,
            timeout=timeout,
            retries=retries
        )

    @property
    def _client_name(self) -> str:
        return "OpenRouter"

    def is_server_running(self) -> bool:
        """Check if OpenRouter API is accessible."""
        is_running = super().is_server_running()
        if not is_running:
            import os
            if not os.environ.get("OPENROUTER_API_KEY"):
                print("\n⚠ OPENROUTER_API_KEY environment variable not set")
                print("Set it in your .env file or export it:")
                print("  export OPENROUTER_API_KEY=sk-or-...")
        return is_running


class SharedOpenAIResponsesAdapter(SharedLLMAdapter):
    """OpenAI Responses adapter using shared LLM client."""

    settings: OpenAIResponsesSettings

    def __init__(
        self,
        settings: OpenAIResponsesSettings,
        timeout: float = 60.0,
        retries: int = 2,
    ):
        super().__init__(
            provider="openai_responses",
            settings=settings,
            timeout=timeout,
            retries=retries,
        )

    @property
    def _client_name(self) -> str:
        return "OpenAI Responses"

    def structured_chat(
        self,
        messages: Sequence[Mapping[str, str]],
        schema: Mapping[str, Any],
        schema_name: Optional[str] = None,
    ) -> BackendResponse:
        """Route the target call through the provider's structured-output path.

        Mirrors production's wire format: the OpenAI Responses API with
        ``text.format = {type: json_schema, name, strict: true, schema}``
        (baseline-fidelity spec §4.2). Output tokens are UNCAPPED
        (``max_tokens=None``) to match production and to avoid medium-reasoning
        starving the response (§4.4). Reasoning effort is carried by the client's
        instance thinking_effort (set at construction from settings, upstream
        #98 path), so it is not passed per-call here.

        The parsed JSON object is wrapped as a string in ``message`` so
        ``build_response_view`` populates BOTH ``$.content`` and
        ``$.content_json`` identically to the chat() path (it JSON-parses a
        string ``content``); a raw dict would leave ``$.content`` empty.

        Args:
            messages: Chat messages (typically a single assembled user message).
            schema: The json_schema BODY (verbatim from the production task).
            schema_name: Schema name (e.g. "GenerateHashtagsSchema"); merged into
                the schema under "name" only when the body does not already set it.

        Returns:
            BackendResponse whose message is the serialized JSON object.

        Raises:
            BackendError: If the structured request fails.
        """
        try:
            start = time.perf_counter()

            payload_schema = dict(schema)
            if schema_name and "name" not in payload_schema:
                payload_schema["name"] = schema_name  # provider reads schema.get("name")

            result = self.client.structured_output(
                messages=list(messages),
                schema=payload_schema,
                temperature=self.settings.temperature,
                max_tokens=None,  # UNCAPPED — production parity (§4.4)
                strict=True,  # production parity (provider default is False)
            )

            latency_s = time.perf_counter() - start

            # Serialize back to a string so $.content and $.content_json both
            # populate via build_response_view's string branch.
            text = json.dumps(result)
            return BackendResponse(
                message=text,
                raw={"content": text, "content_json": result},
                latency_s=latency_s,
            )

        except LLMError as e:
            raise self._create_error(f"Structured chat request failed: {e}")

    def is_server_running(self) -> bool:
        """Check if OpenAI Responses API is accessible."""
        is_running = super().is_server_running()
        if not is_running:
            import os
            if not os.environ.get("OPENAI_API_KEY"):
                print("\nWarning: OPENAI_API_KEY environment variable not set")
                print("Set it in your .env file or export it:")
                print("  export OPENAI_API_KEY=sk-...")
        return is_running


class SharedUnslothAdapter:
    """Unsloth adapter using shared LLM client for direct LoRA inference.

    This adapter wraps the shared UnslothClient to provide the Evaluator's
    BackendClient interface. Unlike HTTP-based adapters, this loads the model
    directly in-process.

    Note: Does not inherit from SharedLLMAdapter because Unsloth requires
    different initialization (adapter path instead of host/port).
    """

    settings: UnslothSettings  # Type narrowing

    def __init__(
        self,
        settings: UnslothSettings,
        timeout: float = 120.0,
        retries: int = 2,
    ):
        """Initialize Unsloth adapter with shared LLM client.

        Args:
            settings: Unsloth settings with adapter path
            timeout: Not used (kept for interface compatibility)
            retries: Not used (kept for interface compatibility)
        """
        self.settings = settings
        self.timeout = timeout
        self.retries = retries
        self.client: BaseLLMClient = None

        # Create shared Unsloth client
        try:
            from shared.llm.providers.unsloth import UnslothClient as SharedUnslothClient

            stage_logger = stage_logger_from_env()
            model_load_details = {
                "backend": "unsloth",
                "adapter_path": settings.model,
                "max_seq_length": settings.max_seq_length,
                "load_in_4bit": settings.load_in_4bit,
            }
            if stage_logger is not None:
                stage_logger.emit(
                    "model_load_started",
                    message="Starting Unsloth model load",
                    details=model_load_details,
                )
            model_load_start = time.perf_counter()
            self.client = SharedUnslothClient(
                adapter_path=settings.model,
                max_seq_length=settings.max_seq_length,
                load_in_4bit=settings.load_in_4bit,
                top_p=settings.top_p,
            )
            if stage_logger is not None:
                completed_details = dict(model_load_details)
                completed_details["elapsed_seconds"] = round(time.perf_counter() - model_load_start, 3)
                stage_logger.emit(
                    "model_load_completed",
                    message="Unsloth model load completed",
                    details=completed_details,
                )
        except LLMError as e:
            raise BackendError(f"Failed to create Unsloth client: {e}")
        except Exception as e:
            raise BackendError(f"Failed to load model: {e}")

    def chat(self, messages: Sequence[Mapping[str, str]]) -> BackendResponse:
        """Send chat request and return BackendResponse.

        Args:
            messages: Chat messages

        Returns:
            BackendResponse with latency and raw data

        Raises:
            BackendError: If inference fails
        """
        if self.client is None:
            raise BackendError("Model not loaded")

        try:
            start = time.perf_counter()

            # Use shared client's chat method
            response_text = self.client.chat(
                messages=list(messages),
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens
            )

            latency_s = time.perf_counter() - start

            return BackendResponse(
                message=response_text,
                raw={
                    "content": response_text,
                    "model": self.settings.model,
                },
                latency_s=latency_s
            )

        except LLMError as e:
            raise BackendError(f"Inference failed: {e}")

    def is_server_running(self) -> bool:
        """Check if model is loaded (no server for Unsloth).

        Returns:
            True if model is loaded and ready
        """
        if self.client is None:
            return False
        try:
            return self.client.test_connection()
        except Exception:
            return False

    def list_models(self) -> list:
        """Return the loaded model/adapter."""
        if self.client:
            return self.client.list_models()
        return []

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self.client and hasattr(self.client, 'unload'):
            self.client.unload()
            self.client = None

    def __del__(self):
        """Cleanup on deletion."""
        self.unload()
