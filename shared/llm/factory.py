"""Factory for creating LLM clients."""

from typing import Optional

from .base import BaseLLMClient
from .config import LLMConfig
from .exceptions import LLMConfigError
from .providers.openrouter import OpenRouterClient
from .providers.openai_responses import OpenAIResponsesClient
from .providers.lmstudio import LMStudioClient
from .providers.ollama import OllamaClient
from .providers.unsloth import UnslothClient


def create_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    config: Optional[LLMConfig] = None,
    env_prefix: str = "IMPROVEMENT",
    config_defaults: Optional[dict] = None
) -> BaseLLMClient:
    """
    Create an LLM client based on configuration.

    Args:
        provider: Provider name ('openrouter', 'openai_responses', 'lmstudio', 'ollama')
                 If None, loads from env using env_prefix
        model: Model name. If None, loads from env using env_prefix
        config: Pre-configured LLMConfig. If provided, overrides provider/model
        env_prefix: Environment variable prefix (default: 'IMPROVEMENT')

    Returns:
        Configured LLM client instance

    Raises:
        LLMConfigError: If configuration is invalid

    Examples:
        # From environment variables (IMPROVEMENT_BACKEND, IMPROVEMENT_MODEL)
        client = create_client()

        # Explicit provider/model
        client = create_client(provider="lmstudio", model="local-model")

        # From custom config
        cfg = LLMConfig.from_env(env_prefix="EVAL")
        client = create_client(config=cfg)

        # Use the client
        response = client.chat([{"role": "user", "content": "Hello"}])
        structured = client.structured_output(messages, schema)
    """
    # Load config if not provided
    if config is None:
        resolved_defaults = dict(config_defaults or {})
        if provider:
            resolved_defaults["provider"] = provider.lower()
        if model:
            resolved_defaults["model"] = model
        config = LLMConfig.from_env(env_prefix=env_prefix, config_defaults=resolved_defaults)

    # Validate config
    config.validate()

    # Create appropriate client
    if config.provider == "openrouter":
        if not config.openrouter_api_key:
            raise LLMConfigError(
                "OpenRouter API key is required. "
                "Set OPENROUTER_API_KEY environment variable."
            )
        return OpenRouterClient(
            api_key=config.openrouter_api_key,
            model=config.model,
            provider=config.provider_routing,  # Pass provider routing (e.g., {"order": ["Groq"]})
            timeout_seconds=config.openrouter_timeout_seconds,
            thinking_effort=config.thinking_effort,
        )

    elif config.provider == "openai_responses":
        if not config.openai_api_key:
            raise LLMConfigError(
                "OpenAI API key is required. "
                "Set OPENAI_API_KEY environment variable."
            )
        return OpenAIResponsesClient(
            api_key=config.openai_api_key,
            model=config.model,
            base_url=config.openai_responses_base_url,
            timeout_seconds=config.openai_responses_timeout_seconds,
            store=config.openai_responses_store,
            structured_output_strict=config.openai_responses_structured_output_strict,
            thinking_effort=config.thinking_effort,
        )

    elif config.provider == "lmstudio":
        return LMStudioClient(
            host=config.lmstudio_host,
            port=config.lmstudio_port,
            model=config.model
        )

    elif config.provider == "ollama":
        return OllamaClient(
            host=config.ollama_host,
            port=config.ollama_port,
            model=config.model
        )

    elif config.provider == "unsloth":
        return UnslothClient(
            adapter_path=config.model,
            max_seq_length=config.unsloth_max_seq_length,
            load_in_4bit=config.unsloth_load_in_4bit,
            top_p=config.unsloth_top_p,
        )

    else:
        raise LLMConfigError(
            f"Unknown provider: {config.provider}. "
            f"Supported: openrouter, openai_responses, lmstudio, ollama, unsloth"
        )


def list_providers() -> list:
    """
    List available LLM providers.

    Returns:
        List of provider names
    """
    return ["openrouter", "openai_responses", "lmstudio", "ollama", "unsloth"]
