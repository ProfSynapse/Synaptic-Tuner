"""Factory for creating LLM clients."""

from typing import Optional

from .base import BaseLLMClient
from .config import LLMConfig
from .exceptions import LLMConfigError
from .providers.openrouter import OpenRouterClient
from .providers.lmstudio import LMStudioClient
from .providers.ollama import OllamaClient


def create_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    config: Optional[LLMConfig] = None,
    env_prefix: str = "IMPROVEMENT"
) -> BaseLLMClient:
    """
    Create an LLM client based on configuration.

    Args:
        provider: Provider name ('openrouter', 'lmstudio', 'ollama')
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
        config = LLMConfig.from_env(env_prefix=env_prefix)

        # Override with explicit parameters if provided
        if provider:
            config.provider = provider.lower()
        if model:
            config.model = model

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
            model=config.model
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

    else:
        raise LLMConfigError(
            f"Unknown provider: {config.provider}. "
            f"Supported: openrouter, lmstudio, ollama"
        )


def list_providers() -> list:
    """
    List available LLM providers.

    Returns:
        List of provider names
    """
    return ["openrouter", "lmstudio", "ollama"]
