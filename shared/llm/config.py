"""Configuration for shared LLM clients."""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv


@dataclass
class LLMConfig:
    """Configuration for LLM clients."""

    # Provider selection
    provider: str  # 'openrouter', 'lmstudio', or 'ollama'
    model: str

    # OpenRouter config
    openrouter_api_key: Optional[str] = None

    # LM Studio config
    lmstudio_host: str = "localhost"
    lmstudio_port: int = 1234

    # Ollama config
    ollama_host: str = "localhost"
    ollama_port: int = 11434

    @classmethod
    def from_env(cls, env_prefix: str = "IMPROVEMENT") -> "LLMConfig":
        """
        Load configuration from environment variables.

        Args:
            env_prefix: Prefix for env vars (e.g., 'IMPROVEMENT' -> 'IMPROVEMENT_BACKEND')

        Environment variables:
            {PREFIX}_BACKEND: Provider name (openrouter, lmstudio, ollama)
            {PREFIX}_MODEL: Model name
            OPENROUTER_API_KEY: API key for OpenRouter
            LMSTUDIO_HOST: LM Studio host (default: localhost)
            LMSTUDIO_PORT: LM Studio port (default: 1234)
            OLLAMA_HOST: Ollama host (default: localhost)
            OLLAMA_PORT: Ollama port (default: 11434)

        Returns:
            LLMConfig instance

        Example .env:
            IMPROVEMENT_BACKEND=openrouter
            IMPROVEMENT_MODEL=openai/gpt-5-mini
            OPENROUTER_API_KEY=sk-or-v1-...

            # Or use LM Studio:
            IMPROVEMENT_BACKEND=lmstudio
            IMPROVEMENT_MODEL=local-model
            LMSTUDIO_HOST=192.168.1.100  # If on Windows from WSL
        """
        # Load .env file if not already loaded
        load_dotenv()

        provider = os.getenv(f"{env_prefix}_BACKEND", "openrouter")
        model = os.getenv(f"{env_prefix}_MODEL", "openai/gpt-5-mini")

        return cls(
            provider=provider.lower(),
            model=model,
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            lmstudio_host=os.getenv("LMSTUDIO_HOST", "localhost"),
            lmstudio_port=int(os.getenv("LMSTUDIO_PORT", "1234")),
            ollama_host=os.getenv("OLLAMA_HOST", "localhost"),
            ollama_port=int(os.getenv("OLLAMA_PORT", "11434")),
        )

    def validate(self) -> None:
        """
        Validate configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        valid_providers = ["openrouter", "lmstudio", "ollama"]
        if self.provider not in valid_providers:
            raise ValueError(
                f"Invalid provider '{self.provider}'. "
                f"Must be one of: {', '.join(valid_providers)}"
            )

        if self.provider == "openrouter" and not self.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is required for OpenRouter"
            )

        if not self.model:
            raise ValueError("Model name is required")
