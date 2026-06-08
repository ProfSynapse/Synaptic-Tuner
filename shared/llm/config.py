"""Configuration for shared LLM clients."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class LLMConfig:
    """Configuration for LLM clients."""

    # Provider selection
    provider: str  # 'openrouter', 'openai_responses', 'lmstudio', 'ollama', or 'unsloth'
    model: str  # Model name or path to LoRA adapter (for unsloth)
    temperature: float = 0.7
    max_tokens: int = 2048

    # OpenRouter config
    openrouter_api_key: Optional[str] = None
    provider_routing: Optional[Dict[str, Any]] = None  # OpenRouter provider routing
    openrouter_timeout_seconds: float = 60.0

    # OpenAI Responses API config
    openai_api_key: Optional[str] = None
    openai_responses_base_url: str = "https://api.openai.com/v1"
    openai_responses_timeout_seconds: float = 60.0
    openai_responses_store: bool = False
    openai_responses_structured_output_strict: bool = False

    # LM Studio config
    lmstudio_host: str = "localhost"
    lmstudio_port: int = 1234

    # Ollama config
    ollama_host: str = "localhost"
    ollama_port: int = 11434

    # Unsloth config (direct LoRA inference)
    unsloth_max_seq_length: int = 4096
    unsloth_load_in_4bit: bool = True
    unsloth_top_p: float = 0.9

    @classmethod
    def from_env(cls, env_prefix: str = "IMPROVEMENT", config_defaults: Optional[Dict[str, Any]] = None) -> "LLMConfig":
        """
        Load configuration from environment variables.

        Args:
            env_prefix: Prefix for env vars (e.g., 'IMPROVEMENT' -> 'IMPROVEMENT_BACKEND')
            config_defaults: Optional dict with defaults (provider/model/temperature/max_tokens)

        Environment variables:
            {PREFIX}_BACKEND: Provider name (openrouter, openai_responses, lmstudio, ollama)
            {PREFIX}_MODEL: Model name
            OPENROUTER_API_KEY: API key for OpenRouter
            OPENAI_API_KEY: API key for OpenAI Responses
            OPENAI_RESPONSES_BASE_URL: Optional OpenAI-compatible Responses base URL
            OPENAI_RESPONSES_TIMEOUT_SECONDS: Optional OpenAI Responses timeout
            OPENAI_RESPONSES_STORE: Optional response storage opt-in (default false)
            OPENAI_RESPONSES_STRUCTURED_OUTPUT_STRICT: Optional JSON Schema strict opt-in
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

            # Or use OpenAI Responses:
            IMPROVEMENT_BACKEND=openai_responses
            IMPROVEMENT_MODEL=gpt-5-mini
            OPENAI_API_KEY=sk-...

            # Or use LM Studio:
            IMPROVEMENT_BACKEND=lmstudio
            IMPROVEMENT_MODEL=local-model
            LMSTUDIO_HOST=192.168.1.100  # If on Windows from WSL
        """
        _load_env_file()

        cfg = config_defaults or {}
        provider = cfg.get("provider", os.getenv(f"{env_prefix}_BACKEND", "openrouter"))
        model = cfg.get("model", os.getenv(f"{env_prefix}_MODEL", "openai/gpt-5-mini"))
        temperature = float(cfg.get("temperature", 0.7))
        max_tokens = int(cfg.get("max_tokens", 2048))

        # Build provider routing config for OpenRouter
        provider_routing = None
        if "provider_routing" in cfg:
            pr = cfg["provider_routing"]
            provider_routing = {}
            if "order" in pr:
                provider_routing["order"] = pr["order"]
            if "allow_fallbacks" in pr:
                provider_routing["allow_fallbacks"] = pr["allow_fallbacks"]
            if "require_parameters" in pr:
                provider_routing["require_parameters"] = pr["require_parameters"]
            if "data_collection" in pr:
                provider_routing["data_collection"] = pr["data_collection"]

        return cls(
            provider=provider.lower(),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            provider_routing=provider_routing,
            openrouter_timeout_seconds=float(
                cfg.get(
                    "timeout_seconds",
                    os.getenv("OPENROUTER_TIMEOUT_SECONDS", "60"),
                )
            ),
            openai_api_key=_env_secret("OPENAI_API_KEY"),
            openai_responses_base_url=cfg.get(
                "base_url",
                os.getenv("OPENAI_RESPONSES_BASE_URL", "https://api.openai.com/v1"),
            ),
            openai_responses_timeout_seconds=float(
                cfg.get(
                    "timeout_seconds",
                    os.getenv("OPENAI_RESPONSES_TIMEOUT_SECONDS", "60"),
                )
            ),
            openai_responses_store=_coerce_bool(
                cfg.get(
                    "store",
                    _env_bool("OPENAI_RESPONSES_STORE", False),
                )
            ),
            openai_responses_structured_output_strict=_coerce_bool(
                cfg.get(
                    "structured_output_strict",
                    _env_bool("OPENAI_RESPONSES_STRUCTURED_OUTPUT_STRICT", False),
                )
            ),
            lmstudio_host=os.getenv("LMSTUDIO_HOST", "localhost"),
            lmstudio_port=int(os.getenv("LMSTUDIO_PORT", "1234")),
            ollama_host=os.getenv("OLLAMA_HOST", "localhost"),
            ollama_port=int(os.getenv("OLLAMA_PORT", "11434")),
            # Unsloth config
            unsloth_max_seq_length=int(cfg.get("max_seq_length", os.getenv("UNSLOTH_MAX_SEQ_LENGTH", "4096"))),
            unsloth_load_in_4bit=cfg.get("load_in_4bit", os.getenv("UNSLOTH_LOAD_IN_4BIT", "true").lower() == "true"),
            unsloth_top_p=float(cfg.get("top_p", os.getenv("UNSLOTH_TOP_P", "0.9"))),
        )

    def validate(self) -> None:
        """
        Validate configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        valid_providers = ["openrouter", "openai_responses", "lmstudio", "ollama", "unsloth"]
        if self.provider not in valid_providers:
            raise ValueError(
                f"Invalid provider '{self.provider}'. "
                f"Must be one of: {', '.join(valid_providers)}"
            )

        if self.provider == "openrouter" and not self.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is required for OpenRouter"
            )

        if self.provider == "openai_responses" and not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for OpenAI Responses"
            )

        if self.provider == "unsloth":
            # For unsloth, model is the adapter path
            adapter_path = Path(self.model)
            if not adapter_path.exists():
                raise ValueError(f"Unsloth adapter path not found: {self.model}")
            if not (adapter_path / "adapter_config.json").exists():
                raise ValueError(
                    f"adapter_config.json not found in {self.model}. "
                    "Is this a valid LoRA adapter directory?"
                )

        if not self.model:
            raise ValueError("Model name is required")


def _load_env_file() -> bool:
    """
    Load environment variables from the repo-root .env file.

    We implement a lightweight loader here to avoid depending on python-dotenv
    inside the shared LLM client stack.
    """
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return False

    try:
        with env_path.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()
        return True
    except Exception:
        # Best-effort load; ignore parsing errors
        return False


def _env_secret(var_name: str) -> Optional[str]:
    """Return a non-empty environment secret without exposing its value."""
    value = os.getenv(var_name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_bool(var_name: str, default: bool) -> bool:
    value = os.getenv(var_name)
    if value is None:
        return default
    return _coerce_bool(value)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
