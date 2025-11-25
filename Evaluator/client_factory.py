"""Factory for creating backend clients.

This module provides a centralized way to create backend clients based on
configuration, following the Factory pattern. This enables:
- Single point of client instantiation
- Easy addition of new backends (Open/Closed Principle)
- Dependency injection for testing
- Type-safe client creation
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Type, Union

from .config import LMStudioSettings, OllamaSettings, VLLMSettings
from .enums import BackendType
from .lmstudio_client import LMStudioClient
from .ollama_client import OllamaClient
from .vllm_client import VLLMClient
from .protocols import BackendClient, BackendSettings

# Type alias for settings types
SettingsType = Union[OllamaSettings, LMStudioSettings, VLLMSettings]
ClientType = Union[OllamaClient, LMStudioClient, VLLMClient]


# Registry mapping backend types to their client and settings classes
_CLIENT_REGISTRY: Dict[BackendType, Type[BackendClient]] = {
    BackendType.OLLAMA: OllamaClient,
    BackendType.LMSTUDIO: LMStudioClient,
    BackendType.VLLM: VLLMClient,
}

_SETTINGS_REGISTRY: Dict[BackendType, Type[BackendSettings]] = {
    BackendType.OLLAMA: OllamaSettings,
    BackendType.LMSTUDIO: LMStudioSettings,
    BackendType.VLLM: VLLMSettings,
}


def create_client(
    backend: BackendType | str,
    settings: BackendSettings,
    timeout: float = 60.0,
    retries: int = 2,
) -> BackendClient:
    """Create a backend client for the specified backend type.

    Args:
        backend: The backend type (ollama, lmstudio, or BackendType enum)
        settings: Backend-specific settings object
        timeout: HTTP request timeout in seconds
        retries: Number of retry attempts

    Returns:
        Configured backend client

    Raises:
        ValueError: If backend type is not supported

    Example:
        settings = OllamaSettings(model="llama2")
        client = create_client(BackendType.OLLAMA, settings)
    """
    # Convert string to enum if needed
    if isinstance(backend, str):
        try:
            backend = BackendType(backend.lower())
        except ValueError:
            raise ValueError(
                f"Unknown backend type: {backend}. "
                f"Supported: {', '.join(b.value for b in BackendType)}"
            )

    client_class = _CLIENT_REGISTRY.get(backend)
    if client_class is None:
        raise ValueError(
            f"No client registered for backend: {backend}. "
            f"Supported: {', '.join(b.value for b in _CLIENT_REGISTRY)}"
        )

    return client_class(settings=settings, timeout=timeout, retries=retries)


def create_settings(
    backend: BackendType | str,
    model: str,
    host: Optional[str] = None,
    port: Optional[int] = None,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_tokens: int = 1024,
    seed: Optional[int] = None,
) -> BackendSettings:
    """Create backend settings for the specified backend type.

    This is a convenience function that creates the appropriate settings
    object with optional overrides for host/port.

    Args:
        backend: The backend type
        model: Model name/ID
        host: Optional host override (uses env var default if not specified)
        port: Optional port override (uses env var default if not specified)
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum output tokens
        seed: Optional random seed for reproducibility

    Returns:
        Configured settings object

    Example:
        settings = create_settings(
            BackendType.LMSTUDIO,
            model="local-model",
            temperature=0.5
        )
    """
    # Convert string to enum if needed
    if isinstance(backend, str):
        try:
            backend = BackendType(backend.lower())
        except ValueError:
            raise ValueError(
                f"Unknown backend type: {backend}. "
                f"Supported: {', '.join(b.value for b in BackendType)}"
            )

    settings_class = _SETTINGS_REGISTRY.get(backend)
    if settings_class is None:
        raise ValueError(
            f"No settings class registered for backend: {backend}. "
            f"Supported: {', '.join(b.value for b in _SETTINGS_REGISTRY)}"
        )

    # Build kwargs, only including overrides if specified
    kwargs: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "seed": seed,
    }
    if host is not None:
        kwargs["host"] = host
    if port is not None:
        kwargs["port"] = port

    return settings_class(**kwargs)


def create_client_from_args(
    backend: BackendType | str,
    model: str,
    host: Optional[str] = None,
    port: Optional[int] = None,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_tokens: int = 1024,
    seed: Optional[int] = None,
    timeout: float = 60.0,
    retries: int = 2,
) -> BackendClient:
    """Create a backend client from individual arguments.

    This is a convenience function that combines create_settings and
    create_client into a single call, useful for CLI applications.

    Args:
        backend: The backend type
        model: Model name/ID
        host: Optional host override
        port: Optional port override
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum output tokens
        seed: Optional random seed
        timeout: HTTP request timeout
        retries: Number of retry attempts

    Returns:
        Configured backend client

    Example:
        client = create_client_from_args(
            backend="lmstudio",
            model="local-model",
            temperature=0.5,
            timeout=120.0
        )
    """
    settings = create_settings(
        backend=backend,
        model=model,
        host=host,
        port=port,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
    )
    return create_client(
        backend=backend,
        settings=settings,
        timeout=timeout,
        retries=retries,
    )


def register_backend(
    backend_type: BackendType,
    client_class: Type[BackendClient],
    settings_class: Type[BackendSettings],
) -> None:
    """Register a new backend type.

    This allows extending the factory with custom backends without
    modifying the module (Open/Closed Principle).

    Args:
        backend_type: The backend type enum value
        client_class: The client class to use
        settings_class: The settings class to use

    Example:
        # Register a custom backend
        register_backend(
            BackendType.CUSTOM,
            CustomClient,
            CustomSettings
        )
    """
    _CLIENT_REGISTRY[backend_type] = client_class
    _SETTINGS_REGISTRY[backend_type] = settings_class


def get_supported_backends() -> list[BackendType]:
    """Return list of supported backend types.

    Returns:
        List of BackendType enum values
    """
    return list(_CLIENT_REGISTRY.keys())
