"""Exceptions for shared LLM client system."""


class LLMError(Exception):
    """Base exception for LLM client errors."""
    pass


class LLMConnectionError(LLMError):
    """Raised when unable to connect to LLM backend."""
    pass


class LLMResponseError(LLMError):
    """Raised when LLM returns invalid or unexpected response."""
    pass


class LLMConfigError(LLMError):
    """Raised when LLM client configuration is invalid."""
    pass
