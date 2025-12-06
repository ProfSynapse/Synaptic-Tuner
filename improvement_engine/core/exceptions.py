"""Custom exceptions for improvement engine."""


class ImprovementEngineError(Exception):
    """Base exception for improvement engine errors."""
    pass


class ValidationError(ImprovementEngineError):
    """Raised when validation fails."""
    pass


class LLMServiceError(ImprovementEngineError):
    """Raised when LLM service encounters an error."""
    pass


class FileHandlerError(ImprovementEngineError):
    """Raised when file operations fail."""
    pass


class ConfigurationError(ImprovementEngineError):
    """Raised when configuration is invalid."""
    pass
