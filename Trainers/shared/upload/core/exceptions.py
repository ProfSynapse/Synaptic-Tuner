"""
Custom exceptions for the upload framework.

Provides specific exception types for different failure modes,
enabling precise error handling and user-friendly error messages.
"""


class UploadError(Exception):
    """Base exception for all upload-related errors."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self):
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


class SaveError(UploadError):
    """Error during model saving."""
    pass


class ConversionError(UploadError):
    """Error during format conversion (e.g., GGUF)."""
    pass


class ValidationError(UploadError):
    """Error during input validation."""
    pass


class GPUMemoryError(UploadError):
    """Insufficient GPU memory for operation."""

    def __init__(self, required_gb: float, available_gb: float, operation: str = ""):
        message = f"Insufficient GPU memory for {operation}: need {required_gb:.1f}GB, have {available_gb:.1f}GB"
        super().__init__(
            message,
            details={
                "required_gb": required_gb,
                "available_gb": available_gb,
                "operation": operation,
            }
        )
        self.required_gb = required_gb
        self.available_gb = available_gb
        self.operation = operation


class NetworkError(UploadError):
    """Error during network operations (upload, download)."""
    pass


class AuthenticationError(UploadError):
    """Error during authentication (invalid token, etc.)."""
    pass


class DependencyError(UploadError):
    """Missing or incompatible dependency."""

    def __init__(self, dependency: str, message: str = ""):
        full_message = f"Missing dependency: {dependency}"
        if message:
            full_message = f"{full_message}. {message}"
        super().__init__(full_message, details={"dependency": dependency})
        self.dependency = dependency
