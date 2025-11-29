"""
Core abstractions for the upload framework.
"""

from .types import ModelPath, RepositoryId, Credential
from .config import UploadConfig, SaveConfig, ConversionConfig, DocumentationConfig
from .exceptions import (
    UploadError,
    SaveError,
    ConversionError,
    ValidationError,
    GPUMemoryError,
)
from .interfaces import (
    ISaveStrategy,
    IConverter,
    IUploader,
    IModelLoader,
    IDocumentationGenerator,
)

__all__ = [
    # Types
    "ModelPath",
    "RepositoryId",
    "Credential",
    # Config
    "UploadConfig",
    "SaveConfig",
    "ConversionConfig",
    "DocumentationConfig",
    # Exceptions
    "UploadError",
    "SaveError",
    "ConversionError",
    "ValidationError",
    "GPUMemoryError",
    # Interfaces
    "ISaveStrategy",
    "IConverter",
    "IUploader",
    "IModelLoader",
    "IDocumentationGenerator",
]
