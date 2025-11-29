"""
Uploaders for model repositories.

Each uploader handles uploading models to a specific platform.
"""

from .base import BaseUploader
from .huggingface import HuggingFaceUploader
from .registry import UploaderRegistry

# Register built-in uploaders
UploaderRegistry.register("huggingface", HuggingFaceUploader)

__all__ = [
    "BaseUploader",
    "HuggingFaceUploader",
    "UploaderRegistry",
]
