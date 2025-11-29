"""
Model loading abstractions.

Provides a unified interface for loading models with different frameworks.
"""

from .base import BaseModelLoader
from .unsloth_loader import UnslothModelLoader
from .registry import ModelLoaderRegistry

# Register built-in loaders
ModelLoaderRegistry.register("unsloth", UnslothModelLoader)

__all__ = [
    "BaseModelLoader",
    "UnslothModelLoader",
    "ModelLoaderRegistry",
]
