"""
Model loading abstractions.

Provides a unified interface for loading models with different frameworks.
"""

# Import merge utilities first (self-contained, no circular deps)
from .merge import (
    is_lora_checkpoint,
    is_merged_model,
    get_base_model_name,
    find_merged_for_run,
    merge_lora_checkpoint,
    find_or_create_merged,
    resolve_model_path,
)

# These have circular import issues with shared.upload - import lazily
def _get_loader_classes():
    """Lazy import to avoid circular dependency with shared.upload."""
    from .base import BaseModelLoader
    from .unsloth_loader import UnslothModelLoader
    from .registry import ModelLoaderRegistry
    return BaseModelLoader, UnslothModelLoader, ModelLoaderRegistry


def __getattr__(name):
    """Lazy attribute access for loader classes."""
    if name in ("BaseModelLoader", "UnslothModelLoader", "ModelLoaderRegistry"):
        BaseModelLoader, UnslothModelLoader, ModelLoaderRegistry = _get_loader_classes()
        # Register built-in loaders on first access
        ModelLoaderRegistry.register("unsloth", UnslothModelLoader)
        globals()["BaseModelLoader"] = BaseModelLoader
        globals()["UnslothModelLoader"] = UnslothModelLoader
        globals()["ModelLoaderRegistry"] = ModelLoaderRegistry
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Merge utilities (always available)
    "is_lora_checkpoint",
    "is_merged_model",
    "get_base_model_name",
    "find_merged_for_run",
    "merge_lora_checkpoint",
    "find_or_create_merged",
    "resolve_model_path",
    # Loader classes (lazy import)
    "BaseModelLoader",
    "UnslothModelLoader",
    "ModelLoaderRegistry",
]
