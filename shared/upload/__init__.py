"""
Upload framework for model deployment.

Provides a modular, extensible architecture for:
- Saving models in various formats (LoRA, merged 16-bit, 4-bit)
- Converting to deployment formats (GGUF)
- Uploading to model hubs (HuggingFace)
- Generating documentation (manifests, model cards)
"""

from .core.config import UploadConfig, SaveConfig, ConversionConfig, DocumentationConfig


def __getattr__(name):
    """Lazy-load orchestrator to avoid circular imports with model_loading."""
    if name == "UploadOrchestrator":
        from .orchestrator import UploadOrchestrator
        globals()["UploadOrchestrator"] = UploadOrchestrator
        return UploadOrchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "UploadOrchestrator",
    "UploadConfig",
    "SaveConfig",
    "ConversionConfig",
    "DocumentationConfig",
]
