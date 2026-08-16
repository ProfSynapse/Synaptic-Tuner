"""
Synaptic Tuner - Unified CLI for training, uploading, and evaluating models.

This package provides a modular, maintainable architecture for the Synaptic Tuner CLI,
transforming the monolithic tuner.py into a well-structured application following SOLID principles.

Key components:
- cli: Command-line interface and routing
- core: Interfaces, configuration models, and exceptions
- handlers: Menu logic and workflow orchestration
- backends: Training and evaluation backend abstractions
- discovery: Resource discovery (training runs, checkpoints, models, prompt sets)
- ui: User interface components
- utils: Cross-cutting utilities

Entry points:
- python -m tuner
- python tuner.py (wrapper)
- ./run.sh (Bash wrapper)
- ./run.ps1 (PowerShell wrapper)
"""

from __future__ import annotations

from importlib import import_module

from synaptic_tuner._version import __version__
__author__ = "Synaptic Tuner Team"

_LAZY_EXPORTS = {
    "ITrainingBackend": ("tuner.core.interfaces", "ITrainingBackend"),
    "IEvaluationBackend": ("tuner.core.interfaces", "IEvaluationBackend"),
    "IHandler": ("tuner.core.interfaces", "IHandler"),
    "IDiscoveryService": ("tuner.core.interfaces", "IDiscoveryService"),
    "TrainingConfig": ("tuner.core.config", "TrainingConfig"),
    "CheckpointInfo": ("tuner.core.config", "CheckpointInfo"),
    "UploadConfig": ("tuner.core.config", "UploadConfig"),
    "EvalConfig": ("tuner.core.config", "EvalConfig"),
    "TunerError": ("tuner.core.exceptions", "TunerError"),
    "ConfigurationError": ("tuner.core.exceptions", "ConfigurationError"),
    "BackendError": ("tuner.core.exceptions", "BackendError"),
    "DiscoveryError": ("tuner.core.exceptions", "DiscoveryError"),
    "ValidationError": ("tuner.core.exceptions", "ValidationError"),
}


def __getattr__(name: str):
    """Resolve legacy convenience exports only when callers request them."""

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

__all__ = [
    # Version
    "__version__",
    "__author__",
    # Interfaces
    "ITrainingBackend",
    "IEvaluationBackend",
    "IHandler",
    "IDiscoveryService",
    # Configuration
    "TrainingConfig",
    "CheckpointInfo",
    "UploadConfig",
    "EvalConfig",
    # Exceptions
    "TunerError",
    "ConfigurationError",
    "BackendError",
    "DiscoveryError",
    "ValidationError",
]
