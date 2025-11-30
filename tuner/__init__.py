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

__version__ = "1.0.0"
__author__ = "Synaptic Tuner Team"

# Export key classes for convenience
from tuner.core.interfaces import (
    ITrainingBackend,
    IEvaluationBackend,
    IHandler,
    IDiscoveryService,
)
from tuner.core.config import (
    TrainingConfig,
    CheckpointInfo,
    UploadConfig,
    EvalConfig,
)
from tuner.core.exceptions import (
    TunerError,
    ConfigurationError,
    BackendError,
    DiscoveryError,
    ValidationError,
)

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
