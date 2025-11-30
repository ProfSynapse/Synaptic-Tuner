"""
Core module - Interfaces, configuration models, and exceptions.

This module defines the fundamental abstractions and data structures used throughout
the tuner package. It provides:

- interfaces.py: Abstract base classes for backends, handlers, and discovery services
- config.py: Configuration dataclasses for training, upload, evaluation
- exceptions.py: Custom exception hierarchy for error handling

Usage:
    from tuner.core import ITrainingBackend, TrainingConfig, ConfigurationError
"""

from __future__ import annotations

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
