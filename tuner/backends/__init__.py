"""
Backend abstractions for Synaptic Tuner.

Location: /mnt/f/Code/Toolset-Training/tuner/backends/__init__.py
Purpose: Export backend implementations and registries

Usage:
    from tuner.backends import RTXBackend, MacBackend
    from tuner.backends import TrainingBackendRegistry, EvaluationBackendRegistry
"""

from .training import (
    ITrainingBackend,
    RTXBackend,
    MacBackend,
)
from .evaluation import (
    IEvaluationBackend,
    OllamaBackend,
    LMStudioBackend,
)
from .registry import (
    TrainingBackendRegistry,
    EvaluationBackendRegistry,
)

__all__ = [
    # Training backends
    "ITrainingBackend",
    "RTXBackend",
    "MacBackend",
    # Evaluation backends
    "IEvaluationBackend",
    "OllamaBackend",
    "LMStudioBackend",
    # Registries
    "TrainingBackendRegistry",
    "EvaluationBackendRegistry",
]
