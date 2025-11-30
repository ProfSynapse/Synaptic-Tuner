"""
Evaluation backend implementations.

Location: /mnt/f/Code/Toolset-Training/tuner/backends/evaluation/__init__.py
Purpose: Export evaluation backend implementations
Used by: EvaluationBackendRegistry, eval_handler

This module provides backends for evaluating models via local inference servers:
- OllamaBackend: Ollama CLI-based model listing and inference
- LMStudioBackend: LM Studio HTTP API-based model listing and inference
"""

from .base import IEvaluationBackend
from .ollama_backend import OllamaBackend
from .lmstudio_backend import LMStudioBackend

__all__ = [
    "IEvaluationBackend",
    "OllamaBackend",
    "LMStudioBackend",
]
