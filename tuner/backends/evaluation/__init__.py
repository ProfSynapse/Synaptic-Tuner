"""
Evaluation backend implementations.

Location: tuner/backends/evaluation/__init__.py
Purpose: Export evaluation backend implementations
Used by: EvaluationBackendRegistry, eval_handler

This module provides backends for evaluating models via local inference:
- OllamaBackend: Ollama CLI-based model listing and inference
- LMStudioBackend: LM Studio HTTP API-based model listing and inference
- LlamaCppBackend: llama.cpp direct GGUF execution via llama-cli
"""

from .base import IEvaluationBackend
from .ollama_backend import OllamaBackend
from .lmstudio_backend import LMStudioBackend
from .llamacpp_backend import LlamaCppBackend

__all__ = [
    "IEvaluationBackend",
    "OllamaBackend",
    "LMStudioBackend",
    "LlamaCppBackend",
]
