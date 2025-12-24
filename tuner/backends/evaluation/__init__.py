"""
Evaluation backend implementations.

Location: tuner/backends/evaluation/__init__.py
Purpose: Export evaluation backend implementations
Used by: EvaluationBackendRegistry, eval_handler

This module provides backends for evaluating models via local inference:
- OllamaBackend: Ollama CLI-based model listing and inference
- LMStudioBackend: LM Studio HTTP API-based model listing and inference
- LlamaCppBackend: llama.cpp direct GGUF execution via llama-cli
- UnslothBackend: Direct Unsloth/LoRA inference (loads adapter on base model)
"""

from .base import IEvaluationBackend
from .ollama_backend import OllamaBackend
from .lmstudio_backend import LMStudioBackend
from .llamacpp_backend import LlamaCppBackend
from .unsloth_backend import UnslothBackend

__all__ = [
    "IEvaluationBackend",
    "OllamaBackend",
    "LMStudioBackend",
    "LlamaCppBackend",
    "UnslothBackend",
]
