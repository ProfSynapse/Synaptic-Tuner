"""Lazy backend registries with metadata-only discovery."""

from __future__ import annotations

from importlib import import_module
from typing import Dict, List, Type

from tuner.core.interfaces import IEvaluationBackend, ITrainingBackend

_BackendEntry = str | type


def _resolve_target(target: str) -> type:
    module_name, separator, attribute = target.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(f"Invalid backend target: {target!r}")
    return getattr(import_module(module_name), attribute)


class TrainingBackendRegistry:
    """Ordered registry that imports only the selected training backend."""

    _backends: Dict[str, _BackendEntry] = {
        "rtx": "tuner.backends.training.rtx_backend:RTXBackend",
        "mac": "tuner.backends.training.mac_backend:MacBackend",
        "hf_jobs": "tuner.backends.training.cloud.hf_jobs_backend:HFJobsBackend",
        "runpod": "tuner.backends.training.cloud.runpod_backend:RunPodBackend",
    }
    _resolved: Dict[str, Type[ITrainingBackend]] = {}

    @classmethod
    def register(cls, name: str, backend: Type[ITrainingBackend]) -> None:
        cls._backends[name] = backend
        cls._resolved.pop(name, None)

    @classmethod
    def get(cls, name: str, **kwargs) -> ITrainingBackend:
        if name not in cls._backends:
            available = ", ".join(cls.list())
            raise ValueError(
                f"Unknown training backend: '{name}'. Available backends: {available}"
            )
        backend_type = cls._resolved.get(name)
        if backend_type is None:
            entry = cls._backends[name]
            backend_type = _resolve_target(entry) if isinstance(entry, str) else entry
            cls._resolved[name] = backend_type
        return backend_type(**kwargs)

    @classmethod
    def list(cls) -> List[str]:
        return list(cls._backends)


class EvaluationBackendRegistry:
    """Ordered registry that imports only the selected evaluation backend."""

    _backends: Dict[str, _BackendEntry] = {
        "ollama": "tuner.backends.evaluation.ollama_backend:OllamaBackend",
        "lmstudio": "tuner.backends.evaluation.lmstudio_backend:LMStudioBackend",
        "llamacpp": "tuner.backends.evaluation.llamacpp_backend:LlamaCppBackend",
        "unsloth": "tuner.backends.evaluation.unsloth_backend:UnslothBackend",
        "mlc": "tuner.backends.evaluation.mlc_backend:MLCBackend",
    }
    _resolved: Dict[str, Type[IEvaluationBackend]] = {}

    @classmethod
    def register(cls, name: str, backend: Type[IEvaluationBackend]) -> None:
        cls._backends[name] = backend
        cls._resolved.pop(name, None)

    @classmethod
    def get(cls, name: str, **kwargs) -> IEvaluationBackend:
        if name not in cls._backends:
            available = ", ".join(cls.list())
            raise ValueError(
                f"Unknown evaluation backend: '{name}'. Available backends: {available}"
            )
        backend_type = cls._resolved.get(name)
        if backend_type is None:
            entry = cls._backends[name]
            backend_type = _resolve_target(entry) if isinstance(entry, str) else entry
            cls._resolved[name] = backend_type
        return backend_type(**kwargs)

    @classmethod
    def list(cls) -> List[str]:
        return list(cls._backends)


__all__ = ["TrainingBackendRegistry", "EvaluationBackendRegistry"]
