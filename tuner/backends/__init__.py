"""Import-light compatibility facade for backend abstractions."""

from __future__ import annotations

from importlib import import_module

_EXPORT_MODULES = {
    "ITrainingBackend": ".training.base",
    "RTXBackend": ".training.rtx_backend",
    "MacBackend": ".training.mac_backend",
    "IEvaluationBackend": ".evaluation.base",
    "OllamaBackend": ".evaluation.ollama_backend",
    "LMStudioBackend": ".evaluation.lmstudio_backend",
    "TrainingBackendRegistry": ".registry",
    "EvaluationBackendRegistry": ".registry",
}

__all__ = [
    "ITrainingBackend", "RTXBackend", "MacBackend",
    "IEvaluationBackend", "OllamaBackend", "LMStudioBackend",
    "TrainingBackendRegistry", "EvaluationBackendRegistry",
]


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORT_MODULES))
