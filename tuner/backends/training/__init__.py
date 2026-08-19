"""Import-light compatibility facade for training backends."""

from __future__ import annotations

from importlib import import_module

_EXPORT_MODULES = {
    "ITrainingBackend": ".base",
    "RTXBackend": ".rtx_backend",
    "MacBackend": ".mac_backend",
    "HFJobsBackend": ".cloud",
    "ModalBackend": ".cloud",
    "RunPodBackend": ".cloud",
    "AVAILABLE_BACKENDS": ".cloud",
}

__all__ = [
    "ITrainingBackend", "RTXBackend", "MacBackend",
    "HFJobsBackend", "ModalBackend", "RunPodBackend", "AVAILABLE_BACKENDS",
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
