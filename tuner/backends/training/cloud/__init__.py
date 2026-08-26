"""Import-light compatibility facade for optional cloud backends."""

from __future__ import annotations

from importlib import import_module

_BACKEND_TARGETS = {
    "hf_jobs": (".hf_jobs_backend", "HFJobsBackend"),
    "runpod": (".runpod_backend", "RunPodBackend"),
}
_EXPORT_TARGETS = {
    "HFJobsBackend": _BACKEND_TARGETS["hf_jobs"],
    "RunPodBackend": _BACKEND_TARGETS["runpod"],
}

__all__ = ["AVAILABLE_BACKENDS", "HFJobsBackend", "RunPodBackend"]


def _resolve_backend(name: str):
    module_name, attribute = _EXPORT_TARGETS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __getattr__(name: str):
    if name in _EXPORT_TARGETS:
        return _resolve_backend(name)
    if name == "AVAILABLE_BACKENDS":
        available = {}
        for backend_id, (_module_name, attribute) in _BACKEND_TARGETS.items():
            try:
                backend = _resolve_backend(attribute)
            except ImportError:
                continue
            if backend is not None:
                available[backend_id] = backend
        globals()[name] = available
        return available
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
