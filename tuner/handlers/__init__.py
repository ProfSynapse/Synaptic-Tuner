"""
Command handlers for the Synaptic Tuner CLI.

This package contains handler implementations for different CLI commands:
- TrainHandler: Training workflow orchestration (STUB - to be implemented)
- UploadHandler: Model upload workflow (STUB - to be implemented)
- EvalHandler: Evaluation workflow
- PipelineHandler: Full pipeline (train -> upload -> eval)
- MainMenuHandler: Interactive main menu
- SynthChatHandler: Synthetic data generation and improvement
- StatusHandler: System status overview for AI assistants
- DoctorHandler: System diagnostics with recommendations and auto-fix

Each handler implements the IHandler interface and can be registered
with the router for command dispatching.
"""

from __future__ import annotations

from importlib import import_module

_LAZY_EXPORTS = {
    "TrainHandler": ("tuner.handlers.train_handler", "TrainHandler"),
    "UploadHandler": ("tuner.handlers.upload_handler", "UploadHandler"),
    "EvalHandler": ("tuner.handlers.eval_handler", "EvalHandler"),
    "PipelineHandler": ("tuner.handlers.pipeline_handler", "PipelineHandler"),
    "MainMenuHandler": ("tuner.handlers.main_menu_handler", "MainMenuHandler"),
    "SynthChatHandler": ("tuner.handlers.synthchat_handler", "SynthChatHandler"),
    "StatusHandler": ("tuner.handlers.status_handler", "StatusHandler"),
    "DoctorHandler": ("tuner.handlers.doctor_handler", "DoctorHandler"),
    "MLHandler": ("tuner.handlers.ml_handler", "MLHandler"),
}


def __getattr__(name: str):
    """Resolve legacy handler exports without importing unrelated handlers."""

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

__all__ = [
    "TrainHandler",
    "UploadHandler",
    "EvalHandler",
    "PipelineHandler",
    "MainMenuHandler",
    "SynthChatHandler",
    "StatusHandler",
    "DoctorHandler",
    "MLHandler",
]
