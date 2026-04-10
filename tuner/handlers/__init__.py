"""
Lazy exports for Synaptic Tuner command handlers.

Avoid eager imports here. Some handlers pull in heavyweight optional
dependencies, and importing the package itself should not force every
runtime path to load them.
"""

from __future__ import annotations

import importlib

_HANDLER_MODULES = {
    "TrainHandler": "tuner.handlers.train_handler",
    "UploadHandler": "tuner.handlers.upload_handler",
    "EvalHandler": "tuner.handlers.eval_handler",
    "PipelineHandler": "tuner.handlers.pipeline_handler",
    "MainMenuHandler": "tuner.handlers.main_menu_handler",
    "SynthChatHandler": "tuner.handlers.synthchat_handler",
    "StatusHandler": "tuner.handlers.status_handler",
    "DoctorHandler": "tuner.handlers.doctor_handler",
    "MLHandler": "tuner.handlers.ml_handler",
}

__all__ = list(_HANDLER_MODULES)


def __getattr__(name: str):
    module_name = _HANDLER_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name)
    return getattr(module, name)
