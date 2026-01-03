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

from tuner.handlers.train_handler import TrainHandler
from tuner.handlers.upload_handler import UploadHandler
from tuner.handlers.eval_handler import EvalHandler
from tuner.handlers.pipeline_handler import PipelineHandler
from tuner.handlers.main_menu_handler import MainMenuHandler
from tuner.handlers.synthchat_handler import SynthChatHandler
from tuner.handlers.status_handler import StatusHandler
from tuner.handlers.doctor_handler import DoctorHandler

__all__ = [
    "TrainHandler",
    "UploadHandler",
    "EvalHandler",
    "PipelineHandler",
    "MainMenuHandler",
    "SynthChatHandler",
    "StatusHandler",
    "DoctorHandler",
]
