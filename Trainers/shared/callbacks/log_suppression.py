"""Noisy-logger suppression helper shared by trainers (currently used by SFT)."""

from __future__ import annotations

import logging
import sys
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

try:
    from shared.ui import suppress_logs
    _SUPPRESS_AVAILABLE = True
except ImportError:
    _SUPPRESS_AVAILABLE = False


_NOISY_LOGGERS = [
    "unsloth",
    "transformers",
    "datasets",
    "accelerate",
    "trl",
    "peft",
    "bitsandbytes",
    "torch",
    "huggingface_hub",
]


def suppress_training_logs():
    """Context manager that quiets common noisy training loggers; no-op if unavailable."""
    if _SUPPRESS_AVAILABLE:
        return suppress_logs(_NOISY_LOGGERS, level=logging.WARNING)
    return nullcontext()
