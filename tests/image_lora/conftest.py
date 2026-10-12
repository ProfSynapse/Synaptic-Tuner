"""Shared setup for the image_lora method tests (CPU only, no Modal SDK, no network).

The trainer's modules live in Trainers/image_lora/src and are imported as
top-level modules, the same way train_image_lora.py imports them.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "Trainers" / "image_lora" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
