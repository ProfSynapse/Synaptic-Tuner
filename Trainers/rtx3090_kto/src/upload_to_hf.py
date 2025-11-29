#!/usr/bin/env python3
"""
KTO model upload entry point.

This is a thin wrapper around the shared upload framework.
All upload logic is in Trainers/shared/upload/.

Usage:
    python src/upload_to_hf_new.py ./kto_output_rtx3090/20251122/final_model username/model-name [options]

For full options, run:
    python src/upload_to_hf_new.py --help
"""

import os
import sys
from pathlib import Path

# Add shared module to path
SHARED_PATH = Path(__file__).parent.parent.parent / "shared"
sys.path.insert(0, str(SHARED_PATH))

# Set default output directory for KTO trainer
os.environ.setdefault("TRAINER_TYPE", "kto")

from upload.cli.upload_cli import main

if __name__ == "__main__":
    sys.exit(main())
