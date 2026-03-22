#!/usr/bin/env python3
"""
Universal model upload CLI for all trainers.

This is the main entry point for uploading models to HuggingFace Hub.
It uses the shared upload framework and works with any trainer.

Usage:
    python3 .skills/upload-deployment/scripts/upload_model.py ./path/to/model username/model-name [options]

Examples:
    # Basic upload with 16-bit merge
    python3 .skills/upload-deployment/scripts/upload_model.py ../sft/sft_output/20251122/final_model username/my-model

    # Upload with GGUF creation
    python3 .skills/upload-deployment/scripts/upload_model.py ./model username/my-model --create-gguf

    # LoRA-only upload (fastest)
    python3 .skills/upload-deployment/scripts/upload_model.py ./model username/my-model --save-method lora
"""

import sys
from pathlib import Path

# Add the repo root to sys.path so the shared package imports cleanly.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from shared.upload.cli.upload_cli import main

if __name__ == "__main__":
    sys.exit(main())
