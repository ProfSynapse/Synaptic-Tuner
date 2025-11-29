#!/usr/bin/env python3
"""
Universal model upload CLI for all trainers.

This is the main entry point for uploading models to HuggingFace Hub.
It uses the shared upload framework and works with any trainer.

Usage:
    python upload_model.py ./path/to/model username/model-name [options]

Examples:
    # Basic upload with 16-bit merge
    python upload_model.py ../rtx3090_sft/sft_output_rtx3090/20251122/final_model username/my-model

    # Upload with GGUF creation
    python upload_model.py ./model username/my-model --create-gguf

    # LoRA-only upload (fastest)
    python upload_model.py ./model username/my-model --save-method lora
"""

import sys
from pathlib import Path

# Add shared module to path
SHARED_PATH = Path(__file__).parent.parent / "shared"
sys.path.insert(0, str(SHARED_PATH))

from upload.cli.upload_cli import main

if __name__ == "__main__":
    sys.exit(main())
