#!/usr/bin/env python3
"""Compatibility entrypoint for the packaged synthetic dataset validator."""
from __future__ import annotations

import sys
from pathlib import Path

# Direct script execution starts with this skill directory on sys.path.
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from shared.validation.dataset_validator import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
