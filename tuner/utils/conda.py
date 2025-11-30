"""
Conda environment utilities.

Location: tuner/utils/conda.py
Purpose: Find Python interpreter from conda environment
Used by: Handlers for executing training/upload scripts
"""

import os
import sys
from pathlib import Path
from .environment import detect_environment


# Standard Unsloth environment - always use unsloth_latest
UNSLOTH_ENV = "unsloth_latest"
UNSLOTH_VERSION = "2025.11.4"


def get_conda_python() -> str:
    """
    Find Python interpreter from unsloth_latest conda environment.

    Searches common conda installation paths based on detected environment.
    Falls back to current Python interpreter if conda env not found.

    Returns:
        str: Path to Python interpreter

    Platform-specific search paths:
        Windows:
            - %USERPROFILE%/miniconda3/envs/unsloth_latest/python.exe
            - %USERPROFILE%/anaconda3/envs/unsloth_latest/python.exe
            - C:/ProgramData/miniconda3/envs/unsloth_latest/python.exe
            - C:/ProgramData/anaconda3/envs/unsloth_latest/python.exe

        Linux/WSL:
            - ~/.conda/envs/unsloth_latest/bin/python
            - ~/miniconda3/envs/unsloth_latest/bin/python
            - ~/anaconda3/envs/unsloth_latest/bin/python
            - /opt/conda/envs/unsloth_latest/bin/python

    Example:
        >>> python = get_conda_python()
        >>> subprocess.run([python, "train_sft.py"])
    """
    env = detect_environment()

    if env == "windows":
        paths = [
            Path(os.environ.get("USERPROFILE", "")) / "miniconda3" / "envs" / UNSLOTH_ENV / "python.exe",
            Path(os.environ.get("USERPROFILE", "")) / "anaconda3" / "envs" / UNSLOTH_ENV / "python.exe",
            Path("C:/ProgramData/miniconda3/envs") / UNSLOTH_ENV / "python.exe",
            Path("C:/ProgramData/anaconda3/envs") / UNSLOTH_ENV / "python.exe",
        ]
    else:
        # Linux and WSL use same paths
        paths = [
            Path.home() / ".conda" / "envs" / UNSLOTH_ENV / "bin" / "python",
            Path.home() / "miniconda3" / "envs" / UNSLOTH_ENV / "bin" / "python",
            Path.home() / "anaconda3" / "envs" / UNSLOTH_ENV / "bin" / "python",
            Path("/opt/conda/envs") / UNSLOTH_ENV / "bin" / "python",
        ]

    # Find first existing path
    for p in paths:
        if p.exists():
            return str(p)

    # Environment not found - warn and fallback
    print(f"Warning: Unsloth environment '{UNSLOTH_ENV}' not found!")
    print(f"Falling back to current Python: {sys.executable}")
    print("For GPU operations, please run setup first:")
    print("  cd Trainers && source activate_unsloth_latest.sh")
    return sys.executable
