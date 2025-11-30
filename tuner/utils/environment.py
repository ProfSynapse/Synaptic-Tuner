"""
Environment detection utilities.

Location: tuner/utils/environment.py
Purpose: Detect runtime environment (WSL, Linux, or Windows)
Used by: Conda path finding, backend validation
"""

import sys
from pathlib import Path


def detect_environment() -> str:
    """
    Detect if running in WSL, native Linux, or Windows.

    Detection logic:
    1. Check sys.platform == "win32" → Windows
    2. Check /mnt/c exists → WSL
    3. Otherwise → Linux

    Returns:
        str: "windows", "wsl", or "linux"

    Example:
        >>> env = detect_environment()
        >>> if env == "wsl":
        ...     print("Running in WSL2")
    """
    if sys.platform == "win32":
        return "windows"
    elif Path("/mnt/c").exists():
        return "wsl"
    else:
        return "linux"
