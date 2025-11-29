"""
Environment variable utilities.
"""

import os
from pathlib import Path
from typing import Optional


def load_env_file(paths: list = None) -> bool:
    """
    Load environment variables from .env file.

    Args:
        paths: List of paths to check for .env file

    Returns:
        True if .env file was loaded, False otherwise
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False

    if paths is None:
        paths = [
            Path.cwd() / ".env",
            Path.cwd().parent / ".env",
            Path.cwd().parent.parent / ".env",
            Path(__file__).parent.parent.parent.parent / ".env",
        ]

    for path in paths:
        if path.exists():
            load_dotenv(path)
            return True

    return False


def get_env_var(name: str, default: str = None, required: bool = False) -> Optional[str]:
    """
    Get environment variable with optional default and requirement check.

    Args:
        name: Environment variable name
        default: Default value if not found
        required: Whether to raise error if not found

    Returns:
        Environment variable value

    Raises:
        ValueError: If required=True and variable not found
    """
    value = os.environ.get(name, default)

    if required and value is None:
        raise ValueError(f"Required environment variable not set: {name}")

    return value


def get_hf_token() -> Optional[str]:
    """
    Get HuggingFace token from environment.

    Checks both HF_TOKEN and HF_API_KEY.

    Returns:
        HuggingFace token or None
    """
    return os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
