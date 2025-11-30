"""
Input validation utilities.

Location: tuner/utils/validation.py
Purpose: Validate user inputs (repo IDs, paths, etc.)
Used by: Handlers for input validation
"""

import os
from pathlib import Path


def validate_repo_id(repo_id: str) -> bool:
    """
    Validate HuggingFace repository ID format.

    A valid repo ID must be in the format "username/model-name".

    Args:
        repo_id: Repository ID to validate

    Returns:
        bool: True if valid format, False otherwise

    Example:
        >>> validate_repo_id("profsynapse/claudesidian-mcp")
        True
        >>> validate_repo_id("invalid-format")
        False
    """
    if not repo_id or not isinstance(repo_id, str):
        return False

    parts = repo_id.split("/")
    if len(parts) != 2:
        return False

    username, model_name = parts
    return bool(username.strip() and model_name.strip())


def validate_path_exists(path: Path) -> bool:
    """
    Check if a path exists.

    Args:
        path: Path to validate

    Returns:
        bool: True if path exists, False otherwise

    Example:
        >>> validate_path_exists(Path("/mnt/f/Code/Toolset-Training"))
        True
        >>> validate_path_exists(Path("/nonexistent/path"))
        False
    """
    if not path:
        return False
    return path.exists()


def load_env_file(env_path: Path) -> bool:
    """
    Load environment variables from .env file.

    Parses .env file and loads variables into os.environ.
    Skips comments (lines starting with #) and empty lines.

    Args:
        env_path: Path to .env file

    Returns:
        bool: True if file was loaded successfully, False if file doesn't exist

    Format:
        KEY=value
        # Comments are ignored
        ANOTHER_KEY=another value

    Example:
        >>> load_env_file(Path(".env"))
        True
        >>> os.environ.get("HF_TOKEN")
        'hf_your_token_here'
    """
    if not env_path.exists():
        return False

    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue
                # Parse KEY=value
                if "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()
        return True
    except Exception:
        return False
