"""
Environment variable utilities.
"""

import os
from pathlib import Path
from typing import Iterable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from tuner.project.context import ProjectContext

_SECRET_NAME_PARTS = ("TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL")


def redact_env_value(name: str, value: str) -> str:
    """Return a display-safe environment value without exposing credentials."""

    upper_name = name.upper()
    if any(part in upper_name for part in _SECRET_NAME_PARTS):
        return "<redacted>" if value else ""
    return value


def load_env_file(
    paths: Iterable[Path | str] | Path | str | None = None,
    *,
    context: "ProjectContext | None" = None,
    explicit_path: Path | str | None = None,
) -> bool:
    """
    Load environment variables from .env file.

    Args:
        paths: Paths to check for an env file. A single path is accepted.
        context: Project context selecting exactly one host or engine ``.env``.
            Process environment variables always take precedence.
        explicit_path: Explicit env file. When provided, it is the only file
            considered and therefore has selection priority over context roots.

    Returns:
        True if .env file was loaded, False otherwise
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False

    if explicit_path is not None:
        candidates = [Path(explicit_path)]
    elif context is not None:
        selected_root = (
            context.project_root if context.mode == "host" else context.engine_root
        )
        candidates = [selected_root / ".env"]
    elif paths is None:
        # env.py is at shared/utilities/env.py
        # So .parent.parent.parent gets us to repo root
        repo_root = Path(__file__).parent.parent.parent
        candidates = [
            Path.cwd() / ".env",
            Path.cwd().parent / ".env",
            Path.cwd().parent.parent / ".env",
            repo_root / ".env",
        ]
    elif isinstance(paths, (str, Path)):
        candidates = [Path(paths)]
    else:
        candidates = [Path(path) for path in paths]

    for path in candidates:
        if path.exists():
            load_dotenv(path, override=False)
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
    for key in ("HF_TOKEN", "HF_API_KEY"):
        value = os.environ.get(key)
        if value is None:
            continue
        value = value.strip()
        if value:
            return value
    return None
