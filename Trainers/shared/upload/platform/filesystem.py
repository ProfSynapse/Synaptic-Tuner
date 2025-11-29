"""
Filesystem utilities for cross-platform compatibility.

Handles differences between Windows, WSL, and native Linux/macOS filesystems.
"""

import gc
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple


def is_windows_filesystem(path: Path) -> bool:
    """
    Check if a path is on a Windows filesystem (mounted via WSL).

    Windows filesystems under WSL have known I/O performance issues
    and file locking problems.

    Args:
        path: Path to check

    Returns:
        True if path is on Windows filesystem (e.g., /mnt/c, /mnt/d)
    """
    resolved = Path(path).resolve()
    path_str = str(resolved)
    return path_str.startswith('/mnt/')


def get_native_temp_dir() -> Path:
    """
    Get a native (non-Windows) temporary directory.

    On WSL, this returns ~/tmp to avoid Windows filesystem I/O issues.
    On native Linux/macOS, this returns the system temp directory.

    Returns:
        Path to native temporary directory
    """
    # Check if we're in WSL
    if Path('/mnt/c').exists():
        # We're in WSL - use home directory
        temp_dir = Path.home() / 'tmp'
    else:
        # Native Linux/macOS - use system temp
        temp_dir = Path(tempfile.gettempdir())

    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def copy_to_native_filesystem(
    source_path: Path,
    prefix: str = "model_"
) -> Tuple[Path, bool]:
    """
    Copy a directory to the native filesystem if needed.

    If the source is already on native filesystem, returns it unchanged.
    If on Windows filesystem (WSL), copies to ~/tmp for better I/O.

    Args:
        source_path: Path to copy
        prefix: Prefix for temporary directory name

    Returns:
        Tuple of (working_path, was_copied)
        - working_path: Path to use for operations
        - was_copied: Whether a copy was made (caller should clean up)
    """
    source_path = Path(source_path).resolve()

    if not is_windows_filesystem(source_path):
        return source_path, False

    print("⚠ Detected Windows filesystem - copying to WSL native filesystem...")

    # Create temp directory
    temp_base = get_native_temp_dir()
    temp_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=str(temp_base)))
    dest_path = temp_dir / source_path.name

    # Copy
    shutil.copytree(str(source_path), str(dest_path))
    print(f"✓ Copied to: {dest_path}")

    return dest_path, True


def cleanup_temp_directory(
    temp_dir: Path,
    max_retries: int = 3,
    delay: float = 2.0,
    silent: bool = False
) -> bool:
    """
    Clean up temporary directory with retries for Windows file locks.

    Args:
        temp_dir: Path to temporary directory to remove
        max_retries: Maximum number of retry attempts
        delay: Seconds to wait between retries
        silent: Whether to suppress output

    Returns:
        True if cleanup succeeded, False otherwise
    """
    if not temp_dir.exists():
        return True

    for attempt in range(max_retries):
        try:
            # Force garbage collection to release file handles
            gc.collect()

            if attempt > 0:
                if not silent:
                    print(f"  Retry {attempt}/{max_retries - 1}...")
                time.sleep(delay)

            shutil.rmtree(temp_dir)
            if not silent:
                print("✓ Temporary files removed")
            return True

        except PermissionError:
            if attempt < max_retries - 1:
                continue
            # Final attempt failed
            if not silent:
                print(f"⚠ Could not remove temporary files (Windows file lock): {temp_dir}")
                print("  You can manually delete this directory later")
                print("  Tip: Run from WSL bash instead of PowerShell to avoid this issue")
            return False

        except Exception as e:
            if not silent:
                print(f"⚠ Error cleaning up: {e}")
            return False

    return False


def ensure_directory(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        The path (for chaining)
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_size_gb(path: Path) -> float:
    """
    Get total size of a file or directory in GB.

    Args:
        path: Path to file or directory

    Returns:
        Size in GB
    """
    if not path.exists():
        return 0.0

    if path.is_file():
        return path.stat().st_size / (1024**3)

    total = 0
    for item in path.rglob('*'):
        if item.is_file():
            total += item.stat().st_size

    return total / (1024**3)
