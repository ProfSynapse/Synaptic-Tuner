"""
Platform-specific utilities for cross-platform compatibility.
"""

from .gpu_memory import (
    get_gpu_memory_info,
    clear_gpu_cache,
    ensure_gpu_memory,
    GPU_MEMORY_REQUIREMENTS,
)
from .windows_patches import apply_windows_patches, is_windows
from .filesystem import (
    cleanup_temp_directory,
    is_windows_filesystem,
    get_native_temp_dir,
    copy_to_native_filesystem,
)

__all__ = [
    # GPU
    "get_gpu_memory_info",
    "clear_gpu_cache",
    "ensure_gpu_memory",
    "GPU_MEMORY_REQUIREMENTS",
    # Windows
    "apply_windows_patches",
    "is_windows",
    # Filesystem
    "cleanup_temp_directory",
    "is_windows_filesystem",
    "get_native_temp_dir",
    "copy_to_native_filesystem",
]
