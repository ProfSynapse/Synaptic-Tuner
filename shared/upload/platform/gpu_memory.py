"""
GPU memory management utilities.

Provides functions for checking, clearing, and managing GPU memory
during model upload operations.
"""

import gc
from typing import Tuple

from ..core.exceptions import GPUMemoryError


def get_gpu_memory_info() -> Tuple[float, float, float]:
    """
    Get GPU memory information in GB.

    Returns:
        Tuple of (used_gb, total_gb, free_gb)
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return (0.0, 0.0, 0.0)

        # Get memory info in bytes
        total = torch.cuda.get_device_properties(0).total_memory
        reserved = torch.cuda.memory_reserved(0)

        # Free memory is total minus reserved (reserved includes allocated + cached)
        free = total - reserved
        used = reserved

        # Convert to GB
        return (used / 1024**3, total / 1024**3, free / 1024**3)
    except ImportError:
        return (0.0, 0.0, 0.0)
    except Exception:
        return (0.0, 0.0, 0.0)


def clear_gpu_cache() -> float:
    """
    Clear GPU cache and return freed memory in GB.

    Returns:
        Amount of memory freed in GB
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0

        before_used, _, _ = get_gpu_memory_info()

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache
        torch.cuda.empty_cache()

        # Synchronize to ensure cache is cleared
        torch.cuda.synchronize()

        after_used, _, _ = get_gpu_memory_info()
        freed = before_used - after_used

        return max(0.0, freed)
    except ImportError:
        return 0.0
    except Exception:
        return 0.0


def ensure_gpu_memory(
    required_gb: float,
    operation_name: str = "operation",
    auto_clear: bool = True,
    raise_on_failure: bool = False
) -> bool:
    """
    Ensure sufficient GPU memory is available for an operation.

    Args:
        required_gb: Required GPU memory in GB
        operation_name: Name of the operation (for logging)
        auto_clear: Whether to automatically clear cache if needed
        raise_on_failure: Whether to raise GPUMemoryError on failure

    Returns:
        True if sufficient memory is available, False otherwise

    Raises:
        GPUMemoryError: If raise_on_failure=True and memory is insufficient
    """
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠ No CUDA GPU available")
            if raise_on_failure:
                raise GPUMemoryError(required_gb, 0.0, operation_name)
            return False
    except ImportError:
        print("⚠ PyTorch not available")
        return False

    used_gb, total_gb, free_gb = get_gpu_memory_info()

    print(f"\n{'='*60}")
    print("GPU MEMORY CHECK")
    print(f"{'='*60}")
    print(f"Operation: {operation_name}")
    print(f"Required: ~{required_gb:.1f} GB")
    print(f"Available: {free_gb:.1f} GB / {total_gb:.1f} GB total")
    print(f"Currently used: {used_gb:.1f} GB")

    if free_gb >= required_gb:
        print("✓ Sufficient GPU memory available")
        return True

    if auto_clear:
        print(f"\n⚠ Insufficient memory ({free_gb:.1f} GB < {required_gb:.1f} GB required)")
        print("Attempting to clear GPU cache...")

        freed_gb = clear_gpu_cache()

        # Recheck after clearing
        used_gb, total_gb, free_gb = get_gpu_memory_info()
        print(f"Freed: {freed_gb:.1f} GB")
        print(f"Available after clearing: {free_gb:.1f} GB")

        if free_gb >= required_gb:
            print("✓ Sufficient GPU memory now available")
            return True

    # Still not enough
    shortfall = required_gb - free_gb
    print(f"\n✗ Still insufficient GPU memory")
    print(f"  Need {shortfall:.1f} GB more")
    print("\nSuggestions:")
    print("  1. Close other applications using GPU (check: nvidia-smi)")
    print("  2. Restart your terminal/Python session")
    print("  3. Use --save-method lora (no GPU needed for upload)")
    print("  4. Run from a fresh terminal session")

    if raise_on_failure:
        raise GPUMemoryError(required_gb, free_gb, operation_name)

    return False


def get_required_memory(model_size: str, operation: str) -> float:
    """
    Get required GPU memory for a specific operation.

    Args:
        model_size: Model size (e.g., "3b", "7b", "13b", "20b")
        operation: Operation type (e.g., "merge_16bit", "merge_4bit", "gguf")

    Returns:
        Required GPU memory in GB
    """
    key = f"{model_size}_{operation}"
    return GPU_MEMORY_REQUIREMENTS.get(key, GPU_MEMORY_REQUIREMENTS.get("default_merge", 14.0))


# Estimated GPU memory requirements (in GB) for different operations
GPU_MEMORY_REQUIREMENTS = {
    # 3B model
    "3b_merge_16bit": 7.0,
    "3b_merge_4bit": 4.0,
    "3b_gguf": 7.0,
    # 7B model
    "7b_merge_16bit": 14.0,
    "7b_merge_4bit": 8.0,
    "7b_gguf": 14.0,
    # 13B model
    "13b_merge_16bit": 26.0,
    "13b_merge_4bit": 14.0,
    "13b_gguf": 26.0,
    # 20B model
    "20b_merge_16bit": 40.0,
    "20b_merge_4bit": 20.0,
    "20b_gguf": 40.0,
    # Defaults (assume 7B)
    "default_merge": 14.0,
    "default_gguf": 14.0,
}
