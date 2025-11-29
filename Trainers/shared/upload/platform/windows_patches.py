"""
Windows compatibility patches.

Applies necessary patches to make the upload system work on Windows,
including fixes for dataclasses, torch.compile, and torch._inductor.
"""

import os
import sys

_patches_applied = False


def is_windows() -> bool:
    """Check if running on Windows."""
    return sys.platform == 'win32'


def apply_windows_patches() -> bool:
    """
    Apply Windows compatibility patches.

    Should be called BEFORE importing unsloth or other problematic libraries.

    Returns:
        True if patches were applied, False if already applied or not on Windows
    """
    global _patches_applied

    if _patches_applied:
        return False

    if not is_windows():
        return False

    print("Applying Windows compatibility patches...")

    # Patch 1: Wrap fields() for non-dataclasses
    # Some libraries call fields() on non-dataclass objects
    try:
        from dataclasses import fields
        import dataclasses

        original_fields = fields

        def patched_fields(class_or_instance):
            try:
                return original_fields(class_or_instance)
            except TypeError:
                return ()

        dataclasses.fields = patched_fields
    except ImportError:
        pass

    # Patch 2: Disable torch.compile (not fully supported on Windows)
    os.environ['PYTORCH_JIT'] = '0'
    os.environ['TORCH_COMPILE_DISABLE'] = '1'

    # Patch 3: Pre-patch torch._inductor
    # This prevents errors when torch tries to access attr_desc_fields
    try:
        import torch._inductor.runtime.hints
        if not hasattr(torch._inductor.runtime.hints, 'attr_desc_fields'):
            torch._inductor.runtime.hints.attr_desc_fields = set()
    except (ImportError, ModuleNotFoundError):
        pass

    _patches_applied = True
    print("✓ Windows patches applied")

    return True


def ensure_windows_compatibility():
    """
    Ensure Windows compatibility by applying patches if needed.

    This is a convenience function that can be called at any point.
    Patches are only applied once regardless of how many times this is called.
    """
    if is_windows():
        apply_windows_patches()
