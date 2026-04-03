"""
shared/env_bootstrap.py

Environment bootstrap for Unsloth-based trainers (SFT, KTO, GRPO).

Must be called early — before importing unsloth, torch, or transformers.
Consolidates duplicated boilerplate from train_sft.py, train_kto.py,
and train_grpo.py into a single init_trainer_env() call.
"""

import os
import sys


def init_trainer_env(
    *,
    disable_torch_compile: bool = True,
    apply_windows_patches: bool = True,
    load_dotenv: bool = True,
    suppress_transformers: bool = True,
    utf8_output: bool = True,
) -> None:
    """One-call environment bootstrap for all trainers.

    Call this at the top of your trainer script, BEFORE importing
    unsloth, torch, or transformers.

    Args:
        disable_torch_compile: Set TORCH_COMPILE_DISABLE, TORCHDYNAMO_DISABLE, PYTORCH_JIT
        apply_windows_patches: Apply Unsloth Windows compatibility patches
        load_dotenv: Load .env file for API keys
        suppress_transformers: Suppress verbose transformers logging
        utf8_output: Force UTF-8 stdout/stderr on Windows
    """
    if disable_torch_compile:
        _disable_torch_compile()

    if utf8_output:
        _setup_utf8_output()

    if load_dotenv:
        _load_env_file()

    if apply_windows_patches:
        _apply_windows_patches()

    if suppress_transformers:
        _suppress_transformers_logging_early()


def _disable_torch_compile() -> None:
    """Disable torch.compile — required for VL models and WSL compatibility.

    Must be set BEFORE importing unsloth or torch.
    """
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("PYTORCH_JIT", "0")


def _setup_utf8_output() -> None:
    """Force UTF-8 stdout/stderr on Windows to handle unicode characters."""
    if sys.platform != "win32":
        return
    import io

    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


def _load_env_file() -> None:
    """Load .env file for API keys (HF_TOKEN, WANDB_API_KEY)."""
    try:
        from dotenv import load_dotenv as _load

        _load()
    except ImportError:
        pass  # dotenv not required


def _apply_windows_patches() -> None:
    """Apply Unsloth Windows compatibility patches.

    Includes dataclasses.fields wrapping, torch.compile disable,
    and torch._inductor pre-patching. Only runs on Windows.
    """
    if sys.platform != "win32":
        return

    print("Applying Windows compatibility patches for Unsloth...")

    from dataclasses import fields
    import dataclasses

    original_fields = fields

    def patched_fields(class_or_instance):
        try:
            return original_fields(class_or_instance)
        except TypeError:
            return ()

    dataclasses.fields = patched_fields

    # Redundant with _disable_torch_compile but ensures Windows-specific
    # forced override (not setdefault).
    os.environ["PYTORCH_JIT"] = "0"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"

    try:
        import torch._inductor.runtime.hints

        if not hasattr(torch._inductor.runtime.hints, "attr_desc_fields"):
            torch._inductor.runtime.hints.attr_desc_fields = set()
    except Exception:
        pass

    print("[OK] Windows patches applied")


def _suppress_transformers_logging_early() -> None:
    """Suppress verbose transformers/trainer logging before transformers is imported.

    Sets Python logging levels so that when transformers is later imported,
    its loggers start at WARNING level instead of INFO.
    """
    import logging

    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers.trainer").setLevel(logging.WARNING)
    logging.getLogger("transformers.trainer_callback").setLevel(logging.WARNING)


def suppress_transformers_logging() -> None:
    """Suppress verbose transformers logging after transformers is imported.

    Safe to call after transformers has been imported. Sets the
    transformers library's internal verbosity to WARNING level.
    """
    try:
        import transformers

        transformers.logging.set_verbosity_warning()
    except ImportError:
        pass
