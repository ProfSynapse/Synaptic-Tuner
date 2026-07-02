"""Generic, no-op-on-CPU GPU peak-memory telemetry for the batch verbs.

Location: tuner/batch/gpu_telemetry.py
Purpose: Give consumers visibility into VRAM headroom so they can tune
    ``--batch-size`` against measured peaks instead of estimates. Strictly
    no-op on CPU (no CUDA calls attempted, no import-time cost): torch is
    imported lazily and every function returns a benign value when CUDA is
    unavailable.
Used by: tuner.batch.runner.

The reset half reuses ``shared.training_capacity.reset_capacity_peaks`` (which
imports only stdlib at module load and lazily imports torch) so the two
codepaths that reset peak stats stay in one place. The read half is a tiny
local helper — two numbers in GiB — to avoid pulling the heavy capacity
snapshot dict for a one-line log parenthetical.
"""

from __future__ import annotations

from typing import Optional, Tuple

_GIB = 1024 ** 3


def _import_torch():
    try:
        import torch  # type: ignore

        return torch
    except Exception:
        return None


def reset_peak(torch_module=None) -> None:
    """Reset CUDA peak-memory stats at stage start. No-op on CPU.

    Delegates to the shared capacity helper so peak-reset logic is not
    duplicated. Safe to call unconditionally.
    """
    try:
        from shared.training_capacity import reset_capacity_peaks
    except Exception:
        # Fallback: reset directly if the shared helper is unavailable.
        torch_module = torch_module or _import_torch()
        if torch_module is not None and torch_module.cuda.is_available():
            try:
                torch_module.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        return
    reset_capacity_peaks(torch_module)


def peak_and_total_gib(torch_module=None) -> Optional[Tuple[float, float]]:
    """Return ``(peak_allocated_gib, total_gib)`` for the active CUDA device.

    Returns ``None`` when CUDA is unavailable (CPU runs), so callers omit the
    parenthetical entirely rather than logging zeros.
    """
    torch_module = torch_module or _import_torch()
    if torch_module is None or not torch_module.cuda.is_available():
        return None
    try:
        device = torch_module.cuda.current_device()
        peak = float(torch_module.cuda.max_memory_allocated(device))
        total = float(torch_module.cuda.get_device_properties(device).total_memory)
    except Exception:
        return None
    return peak / _GIB, total / _GIB


def peak_suffix(torch_module=None) -> str:
    """Return a `` (gpu peak X.X/Y.Y GiB)`` suffix, or ``""`` on CPU.

    The leading space is included so callers can append it directly to a log
    line; on CPU the empty string leaves the line shape unchanged.
    """
    pt = peak_and_total_gib(torch_module)
    if pt is None:
        return ""
    peak, total = pt
    return f" (gpu peak {peak:.1f}/{total:.1f} GiB)"
