"""Per-trainer health-check strategies.

Each concrete class prints identical-format warning blocks (100-char bracket
lines + "Consider:" footer) matching the exact output each trainer produces
today. Preserving this format is a non-goal for elegance but a hard
requirement for zero-behavior-change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


def _grad_norm_warning(logs: Dict[str, Any], max_grad_norm: Optional[float]) -> Optional[str]:
    """Shared grad-norm warning formatter used by SFT + KTO checkers."""
    grad_norm = logs.get("grad_norm", 0.0)
    if grad_norm <= 100.0:
        return None
    if max_grad_norm is not None:
        clipped_norm = min(grad_norm, max_grad_norm)
        return f"⚠ High gradient norm: {grad_norm:.2f} → {clipped_norm:.2f} (clipped)"
    return f"⚠ High gradient norm: {grad_norm:.2f} (may cause instability)"


def _print_warnings(warnings, max_grad_norm, grad_norm):
    """Shared footer logic — prints bracketed block with 'Consider:' hint."""
    if not warnings:
        return
    print("\n" + "!" * 100)
    for w in warnings:
        print(f"  {w}")
    if max_grad_norm is None or grad_norm > max_grad_norm * 10:
        print("  Consider: reducing learning rate or using tighter gradient clipping")
    print("!" * 100 + "\n")


class HealthChecker(ABC):
    @abstractmethod
    def check(self, logs: Dict[str, Any], step: int, max_grad_norm: Optional[float]) -> None: ...


class SFTHealthChecker(HealthChecker):
    """SFT: loss-range, grad-norm-clip, loss-still-high-after-50-steps."""

    def check(self, logs, step, max_grad_norm):
        warnings = []
        loss = logs.get("loss", 0.0)
        if not (0 < loss < 100):
            warnings.append(f"⚠ Unusual loss value: {loss:.4f}")
        grad_warning = _grad_norm_warning(logs, max_grad_norm)
        if grad_warning:
            warnings.append(grad_warning)
        if step > 50 and loss > 2.0:
            warnings.append(
                f"⚠ Loss remains high after {step} steps: {loss:.4f} (may need longer training or LR adjustment)"
            )
        _print_warnings(warnings, max_grad_norm, logs.get("grad_norm", 0.0))


class KTOHealthChecker(HealthChecker):
    """KTO: loss-range, margin < -1.0, reward-collapse, grad-norm-clip."""

    def check(self, logs, step, max_grad_norm):
        warnings = []
        loss = logs.get("loss", 0.0)
        if not (0 < loss < 100):
            warnings.append(f"⚠ Unusual loss value: {loss:.4f}")
        margin = logs.get("rewards/margins", 0.0)
        if margin < -1.0:
            warnings.append(
                f"⚠ Very negative margin: {margin:.4f} (chosen model may be worse than reference)"
            )
        chosen = logs.get("rewards/chosen", 0.0)
        rejected = logs.get("rewards/rejected", 0.0)
        if abs(chosen) < 0.001 and abs(rejected) < 0.001 and step > 10:
            warnings.append("⚠ Reward collapse detected (both rewards near zero)")
        grad_warning = _grad_norm_warning(logs, max_grad_norm)
        if grad_warning:
            warnings.append(grad_warning)
        _print_warnings(warnings, max_grad_norm, logs.get("grad_norm", 0.0))


class NoOpHealthChecker(HealthChecker):
    """GRPO: no health checks today."""

    def check(self, logs, step, max_grad_norm):
        pass
