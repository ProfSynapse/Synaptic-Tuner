"""
Decision-trainer metrics callback over the shared BaseMetricsCallback.

Location: Trainers/decision/decision_core/callbacks.py
Used by:  train_decision.py.

Rows show the total loss plus its parts (``loss_ce``, ``loss_kl_frozen``,
``loss_brier``) logged by DecisionTrainer.log.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Trainers.shared.callbacks.base import BaseMetricsCallback, format_time  # noqa: E402
from Trainers.shared.callbacks.health_checks import (  # noqa: E402
    HealthChecker,
    _grad_norm_warning,
    _print_warnings,
)


class DecisionHealthChecker(HealthChecker):
    """Cross-entropy over K options starts near log K (<= ~3.2 for 24 options)."""

    def check(self, logs: Dict[str, Any], step: int, max_grad_norm: Optional[float]) -> None:
        warnings: list[str] = []
        loss = logs.get("loss", 0.0)
        if not (0 <= loss < 20):
            warnings.append(f"⚠ Unusual loss value: {loss:.4f}")
        grad_warning = _grad_norm_warning(logs, max_grad_norm)
        if grad_warning:
            warnings.append(grad_warning)
        _print_warnings(warnings, max_grad_norm, logs.get("grad_norm", 0.0))


class DecisionMetricsCallback(BaseMetricsCallback):
    default_output_dir = "./decision_output"
    start_banner = "DECISION TRAINING STARTED"
    completion_banner = "DECISION TRAINING COMPLETED"
    training_type_label = "decision"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.health_checker = DecisionHealthChecker()

    def _print_header(self) -> None:
        print("\n" + "=" * 108)
        print(
            f"{'Step':>8} | {'Loss':>9} | {'CE':>8} | {'KL':>8} | {'GradNorm':>9} | "
            f"{'LR':>10} | {'Samp/s':>7} | {'Time':>8} | {'ETA':>8} | {'Progress':>12}"
        )
        print("-" * 108)

    def _print_row(
        self,
        *,
        step: int,
        state: Any,
        args: Any,
        logs: Dict[str, Any],
        capacity_snapshot: Dict[str, Any],
        interval_time: float,
        samples_per_sec: float,
        eta: str,
        progress: str,
    ) -> None:
        print(
            f"{step:>8,} | {logs.get('loss', 0.0):>9.4f} | {logs.get('loss_ce', 0.0):>8.4f} | "
            f"{logs.get('loss_kl_frozen', 0.0):>8.4f} | {logs.get('grad_norm', 0.0):>9.3f} | "
            f"{logs.get('learning_rate', 0.0):>10.2e} | {samples_per_sec:>7.1f} | "
            f"{format_time(interval_time):>8} | {eta:>8} | {progress:>12}"
        )
