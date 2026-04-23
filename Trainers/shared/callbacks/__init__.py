"""Shared training callback package for sft/kto/grpo trainers.

Per-trainer `Trainers/<trainer>/src/training_callbacks.py` modules subclass
`BaseMetricsCallback` + `BaseLiveDashboardCallback` and inject strategies
(HealthChecker, metric extraction, row format). Public symbols — the
callback classes plus `DASHBOARD_AVAILABLE` / `RICH_AVAILABLE` — are
re-exported from the per-trainer modules at unchanged paths.
"""

from __future__ import annotations

from .base import (
    BaseMetricsCallback,
    BaseLiveDashboardCallback,
    append_final_training_summary,
    resolve_cloud_provider,
    format_time,
    DASHBOARD_AVAILABLE,
    RICH_AVAILABLE,
)
from .health_checks import (
    HealthChecker,
    SFTHealthChecker,
    KTOHealthChecker,
    NoOpHealthChecker,
)
from .lr_schedules import TwoStageLRCallback
from .checkpoints import CheckpointMonitorCallback
from .log_suppression import suppress_training_logs

__all__ = [
    "BaseMetricsCallback",
    "BaseLiveDashboardCallback",
    "append_final_training_summary",
    "resolve_cloud_provider",
    "format_time",
    "DASHBOARD_AVAILABLE",
    "RICH_AVAILABLE",
    "HealthChecker",
    "SFTHealthChecker",
    "KTOHealthChecker",
    "NoOpHealthChecker",
    "TwoStageLRCallback",
    "CheckpointMonitorCallback",
    "suppress_training_logs",
]
