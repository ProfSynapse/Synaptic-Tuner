"""Shared training callback package for sft/kto/grpo trainers.

Per-trainer `Trainers/<trainer>/src/training_callbacks.py` modules subclass
`BaseMetricsCallback` + `BaseLiveDashboardCallback` and inject strategies
(HealthChecker, metric extraction, row format). Public symbols — the
callback classes plus `DASHBOARD_AVAILABLE` / `RICH_AVAILABLE` — are
re-exported from the per-trainer modules at unchanged paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Single point of sys.path bootstrap for the callback package. Submodules
# (base.py, log_suppression.py) and per-trainer concrete subclass modules
# (Trainers/<t>/src/training_callbacks.py) rely on the repo-root being on
# sys.path to import `shared.*`. Consolidated here so each submodule doesn't
# repeat the incantation.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

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
