"""
shared/flywheel/readiness.py

ReadinessChecker: evaluates whether enough data has accumulated to justify
triggering a flywheel retrain cycle. Checks minimum example counts, quality
thresholds, and time since last cycle.

Used by: orchestrator.py, CLI (flywheel readiness command)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .catalog import LogCatalog, LogFilter
from .config import FlywheelConfig

logger = logging.getLogger(__name__)


@dataclass
class ReadinessReport:
    """Assessment of whether a retrain cycle should be triggered."""
    ready: bool = False
    new_log_count: int = 0
    estimated_sft: int = 0
    estimated_kto: int = 0
    estimated_grpo: int = 0
    avg_quality_score: float = 0.0
    days_since_last_cycle: float | None = None
    reasons: list[str] = field(default_factory=list)


class ReadinessChecker:
    """Checks if enough data has accumulated to justify a retrain cycle.

    Thresholds from config:
    - min_new_examples: Minimum unprocessed logs since last cycle
    - min_sft_examples: Minimum expected SFT examples
    - min_quality_score: Average fitness score threshold
    - min_days_since_last_cycle: Cooldown period

    Args:
        catalog: LogCatalog instance
        config: FlywheelConfig
    """

    def __init__(
        self,
        catalog: LogCatalog,
        config: FlywheelConfig,
    ) -> None:
        self._catalog = catalog
        self._config = config

    async def check(self) -> ReadinessReport:
        """Evaluate readiness for a retrain cycle.

        Returns:
            ReadinessReport with ready flag and breakdown.
        """
        report = ReadinessReport()
        cfg = self._config

        # Count unused (unprocessed) logs
        report.new_log_count = await self._catalog.count_logs(
            LogFilter(unused_only=True),
        )

        # Count by expected tag distribution
        report.estimated_sft = await self._catalog.count_logs(
            LogFilter(tag="sft", unused_only=True),
        )
        report.estimated_kto = await self._catalog.count_logs(
            LogFilter(tag="kto", unused_only=True),
        )
        report.estimated_grpo = await self._catalog.count_logs(
            LogFilter(tag="grpo", unused_only=True),
        )

        # Compute average quality score via database aggregation
        report.avg_quality_score = await self._catalog.avg_score(
            LogFilter(unused_only=True),
        )

        # Check days since last cycle
        latest_version = await self._catalog.get_latest_dataset_version()
        if latest_version:
            try:
                last_ts = datetime.fromisoformat(
                    latest_version.created_at.replace("Z", "+00:00")
                )
                now = datetime.now(timezone.utc)
                delta = now - last_ts
                report.days_since_last_cycle = delta.total_seconds() / 86400
            except (ValueError, TypeError):
                report.days_since_last_cycle = None

        # Evaluate readiness
        reasons: list[str] = []
        ready = True

        if report.new_log_count < cfg.min_new_examples:
            ready = False
            reasons.append(
                f"Need {cfg.min_new_examples} new logs, have {report.new_log_count}"
            )

        if report.estimated_sft < cfg.min_sft_examples:
            ready = False
            reasons.append(
                f"Need {cfg.min_sft_examples} SFT examples, estimated {report.estimated_sft}"
            )

        if (
            report.avg_quality_score > 0.0
            and report.avg_quality_score < cfg.min_quality_score
        ):
            ready = False
            reasons.append(
                f"Average quality {report.avg_quality_score:.2f} below "
                f"threshold {cfg.min_quality_score}"
            )

        if (
            report.days_since_last_cycle is not None
            and cfg.min_days_since_last_cycle > 0
            and report.days_since_last_cycle < cfg.min_days_since_last_cycle
        ):
            ready = False
            reasons.append(
                f"Only {report.days_since_last_cycle:.1f} days since last cycle, "
                f"need {cfg.min_days_since_last_cycle}"
            )

        if ready:
            reasons.append("All readiness criteria met")

        report.ready = ready
        report.reasons = reasons

        logger.info(
            "Readiness check: %s (%d new logs, avg quality %.2f)",
            "READY" if ready else "NOT READY",
            report.new_log_count, report.avg_quality_score,
        )
        return report
