"""Tests for shared.flywheel.readiness — ReadinessChecker."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from shared.flywheel.catalog import DatasetVersion, InferenceLogRecord, LogFilter
from shared.flywheel.config import FlywheelConfig
from shared.flywheel.readiness import ReadinessChecker, ReadinessReport


def _make_record(
    log_id: str, fitness_score: float | None = None, **kwargs,
) -> InferenceLogRecord:
    defaults = dict(
        timestamp="2026-01-15T12:00:00Z",
        model_id="test-model",
        source_file="test.jsonl",
        line_number=0,
    )
    defaults.update(kwargs)
    return InferenceLogRecord(log_id=log_id, fitness_score=fitness_score, **defaults)


class TestReadinessReport:
    """ReadinessReport dataclass defaults."""

    def test_defaults(self):
        r = ReadinessReport()
        assert r.ready is False
        assert r.new_log_count == 0
        assert r.reasons == []


@pytest.mark.asyncio
class TestReadinessChecker:
    """ReadinessChecker evaluates all criteria for triggering a retrain cycle."""

    def _make_checker(self, catalog, **config_kwargs):
        cfg = FlywheelConfig(**config_kwargs)
        return ReadinessChecker(catalog, cfg)

    async def test_ready_when_all_criteria_met(self):
        """Returns ready=True when enough examples, quality, and SFT count."""
        catalog = AsyncMock()
        catalog.count_logs = AsyncMock(side_effect=[
            600,   # unused total
            150,   # estimated_sft
            100,   # estimated_kto
            50,    # estimated_grpo
        ])
        catalog.avg_score = AsyncMock(return_value=0.8)
        catalog.get_latest_dataset_version = AsyncMock(return_value=None)

        checker = self._make_checker(catalog, min_new_examples=500, min_sft_examples=100)
        report = await checker.check()

        assert report.ready is True
        assert report.new_log_count == 600
        assert report.estimated_sft == 150
        assert "All readiness criteria met" in report.reasons

    async def test_not_ready_too_few_examples(self):
        """Returns ready=False when below min_new_examples."""
        catalog = AsyncMock()
        catalog.count_logs = AsyncMock(side_effect=[100, 50, 30, 10])
        catalog.avg_score = AsyncMock(return_value=0.9)
        catalog.get_latest_dataset_version = AsyncMock(return_value=None)

        checker = self._make_checker(catalog, min_new_examples=500)
        report = await checker.check()

        assert report.ready is False
        assert any("Need 500 new logs" in r for r in report.reasons)

    async def test_not_ready_too_few_sft(self):
        """Returns ready=False when below min_sft_examples."""
        catalog = AsyncMock()
        catalog.count_logs = AsyncMock(side_effect=[
            600,  # unused total (above threshold)
            10,   # estimated_sft (below threshold)
            100,  # estimated_kto
            50,   # estimated_grpo
        ])
        catalog.avg_score = AsyncMock(return_value=0.9)
        catalog.get_latest_dataset_version = AsyncMock(return_value=None)

        checker = self._make_checker(catalog, min_new_examples=500, min_sft_examples=100)
        report = await checker.check()

        assert report.ready is False
        assert any("SFT examples" in r for r in report.reasons)

    async def test_not_ready_low_quality(self):
        """Returns ready=False when average quality below threshold."""
        catalog = AsyncMock()
        catalog.count_logs = AsyncMock(side_effect=[600, 150, 100, 50])
        catalog.avg_score = AsyncMock(return_value=0.3)
        catalog.get_latest_dataset_version = AsyncMock(return_value=None)

        checker = self._make_checker(
            catalog, min_new_examples=500, min_sft_examples=100, min_quality_score=0.6,
        )
        report = await checker.check()

        assert report.ready is False
        assert report.avg_quality_score == pytest.approx(0.3)
        assert any("quality" in r.lower() for r in report.reasons)

    async def test_not_ready_cooldown_period(self):
        """Returns ready=False when within min_days_since_last_cycle."""
        recent_version = DatasetVersion(
            version_id="v001",
            created_at=datetime.now(timezone.utc).isoformat(),
            source_model_id="m",
            record_counts={},
            file_paths={},
            content_hash="",
        )

        catalog = AsyncMock()
        catalog.count_logs = AsyncMock(side_effect=[600, 150, 100, 50])
        catalog.avg_score = AsyncMock(return_value=0.9)
        catalog.get_latest_dataset_version = AsyncMock(return_value=recent_version)

        checker = self._make_checker(
            catalog,
            min_new_examples=500,
            min_sft_examples=100,
            min_days_since_last_cycle=7,
        )
        report = await checker.check()

        assert report.ready is False
        assert report.days_since_last_cycle is not None
        assert report.days_since_last_cycle < 1  # Just created

    async def test_no_scored_logs_still_checks_counts(self):
        """With no scored logs, quality check is skipped but count checks apply."""
        catalog = AsyncMock()
        catalog.count_logs = AsyncMock(side_effect=[600, 150, 100, 50])
        catalog.avg_score = AsyncMock(return_value=0.0)  # no scored logs
        catalog.get_latest_dataset_version = AsyncMock(return_value=None)

        checker = self._make_checker(catalog, min_new_examples=500, min_sft_examples=100)
        report = await checker.check()

        assert report.ready is True
        assert report.avg_quality_score == 0.0

    async def test_zero_cooldown_always_passes(self):
        """min_days_since_last_cycle=0 means cooldown check is skipped."""
        recent_version = DatasetVersion(
            version_id="v001",
            created_at=datetime.now(timezone.utc).isoformat(),
            source_model_id="m",
            record_counts={},
            file_paths={},
            content_hash="",
        )

        catalog = AsyncMock()
        catalog.count_logs = AsyncMock(side_effect=[600, 150, 100, 50])
        catalog.avg_score = AsyncMock(return_value=0.9)
        catalog.get_latest_dataset_version = AsyncMock(return_value=recent_version)

        checker = self._make_checker(
            catalog,
            min_new_examples=500,
            min_sft_examples=100,
            min_days_since_last_cycle=0,
        )
        report = await checker.check()

        assert report.ready is True
