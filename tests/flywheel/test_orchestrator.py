"""Tests for shared.flywheel.orchestrator — FlywheelOrchestrator pipeline."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.flywheel.catalog import DatasetVersion, LogFilter
from shared.flywheel.cleaner import CleaningResult
from shared.flywheel.config import FlywheelConfig
from shared.flywheel.orchestrator import (
    CycleResult,
    FlywheelOrchestrator,
    RetrainMode,
    TrainingResult,
)
from shared.flywheel.readiness import ReadinessReport
from shared.flywheel.stager import StagingResult
from shared.flywheel.tagger import TaggingResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_orchestrator(
    *,
    catalog: AsyncMock | None = None,
    config: FlywheelConfig | None = None,
    cleaner: AsyncMock | None = None,
    tagger: AsyncMock | None = None,
    stager: AsyncMock | None = None,
) -> FlywheelOrchestrator:
    """Build a FlywheelOrchestrator with mocked dependencies."""
    cat = catalog or AsyncMock()
    cfg = config or FlywheelConfig()
    cln = cleaner or AsyncMock()
    tag = tagger or AsyncMock()
    stg = stager or AsyncMock()
    return FlywheelOrchestrator(cat, cfg, cln, tag, stg)


# ---------------------------------------------------------------------------
# RetrainMode enum
# ---------------------------------------------------------------------------


class TestRetrainMode:
    """RetrainMode enum values."""

    def test_gpu_mutex_value(self):
        assert RetrainMode.GPU_MUTEX.value == "gpu_mutex"

    def test_hot_swap_value(self):
        assert RetrainMode.HOT_SWAP.value == "hot_swap"

    def test_cloud_value(self):
        assert RetrainMode.CLOUD.value == "cloud"


# ---------------------------------------------------------------------------
# CycleResult / TrainingResult dataclasses
# ---------------------------------------------------------------------------


class TestDataclasses:
    """Dataclass defaults are sane."""

    def test_cycle_result_defaults(self):
        r = CycleResult()
        assert r.cleaning is None
        assert r.tagging is None
        assert r.staging is None
        assert r.training is None
        assert r.hot_swap_success is None
        assert r.total_duration_seconds == 0.0

    def test_training_result_defaults(self):
        r = TrainingResult()
        assert r.success is False
        assert r.run_id == ""
        assert r.adapter_path == ""
        assert r.error is None


# ---------------------------------------------------------------------------
# run_cycle happy path
# ---------------------------------------------------------------------------


class TestRunCycleHappyPath:
    """run_cycle executes clean -> tag -> stage in sequence."""

    @pytest.mark.asyncio
    async def test_full_cycle_skip_retrain(self):
        """Clean -> tag -> stage all called; retrain skipped."""
        cleaner = AsyncMock()
        cleaner.clean_logs.return_value = CleaningResult(
            total_processed=10, scored=10,
        )
        tagger = AsyncMock()
        tagger.tag_logs.return_value = TaggingResult(
            total_processed=10, sft_count=5, kto_count=3,
        )
        stager = AsyncMock()
        stager.stage_dataset.return_value = StagingResult(
            version_id="v001", sft_count=5, total_records=8,
        )

        orch = _make_orchestrator(cleaner=cleaner, tagger=tagger, stager=stager)
        result = await orch.run_cycle(skip_retrain=True)

        cleaner.clean_logs.assert_awaited_once()
        tagger.tag_logs.assert_awaited_once()
        stager.stage_dataset.assert_awaited_once()
        assert result.cleaning.scored == 10
        assert result.tagging.sft_count == 5
        assert result.staging.version_id == "v001"
        assert result.training is None
        assert result.total_duration_seconds > 0

    @pytest.mark.asyncio
    async def test_dry_run_skips_all_stages(self):
        """Dry run only checks readiness, does not execute stages."""
        catalog = AsyncMock()
        catalog.count_logs = AsyncMock(return_value=0)
        catalog.avg_score = AsyncMock(return_value=0.0)
        catalog.get_latest_dataset_version = AsyncMock(return_value=None)

        cleaner = AsyncMock()
        tagger = AsyncMock()
        stager = AsyncMock()

        orch = _make_orchestrator(
            catalog=catalog, cleaner=cleaner, tagger=tagger, stager=stager,
        )
        result = await orch.run_cycle(dry_run=True)

        cleaner.clean_logs.assert_not_awaited()
        tagger.tag_logs.assert_not_awaited()
        stager.stage_dataset.assert_not_awaited()
        assert result.cleaning is None


# ---------------------------------------------------------------------------
# Zero-scored-logs short-circuit
# ---------------------------------------------------------------------------


class TestZeroScoredLogsWarning:
    """When cleaning scores 0 logs, a warning is logged but pipeline continues."""

    @pytest.mark.asyncio
    async def test_zero_scored_continues_to_tag_and_stage(self):
        """Pipeline continues even when cleaner scores zero logs."""
        cleaner = AsyncMock()
        cleaner.clean_logs.return_value = CleaningResult(
            total_processed=5, scored=0, errors=5,
        )
        tagger = AsyncMock()
        tagger.tag_logs.return_value = TaggingResult()
        stager = AsyncMock()
        stager.stage_dataset.return_value = StagingResult()

        orch = _make_orchestrator(cleaner=cleaner, tagger=tagger, stager=stager)
        result = await orch.run_cycle(skip_retrain=True)

        # Tag and stage still called even when zero scored
        tagger.tag_logs.assert_awaited_once()
        stager.stage_dataset.assert_awaited_once()
        assert result.cleaning.scored == 0


# ---------------------------------------------------------------------------
# Stage error handling
# ---------------------------------------------------------------------------


class TestStageErrorHandling:
    """Each pipeline stage catches exceptions independently."""

    @pytest.mark.asyncio
    async def test_cleaning_failure_does_not_block_tagging(self):
        """If cleaner throws, tagging and staging still execute."""
        cleaner = AsyncMock()
        cleaner.clean_logs.side_effect = RuntimeError("evaluator crash")
        tagger = AsyncMock()
        tagger.tag_logs.return_value = TaggingResult()
        stager = AsyncMock()
        stager.stage_dataset.return_value = StagingResult()

        orch = _make_orchestrator(cleaner=cleaner, tagger=tagger, stager=stager)
        result = await orch.run_cycle(skip_retrain=True)

        assert result.cleaning is None  # Failed, so not assigned
        tagger.tag_logs.assert_awaited_once()
        stager.stage_dataset.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_tagging_failure_does_not_block_staging(self):
        """If tagger throws, staging still executes."""
        cleaner = AsyncMock()
        cleaner.clean_logs.return_value = CleaningResult(scored=5)
        tagger = AsyncMock()
        tagger.tag_logs.side_effect = RuntimeError("tagger crash")
        stager = AsyncMock()
        stager.stage_dataset.return_value = StagingResult()

        orch = _make_orchestrator(cleaner=cleaner, tagger=tagger, stager=stager)
        result = await orch.run_cycle(skip_retrain=True)

        assert result.tagging is None
        stager.stage_dataset.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_staging_failure_does_not_crash_cycle(self):
        """If stager throws, cycle completes with staging=None."""
        cleaner = AsyncMock()
        cleaner.clean_logs.return_value = CleaningResult(scored=5)
        tagger = AsyncMock()
        tagger.tag_logs.return_value = TaggingResult(sft_count=5)
        stager = AsyncMock()
        stager.stage_dataset.side_effect = RuntimeError("disk full")

        orch = _make_orchestrator(cleaner=cleaner, tagger=tagger, stager=stager)
        result = await orch.run_cycle(skip_retrain=True)

        assert result.staging is None
        assert result.total_duration_seconds > 0


# ---------------------------------------------------------------------------
# check_readiness delegation
# ---------------------------------------------------------------------------


class TestCheckReadiness:
    """check_readiness delegates to ReadinessChecker."""

    @pytest.mark.asyncio
    async def test_delegates_to_readiness_checker(self):
        """check_readiness calls ReadinessChecker.check()."""
        expected_report = ReadinessReport(ready=True, reasons=["All met"])
        catalog = AsyncMock()
        catalog.count_logs = AsyncMock(return_value=600)
        catalog.find_logs = AsyncMock(return_value=[])
        catalog.get_latest_dataset_version = AsyncMock(return_value=None)

        orch = _make_orchestrator(catalog=catalog)

        with patch.object(orch._readiness, "check", return_value=expected_report):
            report = await orch.check_readiness()

        assert report.ready is True
        assert "All met" in report.reasons


# ---------------------------------------------------------------------------
# _select_trainer logic
# ---------------------------------------------------------------------------


class TestSelectTrainer:
    """_select_trainer picks script and args based on dataset composition."""

    def test_auto_selects_sft_when_sft_records_present(self):
        """auto mode picks SFT if sft count > 0."""
        cfg = FlywheelConfig(retrain_trainer="auto")
        orch = _make_orchestrator(config=cfg)

        version = DatasetVersion(
            version_id="v001",
            created_at="2026-01-01T00:00:00Z",
            source_model_id="test",
            record_counts={"sft": 50, "kto_pos": 0, "kto_neg": 0},
            file_paths={"sft": "/data/sft.jsonl"},
        )
        script, args = orch._select_trainer(version)
        assert "sft" in script.lower()
        assert "--dataset-file" in args

    def test_auto_selects_kto_when_only_kto_records(self):
        """auto mode picks KTO when no SFT but KTO records exist."""
        cfg = FlywheelConfig(retrain_trainer="auto")
        orch = _make_orchestrator(config=cfg)

        version = DatasetVersion(
            version_id="v002",
            created_at="2026-01-01T00:00:00Z",
            source_model_id="test",
            record_counts={"sft": 0, "kto_pos": 30, "kto_neg": 20},
            file_paths={"kto": "/data/kto.jsonl"},
        )
        script, args = orch._select_trainer(version)
        assert "kto" in script.lower()

    def test_auto_falls_back_to_sft_when_empty(self):
        """auto mode defaults to SFT when no records present."""
        cfg = FlywheelConfig(retrain_trainer="auto")
        orch = _make_orchestrator(config=cfg)

        version = DatasetVersion(
            version_id="v003",
            created_at="2026-01-01T00:00:00Z",
            source_model_id="test",
            record_counts={},
            file_paths={},
        )
        script, args = orch._select_trainer(version)
        assert "sft" in script.lower()

    def test_explicit_kto_trainer(self):
        """Explicit kto trainer overrides auto detection."""
        cfg = FlywheelConfig(retrain_trainer="kto")
        orch = _make_orchestrator(config=cfg)

        version = DatasetVersion(
            version_id="v004",
            created_at="2026-01-01T00:00:00Z",
            source_model_id="test",
            record_counts={"sft": 100},
            file_paths={"kto": "/data/kto.jsonl"},
        )
        script, args = orch._select_trainer(version)
        assert "kto" in script.lower()

    def test_unknown_trainer_defaults_to_sft(self):
        """Unknown trainer value falls through to SFT."""
        cfg = FlywheelConfig(retrain_trainer="unknown_trainer")
        orch = _make_orchestrator(config=cfg)

        version = DatasetVersion(
            version_id="v005",
            created_at="2026-01-01T00:00:00Z",
            source_model_id="test",
            record_counts={},
            file_paths={"sft": "/data/sft.jsonl"},
        )
        script, args = orch._select_trainer(version)
        assert "sft" in script.lower()


# ---------------------------------------------------------------------------
# Retrain mode selection in run_cycle
# ---------------------------------------------------------------------------


class TestRetrainModeSelection:
    """run_cycle uses retrain_mode to determine retraining strategy."""

    @pytest.mark.asyncio
    async def test_retrain_mode_override(self):
        """Explicit retrain_mode overrides config default."""
        cleaner = AsyncMock()
        cleaner.clean_logs.return_value = CleaningResult(scored=10)
        tagger = AsyncMock()
        tagger.tag_logs.return_value = TaggingResult(sft_count=10)
        stager = AsyncMock()
        stager.stage_dataset.return_value = StagingResult(
            version_id="v001", sft_count=10,
        )

        catalog = AsyncMock()
        catalog.get_dataset_version = AsyncMock(return_value=DatasetVersion(
            version_id="v001",
            created_at="2026-01-01T00:00:00Z",
            source_model_id="test",
            record_counts={"sft": 10},
            file_paths={"sft": "/data/sft.jsonl"},
        ))

        cfg = FlywheelConfig(retrain_mode="gpu_mutex")
        orch = _make_orchestrator(
            catalog=catalog, config=cfg,
            cleaner=cleaner, tagger=tagger, stager=stager,
        )

        # Mock the training path to avoid subprocess calls
        with patch.object(orch, "_run_training", return_value=TrainingResult(
            success=True, adapter_path="/adapters/v001",
        )) as mock_train, patch.object(orch, "_start_vllm", return_value=True):
            result = await orch.run_cycle(
                retrain_mode=RetrainMode.HOT_SWAP,
            )

        # Training was called — the mode is passed through
        mock_train.assert_awaited_once()
        # Verify the mode passed was HOT_SWAP, not the config default
        call_args = mock_train.call_args
        assert call_args[0][1] == RetrainMode.HOT_SWAP

    @pytest.mark.asyncio
    async def test_skip_retrain_when_no_staging_version(self):
        """Retrain is skipped when staging produces no version_id."""
        cleaner = AsyncMock()
        cleaner.clean_logs.return_value = CleaningResult(scored=5)
        tagger = AsyncMock()
        tagger.tag_logs.return_value = TaggingResult()
        stager = AsyncMock()
        stager.stage_dataset.return_value = StagingResult()  # No version_id

        orch = _make_orchestrator(cleaner=cleaner, tagger=tagger, stager=stager)

        with patch.object(orch, "_run_training") as mock_train:
            result = await orch.run_cycle()

        mock_train.assert_not_awaited()
        assert result.training is None

    @pytest.mark.asyncio
    async def test_retrain_failure_sets_error(self):
        """When _run_training raises, training result captures the error."""
        cleaner = AsyncMock()
        cleaner.clean_logs.return_value = CleaningResult(scored=10)
        tagger = AsyncMock()
        tagger.tag_logs.return_value = TaggingResult(sft_count=10)
        stager = AsyncMock()
        stager.stage_dataset.return_value = StagingResult(version_id="v001")

        catalog = AsyncMock()
        catalog.get_dataset_version = AsyncMock(side_effect=RuntimeError("DB gone"))

        orch = _make_orchestrator(
            catalog=catalog, cleaner=cleaner, tagger=tagger, stager=stager,
        )
        result = await orch.run_cycle()

        assert result.training is not None
        assert result.training.success is False
        assert "DB gone" in result.training.error
