"""Tests for shared.checkpoint_eval — CheckpointEvaluator and helpers."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from shared.checkpoint_eval import (
    CheckpointEvaluator,
    CheckpointInfo,
    CheckpointReport,
    CheckpointResult,
)
from shared.eval_backend import EvalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run_dir(tmp_path: Path, checkpoint_steps: list[int]) -> Path:
    """Create a mock training run directory with checkpoint subdirectories."""
    run_dir = tmp_path / "run_20260320_120000"
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True)

    for step in checkpoint_steps:
        ckpt_dir = checkpoints_dir / f"checkpoint-{step}"
        ckpt_dir.mkdir()
        # Write a dummy file so copytree has something to copy
        (ckpt_dir / "adapter_model.bin").write_text("dummy")

    return run_dir


def _write_training_log(
    run_dir: Path, entries: list[dict], filename: str = "training_latest.jsonl"
) -> Path:
    """Write training log JSONL entries to run_dir/logs/."""
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / filename
    with open(log_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return log_path


def _make_mock_backend(scores: dict[str, float] | None = None) -> AsyncMock:
    """Create a mock EvalBackend that returns scores keyed by adapter path suffix.

    If scores is None, returns 0.5 for everything.
    """
    backend = AsyncMock()

    async def mock_run_eval(adapter_path: str, scenario: str) -> EvalResult:
        if scores is not None:
            # Match by the last path component
            name = Path(adapter_path).name
            score = scores.get(name, 0.0)
        else:
            score = 0.5
        return EvalResult(eval_score=score)

    backend.run_eval = mock_run_eval
    return backend


# ---------------------------------------------------------------------------
# discover_checkpoints
# ---------------------------------------------------------------------------


class TestDiscoverCheckpoints:
    """CheckpointEvaluator.discover_checkpoints finds checkpoint dirs."""

    def test_finds_checkpoint_dirs(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, [100, 200, 300])
        evaluator = CheckpointEvaluator(
            str(run_dir), AsyncMock(), "scenario.yaml"
        )
        checkpoints = evaluator.discover_checkpoints()

        assert len(checkpoints) == 3
        steps = [c.step for c in checkpoints]
        assert steps == [100, 200, 300]

    def test_returns_empty_when_no_checkpoints_dir(self, tmp_path):
        run_dir = tmp_path / "empty_run"
        run_dir.mkdir()
        evaluator = CheckpointEvaluator(
            str(run_dir), AsyncMock(), "scenario.yaml"
        )
        checkpoints = evaluator.discover_checkpoints()

        assert checkpoints == []

    def test_ignores_non_checkpoint_dirs(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, [100])
        # Add non-checkpoint directory
        (run_dir / "checkpoints" / "other_dir").mkdir()
        # Add a file (not a dir)
        (run_dir / "checkpoints" / "checkpoint-info.txt").write_text("info")

        evaluator = CheckpointEvaluator(
            str(run_dir), AsyncMock(), "scenario.yaml"
        )
        checkpoints = evaluator.discover_checkpoints()

        assert len(checkpoints) == 1
        assert checkpoints[0].step == 100

    def test_handles_unparseable_step(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, [100])
        # Add checkpoint with non-numeric step
        (run_dir / "checkpoints" / "checkpoint-abc").mkdir()

        evaluator = CheckpointEvaluator(
            str(run_dir), AsyncMock(), "scenario.yaml"
        )
        checkpoints = evaluator.discover_checkpoints()

        assert len(checkpoints) == 1
        assert checkpoints[0].step == 100

    def test_sorted_by_step(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, [300, 100, 200])
        evaluator = CheckpointEvaluator(
            str(run_dir), AsyncMock(), "scenario.yaml"
        )
        checkpoints = evaluator.discover_checkpoints()

        steps = [c.step for c in checkpoints]
        assert steps == sorted(steps)


# ---------------------------------------------------------------------------
# read_checkpoint_losses
# ---------------------------------------------------------------------------


class TestReadCheckpointLosses:
    """CheckpointEvaluator.read_checkpoint_losses reads training log."""

    def test_matches_steps_to_losses(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, [100, 200, 300])
        _write_training_log(
            run_dir,
            [
                {"step": 100, "loss": 2.5},
                {"step": 200, "loss": 1.8},
                {"step": 300, "loss": 1.2},
            ],
        )

        evaluator = CheckpointEvaluator(
            str(run_dir), AsyncMock(), "scenario.yaml"
        )
        checkpoints = evaluator.discover_checkpoints()
        checkpoints = evaluator.read_checkpoint_losses(checkpoints)

        assert checkpoints[0].training_loss == 2.5
        assert checkpoints[1].training_loss == 1.8
        assert checkpoints[2].training_loss == 1.2

    def test_handles_global_step_key(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, [100])
        _write_training_log(
            run_dir, [{"global_step": 100, "train_loss": 1.5}]
        )

        evaluator = CheckpointEvaluator(
            str(run_dir), AsyncMock(), "scenario.yaml"
        )
        checkpoints = evaluator.discover_checkpoints()
        checkpoints = evaluator.read_checkpoint_losses(checkpoints)

        assert checkpoints[0].training_loss == 1.5

    def test_finds_closest_step(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, [100])
        # Log has steps 95 and 105, checkpoint is at 100
        _write_training_log(
            run_dir,
            [
                {"step": 95, "loss": 2.0},
                {"step": 105, "loss": 1.9},
            ],
        )

        evaluator = CheckpointEvaluator(
            str(run_dir), AsyncMock(), "scenario.yaml"
        )
        checkpoints = evaluator.discover_checkpoints()
        checkpoints = evaluator.read_checkpoint_losses(checkpoints)

        # Closest step to 100 is 105 (distance 5 vs 5, min picks first equal)
        assert checkpoints[0].training_loss is not None

    def test_handles_missing_log(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, [100])
        # No logs directory

        evaluator = CheckpointEvaluator(
            str(run_dir), AsyncMock(), "scenario.yaml"
        )
        checkpoints = evaluator.discover_checkpoints()
        checkpoints = evaluator.read_checkpoint_losses(checkpoints)

        assert checkpoints[0].training_loss is None

    def test_handles_malformed_log_entries(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, [100])
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True)
        log_path = logs_dir / "training_latest.jsonl"
        with open(log_path, "w") as f:
            f.write("not json\n")
            f.write('{"step": 100, "loss": 1.5}\n')
            f.write('{"no_step": true}\n')

        evaluator = CheckpointEvaluator(
            str(run_dir), AsyncMock(), "scenario.yaml"
        )
        checkpoints = evaluator.discover_checkpoints()
        checkpoints = evaluator.read_checkpoint_losses(checkpoints)

        assert checkpoints[0].training_loss == 1.5

    def test_empty_log_file(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, [100])
        _write_training_log(run_dir, [])

        evaluator = CheckpointEvaluator(
            str(run_dir), AsyncMock(), "scenario.yaml"
        )
        checkpoints = evaluator.discover_checkpoints()
        checkpoints = evaluator.read_checkpoint_losses(checkpoints)

        assert checkpoints[0].training_loss is None


# ---------------------------------------------------------------------------
# prefilter_by_loss
# ---------------------------------------------------------------------------


class TestPrefilterByLoss:
    """CheckpointEvaluator.prefilter_by_loss selects top N by lowest loss."""

    def test_selects_top_n_by_lowest_loss(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, [])
        evaluator = CheckpointEvaluator(
            str(run_dir), AsyncMock(), "scenario.yaml"
        )

        checkpoints = [
            CheckpointInfo(path=Path("a"), step=100, training_loss=2.5),
            CheckpointInfo(path=Path("b"), step=200, training_loss=1.0),
            CheckpointInfo(path=Path("c"), step=300, training_loss=1.5),
            CheckpointInfo(path=Path("d"), step=400, training_loss=0.8),
        ]

        selected = evaluator.prefilter_by_loss(checkpoints, top_n=2)

        assert len(selected) == 2
        steps = {c.step for c in selected}
        assert steps == {200, 400}  # losses 1.0 and 0.8

    def test_includes_unknowns_when_not_enough_with_loss(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, [])
        evaluator = CheckpointEvaluator(
            str(run_dir), AsyncMock(), "scenario.yaml"
        )

        checkpoints = [
            CheckpointInfo(path=Path("a"), step=100, training_loss=2.5),
            CheckpointInfo(path=Path("b"), step=200, training_loss=None),
            CheckpointInfo(path=Path("c"), step=300, training_loss=None),
        ]

        selected = evaluator.prefilter_by_loss(checkpoints, top_n=3)

        assert len(selected) == 3

    def test_top_n_larger_than_available(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, [])
        evaluator = CheckpointEvaluator(
            str(run_dir), AsyncMock(), "scenario.yaml"
        )

        checkpoints = [
            CheckpointInfo(path=Path("a"), step=100, training_loss=1.0),
            CheckpointInfo(path=Path("b"), step=200, training_loss=2.0),
        ]

        selected = evaluator.prefilter_by_loss(checkpoints, top_n=10)

        assert len(selected) == 2

    def test_all_unknown_losses(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, [])
        evaluator = CheckpointEvaluator(
            str(run_dir), AsyncMock(), "scenario.yaml"
        )

        checkpoints = [
            CheckpointInfo(path=Path("a"), step=100, training_loss=None),
            CheckpointInfo(path=Path("b"), step=200, training_loss=None),
        ]

        selected = evaluator.prefilter_by_loss(checkpoints, top_n=1)

        assert len(selected) == 1

    def test_preserves_loss_ordering(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, [])
        evaluator = CheckpointEvaluator(
            str(run_dir), AsyncMock(), "scenario.yaml"
        )

        checkpoints = [
            CheckpointInfo(path=Path("a"), step=100, training_loss=3.0),
            CheckpointInfo(path=Path("b"), step=200, training_loss=1.0),
            CheckpointInfo(path=Path("c"), step=300, training_loss=2.0),
        ]

        selected = evaluator.prefilter_by_loss(checkpoints, top_n=3)

        losses = [c.training_loss for c in selected]
        assert losses == [1.0, 2.0, 3.0]  # sorted ascending


# ---------------------------------------------------------------------------
# CheckpointResult / CheckpointReport dataclasses
# ---------------------------------------------------------------------------


class TestDataclasses:
    """CheckpointResult and CheckpointReport dataclass behavior."""

    def test_checkpoint_result_defaults(self):
        r = CheckpointResult(
            path=Path("a"), step=100, training_loss=1.5, eval_score=0.8
        )
        assert r.rank == 0

    def test_checkpoint_report_fields(self):
        best = CheckpointResult(
            path=Path("a"), step=100, training_loss=1.0, eval_score=0.9, rank=1
        )
        report = CheckpointReport(
            checkpoints_evaluated=3,
            best=best,
            final_model_rank=2,
            results=[best],
        )
        assert report.checkpoints_evaluated == 3
        assert report.best.eval_score == 0.9
        assert report.final_model_rank == 2


# ---------------------------------------------------------------------------
# evaluate_checkpoints (end-to-end)
# ---------------------------------------------------------------------------


class TestEvaluateCheckpointsE2E:
    """End-to-end CheckpointEvaluator with mock EvalBackend."""

    @pytest.mark.asyncio
    async def test_finds_best_checkpoint(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, [100, 200, 300])
        _write_training_log(
            run_dir,
            [
                {"step": 100, "loss": 2.0},
                {"step": 200, "loss": 1.5},
                {"step": 300, "loss": 1.0},
            ],
        )

        # Checkpoint 200 has the best eval score despite not having lowest loss
        backend = _make_mock_backend(
            {
                "checkpoint-100": 0.60,
                "checkpoint-200": 0.90,
                "checkpoint-300": 0.75,
            }
        )

        evaluator = CheckpointEvaluator(
            str(run_dir), backend, "tool_prompts.yaml"
        )
        report = await evaluator.evaluate_checkpoints(top_n=0)

        assert report.best.step == 200
        assert report.best.eval_score == 0.90
        assert report.checkpoints_evaluated == 3

    @pytest.mark.asyncio
    async def test_includes_final_model(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, [100, 200])
        # Create final_model directory
        final_model = run_dir / "final_model"
        final_model.mkdir()
        (final_model / "adapter_model.bin").write_text("dummy")

        _write_training_log(
            run_dir,
            [
                {"step": 100, "loss": 2.0},
                {"step": 200, "loss": 1.5},
            ],
        )

        backend = _make_mock_backend(
            {
                "checkpoint-100": 0.60,
                "checkpoint-200": 0.80,
                "final_model": 0.70,
            }
        )

        evaluator = CheckpointEvaluator(
            str(run_dir), backend, "tool_prompts.yaml"
        )
        report = await evaluator.evaluate_checkpoints(top_n=0)

        # final_model should be included and ranked
        assert report.final_model_rank > 0
        assert report.checkpoints_evaluated == 3

    @pytest.mark.asyncio
    async def test_prefilters_by_loss(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, [100, 200, 300, 400, 500])
        _write_training_log(
            run_dir,
            [
                {"step": 100, "loss": 5.0},
                {"step": 200, "loss": 4.0},
                {"step": 300, "loss": 1.0},
                {"step": 400, "loss": 2.0},
                {"step": 500, "loss": 3.0},
            ],
        )

        backend = _make_mock_backend(
            {
                "checkpoint-300": 0.90,
                "checkpoint-400": 0.85,
            }
        )

        evaluator = CheckpointEvaluator(
            str(run_dir), backend, "tool_prompts.yaml"
        )
        report = await evaluator.evaluate_checkpoints(top_n=2)

        # Only 2 checkpoints evaluated (steps 300 and 400 with lowest loss)
        assert report.checkpoints_evaluated == 2
        assert report.best.step == 300

    @pytest.mark.asyncio
    async def test_copies_best_to_best_checkpoint_dir(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, [100, 200])
        _write_training_log(
            run_dir,
            [
                {"step": 100, "loss": 2.0},
                {"step": 200, "loss": 1.0},
            ],
        )

        backend = _make_mock_backend(
            {
                "checkpoint-100": 0.90,
                "checkpoint-200": 0.60,
            }
        )

        evaluator = CheckpointEvaluator(
            str(run_dir), backend, "tool_prompts.yaml"
        )
        report = await evaluator.evaluate_checkpoints(top_n=0)

        best_dir = run_dir / "best_checkpoint"
        assert best_dir.exists()
        assert (best_dir / "adapter_model.bin").exists()

    @pytest.mark.asyncio
    async def test_writes_results_tsv(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, [100, 200])
        _write_training_log(
            run_dir,
            [
                {"step": 100, "loss": 2.0},
                {"step": 200, "loss": 1.0},
            ],
        )

        backend = _make_mock_backend(
            {
                "checkpoint-100": 0.70,
                "checkpoint-200": 0.80,
            }
        )

        evaluator = CheckpointEvaluator(
            str(run_dir), backend, "tool_prompts.yaml"
        )
        await evaluator.evaluate_checkpoints(top_n=0)

        tsv_path = run_dir / "checkpoint_eval_results.tsv"
        assert tsv_path.exists()

        lines = tsv_path.read_text().strip().split("\n")
        assert len(lines) == 3  # header + 2 results
        assert lines[0] == "rank\tstep\tpath\ttraining_loss\teval_score"

    @pytest.mark.asyncio
    async def test_raises_when_no_checkpoints(self, tmp_path):
        run_dir = tmp_path / "empty_run"
        run_dir.mkdir()

        evaluator = CheckpointEvaluator(
            str(run_dir), AsyncMock(), "scenario.yaml"
        )

        with pytest.raises(ValueError, match="No checkpoints found"):
            await evaluator.evaluate_checkpoints()

    @pytest.mark.asyncio
    async def test_handles_eval_failure_gracefully(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, [100, 200])
        _write_training_log(
            run_dir,
            [
                {"step": 100, "loss": 2.0},
                {"step": 200, "loss": 1.0},
            ],
        )

        # Backend that fails on checkpoint-100 but succeeds on checkpoint-200
        backend = AsyncMock()
        call_count = 0

        async def mock_run_eval(adapter_path: str, scenario: str) -> EvalResult:
            nonlocal call_count
            call_count += 1
            if "checkpoint-100" in adapter_path:
                raise RuntimeError("GPU error")
            return EvalResult(eval_score=0.85)

        backend.run_eval = mock_run_eval

        evaluator = CheckpointEvaluator(
            str(run_dir), backend, "tool_prompts.yaml"
        )
        report = await evaluator.evaluate_checkpoints(top_n=0)

        # Failed checkpoint gets score 0.0, other one gets 0.85
        assert report.best.eval_score == 0.85
        assert report.best.step == 200
        assert report.checkpoints_evaluated == 2

    @pytest.mark.asyncio
    async def test_final_model_rank_when_not_best(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, [100])
        final_model = run_dir / "final_model"
        final_model.mkdir()
        (final_model / "adapter_model.bin").write_text("dummy")

        _write_training_log(run_dir, [{"step": 100, "loss": 1.0}])

        backend = _make_mock_backend(
            {
                "checkpoint-100": 0.95,
                "final_model": 0.80,
            }
        )

        evaluator = CheckpointEvaluator(
            str(run_dir), backend, "tool_prompts.yaml"
        )
        report = await evaluator.evaluate_checkpoints(top_n=0)

        assert report.best.step == 100
        assert report.final_model_rank == 2

    @pytest.mark.asyncio
    async def test_final_model_rank_negative_when_no_final(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, [100])
        _write_training_log(run_dir, [{"step": 100, "loss": 1.0}])

        backend = _make_mock_backend({"checkpoint-100": 0.90})

        evaluator = CheckpointEvaluator(
            str(run_dir), backend, "tool_prompts.yaml"
        )
        report = await evaluator.evaluate_checkpoints(top_n=0)

        assert report.final_model_rank == -1
