"""
Tests for EvalHandler TableSpec/ColumnSpec DRY refactoring.

Verifies:
- _TableSpec and _ColumnSpec dataclass construction
- _display_table renders both Rich and plain-text paths
- Each thin wrapper builds correct TableSpec
- Conditional columns (checkpoint display varies by trainer_type)
- Edge cases: empty items list, single item
"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest

from tuner.handlers.eval_handler import EvalHandler


# =========================================================
# Helpers
# =========================================================

def _make_handler(repo_root: Path) -> EvalHandler:
    """Build an EvalHandler with a mocked Namespace for args."""
    args = SimpleNamespace(json=False, backend=None, model=None, scenario=None)
    handler = EvalHandler(args=args)
    handler._repo_root = repo_root
    return handler


# =========================================================
# TableSpec / ColumnSpec Dataclass Construction
# =========================================================

class TestTableSpecConstruction:
    def test_column_spec_defaults(self, tmp_path):
        handler = _make_handler(tmp_path)
        col = handler._ColumnSpec("Name")
        assert col.header == "Name"
        assert col.style == "white"
        assert col.width is None
        assert col.justify == "left"

    def test_column_spec_overrides(self, tmp_path):
        handler = _make_handler(tmp_path)
        col = handler._ColumnSpec("Size", style="dim", width=10, justify="right")
        assert col.header == "Size"
        assert col.style == "dim"
        assert col.width == 10
        assert col.justify == "right"

    def test_table_spec_defaults(self, tmp_path):
        handler = _make_handler(tmp_path)
        spec = handler._TableSpec(title="Test Table")
        assert spec.title == "Test Table"
        assert spec.columns == []
        assert callable(spec.row_extractor)
        assert callable(spec.plain_formatter)

    def test_table_spec_with_columns(self, tmp_path):
        handler = _make_handler(tmp_path)
        cols = [
            handler._ColumnSpec("A"),
            handler._ColumnSpec("B", style="dim"),
        ]
        spec = handler._TableSpec(title="T", columns=cols)
        assert len(spec.columns) == 2
        assert spec.columns[0].header == "A"
        assert spec.columns[1].style == "dim"


# =========================================================
# _display_table Plain-Text Path
# =========================================================

class TestDisplayTablePlainText:
    def test_renders_plain_text_when_rich_unavailable(self, tmp_path, capsys):
        handler = _make_handler(tmp_path)
        spec = handler._TableSpec(
            title="My Table",
            columns=[handler._ColumnSpec("Name")],
            row_extractor=lambda i, x: [x],
            plain_formatter=lambda i, x: x,
        )
        with patch("tuner.handlers.eval_handler.RICH_AVAILABLE", False):
            handler._display_table(["alpha", "beta", "gamma"], spec)

        captured = capsys.readouterr()
        assert "My Table:" in captured.out
        assert "[1] alpha" in captured.out
        assert "[2] beta" in captured.out
        assert "[3] gamma" in captured.out

    def test_renders_empty_list(self, tmp_path, capsys):
        handler = _make_handler(tmp_path)
        spec = handler._TableSpec(
            title="Empty Table",
            columns=[],
            row_extractor=lambda i, x: [],
            plain_formatter=lambda i, x: str(x),
        )
        with patch("tuner.handlers.eval_handler.RICH_AVAILABLE", False):
            handler._display_table([], spec)

        captured = capsys.readouterr()
        assert "Empty Table:" in captured.out
        # No items
        assert "[1]" not in captured.out

    def test_renders_single_item(self, tmp_path, capsys):
        handler = _make_handler(tmp_path)
        spec = handler._TableSpec(
            title="Single",
            columns=[handler._ColumnSpec("Val")],
            row_extractor=lambda i, x: [str(x)],
            plain_formatter=lambda i, x: str(x),
        )
        with patch("tuner.handlers.eval_handler.RICH_AVAILABLE", False):
            handler._display_table([42], spec)

        captured = capsys.readouterr()
        assert "[1] 42" in captured.out


# =========================================================
# _display_table Rich Path
# =========================================================

class TestDisplayTableRich:
    def test_creates_rich_table_with_columns(self, tmp_path):
        handler = _make_handler(tmp_path)
        spec = handler._TableSpec(
            title="Rich Test",
            columns=[
                handler._ColumnSpec("Name"),
                handler._ColumnSpec("Value", style="dim", justify="right"),
            ],
            row_extractor=lambda i, x: [x[0], x[1]],
            plain_formatter=lambda i, x: f"{x[0]}: {x[1]}",
        )
        # Mock Rich console to capture output
        mock_console = MagicMock()
        with patch("tuner.handlers.eval_handler.RICH_AVAILABLE", True), \
             patch("tuner.handlers.eval_handler.console", mock_console):
            handler._display_table([("alpha", "1"), ("beta", "2")], spec)

        # console.print should be called (for spacing + table)
        assert mock_console.print.call_count >= 1


# =========================================================
# Thin Wrappers
# =========================================================

class TestDisplayModelsTable:
    def test_models_table_spec(self, tmp_path, capsys):
        handler = _make_handler(tmp_path)
        with patch("tuner.handlers.eval_handler.RICH_AVAILABLE", False):
            handler._display_models_table("ollama", ["model-a", "model-b"])

        captured = capsys.readouterr()
        assert "Available Ollama Models:" in captured.out
        assert "model-a" in captured.out
        assert "model-b" in captured.out


class TestDisplayTrainingRunsTable:
    def test_training_runs_table(self, tmp_path, capsys):
        handler = _make_handler(tmp_path)
        run1 = tmp_path / "run-20260321"
        run1.mkdir()
        (run1 / "final_model").mkdir()
        run2 = tmp_path / "run-20260322"
        run2.mkdir()

        with patch("tuner.handlers.eval_handler.RICH_AVAILABLE", False):
            handler._display_training_runs_table([run1, run2], "sft")

        captured = capsys.readouterr()
        assert "Available SFT Training Runs:" in captured.out
        assert "run-20260321" in captured.out
        assert "run-20260322" in captured.out


# =========================================================
# Conditional Columns (Checkpoints)
# =========================================================

class TestCheckpointConditionalColumns:
    def _make_checkpoint(self, step=100, is_final=False, metrics=None):
        if metrics is None:
            metrics = {"loss": 0.5, "epoch": 1.0}
        return SimpleNamespace(
            step=step,
            is_final=is_final,
            metrics=metrics,
        )

    def test_sft_checkpoint_columns(self, tmp_path):
        handler = _make_handler(tmp_path)
        cp = self._make_checkpoint()
        row = handler._checkpoint_row(cp, "sft")
        # SFT: [name, step, loss, epoch] = 4 columns
        assert len(row) == 4

    def test_kto_checkpoint_has_extra_columns(self, tmp_path):
        handler = _make_handler(tmp_path)
        cp = self._make_checkpoint(metrics={"loss": 0.3, "epoch": 1.0, "kl": 0.01, "rewards/margins": 0.5})
        row = handler._checkpoint_row(cp, "kto")
        # KTO: [name, step, loss, kl, margin, epoch] = 6 columns
        assert len(row) == 6

    def test_grpo_checkpoint_has_reward_column(self, tmp_path):
        handler = _make_handler(tmp_path)
        cp = self._make_checkpoint(metrics={"loss": 0.2, "epoch": 1.0, "reward": 0.8})
        row = handler._checkpoint_row(cp, "grpo")
        # GRPO: [name, step, loss, reward, epoch] = 5 columns
        assert len(row) == 5

    def test_final_model_checkpoint_display(self, tmp_path):
        handler = _make_handler(tmp_path)
        cp = self._make_checkpoint(is_final=True, metrics={"loss": 0.1, "epoch": 3.0})
        row = handler._checkpoint_row(cp, "sft")
        assert "\u2605" in row[0]  # Star for final model
        assert row[1] == "-"  # Step is dash for final

    def test_missing_metrics_show_dashes(self, tmp_path):
        handler = _make_handler(tmp_path)
        cp = self._make_checkpoint(metrics={})
        row = handler._checkpoint_row(cp, "kto")
        # loss, kl, margin, epoch should all be "-"
        assert row[2] == "-"  # loss
        assert row[3] == "-"  # kl
        assert row[4] == "-"  # margin
        assert row[5] == "-"  # epoch

    def test_grpo_falls_back_to_rewards_mean(self, tmp_path):
        handler = _make_handler(tmp_path)
        cp = self._make_checkpoint(metrics={"loss": 0.2, "epoch": 1.0, "rewards/mean": 0.75})
        row = handler._checkpoint_row(cp, "grpo")
        assert "0.75" in row[3]  # reward column
