"""Tests for PivotRL pivot profiler."""

import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Trainers" / "grpo" / "src"))

# pivot_profiler imports torch at module level; stub it when unavailable.
if "torch" not in sys.modules:
    sys.modules["torch"] = MagicMock()

from pivot_profiler import (
    PivotCandidate,
    PivotResult,
    extract_candidates,
    filter_pivots,
    pivots_to_dataset,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _make_result(std: float, mean: float = 0.5) -> PivotResult:
    cand = PivotCandidate(
        source_file="f.jsonl", source_line=1, turn_index=1,
        state_messages=[{"role": "user", "content": "hi"}],
        reference_action="ok", ground_truth_tool="tool",
        ground_truth_args_json='{"a":1}', jsonl_hash="abcd1234",
    )
    return PivotResult(
        candidate=cand, reward_mean=mean, reward_std=std,
        reward_min=mean - std, reward_max=mean + std,
        num_rollouts=8, is_pivot=False,
    )


# ---------------------------------------------------------------------------
# extract_candidates
# ---------------------------------------------------------------------------

class TestExtractCandidates:

    def test_single_turn(self, tmp_path):
        _write_jsonl(tmp_path / "data.jsonl", [
            {"conversations": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]}
        ])
        cands = extract_candidates(tmp_path / "data.jsonl")
        assert len(cands) == 1
        assert cands[0].state_messages == [{"role": "user", "content": "Hello"}]
        assert cands[0].reference_action == "Hi there"

    def test_multi_turn(self, tmp_path):
        _write_jsonl(tmp_path / "data.jsonl", [
            {"conversations": [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
                {"role": "user", "content": "Q3"},
                {"role": "assistant", "content": "A3"},
            ]}
        ])
        cands = extract_candidates(tmp_path / "data.jsonl")
        assert len(cands) == 3
        # First candidate sees only the first user message
        assert len(cands[0].state_messages) == 1
        # Third candidate sees all preceding messages (5 messages before index 5)
        assert len(cands[2].state_messages) == 5

    def test_with_system_message(self, tmp_path):
        _write_jsonl(tmp_path / "data.jsonl", [
            {"conversations": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ]}
        ])
        cands = extract_candidates(tmp_path / "data.jsonl")
        assert len(cands) == 1
        assert cands[0].state_messages[0] == {"role": "system", "content": "You are helpful."}
        assert len(cands[0].state_messages) == 2  # system + user


# ---------------------------------------------------------------------------
# filter_pivots
# ---------------------------------------------------------------------------

class TestFilterPivots:

    def test_variance_threshold(self):
        results = [_make_result(std=s) for s in [0.05, 0.10, 0.15, 0.20, 0.25]]
        filtered = filter_pivots(results, variance_threshold=0.15, min_candidates=0)
        assert len(filtered) == 3
        assert all(r.reward_std >= 0.15 for r in filtered)

    def test_mean_reward_range(self):
        results = [_make_result(std=0.2, mean=m) for m in [0.1, 0.3, 0.5, 0.7, 0.9]]
        filtered = filter_pivots(
            results, variance_threshold=0.0, min_candidates=0,
            mean_reward_range=(0.2, 0.8),
        )
        assert len(filtered) == 3
        assert all(0.2 <= r.reward_mean <= 0.8 for r in filtered)

    def test_max_candidates(self):
        results = [_make_result(std=0.1 * (i + 1)) for i in range(10)]
        filtered = filter_pivots(
            results, variance_threshold=0.0, min_candidates=0, max_candidates=3,
        )
        assert len(filtered) == 3
        # Should keep top-3 by variance (descending)
        stds = [r.reward_std for r in filtered]
        assert stds == sorted(stds, reverse=True)
        assert stds[0] == pytest.approx(1.0)

    def test_min_candidates_warning(self, caplog):
        results = [_make_result(std=0.3), _make_result(std=0.4)]
        with caplog.at_level(logging.WARNING):
            filtered = filter_pivots(results, variance_threshold=0.1, min_candidates=10)
        assert len(filtered) == 2
        assert "Only 2 pivots found" in caplog.text


# ---------------------------------------------------------------------------
# pivots_to_dataset
# ---------------------------------------------------------------------------

class TestPivotsToDataset:

    def test_format(self):
        pr = _make_result(std=0.3, mean=0.6)
        pr.is_pivot = True
        rows = pivots_to_dataset([pr])
        assert len(rows) == 1
        row = rows[0]
        assert isinstance(row["prompt"], list)
        assert row["prompt"][0]["role"] == "user"
        assert row["ground_truth_tool"] == "tool"
        assert row["ground_truth_args_json"] == '{"a":1}'
        meta = row["pivot_metadata"]
        assert meta["reward_mean"] == pytest.approx(0.6)
        assert meta["reward_std"] == pytest.approx(0.3)
        assert meta["num_rollouts"] == 8
        assert "source_file" in meta
        assert "jsonl_hash" in meta
