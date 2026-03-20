"""Tests for shared/experiment_tracking/registry.py — RunRegistry JSONL store."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.experiment_tracking.registry import RunRegistry
from shared.experiment_tracking.schema import RunFilter, RunRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(**overrides) -> RunRecord:
    """Build a RunRecord with sensible defaults."""
    defaults = dict(
        run_id="run-001",
        run_type="sft",
        name="SFT run",
        timestamp="2026-03-14T18:00:00+00:00",
        status="completed",
        output_dir="/runs/sft_20260314",
    )
    defaults.update(overrides)
    return RunRecord(**defaults)


# ===========================================================================
# RunRegistry — Core Operations
# ===========================================================================

class TestRunRegistryCore:
    """Register, read back, and query runs."""

    def test_register_and_read_back(self, tmp_path: Path):
        registry = RunRegistry(tmp_path / "registry.jsonl")
        record = _make_record()
        run_id = registry.register_run(record)

        assert run_id == "run-001"
        runs = registry.find_runs()
        assert len(runs) == 1
        assert runs[0].run_id == "run-001"
        assert runs[0].run_type == "sft"

    def test_register_multiple_runs(self, tmp_path: Path):
        registry = RunRegistry(tmp_path / "registry.jsonl")
        registry.register_run(_make_record(run_id="run-001", run_type="sft"))
        registry.register_run(_make_record(run_id="run-002", run_type="kto"))
        registry.register_run(_make_record(run_id="run-003", run_type="ml"))

        runs = registry.find_runs()
        assert len(runs) == 3
        assert [r.run_id for r in runs] == ["run-001", "run-002", "run-003"]

    def test_get_run_found(self, tmp_path: Path):
        registry = RunRegistry(tmp_path / "registry.jsonl")
        registry.register_run(_make_record(run_id="run-001"))
        registry.register_run(_make_record(run_id="run-002"))

        result = registry.get_run("run-002")
        assert result is not None
        assert result.run_id == "run-002"

    def test_get_run_not_found(self, tmp_path: Path):
        registry = RunRegistry(tmp_path / "registry.jsonl")
        registry.register_run(_make_record(run_id="run-001"))

        result = registry.get_run("nonexistent")
        assert result is None

    def test_creates_parent_directories(self, tmp_path: Path):
        deep_path = tmp_path / "deep" / "nested" / "registry.jsonl"
        registry = RunRegistry(deep_path)
        registry.register_run(_make_record())

        assert deep_path.exists()

    def test_registry_file_is_valid_jsonl(self, tmp_path: Path):
        path = tmp_path / "registry.jsonl"
        registry = RunRegistry(path)
        registry.register_run(_make_record(run_id="run-001"))
        registry.register_run(_make_record(run_id="run-002"))

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            data = json.loads(line)
            assert "run_id" in data


# ===========================================================================
# RunRegistry — Filtering
# ===========================================================================

class TestRunRegistryFiltering:
    """Test find_runs with various filters."""

    @pytest.fixture
    def populated_registry(self, tmp_path: Path) -> RunRegistry:
        registry = RunRegistry(tmp_path / "registry.jsonl")
        registry.register_run(_make_record(
            run_id="sft-001", run_type="sft", status="completed",
            timestamp="2026-03-14T10:00:00+00:00",
            model_name="unsloth/Qwen2.5-7B",
            tags={"method": "sft", "provider": "local"},
        ))
        registry.register_run(_make_record(
            run_id="kto-001", run_type="kto", status="completed",
            timestamp="2026-03-14T15:00:00+00:00",
            model_name="unsloth/Qwen2.5-7B-SFT",
            tags={"method": "kto", "provider": "local"},
        ))
        registry.register_run(_make_record(
            run_id="ml-001", run_type="ml", status="completed",
            timestamp="2026-03-14T12:00:00+00:00",
            model_name="lightgbm",
            tags={"method": "ml", "algorithm": "lightgbm"},
        ))
        registry.register_run(_make_record(
            run_id="sft-002", run_type="sft", status="failed",
            timestamp="2026-03-15T08:00:00+00:00",
            model_name="unsloth/Qwen2.5-7B",
            tags={"method": "sft", "provider": "cloud"},
        ))
        return registry

    def test_filter_by_run_type(self, populated_registry: RunRegistry):
        runs = populated_registry.find_runs(RunFilter(run_type="sft"))
        assert len(runs) == 2
        assert all(r.run_type == "sft" for r in runs)

    def test_filter_by_status(self, populated_registry: RunRegistry):
        runs = populated_registry.find_runs(RunFilter(status="failed"))
        assert len(runs) == 1
        assert runs[0].run_id == "sft-002"

    def test_filter_by_model_name(self, populated_registry: RunRegistry):
        runs = populated_registry.find_runs(RunFilter(model_name="qwen"))
        assert len(runs) == 3  # All Qwen models

    def test_filter_by_timestamp_range(self, populated_registry: RunRegistry):
        runs = populated_registry.find_runs(RunFilter(
            since="2026-03-14T11:00:00+00:00",
            until="2026-03-14T16:00:00+00:00",
        ))
        assert len(runs) == 2
        assert {r.run_id for r in runs} == {"kto-001", "ml-001"}

    def test_filter_by_tags(self, populated_registry: RunRegistry):
        runs = populated_registry.find_runs(RunFilter(tags={"provider": "cloud"}))
        assert len(runs) == 1
        assert runs[0].run_id == "sft-002"

    def test_filter_combined(self, populated_registry: RunRegistry):
        runs = populated_registry.find_runs(RunFilter(
            run_type="sft", status="completed",
        ))
        assert len(runs) == 1
        assert runs[0].run_id == "sft-001"

    def test_no_filter_returns_all(self, populated_registry: RunRegistry):
        runs = populated_registry.find_runs()
        assert len(runs) == 4

    def test_filter_no_matches(self, populated_registry: RunRegistry):
        runs = populated_registry.find_runs(RunFilter(run_type="grpo"))
        assert runs == []


# ===========================================================================
# RunRegistry — Linkage
# ===========================================================================

class TestRunRegistryLinkage:
    """Test link_runs and get_linked_runs."""

    def test_link_and_query_child(self, tmp_path: Path):
        registry = RunRegistry(tmp_path / "registry.jsonl")
        registry.register_run(_make_record(run_id="train-001", run_type="sft"))
        registry.register_run(_make_record(run_id="eval-001", run_type="evaluation"))
        registry.link_runs(child_run_id="eval-001", parent_run_id="train-001")

        # Query from parent side → find child
        linked = registry.get_linked_runs("train-001")
        assert len(linked) == 1
        assert linked[0].run_id == "eval-001"

    def test_link_and_query_parent(self, tmp_path: Path):
        registry = RunRegistry(tmp_path / "registry.jsonl")
        registry.register_run(_make_record(run_id="train-001", run_type="sft"))
        registry.register_run(_make_record(run_id="eval-001", run_type="evaluation"))
        registry.link_runs(child_run_id="eval-001", parent_run_id="train-001")

        # Query from child side → find parent
        linked = registry.get_linked_runs("eval-001")
        assert len(linked) == 1
        assert linked[0].run_id == "train-001"

    def test_multiple_links(self, tmp_path: Path):
        registry = RunRegistry(tmp_path / "registry.jsonl")
        registry.register_run(_make_record(run_id="train-001", run_type="sft"))
        registry.register_run(_make_record(run_id="eval-001", run_type="evaluation"))
        registry.register_run(_make_record(run_id="eval-002", run_type="evaluation"))
        registry.link_runs(child_run_id="eval-001", parent_run_id="train-001")
        registry.link_runs(child_run_id="eval-002", parent_run_id="train-001")

        linked = registry.get_linked_runs("train-001")
        assert len(linked) == 2
        assert {r.run_id for r in linked} == {"eval-001", "eval-002"}

    def test_link_with_relationship_filter(self, tmp_path: Path):
        registry = RunRegistry(tmp_path / "registry.jsonl")
        registry.register_run(_make_record(run_id="train-001", run_type="sft"))
        registry.register_run(_make_record(run_id="eval-001", run_type="evaluation"))
        registry.register_run(_make_record(run_id="derived-001", run_type="kto"))
        registry.link_runs("eval-001", "train-001", relationship="parent")
        registry.link_runs("derived-001", "train-001", relationship="derived_from")

        # Filter by relationship
        parent_linked = registry.get_linked_runs("train-001", relationship="parent")
        assert len(parent_linked) == 1
        assert parent_linked[0].run_id == "eval-001"

        derived_linked = registry.get_linked_runs("train-001", relationship="derived_from")
        assert len(derived_linked) == 1
        assert derived_linked[0].run_id == "derived-001"

    def test_no_links_returns_empty(self, tmp_path: Path):
        registry = RunRegistry(tmp_path / "registry.jsonl")
        registry.register_run(_make_record(run_id="train-001"))

        linked = registry.get_linked_runs("train-001")
        assert linked == []

    def test_links_coexist_with_records_in_jsonl(self, tmp_path: Path):
        """Links are stored in the same JSONL file but don't interfere with records."""
        path = tmp_path / "registry.jsonl"
        registry = RunRegistry(path)
        registry.register_run(_make_record(run_id="train-001"))
        registry.register_run(_make_record(run_id="eval-001"))
        registry.link_runs("eval-001", "train-001")

        # Records should still load correctly (link lines are skipped)
        runs = registry.find_runs()
        assert len(runs) == 2

        # Verify file has 3 lines (2 records + 1 link)
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 3


# ===========================================================================
# RunRegistry — Edge Cases and Robustness
# ===========================================================================

class TestRunRegistryEdgeCases:
    """Empty registry, malformed lines, concurrent-ish writes."""

    def test_empty_registry_returns_empty_list(self, tmp_path: Path):
        registry = RunRegistry(tmp_path / "registry.jsonl")
        assert registry.find_runs() == []

    def test_nonexistent_file_returns_empty_list(self, tmp_path: Path):
        registry = RunRegistry(tmp_path / "nonexistent" / "registry.jsonl")
        assert registry.find_runs() == []

    def test_malformed_lines_skipped(self, tmp_path: Path):
        """Malformed JSON lines are skipped; valid lines still load."""
        path = tmp_path / "registry.jsonl"
        record = _make_record(run_id="good-001")
        good_line = record.to_json_line()

        # Write a mix of valid, malformed, and empty lines
        path.write_text(
            f"{good_line}\n"
            "this is not valid json\n"
            "\n"
            '{"run_id": "good-002", "run_type": "kto", "name": "KTO", '
            '"timestamp": "2026-01-01T00:00:00Z", "status": "completed", '
            '"output_dir": "/out"}\n'
        )

        registry = RunRegistry(path)
        runs = registry.find_runs()
        assert len(runs) == 2
        assert runs[0].run_id == "good-001"
        assert runs[1].run_id == "good-002"

    def test_blank_lines_skipped(self, tmp_path: Path):
        path = tmp_path / "registry.jsonl"
        record = _make_record(run_id="run-001")
        path.write_text(f"\n\n{record.to_json_line()}\n\n")

        registry = RunRegistry(path)
        runs = registry.find_runs()
        assert len(runs) == 1

    def test_multiple_appends_preserve_order(self, tmp_path: Path):
        registry = RunRegistry(tmp_path / "registry.jsonl")
        for i in range(10):
            registry.register_run(_make_record(run_id=f"run-{i:03d}"))

        runs = registry.find_runs()
        assert [r.run_id for r in runs] == [f"run-{i:03d}" for i in range(10)]

    def test_get_linked_runs_no_registry_file(self, tmp_path: Path):
        registry = RunRegistry(tmp_path / "nonexistent.jsonl")
        assert registry.get_linked_runs("any-id") == []
