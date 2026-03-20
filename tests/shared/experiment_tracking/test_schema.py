"""Tests for shared/experiment_tracking/schema.py — RunRecord and RunFilter."""
from __future__ import annotations

import json

import pytest

from shared.experiment_tracking.schema import RunFilter, RunRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(**overrides) -> RunRecord:
    """Build a RunRecord with sensible defaults, overriding as needed."""
    defaults = dict(
        run_id="run-001",
        run_type="sft",
        name="SFT run 2026-03-14",
        timestamp="2026-03-14T18:00:00+00:00",
        status="completed",
        output_dir="/runs/sft_20260314",
    )
    defaults.update(overrides)
    return RunRecord(**defaults)


# ===========================================================================
# RunRecord
# ===========================================================================

class TestRunRecord:
    """Unit tests for RunRecord dataclass."""

    def test_required_fields_present(self):
        record = _make_record()
        assert record.run_id == "run-001"
        assert record.run_type == "sft"
        assert record.name == "SFT run 2026-03-14"
        assert record.timestamp == "2026-03-14T18:00:00+00:00"
        assert record.status == "completed"
        assert record.output_dir == "/runs/sft_20260314"

    def test_optional_fields_default_to_none(self):
        record = _make_record()
        assert record.parent_run_id is None
        assert record.model_name is None
        assert record.dataset_source is None
        assert record.primary_metric is None
        assert record.primary_metric_name is None
        assert record.hardware is None

    def test_tags_default_to_empty_dict(self):
        record = _make_record()
        assert record.tags == {}

    def test_schema_version_default(self):
        record = _make_record()
        assert record.schema_version == 1

    def test_all_optional_fields_populated(self):
        record = _make_record(
            parent_run_id="parent-001",
            tags={"method": "sft"},
            model_name="unsloth/Qwen2.5-7B",
            dataset_source="tools_v1.8.jsonl",
            primary_metric=0.45,
            primary_metric_name="final_loss",
            hardware="NVIDIA GeForce RTX 3090",
        )
        assert record.parent_run_id == "parent-001"
        assert record.model_name == "unsloth/Qwen2.5-7B"
        assert record.primary_metric == 0.45
        assert record.hardware == "NVIDIA GeForce RTX 3090"

    # -- Serialization round-trip ------------------------------------------

    def test_to_json_line_produces_valid_json(self):
        record = _make_record(tags={"method": "sft"})
        line = record.to_json_line()
        parsed = json.loads(line)
        assert parsed["run_id"] == "run-001"
        assert parsed["tags"] == {"method": "sft"}

    def test_to_json_line_no_trailing_newline(self):
        record = _make_record()
        line = record.to_json_line()
        assert "\n" not in line

    def test_from_json_line_round_trip(self):
        original = _make_record(
            tags={"method": "sft"},
            model_name="test-model",
            primary_metric=0.45,
        )
        line = original.to_json_line()
        restored = RunRecord.from_json_line(line)
        assert restored.run_id == original.run_id
        assert restored.run_type == original.run_type
        assert restored.tags == original.tags
        assert restored.model_name == original.model_name
        assert restored.primary_metric == original.primary_metric

    def test_from_dict_ignores_unknown_fields(self):
        """Forward compatibility: unknown fields are silently dropped."""
        data = {
            "run_id": "run-002",
            "run_type": "sft",
            "name": "test",
            "timestamp": "2026-01-01T00:00:00Z",
            "status": "completed",
            "output_dir": "/out",
            "future_field": "should_be_ignored",
            "another_unknown": 42,
        }
        record = RunRecord.from_dict(data)
        assert record.run_id == "run-002"
        assert not hasattr(record, "future_field")

    def test_from_dict_preserves_schema_version(self):
        data = {
            "run_id": "run-003",
            "run_type": "ml",
            "name": "ML run",
            "timestamp": "2026-01-01T00:00:00Z",
            "status": "completed",
            "output_dir": "/out",
            "schema_version": 2,
        }
        record = RunRecord.from_dict(data)
        assert record.schema_version == 2

    def test_from_dict_missing_optional_fields(self):
        data = {
            "run_id": "run-004",
            "run_type": "kto",
            "name": "KTO run",
            "timestamp": "2026-01-01T00:00:00Z",
            "status": "completed",
            "output_dir": "/out",
        }
        record = RunRecord.from_dict(data)
        assert record.parent_run_id is None
        assert record.tags == {}

    def test_from_dict_missing_required_field_raises(self):
        data = {"run_type": "sft", "name": "incomplete"}
        with pytest.raises(TypeError):
            RunRecord.from_dict(data)


# ===========================================================================
# RunFilter
# ===========================================================================

class TestRunFilter:
    """Unit tests for RunFilter.matches()."""

    def test_empty_filter_matches_everything(self):
        filt = RunFilter()
        assert filt.matches(_make_record(run_type="sft"))
        assert filt.matches(_make_record(run_type="kto"))
        assert filt.matches(_make_record(run_type="ml"))

    def test_filter_by_run_type_string(self):
        filt = RunFilter(run_type="sft")
        assert filt.matches(_make_record(run_type="sft"))
        assert not filt.matches(_make_record(run_type="kto"))

    def test_filter_by_run_type_list(self):
        filt = RunFilter(run_type=["sft", "kto"])
        assert filt.matches(_make_record(run_type="sft"))
        assert filt.matches(_make_record(run_type="kto"))
        assert not filt.matches(_make_record(run_type="ml"))

    def test_filter_by_status(self):
        filt = RunFilter(status="completed")
        assert filt.matches(_make_record(status="completed"))
        assert not filt.matches(_make_record(status="failed"))

    def test_filter_by_model_name_case_insensitive(self):
        filt = RunFilter(model_name="qwen")
        assert filt.matches(_make_record(model_name="unsloth/Qwen2.5-7B"))
        assert not filt.matches(_make_record(model_name="llama-3"))

    def test_filter_by_model_name_none_record(self):
        """Record with no model_name should not match a model_name filter."""
        filt = RunFilter(model_name="qwen")
        assert not filt.matches(_make_record(model_name=None))

    def test_filter_by_since(self):
        filt = RunFilter(since="2026-03-14T00:00:00+00:00")
        assert filt.matches(_make_record(timestamp="2026-03-14T18:00:00+00:00"))
        assert filt.matches(_make_record(timestamp="2026-03-14T00:00:00+00:00"))
        assert not filt.matches(_make_record(timestamp="2026-03-13T23:59:59+00:00"))

    def test_filter_by_until(self):
        filt = RunFilter(until="2026-03-14T18:00:00+00:00")
        assert filt.matches(_make_record(timestamp="2026-03-14T18:00:00+00:00"))
        assert not filt.matches(_make_record(timestamp="2026-03-14T18:00:01+00:00"))

    def test_filter_by_since_and_until_range(self):
        filt = RunFilter(
            since="2026-03-14T00:00:00+00:00",
            until="2026-03-14T23:59:59+00:00",
        )
        assert filt.matches(_make_record(timestamp="2026-03-14T12:00:00+00:00"))
        assert not filt.matches(_make_record(timestamp="2026-03-15T00:00:00+00:00"))

    def test_filter_by_tags(self):
        filt = RunFilter(tags={"method": "sft"})
        assert filt.matches(_make_record(tags={"method": "sft", "provider": "local"}))
        assert not filt.matches(_make_record(tags={"method": "kto"}))
        assert not filt.matches(_make_record(tags={}))

    def test_filter_multiple_tags_all_must_match(self):
        filt = RunFilter(tags={"method": "sft", "provider": "cloud"})
        assert filt.matches(
            _make_record(tags={"method": "sft", "provider": "cloud", "extra": "yes"})
        )
        assert not filt.matches(_make_record(tags={"method": "sft", "provider": "local"}))

    def test_filter_and_logic_all_fields(self):
        """Multiple filter fields combine with AND logic."""
        filt = RunFilter(run_type="sft", status="completed", model_name="qwen")
        assert filt.matches(
            _make_record(run_type="sft", status="completed", model_name="Qwen2.5-7B")
        )
        # Fails on status
        assert not filt.matches(
            _make_record(run_type="sft", status="failed", model_name="Qwen2.5-7B")
        )
        # Fails on run_type
        assert not filt.matches(
            _make_record(run_type="kto", status="completed", model_name="Qwen2.5-7B")
        )
