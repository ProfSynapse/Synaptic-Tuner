"""Tests for flywheel_cycle_to_run_record adapter in shared.experiment_tracking.adapters."""
from __future__ import annotations

import pytest

from shared.experiment_tracking.adapters import flywheel_cycle_to_run_record
from shared.experiment_tracking.registry import RunRecord


class TestFlywheelCycleToRunRecord:
    """flywheel_cycle_to_run_record() produces valid RunRecord."""

    def _make_cycle_data(self, **overrides) -> dict:
        data = {
            "version_id": "v003",
            "record_counts": {"sft": 100, "kto_pos": 50, "kto_neg": 30, "grpo": 20},
            "filter_criteria": {"sft_threshold": 0.8, "kto_min_threshold": 0.3},
            "source_model_id": "llama-3.1-8b",
            "content_hash": "abc123def456",
        }
        data.update(overrides)
        return data

    def test_produces_run_record(self):
        cycle_data = self._make_cycle_data()
        record = flywheel_cycle_to_run_record(cycle_data, output_dir="/data/flywheel/v003")
        assert isinstance(record, RunRecord)

    def test_run_type_is_flywheel_cycle(self):
        record = flywheel_cycle_to_run_record(
            self._make_cycle_data(), output_dir="/data/flywheel/v003",
        )
        assert record.run_type == "flywheel_cycle"

    def test_status_is_completed(self):
        record = flywheel_cycle_to_run_record(
            self._make_cycle_data(), output_dir="/data/flywheel/v003",
        )
        assert record.status == "completed"

    def test_name_includes_version_and_count(self):
        record = flywheel_cycle_to_run_record(
            self._make_cycle_data(), output_dir="/data/flywheel/v003",
        )
        assert "v003" in record.name
        assert "200" in record.name  # 100+50+30+20 = 200

    def test_output_dir_set(self):
        record = flywheel_cycle_to_run_record(
            self._make_cycle_data(), output_dir="/data/flywheel/v003",
        )
        assert record.output_dir == "/data/flywheel/v003"

    def test_model_name_from_source_model(self):
        record = flywheel_cycle_to_run_record(
            self._make_cycle_data(), output_dir="/data/flywheel/v003",
        )
        assert record.model_name == "llama-3.1-8b"

    def test_dataset_source_includes_version(self):
        record = flywheel_cycle_to_run_record(
            self._make_cycle_data(), output_dir="/data/flywheel/v003",
        )
        assert record.dataset_source == "flywheel/v003"

    def test_tags_contain_method_and_hash(self):
        record = flywheel_cycle_to_run_record(
            self._make_cycle_data(), output_dir="/data/flywheel/v003",
        )
        assert record.tags["method"] == "flywheel"
        assert record.tags["version"] == "v003"
        assert record.tags["content_hash"] == "abc123def456"

    def test_primary_metric_is_total_examples(self):
        record = flywheel_cycle_to_run_record(
            self._make_cycle_data(), output_dir="/data/flywheel/v003",
        )
        assert record.primary_metric == 200.0
        assert record.primary_metric_name == "total_examples"

    def test_custom_run_id(self):
        record = flywheel_cycle_to_run_record(
            self._make_cycle_data(),
            output_dir="/data/flywheel/v003",
            run_id="custom-id-123",
        )
        assert record.run_id == "custom-id-123"

    def test_parent_run_id(self):
        record = flywheel_cycle_to_run_record(
            self._make_cycle_data(),
            output_dir="/data/flywheel/v003",
            parent_run_id="parent-run-456",
        )
        assert record.parent_run_id == "parent-run-456"

    def test_auto_generated_run_id(self):
        record = flywheel_cycle_to_run_record(
            self._make_cycle_data(), output_dir="/data/flywheel/v003",
        )
        assert record.run_id  # Non-empty
        assert len(record.run_id) > 10  # UUID-length

    def test_empty_record_counts_handled(self):
        cycle_data = self._make_cycle_data(record_counts={})
        record = flywheel_cycle_to_run_record(cycle_data, output_dir="/data")
        assert record.primary_metric == 0.0
        assert "0" in record.name

    def test_timestamp_is_set(self):
        record = flywheel_cycle_to_run_record(
            self._make_cycle_data(), output_dir="/data/flywheel/v003",
        )
        assert record.timestamp  # Non-empty ISO string
        assert "T" in record.timestamp
