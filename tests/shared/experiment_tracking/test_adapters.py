"""Tests for shared/experiment_tracking/adapters.py — lineage-to-RunRecord conversion."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.experiment_tracking.adapters import (
    eval_to_run_record,
    grpo_log_to_run_record,
    kto_lineage_to_run_record,
    manifest_to_run_record,
    ml_tracking_to_run_record,
    sft_lineage_to_run_record,
)
from shared.experiment_tracking.schema import RunRecord

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures" / "tracking"


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text())


# ===========================================================================
# SFT Adapter
# ===========================================================================

class TestSFTAdapter:
    def test_basic_conversion(self):
        lineage = _load_fixture("sft_training_lineage.json")
        record = sft_lineage_to_run_record(lineage, "/runs/sft_20260314")

        assert isinstance(record, RunRecord)
        assert record.run_type == "sft"
        assert record.status == "completed"
        assert record.output_dir == "/runs/sft_20260314"
        assert record.model_name == "unsloth/Qwen2.5-7B"
        assert record.dataset_source == "Datasets/tools_datasets/thinking/agentManager/tools_v1.8.jsonl"
        assert record.primary_metric == 0.4523
        assert record.primary_metric_name == "final_loss"
        assert record.hardware == "NVIDIA GeForce RTX 3090"
        assert record.timestamp == "2026-03-14T18:00:00+00:00"

    def test_cloud_flag(self):
        lineage = _load_fixture("sft_training_lineage.json")
        record = sft_lineage_to_run_record(lineage, "/runs/cloud_sft", cloud=True)

        assert record.run_type == "cloud_sft"
        assert record.tags["provider"] == "cloud"

    def test_custom_run_id(self):
        lineage = _load_fixture("sft_training_lineage.json")
        record = sft_lineage_to_run_record(lineage, "/runs/sft", run_id="custom-id")

        assert record.run_id == "custom-id"

    def test_auto_generated_run_id(self):
        lineage = _load_fixture("sft_training_lineage.json")
        record = sft_lineage_to_run_record(lineage, "/runs/sft")

        assert record.run_id  # Non-empty
        assert len(record.run_id) == 36  # UUID4 format

    def test_missing_results(self):
        lineage = {"timestamp": "2026-01-01T00:00:00Z"}
        record = sft_lineage_to_run_record(lineage, "/runs/sft")

        assert record.primary_metric is None
        assert record.primary_metric_name is None
        assert record.model_name is None

    def test_tags_include_method(self):
        lineage = _load_fixture("sft_training_lineage.json")
        record = sft_lineage_to_run_record(lineage, "/runs/sft")

        assert record.tags["method"] == "sft"
        assert record.tags["provider"] == "local"


# ===========================================================================
# KTO Adapter
# ===========================================================================

class TestKTOAdapter:
    def test_basic_conversion(self):
        lineage = _load_fixture("kto_training_lineage.json")
        record = kto_lineage_to_run_record(lineage, "/runs/kto_20260314")

        assert record.run_type == "kto"
        assert record.model_name == "unsloth/Qwen2.5-7B-SFT-v1"
        assert record.primary_metric == 0.6891
        assert record.primary_metric_name == "final_loss"
        assert record.timestamp == "2026-03-14T20:00:00+00:00"

    def test_cloud_flag(self):
        lineage = _load_fixture("kto_training_lineage.json")
        record = kto_lineage_to_run_record(lineage, "/runs/kto", cloud=True)

        assert record.run_type == "cloud_kto"
        assert record.tags["provider"] == "cloud"

    def test_custom_run_id(self):
        lineage = _load_fixture("kto_training_lineage.json")
        record = kto_lineage_to_run_record(lineage, "/runs/kto", run_id="kto-custom")

        assert record.run_id == "kto-custom"


# ===========================================================================
# ML Tracking Adapter
# ===========================================================================

class TestMLAdapter:
    def test_basic_conversion(self):
        tracking = _load_fixture("ml_tracking.json")
        record = ml_tracking_to_run_record(tracking, "/output")

        assert record.run_type == "ml"
        assert record.name == "lightgbm_iris_001"
        assert record.model_name == "lightgbm"
        assert record.primary_metric == 0.9667  # accuracy
        assert record.primary_metric_name == "accuracy"
        assert record.tags["algorithm"] == "lightgbm"
        assert record.tags["task_type"] == "classification"

    def test_metric_priority_order(self):
        """accuracy > f1 > r2 > rmse > mse."""
        tracking = {
            "run_name": "test",
            "started_at": "2026-01-01T00:00:00Z",
            "params": {"algorithm": "xgboost", "task_type": "regression"},
            "metrics": {"rmse": 0.5, "r2": 0.95, "mse": 0.25},
        }
        record = ml_tracking_to_run_record(tracking, "/out")
        assert record.primary_metric == 0.95
        assert record.primary_metric_name == "r2"

    def test_no_matching_metric(self):
        tracking = {
            "run_name": "test",
            "started_at": "2026-01-01T00:00:00Z",
            "params": {"algorithm": "custom"},
            "metrics": {"custom_metric": 0.99},
        }
        record = ml_tracking_to_run_record(tracking, "/out")
        assert record.primary_metric is None
        assert record.primary_metric_name is None

    def test_missing_params(self):
        tracking = {
            "run_name": "bare",
            "started_at": "2026-01-01T00:00:00Z",
            "params": {},
            "metrics": {},
        }
        record = ml_tracking_to_run_record(tracking, "/out")
        assert record.tags["algorithm"] == "unknown"
        assert record.tags["task_type"] == "unknown"


# ===========================================================================
# Eval Adapter
# ===========================================================================

class TestEvalAdapter:
    def test_basic_conversion(self):
        lineage = _load_fixture("evaluation_lineage.json")
        record = eval_to_run_record(lineage, "/eval/output")

        assert record.run_type == "evaluation"
        assert record.model_name == "unsloth/Qwen2.5-7B-SFT-v1"
        assert record.primary_metric == 0.85
        assert record.primary_metric_name == "pass_rate"
        assert record.timestamp == "2026-03-14T22:00:00+00:00"
        assert "tool_prompts" in record.tags["suite"]

    def test_parent_run_id(self):
        lineage = _load_fixture("evaluation_lineage.json")
        record = eval_to_run_record(
            lineage, "/eval/output", parent_run_id="train-001"
        )

        assert record.parent_run_id == "train-001"

    def test_no_parent_run_id(self):
        lineage = _load_fixture("evaluation_lineage.json")
        record = eval_to_run_record(lineage, "/eval/output")

        assert record.parent_run_id is None

    def test_empty_results(self):
        lineage = {
            "evaluation_timestamp": "2026-01-01T00:00:00Z",
            "model_evaluated": "test-model",
            "test_config": {"test_suites": []},
            "results_summary": {},
        }
        record = eval_to_run_record(lineage, "/out")
        assert record.primary_metric is None


# ===========================================================================
# Cloud Manifest Adapter
# ===========================================================================

class TestManifestAdapter:
    def test_basic_conversion(self):
        manifest = _load_fixture("cloud_manifest.json")
        record = manifest_to_run_record(manifest)

        assert record.run_type == "cloud_sft"
        assert record.status == "completed"
        assert record.output_dir == "/runs/cloud_sft_20260314"
        assert record.tags["provider"] == "hf_jobs"
        assert record.tags["method"] == "sft"
        assert record.timestamp == "2026-03-14T19:00:00+00:00"

    def test_kto_method(self):
        manifest = {
            "generated_at": "2026-01-01T00:00:00Z",
            "method": "kto",
            "provider": "runpod",
            "paths": {"run_dir": "/runs/kto"},
        }
        record = manifest_to_run_record(manifest)
        assert record.run_type == "cloud_kto"

    def test_custom_run_id(self):
        manifest = _load_fixture("cloud_manifest.json")
        record = manifest_to_run_record(manifest, run_id="cloud-custom")
        assert record.run_id == "cloud-custom"

    def test_missing_paths(self):
        manifest = {"method": "sft", "generated_at": "2026-01-01T00:00:00Z"}
        record = manifest_to_run_record(manifest)
        assert record.output_dir == ""


# ===========================================================================
# GRPO Adapter
# ===========================================================================

class TestGRPOAdapter:
    def test_basic_conversion(self):
        log_entries = [
            {"step": 10, "loss": 2.5, "reward": 0.3, "timestamp": "2026-01-01T00:00:00Z"},
            {"step": 50, "loss": 1.8, "reward": 0.7, "timestamp": "2026-01-01T01:00:00Z"},
            {"event": "train_end", "total_steps": 50, "timestamp": "2026-01-01T01:05:00Z"},
        ]
        record = grpo_log_to_run_record(
            log_entries, "/runs/grpo",
            model_name="test-model", dataset_source="data.jsonl",
        )

        assert record.run_type == "grpo"
        assert record.primary_metric == 0.7  # Last step reward
        assert record.primary_metric_name == "reward"
        assert record.model_name == "test-model"
        assert "50 steps" in record.name

    def test_cloud_flag(self):
        log_entries = [
            {"step": 10, "loss": 2.0},
            {"event": "train_end", "total_steps": 10},
        ]
        record = grpo_log_to_run_record(log_entries, "/runs/grpo", cloud=True)
        assert record.run_type == "cloud_grpo"
        assert record.tags["provider"] == "cloud"

    def test_fallback_to_loss_when_no_reward(self):
        log_entries = [
            {"step": 10, "loss": 2.5},
            {"step": 50, "loss": 1.2},
            {"event": "train_end", "total_steps": 50},
        ]
        record = grpo_log_to_run_record(log_entries, "/runs/grpo")
        assert record.primary_metric == 1.2
        assert record.primary_metric_name == "loss"

    def test_empty_log_entries(self):
        record = grpo_log_to_run_record([], "/runs/grpo")
        assert record.run_type == "grpo"
        assert record.primary_metric is None
        assert record.name == "GRPO run"

    def test_no_train_end_event(self):
        log_entries = [
            {"step": 10, "loss": 2.5, "reward": 0.3},
            {"step": 20, "loss": 2.0, "reward": 0.5},
        ]
        record = grpo_log_to_run_record(log_entries, "/runs/grpo")
        assert record.primary_metric == 0.5
        assert "20 steps" in record.name  # Falls back to last step
