"""Tests for shared.flywheel.config — FlywheelConfig and load_flywheel_config."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from shared.flywheel.config import FlywheelConfig, load_flywheel_config


class TestFlywheelConfigDefaults:
    """FlywheelConfig default values match expected operational thresholds."""

    def test_scoring_defaults(self):
        cfg = FlywheelConfig()
        assert cfg.sft_threshold == 0.8
        assert cfg.kto_min_threshold == 0.3
        assert cfg.ambiguous_min == 0.4
        assert cfg.ambiguous_max == 0.7
        assert cfg.max_errors_before_zero == 5
        assert cfg.scoring_method == "error_count"

    def test_text_response_policy_default(self):
        cfg = FlywheelConfig()
        assert cfg.text_response_policy == "skip"

    def test_grpo_defaults(self):
        cfg = FlywheelConfig()
        assert cfg.grpo_enabled is True
        assert cfg.grpo_reward_scale == 1.0

    def test_storage_defaults(self):
        cfg = FlywheelConfig()
        assert cfg.catalog_backend == "sqlite"
        assert cfg.catalog_path == ".tracking/flywheel.db"
        assert cfg.datasets_dir == "Datasets/flywheel"

    def test_retrain_defaults(self):
        cfg = FlywheelConfig()
        assert cfg.min_new_examples == 500
        assert cfg.min_sft_examples == 100
        assert cfg.min_quality_score == 0.6

    def test_proxy_defaults(self):
        cfg = FlywheelConfig()
        assert cfg.proxy_port == 8080
        assert cfg.vllm_host == "localhost"
        assert cfg.vllm_port == 8000


class TestFlywheelConfigOverrides:
    """FlywheelConfig can be constructed with custom values."""

    def test_override_thresholds(self):
        cfg = FlywheelConfig(sft_threshold=0.9, kto_min_threshold=0.5)
        assert cfg.sft_threshold == 0.9
        assert cfg.kto_min_threshold == 0.5

    def test_text_response_policy_options(self):
        for policy in ("sft", "kto", "skip"):
            cfg = FlywheelConfig(text_response_policy=policy)
            assert cfg.text_response_policy == policy

    def test_fitness_config_path(self):
        cfg = FlywheelConfig(fitness_config_path="/path/to/rules.yaml")
        assert cfg.fitness_config_path == "/path/to/rules.yaml"


class TestToFitnessConfig:
    """to_fitness_config() produces correct FitnessEvaluator config dict."""

    def test_inline_rules(self):
        rules = [{"type": "xml", "name": "test_rule"}]
        cfg = FlywheelConfig(validation_rules=rules)
        fitness = cfg.to_fitness_config()
        assert fitness["validations"] == rules
        assert fitness["scoring"]["method"] == "error_count"
        assert fitness["scoring"]["no_tool_call_score"] == 0.0
        assert "params" in fitness["scoring"]

    def test_empty_rules_produces_empty_validations(self):
        cfg = FlywheelConfig()
        fitness = cfg.to_fitness_config()
        assert fitness["validations"] == []

    def test_external_config_file(self, tmp_path):
        rules_data = {
            "validations": [{"type": "json", "name": "schema_check"}],
            "scoring": {"method": "weighted"},
        }
        rules_file = tmp_path / "rules.yaml"
        with open(rules_file, "w") as f:
            yaml.dump(rules_data, f)

        cfg = FlywheelConfig(fitness_config_path=str(rules_file))
        fitness = cfg.to_fitness_config()
        assert fitness["validations"] == rules_data["validations"]
        assert fitness["scoring"]["method"] == "weighted"

    def test_missing_external_config_falls_back_to_inline(self):
        cfg = FlywheelConfig(
            fitness_config_path="/nonexistent/path.yaml",
            validation_rules=[{"type": "regex"}],
        )
        fitness = cfg.to_fitness_config()
        assert fitness["validations"] == [{"type": "regex"}]


class TestLoadFlywheelConfig:
    """load_flywheel_config() loads from YAML and handles missing files."""

    def test_missing_file_returns_defaults(self):
        cfg = load_flywheel_config("/nonexistent/config.yaml")
        assert cfg.sft_threshold == 0.8
        assert cfg.catalog_backend == "sqlite"

    def test_loads_from_yaml(self, tmp_path):
        config_data = {
            "sft_threshold": 0.9,
            "kto_min_threshold": 0.4,
            "text_response_policy": "sft",
            "proxy_port": 9090,
        }
        config_file = tmp_path / "flywheel.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        cfg = load_flywheel_config(config_file)
        assert cfg.sft_threshold == 0.9
        assert cfg.kto_min_threshold == 0.4
        assert cfg.text_response_policy == "sft"
        assert cfg.proxy_port == 9090

    def test_ignores_unknown_fields(self, tmp_path):
        config_data = {
            "sft_threshold": 0.85,
            "unknown_field": "should_be_ignored",
            "another_future_field": 42,
        }
        config_file = tmp_path / "flywheel.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        cfg = load_flywheel_config(config_file)
        assert cfg.sft_threshold == 0.85
        assert not hasattr(cfg, "unknown_field")

    def test_partial_override_keeps_defaults(self, tmp_path):
        config_data = {"proxy_port": 7777}
        config_file = tmp_path / "flywheel.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        cfg = load_flywheel_config(config_file)
        assert cfg.proxy_port == 7777
        assert cfg.sft_threshold == 0.8  # default preserved
