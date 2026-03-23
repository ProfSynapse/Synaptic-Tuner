"""Tests for shared.flywheel.experiment_config — ExperimentConfig dataclass + YAML loading."""

from pathlib import Path
from unittest.mock import patch

import pytest

from shared.flywheel.experiment_config import ExperimentConfig, load_experiment_config


class TestExperimentConfigDefaults:
    def test_default_values(self):
        cfg = ExperimentConfig()
        assert cfg.max_experiments == 20
        assert cfg.trainer_type == "sft"
        assert cfg.search_strategy == "llm_surrogate"
        assert cfg.search_space == {}

    def test_to_dict_roundtrip(self):
        cfg = ExperimentConfig(max_experiments=5, trainer_type="kto")
        d = cfg.to_dict()
        assert d["max_experiments"] == 5
        assert d["trainer_type"] == "kto"


class TestExperimentConfigFromDict:
    def test_flat_keys(self):
        data = {"max_experiments": 10, "trainer_type": "kto", "search_strategy": "random"}
        cfg = ExperimentConfig.from_dict(data)
        assert cfg.max_experiments == 10
        assert cfg.trainer_type == "kto"
        assert cfg.search_strategy == "random"

    def test_nested_experiment_loop_key(self):
        data = {"experiment_loop": {"max_experiments": 3, "output_dir": "/tmp/exp"}}
        cfg = ExperimentConfig.from_dict(data)
        assert cfg.max_experiments == 3
        assert cfg.output_dir == "/tmp/exp"

    def test_ignores_unknown_keys(self):
        data = {"max_experiments": 5, "bogus_key": "ignored", "another": 42}
        cfg = ExperimentConfig.from_dict(data)
        assert cfg.max_experiments == 5
        assert not hasattr(cfg, "bogus_key")


class TestExperimentConfigValidate:
    def test_valid_config_no_issues(self):
        cfg = ExperimentConfig()
        assert cfg.validate() == []

    def test_max_experiments_below_one(self):
        cfg = ExperimentConfig(max_experiments=0)
        issues = cfg.validate()
        assert any("max_experiments" in i for i in issues)

    def test_invalid_trainer_type(self):
        cfg = ExperimentConfig(trainer_type="grpo")
        issues = cfg.validate()
        assert any("trainer_type" in i for i in issues)

    def test_invalid_search_strategy(self):
        cfg = ExperimentConfig(search_strategy="bayesian")
        issues = cfg.validate()
        assert any("search_strategy" in i for i in issues)

    def test_timeout_too_short(self):
        cfg = ExperimentConfig(training_timeout_seconds=30)
        issues = cfg.validate()
        assert any("training_timeout_seconds" in i for i in issues)

    def test_timeout_warning_for_many_steps(self):
        cfg = ExperimentConfig(max_steps_per_experiment=500, training_timeout_seconds=60)
        issues = cfg.validate()
        assert any("may be too short" in i for i in issues)

    def test_empty_search_space_list(self):
        cfg = ExperimentConfig(search_space={"lr": []})
        issues = cfg.validate()
        assert any("search_space" in i and "lr" in i for i in issues)

    def test_non_list_search_space_value(self):
        cfg = ExperimentConfig(search_space={"lr": 0.001})
        issues = cfg.validate()
        assert any("search_space" in i for i in issues)

    def test_surrogate_retrain_every_zero(self):
        cfg = ExperimentConfig(surrogate_retrain_every=0)
        issues = cfg.validate()
        assert any("surrogate_retrain_every" in i for i in issues)


class TestLoadExperimentConfig:
    def test_missing_file_returns_defaults(self, tmp_path):
        cfg = load_experiment_config(tmp_path / "nonexistent.yaml")
        assert isinstance(cfg, ExperimentConfig)
        assert cfg.max_experiments == 20

    def test_loads_from_yaml(self, tmp_path):
        yaml_path = tmp_path / "experiment_loop.yaml"
        yaml_path.write_text(
            "experiment_loop:\n  max_experiments: 7\n  trainer_type: kto\n",
            encoding="utf-8",
        )
        cfg = load_experiment_config(yaml_path)
        assert cfg.max_experiments == 7
        assert cfg.trainer_type == "kto"
