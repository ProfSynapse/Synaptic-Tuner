"""
tests/flywheel/test_experiment_loop.py

Tests for the autonomous experiment loop:
- ExperimentConfig validation and loading
- Random config sampling
- LLM advisor prompt construction and YAML extraction
- Surrogate model fit/predict cycle
- Phase transition logic
- Results TSV writing and parsing
- Config merge logic
- Feature importance
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from shared.flywheel.experiment_config import (
    ExperimentConfig,
    load_experiment_config,
)
from shared.flywheel.experiment_loop import (
    ExperimentLoop,
    ExperimentResult,
    LLMAdvisor,
    SurrogateModel,
    _flatten_config,
    _merge_config_overrides,
    _random_sample,
    _trainer_script,
)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def sample_search_space():
    return {
        "learning_rate": [1e-4, 2e-4, 5e-4],
        "r": [8, 16, 32],
        "lora_alpha": [16, 32],
    }


@pytest.fixture
def sample_config(sample_search_space):
    return ExperimentConfig(
        max_experiments=5,
        max_steps_per_experiment=50,
        trainer_type="sft",
        search_space=sample_search_space,
        search_strategy="random",
        output_dir="/tmp/test_experiments",
    )


@pytest.fixture
def sample_results():
    return [
        ExperimentResult(
            experiment_id="exp_0000_abc",
            config={"learning_rate": 1e-4, "r": 16, "lora_alpha": 32},
            eval_score=0.75,
            training_loss=0.45,
            duration_seconds=120.0,
            status="completed",
        ),
        ExperimentResult(
            experiment_id="exp_0001_def",
            config={"learning_rate": 2e-4, "r": 8, "lora_alpha": 16},
            eval_score=0.82,
            training_loss=0.38,
            duration_seconds=115.0,
            status="completed",
        ),
        ExperimentResult(
            experiment_id="exp_0002_ghi",
            config={"learning_rate": 5e-4, "r": 32, "lora_alpha": 64},
            eval_score=0.0,
            training_loss=float("inf"),
            duration_seconds=10.0,
            status="failed",
        ),
    ]


# -----------------------------------------------------------------------
# ExperimentConfig tests
# -----------------------------------------------------------------------

class TestExperimentConfig:
    """Tests for ExperimentConfig dataclass."""

    def test_defaults(self):
        cfg = ExperimentConfig()
        assert cfg.max_experiments == 20
        assert cfg.trainer_type == "sft"
        assert cfg.search_strategy == "llm_surrogate"

    def test_validate_valid(self, sample_config):
        issues = sample_config.validate()
        assert issues == []

    def test_validate_bad_max_experiments(self):
        cfg = ExperimentConfig(max_experiments=0)
        issues = cfg.validate()
        assert any("max_experiments" in i for i in issues)

    def test_validate_bad_trainer_type(self):
        cfg = ExperimentConfig(trainer_type="grpo")
        issues = cfg.validate()
        assert any("trainer_type" in i for i in issues)

    def test_validate_bad_strategy(self):
        cfg = ExperimentConfig(search_strategy="bayesian")
        issues = cfg.validate()
        assert any("search_strategy" in i for i in issues)

    def test_validate_empty_search_space_value(self):
        cfg = ExperimentConfig(search_space={"lr": []})
        issues = cfg.validate()
        assert any("search_space" in i for i in issues)

    def test_from_dict_flat(self, sample_search_space):
        data = {
            "max_experiments": 10,
            "trainer_type": "kto",
            "search_space": sample_search_space,
        }
        cfg = ExperimentConfig.from_dict(data)
        assert cfg.max_experiments == 10
        assert cfg.trainer_type == "kto"

    def test_from_dict_nested(self, sample_search_space):
        data = {
            "experiment_loop": {
                "max_experiments": 15,
                "search_space": sample_search_space,
            }
        }
        cfg = ExperimentConfig.from_dict(data)
        assert cfg.max_experiments == 15

    def test_from_dict_ignores_unknown_fields(self):
        data = {"max_experiments": 5, "unknown_field": True}
        cfg = ExperimentConfig.from_dict(data)
        assert cfg.max_experiments == 5
        assert not hasattr(cfg, "unknown_field")

    def test_to_dict(self, sample_config):
        d = sample_config.to_dict()
        assert d["max_experiments"] == 5
        assert d["trainer_type"] == "sft"
        assert isinstance(d["search_space"], dict)


class TestLoadExperimentConfig:
    """Tests for load_experiment_config."""

    def test_load_from_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({
            "experiment_loop": {
                "max_experiments": 3,
                "trainer_type": "kto",
                "search_space": {"lr": [1e-4, 2e-4]},
            }
        }))
        cfg = load_experiment_config(config_file)
        assert cfg.max_experiments == 3
        assert cfg.trainer_type == "kto"

    def test_load_missing_file_returns_defaults(self, tmp_path):
        cfg = load_experiment_config(tmp_path / "nonexistent.yaml")
        assert cfg.max_experiments == 20

    def test_load_default_path_missing(self):
        # Default path won't exist in test env
        cfg = load_experiment_config(Path("/nonexistent/path.yaml"))
        assert isinstance(cfg, ExperimentConfig)


# -----------------------------------------------------------------------
# Random sampling
# -----------------------------------------------------------------------

class TestRandomSample:
    """Tests for random config sampling."""

    def test_keys_match(self, sample_search_space):
        sampled = _random_sample(sample_search_space)
        assert set(sampled.keys()) == set(sample_search_space.keys())

    def test_values_from_space(self, sample_search_space):
        for _ in range(20):
            sampled = _random_sample(sample_search_space)
            for k, v in sampled.items():
                assert v in sample_search_space[k]


# -----------------------------------------------------------------------
# Config merge logic
# -----------------------------------------------------------------------

class TestMergeConfigOverrides:
    """Tests for dot-notation config merging."""

    def test_flat_override(self):
        base = {"r": 8, "lr": 1e-4}
        overrides = {"r": 16}
        result = _merge_config_overrides(base, overrides)
        assert result["r"] == 16
        assert result["lr"] == 1e-4

    def test_dot_notation_override(self):
        base = {"evolutionary": {"enabled": False, "noise_scale": 0.1}}
        overrides = {"evolutionary.enabled": True}
        result = _merge_config_overrides(base, overrides)
        assert result["evolutionary"]["enabled"] is True
        assert result["evolutionary"]["noise_scale"] == 0.1

    def test_creates_nested_keys(self):
        base = {}
        overrides = {"training.lr": 2e-4}
        result = _merge_config_overrides(base, overrides)
        assert result["training"]["lr"] == 2e-4

    def test_does_not_mutate_base(self):
        base = {"r": 8}
        overrides = {"r": 16}
        _merge_config_overrides(base, overrides)
        assert base["r"] == 8


class TestFlattenConfig:
    """Tests for config flattening."""

    def test_flat_dict(self):
        assert _flatten_config({"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_nested_dict(self):
        result = _flatten_config({"a": {"x": 1, "y": 2}, "b": 3})
        assert result == {"a.x": 1, "a.y": 2, "b": 3}


# -----------------------------------------------------------------------
# LLM Advisor
# -----------------------------------------------------------------------

class TestLLMAdvisor:
    """Tests for the LLM advisor."""

    def test_extract_yaml_valid(self):
        text = textwrap.dedent("""\
            Here is my suggestion:
            ```yaml
            learning_rate: 0.0002
            r: 16
            lora_alpha: 32
            ```
            This should work well.
        """)
        result = LLMAdvisor._extract_yaml(text)
        assert result == {"learning_rate": 0.0002, "r": 16, "lora_alpha": 32}

    def test_extract_yaml_no_yaml_tag(self):
        text = "```\nlearning_rate: 0.0001\n```"
        result = LLMAdvisor._extract_yaml(text)
        assert result == {"learning_rate": 0.0001}

    def test_extract_yaml_no_block_raises(self):
        with pytest.raises(ValueError, match="No YAML block"):
            LLMAdvisor._extract_yaml("no code blocks here")

    def test_extract_yaml_non_dict_raises(self):
        with pytest.raises(ValueError, match="did not parse to a dict"):
            LLMAdvisor._extract_yaml("```yaml\n- item1\n- item2\n```")

    def test_propose_config_calls_llm(self, sample_search_space):
        import pandas as pd

        advisor = LLMAdvisor(
            search_space=sample_search_space,
            llm_backend="openrouter",
        )
        mock_client = MagicMock()
        mock_client.chat.return_value = (
            "```yaml\nlearning_rate: 0.0002\nr: 16\nlora_alpha: 32\n```"
        )
        advisor._client = mock_client

        history = pd.DataFrame(columns=["eval_score", "learning_rate", "r"])
        result = advisor.propose_config(history)
        assert result["learning_rate"] == 0.0002
        mock_client.chat.assert_called_once()

    def test_propose_config_fallback_on_error(self, sample_search_space):
        import pandas as pd

        advisor = LLMAdvisor(
            search_space=sample_search_space,
            llm_backend="openrouter",
        )
        mock_client = MagicMock()
        mock_client.chat.side_effect = RuntimeError("connection failed")
        advisor._client = mock_client

        history = pd.DataFrame(columns=["eval_score"])
        result = advisor.propose_config(history)
        # Should fall back to random sampling
        assert set(result.keys()) == set(sample_search_space.keys())

    def test_propose_candidates_returns_n(self, sample_search_space):
        import pandas as pd

        advisor = LLMAdvisor(
            search_space=sample_search_space,
            llm_backend="openrouter",
        )
        mock_client = MagicMock()
        call_count = 0

        def mock_chat(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return f"```yaml\nlearning_rate: {call_count}e-4\nr: 16\nlora_alpha: 32\n```"

        mock_client.chat.side_effect = mock_chat
        advisor._client = mock_client

        history = pd.DataFrame(columns=["eval_score"])
        candidates = advisor.propose_candidates(history, n=3)
        assert len(candidates) == 3

    def test_prompt_includes_search_space(self, sample_search_space):
        import pandas as pd

        advisor = LLMAdvisor(
            search_space=sample_search_space,
            program_md="Focus on learning rate.",
        )
        history = pd.DataFrame(columns=["eval_score"])
        prompt = advisor._build_prompt(history)
        assert "learning_rate" in prompt
        assert "Focus on learning rate" in prompt
        assert "Search Space" in prompt

    def test_prompt_includes_history(self, sample_search_space):
        import pandas as pd

        advisor = LLMAdvisor(search_space=sample_search_space)
        history = pd.DataFrame([
            {"eval_score": 0.8, "learning_rate": 0.0001, "r": 16},
            {"eval_score": 0.7, "learning_rate": 0.0002, "r": 8},
        ])
        prompt = advisor._build_prompt(history)
        assert "Experiment History" in prompt
        assert "0.8" in prompt


# -----------------------------------------------------------------------
# Surrogate Model
# -----------------------------------------------------------------------

class TestSurrogateModel:
    """Tests for the LightGBM surrogate model."""

    def test_available_property(self):
        surrogate = SurrogateModel()
        # Just verify it returns a bool without error
        assert isinstance(surrogate.available, bool)

    @pytest.mark.skipif(
        not SurrogateModel().available,
        reason="lightgbm not installed",
    )
    def test_fit_predict_cycle(self):
        import pandas as pd

        surrogate = SurrogateModel()
        df = pd.DataFrame([
            {"learning_rate": 1e-4, "r": 8, "eval_score": 0.6},
            {"learning_rate": 2e-4, "r": 16, "eval_score": 0.75},
            {"learning_rate": 5e-4, "r": 32, "eval_score": 0.5},
            {"learning_rate": 1e-4, "r": 32, "eval_score": 0.7},
            {"learning_rate": 2e-4, "r": 8, "eval_score": 0.65},
        ])
        surrogate.fit(df)
        assert surrogate._pipeline is not None

        candidates = [
            {"learning_rate": 1e-4, "r": 16},
            {"learning_rate": 5e-4, "r": 8},
        ]
        scores = surrogate.predict_candidates(candidates)
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)

    @pytest.mark.skipif(
        not SurrogateModel().available,
        reason="lightgbm not installed",
    )
    def test_feature_importance(self):
        import pandas as pd

        surrogate = SurrogateModel()
        df = pd.DataFrame([
            {"lr": 1e-4, "r": 8, "eval_score": 0.6},
            {"lr": 2e-4, "r": 16, "eval_score": 0.75},
            {"lr": 5e-4, "r": 32, "eval_score": 0.5},
            {"lr": 1e-4, "r": 32, "eval_score": 0.7},
        ])
        surrogate.fit(df)
        importance = surrogate.feature_importance()
        assert "lr" in importance
        assert "r" in importance
        assert all(isinstance(v, float) for v in importance.values())

    def test_predict_without_fit_returns_zeros(self):
        surrogate = SurrogateModel()
        scores = surrogate.predict_candidates([{"lr": 1e-4}])
        assert scores == [0.0]

    def test_feature_importance_without_fit_returns_empty(self):
        surrogate = SurrogateModel()
        assert surrogate.feature_importance() == {}


# -----------------------------------------------------------------------
# Results persistence
# -----------------------------------------------------------------------

class TestResultsPersistence:
    """Tests for results TSV writing and parsing."""

    def test_save_results_creates_tsv(self, tmp_path, sample_config, sample_results):
        sample_config.output_dir = str(tmp_path)
        loop = ExperimentLoop(sample_config)
        loop.results = sample_results
        loop._save_results(tmp_path)

        tsv_file = tmp_path / "results.tsv"
        assert tsv_file.exists()

        lines = tsv_file.read_text().strip().split("\n")
        assert len(lines) == 4  # header + 3 results

        header = lines[0].split("\t")
        assert "experiment_id" in header
        assert "eval_score" in header
        assert "status" in header

    def test_save_results_parseable(self, tmp_path, sample_config, sample_results):
        sample_config.output_dir = str(tmp_path)
        loop = ExperimentLoop(sample_config)
        loop.results = sample_results
        loop._save_results(tmp_path)

        tsv_file = tmp_path / "results.tsv"
        lines = tsv_file.read_text().strip().split("\n")
        header = lines[0].split("\t")
        data_row = lines[1].split("\t")

        # Find eval_score column
        score_idx = header.index("eval_score")
        assert float(data_row[score_idx]) == pytest.approx(0.75)

    def test_save_best_config(self, tmp_path, sample_config):
        sample_config.output_dir = str(tmp_path)
        loop = ExperimentLoop(sample_config)
        loop.best_score = 0.82
        loop.best_config = {"learning_rate": 2e-4, "r": 8}
        loop._save_best_config(tmp_path)

        best_file = tmp_path / "best_config.yaml"
        assert best_file.exists()
        loaded = yaml.safe_load(best_file.read_text())
        assert loaded["best_score"] == pytest.approx(0.82)
        assert loaded["config"]["r"] == 8

    def test_save_results_handles_empty(self, tmp_path, sample_config):
        sample_config.output_dir = str(tmp_path)
        loop = ExperimentLoop(sample_config)
        loop._save_results(tmp_path)
        # Should not create file if no results
        assert not (tmp_path / "results.tsv").exists()


# -----------------------------------------------------------------------
# Phase transition
# -----------------------------------------------------------------------

class TestPhaseTransition:
    """Tests for LLM-only vs LLM+surrogate phase switching."""

    def test_first_experiment_is_random(self, sample_config):
        sample_config.search_strategy = "llm_surrogate"
        loop = ExperimentLoop(sample_config)
        # No results yet -> should use random
        config = loop._select_next_config()
        assert set(config.keys()) == set(sample_config.search_space.keys())

    def test_phase1_uses_llm(self, sample_config, sample_results):
        import pandas as pd

        sample_config.search_strategy = "llm_surrogate"
        sample_config.surrogate_phase_threshold = 10
        loop = ExperimentLoop(sample_config)
        loop.results = sample_results[:2]  # 2 < threshold of 10

        mock_advisor = MagicMock()
        mock_advisor.propose_config.return_value = {
            "learning_rate": 2e-4, "r": 16, "lora_alpha": 32,
        }
        loop._advisor = mock_advisor

        config = loop._select_next_config()
        mock_advisor.propose_config.assert_called_once()
        assert config["learning_rate"] == 2e-4

    def test_phase2_uses_surrogate(self, sample_config):
        import pandas as pd

        # Create enough results to pass threshold
        sample_config.search_strategy = "llm_surrogate"
        sample_config.surrogate_phase_threshold = 3
        loop = ExperimentLoop(sample_config)

        # Add 3 completed results
        for i in range(3):
            loop.results.append(ExperimentResult(
                experiment_id=f"exp_{i:04d}",
                config={"learning_rate": (i + 1) * 1e-4, "r": 16, "lora_alpha": 32},
                eval_score=0.5 + i * 0.1,
                training_loss=0.5 - i * 0.05,
                duration_seconds=100.0,
                status="completed",
            ))

        mock_advisor = MagicMock()
        mock_advisor.propose_candidates.return_value = [
            {"learning_rate": 1e-4, "r": 8, "lora_alpha": 16},
            {"learning_rate": 2e-4, "r": 16, "lora_alpha": 32},
        ]
        loop._advisor = mock_advisor

        mock_surrogate = MagicMock()
        mock_surrogate.available = True
        mock_surrogate._pipeline = MagicMock()  # pretend it's fitted
        mock_surrogate.predict_candidates.return_value = [0.6, 0.8]
        loop._surrogate = mock_surrogate

        config = loop._select_next_config()
        mock_advisor.propose_candidates.assert_called_once()
        mock_surrogate.predict_candidates.assert_called_once()
        # Should pick candidate with higher predicted score (index 1)
        assert config["learning_rate"] == 2e-4


# -----------------------------------------------------------------------
# Single experiment result recording
# -----------------------------------------------------------------------

class TestExperimentResultRecording:
    """Tests for recording individual experiment results."""

    def test_result_dataclass_fields(self):
        result = ExperimentResult(
            experiment_id="test_001",
            config={"lr": 1e-4},
            eval_score=0.85,
            training_loss=0.25,
            duration_seconds=60.0,
            status="completed",
        )
        assert result.experiment_id == "test_001"
        assert result.eval_score == 0.85
        assert result.status == "completed"

    def test_failed_result(self):
        result = ExperimentResult(
            experiment_id="test_002",
            config={"lr": 5e-4},
            eval_score=0.0,
            training_loss=float("inf"),
            duration_seconds=5.0,
            status="failed",
        )
        assert result.status == "failed"
        assert result.training_loss == float("inf")


# -----------------------------------------------------------------------
# Max steps enforcement
# -----------------------------------------------------------------------

class TestMaxStepsEnforcement:
    """Tests that max_steps is written into the temp config."""

    def test_max_steps_in_temp_config(self, tmp_path, sample_config):
        sample_config.output_dir = str(tmp_path)
        sample_config.max_steps_per_experiment = 100

        loop = ExperimentLoop(sample_config)

        # Mock subprocess to avoid actual training
        with patch("shared.flywheel.experiment_loop.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")

            result = loop._run_single_experiment(
                "exp_test", {"learning_rate": 1e-4},
            )

        # Check the config file was written with max_steps
        config_file = tmp_path / "exp_test" / "config.yaml"
        assert config_file.exists()
        written = yaml.safe_load(config_file.read_text())
        assert written["max_steps"] == 100


# -----------------------------------------------------------------------
# Trainer script mapping
# -----------------------------------------------------------------------

class TestTrainerScript:
    """Tests for trainer script path resolution."""

    def test_sft_script(self):
        assert _trainer_script("sft") == "Trainers/sft/train_sft.py"

    def test_kto_script(self):
        assert _trainer_script("kto") == "Trainers/kto/train_kto.py"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown trainer_type"):
            _trainer_script("grpo")


# -----------------------------------------------------------------------
# Validation on run()
# -----------------------------------------------------------------------

class TestExperimentLoopValidation:
    """Tests that run() rejects invalid configs."""

    def test_run_rejects_invalid_config(self):
        cfg = ExperimentConfig(max_experiments=0)
        loop = ExperimentLoop(cfg)
        with pytest.raises(ValueError, match="Invalid ExperimentConfig"):
            loop.run()
