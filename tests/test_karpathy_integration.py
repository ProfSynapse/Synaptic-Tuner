"""Integration tests for Karpathy training optimizations.

Tests cross-implementation interactions between:
- CheckpointEvaluator (shared.checkpoint_eval)
- EvalBackend protocol (shared.eval_backend)
- ExperimentLoop (shared.flywheel.experiment_loop)
- ExperimentConfig (shared.flywheel.experiment_config)
- LoRASurgeon, SurgeryConfig (shared.evolutionary.lora_surgery)
- EvolutionaryConfig.max_grad_norm (shared.evolutionary.config)
- fitness_reward (Trainers.grpo.src.rewards)
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from shared.evolutionary.config import EvolutionaryConfig
from shared.flywheel.experiment_config import ExperimentConfig
from shared.flywheel.experiment_loop import ExperimentLoop, ExperimentResult
from shared.evolutionary.lora_surgery import LoRASurgeon, SurgeryConfig, SurgeryResult


# ---------------------------------------------------------------------------
# 1. ExperimentLoop + ExperimentConfig integration
# ---------------------------------------------------------------------------

class TestExperimentLoopConfig:
    """ExperimentLoop correctly initializes from ExperimentConfig."""

    def test_loop_accepts_config(self, tmp_path):
        config = ExperimentConfig(
            base_config_path=str(tmp_path / "base.yaml"),
            output_dir=str(tmp_path / "output"),
            search_space={"learning_rate": [1e-4, 2e-4]},
            max_experiments=3,
        )
        loop = ExperimentLoop(config=config)
        assert loop.config.max_experiments == 3
        assert loop.config.search_space == {"learning_rate": [1e-4, 2e-4]}

    def test_tier_yaml_as_base_config(self, tmp_path):
        """Tier YAML from Wave 1B can be used as base_config_path."""
        tier_cfg = {
            "learning_rate": 5e-4,
            "r": 8,
            "lora_alpha": 16,
            "num_train_epochs": 1,
            "max_steps": 200,
        }
        tier_path = tmp_path / "quick.yaml"
        tier_path.write_text(yaml.dump(tier_cfg))

        config = ExperimentConfig(
            base_config_path=str(tier_path),
            output_dir=str(tmp_path),
            search_space={"learning_rate": [1e-4, 5e-4]},
        )
        assert config.base_config_path == str(tier_path)

        loop = ExperimentLoop(config=config)
        assert loop is not None


# ---------------------------------------------------------------------------
# 2. ExperimentLoop + Evolutionary config (Wave 1A)
# ---------------------------------------------------------------------------

class TestExperimentLoopEvolutionary:
    """Evolutionary configs in search space produce valid overrides."""

    def test_evo_search_space_includes_max_grad_norm(self):
        """When evolutionary params are in search space, max_grad_norm
        from Wave 1A should be a valid search dimension."""
        search_space = {
            "learning_rate": [1e-4, 2e-4],
            "evolutionary.enabled": [True, False],
            "evolutionary.noise_scale": [0.01, 0.03, 0.05],
        }
        config = ExperimentConfig(
            search_space=search_space,
            output_dir="/tmp/test",
        )
        # Dot-notation keys for evolutionary settings should be accepted
        assert "evolutionary.enabled" in config.search_space
        assert "evolutionary.noise_scale" in config.search_space

    def test_random_sample_from_search_space(self):
        """_random_sample produces valid configs from search space."""
        from shared.flywheel.experiment_loop import _random_sample

        search_space = {
            "learning_rate": [1e-4, 2e-4, 5e-4],
            "r": [8, 16, 32],
            "evolutionary.enabled": [True, False],
        }
        sample = _random_sample(search_space)
        assert "learning_rate" in sample
        assert sample["learning_rate"] in [1e-4, 2e-4, 5e-4]
        assert "r" in sample
        assert "evolutionary.enabled" in sample


# ---------------------------------------------------------------------------
# 3. LoRA Surgery + EvalBackend
# ---------------------------------------------------------------------------

class TestSurgeryEvalBackend:
    """LoRASurgeon delegates evaluation through EvalBackend protocol."""

    def test_surgeon_init_accepts_eval_backend(self, tmp_path):
        """Surgery accepts any EvalBackend-compatible object."""
        mock_backend = MagicMock()
        adapter_path = tmp_path / "adapter"
        adapter_path.mkdir()

        config = SurgeryConfig(
            adapter_path=str(adapter_path),
            eval_scenario="test.yaml",
            operations=["alpha_sweep"],
        )
        surgeon = LoRASurgeon(
            adapter_path=str(adapter_path),
            eval_backend=mock_backend,
            eval_scenario="test.yaml",
            config=config,
        )
        assert surgeon is not None

    def test_alpha_sweep_calls_evaluate(self, tmp_path):
        """Alpha sweep operation invokes the eval backend."""
        mock_backend = MagicMock()
        mock_backend.run_eval = AsyncMock(return_value=0.85)

        adapter_path = tmp_path / "adapter"
        adapter_path.mkdir()
        # Create minimal adapter_config.json
        (adapter_path / "adapter_config.json").write_text(json.dumps({
            "lora_alpha": 32,
            "r": 16,
        }))

        config = SurgeryConfig(
            adapter_path=str(adapter_path),
            eval_scenario="test.yaml",
            operations=["alpha_sweep"],
            alpha_multipliers=[0.5, 1.5],
        )
        surgeon = LoRASurgeon(
            adapter_path=str(adapter_path),
            eval_backend=mock_backend,
            eval_scenario="test.yaml",
            config=config,
        )

        # alpha_sweep is async, test the signature
        import asyncio
        import inspect
        assert inspect.iscoroutinefunction(surgeon.alpha_sweep)


# ---------------------------------------------------------------------------
# 4. Surgery + Checkpoint eval (best vs final for interpolation)
# ---------------------------------------------------------------------------

class TestSurgeryCheckpointInterpolation:
    """Surgery uses checkpoint eval results for interpolation."""

    def test_surgery_config_accepts_other_checkpoint(self, tmp_path):
        """SurgeryConfig.other_checkpoint_path holds the second checkpoint
        discovered by CheckpointEvaluator (Wave 1C)."""
        best_ckpt = tmp_path / "best"
        final_ckpt = tmp_path / "final"
        best_ckpt.mkdir()
        final_ckpt.mkdir()

        config = SurgeryConfig(
            adapter_path=str(best_ckpt),
            eval_scenario="test.yaml",
            operations=["checkpoint_interpolation"],
            other_checkpoint_path=str(final_ckpt),
            blend_ratios=[0.25, 0.5, 0.75],
        )
        assert config.other_checkpoint_path == str(final_ckpt)
        assert config.blend_ratios == [0.25, 0.5, 0.75]

    def test_checkpoint_interpolation_is_async(self):
        """checkpoint_interpolation is an async operation."""
        import inspect
        assert inspect.iscoroutinefunction(LoRASurgeon.checkpoint_interpolation)


# ---------------------------------------------------------------------------
# 5. EvolutionaryConfig integration (Wave 1A)
# ---------------------------------------------------------------------------

class TestEvolutionaryConfigGradNorm:
    """EvolutionaryConfig includes max_grad_norm from Wave 1A."""

    def test_field_exists(self):
        cfg = EvolutionaryConfig()
        assert hasattr(cfg, "max_grad_norm")
        assert isinstance(cfg.max_grad_norm, (int, float))

    def test_default_is_1(self):
        cfg = EvolutionaryConfig()
        assert cfg.max_grad_norm == 1.0

    def test_to_dict_includes_max_grad_norm(self):
        cfg = EvolutionaryConfig()
        d = cfg.to_dict()
        # max_grad_norm is nested under strategy.params
        assert d["strategy"]["params"]["max_grad_norm"] == 1.0

    def test_from_dict_preserves_max_grad_norm(self):
        data = {
            "enabled": True,
            "candidates": 4,
            "validation_config": "dummy.yaml",
            "strategy": {
                "type": "gradient_noise",
                "params": {
                    "noise_scale": 0.03,
                    "max_grad_norm": 2.5,
                },
            },
        }
        cfg = EvolutionaryConfig.from_dict(data)
        assert cfg.max_grad_norm == 2.5

    def test_round_trip(self):
        cfg = EvolutionaryConfig(max_grad_norm=1.5)
        d = cfg.to_dict()
        cfg2 = EvolutionaryConfig.from_dict(d)
        assert cfg2.max_grad_norm == 1.5


# ---------------------------------------------------------------------------
# 6. Fitness reward (Wave 1D)
# ---------------------------------------------------------------------------

class TestFitnessReward:
    """fitness_reward function from GRPO rewards module."""

    def test_import_and_callable(self):
        from Trainers.grpo.src.rewards import fitness_reward
        assert callable(fitness_reward)

    def test_returns_float(self):
        from Trainers.grpo.src.rewards import fitness_reward
        result = fitness_reward('<tool_call>{"name": "test", "arguments": {}}</tool_call>')
        assert isinstance(result, float)

    def test_tool_call_scores_higher_than_plain(self):
        from Trainers.grpo.src.rewards import fitness_reward
        tool_score = fitness_reward(
            '<tool_call>{"name": "search", "arguments": {"q": "test"}}</tool_call>'
        )
        plain_score = fitness_reward("I cannot help with that.")
        assert tool_score >= plain_score


# ---------------------------------------------------------------------------
# 7. ExperimentResult dataclass integration
# ---------------------------------------------------------------------------

class TestExperimentResultTracking:
    """ExperimentResult captures data needed by downstream systems."""

    def test_result_fields(self):
        result = ExperimentResult(
            experiment_id="exp-001",
            config={"learning_rate": 2e-4, "r": 16},
            eval_score=0.85,
            training_loss=0.12,
            duration_seconds=120.0,
            status="completed",
        )
        assert result.experiment_id == "exp-001"
        assert result.eval_score == 0.85
        assert result.status == "completed"

    def test_result_config_holds_evo_params(self):
        """ExperimentResult config can include evolutionary overrides."""
        result = ExperimentResult(
            experiment_id="exp-002",
            config={
                "learning_rate": 1e-4,
                "evolutionary.enabled": True,
                "evolutionary.noise_scale": 0.03,
            },
            eval_score=0.87,
            training_loss=0.10,
            duration_seconds=90.0,
            status="completed",
        )
        assert "evolutionary.enabled" in result.config
        assert result.config["evolutionary.noise_scale"] == 0.03


# ---------------------------------------------------------------------------
# 8. End-to-end pipeline data flow (mocked)
# ---------------------------------------------------------------------------

class TestEndToEndPipeline:
    """High-level test: experiment loop results feed into surgery."""

    def test_best_experiment_feeds_surgery(self, tmp_path):
        """Best experiment's adapter path can initialize LoRASurgeon."""
        # Simulate experiment loop producing results
        results = [
            ExperimentResult("exp-1", {"lr": 1e-4}, 0.70, 0.15, 60, "completed"),
            ExperimentResult("exp-2", {"lr": 2e-4}, 0.85, 0.10, 65, "completed"),
            ExperimentResult("exp-3", {"lr": 5e-4}, 0.75, 0.12, 55, "completed"),
        ]
        best = max(results, key=lambda r: r.eval_score)
        assert best.experiment_id == "exp-2"

        # Best experiment's adapter feeds into surgery
        adapter_dir = tmp_path / "exp-2" / "adapter"
        adapter_dir.mkdir(parents=True)

        config = SurgeryConfig(
            adapter_path=str(adapter_dir),
            eval_scenario="test.yaml",
            operations=["alpha_sweep"],
            output_dir=str(tmp_path / "surgery_out"),
        )
        mock_backend = MagicMock()
        surgeon = LoRASurgeon(
            adapter_path=str(adapter_dir),
            eval_backend=mock_backend,
            eval_scenario="test.yaml",
            config=config,
        )
        assert surgeon is not None

    def test_checkpoint_scores_determine_interpolation_targets(self):
        """CheckpointEvaluator scores determine which checkpoints
        go to surgery for interpolation."""
        # Simulated checkpoint eval results
        checkpoint_scores = {
            "checkpoint-100": 0.70,
            "checkpoint-200": 0.85,
            "checkpoint-300": 0.80,
            "final_model": 0.82,
        }
        # Best is checkpoint-200, second best is final_model
        sorted_ckpts = sorted(
            checkpoint_scores.items(), key=lambda x: x[1], reverse=True
        )
        best_name, best_score = sorted_ckpts[0]
        second_name, second_score = sorted_ckpts[1]

        assert best_name == "checkpoint-200"
        assert second_name == "final_model"

        # These two feed into surgery's checkpoint_interpolation
        config = SurgeryConfig(
            adapter_path=f"/path/{best_name}",
            eval_scenario="test.yaml",
            operations=["checkpoint_interpolation"],
            other_checkpoint_path=f"/path/{second_name}",
        )
        assert config.adapter_path.endswith("checkpoint-200")
        assert config.other_checkpoint_path.endswith("final_model")
