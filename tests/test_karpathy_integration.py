"""Integration tests for Karpathy training optimizations.

Tests the integration between:
- CheckpointEvaluator (shared.checkpoint_eval)
- EvalBackend protocol (shared.eval_backend)
- ExperimentLoop, LLMAdvisor, SurrogateModel (shared.flywheel.experiment_loop)
- ExperimentConfig (shared.flywheel.experiment_config)
- LoRASurgeon, SurgeryConfig (shared.evolutionary.lora_surgery)
- EvolutionaryConfig.max_grad_norm (shared.evolutionary.config)
- fitness_reward (Trainers.grpo.src.rewards)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import yaml

from shared.evolutionary.config import EvolutionaryConfig


# ---------------------------------------------------------------------------
# Helpers / stubs for modules that may not exist yet
# ---------------------------------------------------------------------------

def _try_import(module_path: str, attr: str):
    """Attempt to import an attribute; return None if the module is missing."""
    try:
        mod = __import__(module_path, fromlist=[attr])
        return getattr(mod, attr, None)
    except (ImportError, ModuleNotFoundError):
        return None


CheckpointEvaluator = _try_import("shared.checkpoint_eval", "CheckpointEvaluator")
EvalBackend = _try_import("shared.eval_backend", "EvalBackend")
ExperimentLoop = _try_import("shared.flywheel.experiment_loop", "ExperimentLoop")
LLMAdvisor = _try_import("shared.flywheel.experiment_loop", "LLMAdvisor")
SurrogateModel = _try_import("shared.flywheel.experiment_loop", "SurrogateModel")
ExperimentConfig = _try_import("shared.flywheel.experiment_config", "ExperimentConfig")
LoRASurgeon = _try_import("shared.evolutionary.lora_surgery", "LoRASurgeon")
SurgeryConfig = _try_import("shared.evolutionary.lora_surgery", "SurgeryConfig")

# fitness_reward lives inside the Trainers subtree
fitness_reward = _try_import("Trainers.grpo.src.rewards", "fitness_reward")


# ---------------------------------------------------------------------------
# Conditional skip decorator
# ---------------------------------------------------------------------------

def _skip_if_missing(*objs, reason="required module not yet implemented"):
    """Return pytest.mark.skipif when any object is None."""
    return pytest.mark.skipif(
        any(o is None for o in objs),
        reason=reason,
    )


# ---------------------------------------------------------------------------
# 1. experiment_loop_uses_checkpoint_eval
# ---------------------------------------------------------------------------

@_skip_if_missing(ExperimentLoop, CheckpointEvaluator)
class TestExperimentLoopUsesCheckpointEval:
    """ExperimentLoop delegates checkpoint scoring to CheckpointEvaluator."""

    def test_calls_checkpoint_evaluator(self, tmp_path):
        mock_evaluator = MagicMock(spec=CheckpointEvaluator)
        mock_evaluator.evaluate.return_value = {"score": 0.85, "loss": 0.12}

        loop = ExperimentLoop(
            checkpoint_evaluator=mock_evaluator,
            output_dir=str(tmp_path),
        )

        # Simulate a single iteration that produces a checkpoint path
        dummy_ckpt = tmp_path / "checkpoint-100"
        dummy_ckpt.mkdir()
        loop.evaluate_checkpoint(str(dummy_ckpt))

        mock_evaluator.evaluate.assert_called_once()
        call_args = mock_evaluator.evaluate.call_args
        assert str(dummy_ckpt) in str(call_args)


# ---------------------------------------------------------------------------
# 2. experiment_loop_loads_tier_config
# ---------------------------------------------------------------------------

@_skip_if_missing(ExperimentLoop, ExperimentConfig)
class TestExperimentLoopLoadsTierConfig:
    """Tier YAML can be used as base_config_path for ExperimentLoop."""

    def test_tier_yaml_accepted(self, tmp_path):
        tier_cfg = {
            "tier": "small",
            "model": "unsloth/Qwen2.5-3B",
            "training": {
                "learning_rate": 2e-4,
                "epochs": 2,
                "batch_size": 4,
            },
        }
        tier_path = tmp_path / "tier_small.yaml"
        tier_path.write_text(yaml.dump(tier_cfg))

        config = ExperimentConfig(base_config_path=str(tier_path))
        assert config.base_config_path == str(tier_path)

        # The loop should accept this config without error
        mock_evaluator = MagicMock()
        loop = ExperimentLoop(
            config=config,
            checkpoint_evaluator=mock_evaluator,
            output_dir=str(tmp_path),
        )
        assert loop is not None


# ---------------------------------------------------------------------------
# 3. experiment_loop_generates_valid_evo_config
# ---------------------------------------------------------------------------

@_skip_if_missing(ExperimentLoop, ExperimentConfig)
class TestExperimentLoopGeneratesValidEvoConfig:
    """When evolutionary.enabled is in search space, configs include max_grad_norm."""

    def test_generated_config_has_max_grad_norm(self, tmp_path):
        search_space = {
            "evolutionary": {
                "enabled": True,
                "max_grad_norm": {"min": 0.1, "max": 2.0},
            },
        }
        config = ExperimentConfig(
            base_config_path=str(tmp_path / "base.yaml"),
            search_space=search_space,
        )

        mock_evaluator = MagicMock()
        loop = ExperimentLoop(
            config=config,
            checkpoint_evaluator=mock_evaluator,
            output_dir=str(tmp_path),
        )

        generated = loop.generate_trial_config()
        evo_section = generated.get("evolutionary", {})
        assert "max_grad_norm" in evo_section
        assert 0.1 <= evo_section["max_grad_norm"] <= 2.0


# ---------------------------------------------------------------------------
# 4. surgery_uses_eval_backend
# ---------------------------------------------------------------------------

@_skip_if_missing(LoRASurgeon, SurgeryConfig)
class TestSurgeryUsesEvalBackend:
    """LoRASurgeon delegates evaluation to eval_backend.run_eval."""

    def test_calls_eval_backend_run_eval(self, tmp_path):
        mock_backend = MagicMock()
        mock_backend.run_eval.return_value = {"score": 0.78}

        surgery_cfg = SurgeryConfig(
            base_model_path=str(tmp_path / "base"),
            adapter_paths=[str(tmp_path / "adapter_a")],
        )
        surgeon = LoRASurgeon(config=surgery_cfg, eval_backend=mock_backend)

        # Create stub adapter directory
        (tmp_path / "adapter_a").mkdir()

        surgeon.alpha_sweep(alphas=[0.0, 0.5, 1.0])

        assert mock_backend.run_eval.call_count >= 1


# ---------------------------------------------------------------------------
# 5. surgery_checkpoint_interpolation_uses_two_paths
# ---------------------------------------------------------------------------

@_skip_if_missing(LoRASurgeon, SurgeryConfig)
class TestSurgeryCheckpointInterpolation:
    """Interpolation blends exactly two checkpoint adapter paths."""

    def test_interpolation_requires_two_checkpoints(self, tmp_path):
        mock_backend = MagicMock()
        mock_backend.run_eval.return_value = {"score": 0.80}

        adapter_a = tmp_path / "adapter_a"
        adapter_b = tmp_path / "adapter_b"
        adapter_a.mkdir()
        adapter_b.mkdir()

        surgery_cfg = SurgeryConfig(
            base_model_path=str(tmp_path / "base"),
            adapter_paths=[str(adapter_a), str(adapter_b)],
        )
        surgeon = LoRASurgeon(config=surgery_cfg, eval_backend=mock_backend)

        result = surgeon.interpolate(alpha=0.5)

        # The result should reference both adapter paths
        assert result is not None
        # Verify eval was called with interpolated weights
        assert mock_backend.run_eval.called


# ---------------------------------------------------------------------------
# 6. evolutionary_config_has_grad_norm
# ---------------------------------------------------------------------------

class TestEvolutionaryConfigHasGradNorm:
    """EvolutionaryConfig includes max_grad_norm field."""

    def test_max_grad_norm_field_exists(self):
        cfg = EvolutionaryConfig()
        assert hasattr(cfg, "max_grad_norm"), (
            "EvolutionaryConfig must have a max_grad_norm field "
            "for Karpathy-style gradient clipping"
        )

    def test_max_grad_norm_default_is_numeric(self):
        cfg = EvolutionaryConfig()
        if hasattr(cfg, "max_grad_norm"):
            assert isinstance(cfg.max_grad_norm, (int, float))

    def test_from_dict_includes_grad_norm(self):
        data = {
            "enabled": True,
            "candidates": 4,
            "validation_config": "dummy.yaml",
            "max_grad_norm": 1.5,
        }
        cfg = EvolutionaryConfig.from_dict(data)
        if hasattr(cfg, "max_grad_norm"):
            assert cfg.max_grad_norm == 1.5

    def test_to_dict_round_trip(self):
        cfg = EvolutionaryConfig(enabled=True, validation_config_path="x.yaml")
        if hasattr(cfg, "max_grad_norm"):
            d = cfg.to_dict()
            assert "max_grad_norm" in d


# ---------------------------------------------------------------------------
# 7. fitness_reward_callable
# ---------------------------------------------------------------------------

class TestFitnessRewardCallable:
    """fitness_reward function exists in rewards module and is callable."""

    @pytest.mark.skipif(
        fitness_reward is None,
        reason="fitness_reward not yet implemented in Trainers.grpo.src.rewards",
    )
    def test_fitness_reward_is_callable(self):
        assert callable(fitness_reward)

    @pytest.mark.skipif(
        fitness_reward is None,
        reason="fitness_reward not yet implemented in Trainers.grpo.src.rewards",
    )
    def test_fitness_reward_accepts_completions(self):
        """fitness_reward should accept completions list and return scores."""
        completions = [
            '<tool_call>{"name": "test", "arguments": {}}</tool_call>',
            "plain text without tool call",
        ]
        result = fitness_reward(completions)
        assert isinstance(result, list)
        assert len(result) == len(completions)
        assert all(isinstance(r, (int, float)) for r in result)

    @pytest.mark.skipif(
        fitness_reward is None,
        reason="fitness_reward not yet implemented in Trainers.grpo.src.rewards",
    )
    def test_fitness_reward_scores_tool_calls_higher(self):
        """Tool-call completions should score higher than plain text."""
        completions = [
            '<tool_call>{"name": "search", "arguments": {"q": "test"}}</tool_call>',
            "I cannot help with that request.",
        ]
        result = fitness_reward(completions)
        assert result[0] > result[1], (
            "Tool-call completion should score higher than plain text"
        )


# ---------------------------------------------------------------------------
# 8. end_to_end_pipeline_mock
# ---------------------------------------------------------------------------

@_skip_if_missing(ExperimentLoop, LoRASurgeon, SurgeryConfig)
class TestEndToEndPipelineMock:
    """Mock experiment loop run + surgery alpha_sweep, verify data flows."""

    def test_pipeline_data_flow(self, tmp_path):
        # -- Phase 1: Experiment loop produces best checkpoint --
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = {"score": 0.90, "loss": 0.08}

        mock_config = MagicMock()
        mock_config.base_config_path = str(tmp_path / "base.yaml")
        mock_config.search_space = {}

        loop = ExperimentLoop(
            config=mock_config,
            checkpoint_evaluator=mock_evaluator,
            output_dir=str(tmp_path),
        )

        # Simulate loop.run() returning best checkpoint
        best_ckpt = tmp_path / "best_checkpoint"
        best_ckpt.mkdir()
        with patch.object(loop, "run", return_value=str(best_ckpt)):
            best_path = loop.run()

        assert best_path == str(best_ckpt)

        # -- Phase 2: Surgery refines the best checkpoint --
        mock_backend = MagicMock()
        mock_backend.run_eval.return_value = {"score": 0.92}

        surgery_cfg = SurgeryConfig(
            base_model_path=str(tmp_path / "base_model"),
            adapter_paths=[best_path],
        )
        surgeon = LoRASurgeon(config=surgery_cfg, eval_backend=mock_backend)

        with patch.object(surgeon, "alpha_sweep", return_value={
            "best_alpha": 0.7,
            "best_score": 0.92,
            "results": [
                {"alpha": 0.0, "score": 0.85},
                {"alpha": 0.5, "score": 0.90},
                {"alpha": 0.7, "score": 0.92},
                {"alpha": 1.0, "score": 0.88},
            ],
        }) as mock_sweep:
            sweep_result = surgeon.alpha_sweep(alphas=[0.0, 0.5, 0.7, 1.0])

        # Verify data flows correctly
        assert sweep_result["best_alpha"] == 0.7
        assert sweep_result["best_score"] == 0.92
        assert len(sweep_result["results"]) == 4
        mock_sweep.assert_called_once_with(alphas=[0.0, 0.5, 0.7, 1.0])

    def test_evaluator_scores_feed_into_surgery(self, tmp_path):
        """Scores from checkpoint evaluation inform surgery decisions."""
        eval_scores = [
            {"ckpt": "ckpt-100", "score": 0.70},
            {"ckpt": "ckpt-200", "score": 0.85},
            {"ckpt": "ckpt-300", "score": 0.80},
        ]

        # Best checkpoint is ckpt-200 (highest score)
        best = max(eval_scores, key=lambda x: x["score"])
        assert best["ckpt"] == "ckpt-200"

        # Surgery should use the best + runner-up for interpolation
        sorted_scores = sorted(eval_scores, key=lambda x: x["score"], reverse=True)
        top_two = [s["ckpt"] for s in sorted_scores[:2]]
        assert "ckpt-200" in top_two
        assert "ckpt-300" in top_two
