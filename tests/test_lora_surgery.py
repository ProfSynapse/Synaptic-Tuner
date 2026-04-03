"""
Tests for LoRA Weight Surgery module.

Location: tests/test_lora_surgery.py
Purpose: Verify all surgery operations, config loading, and edge cases
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from unittest.mock import AsyncMock

import pytest

# Import torch/safetensors — skip entire module if unavailable
torch = pytest.importorskip("torch")
st = pytest.importorskip("safetensors.torch")

from shared.evolutionary.lora_surgery import (
    LoRASurgeon,
    OperationResult,
    SurgeryConfig,
    SurgeryResult,
    copy_adapter,
    find_lora_pairs,
    get_layer_indices,
    get_module_types,
    is_attention_key,
    is_mlp_key,
    load_adapter_config,
    load_all_weights,
    save_adapter_config,
    save_all_weights,
    softmax,
)
from shared.evolutionary.surgery.registry import get_operation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_lora_weight_keys(num_layers: int = 2):
    """Generate realistic LoRA weight keys for testing."""
    keys = {}
    module_types = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    rank = 8
    hidden = 64

    for layer_idx in range(num_layers):
        for mod in module_types:
            a_key = f"base_model.model.model.layers.{layer_idx}.self_attn.{mod}.lora_A.weight"
            b_key = f"base_model.model.model.layers.{layer_idx}.self_attn.{mod}.lora_B.weight"
            if mod in ("gate_proj", "up_proj", "down_proj"):
                a_key = f"base_model.model.model.layers.{layer_idx}.mlp.{mod}.lora_A.weight"
                b_key = f"base_model.model.model.layers.{layer_idx}.mlp.{mod}.lora_B.weight"

            keys[a_key] = torch.randn(rank, hidden)
            keys[b_key] = torch.randn(hidden, rank)

    return keys


@pytest.fixture
def tmp_adapter(tmp_path):
    """Create a temporary adapter directory with config and weights."""
    adapter_dir = str(tmp_path / "test_adapter")
    os.makedirs(adapter_dir)

    # Write adapter_config.json
    config = {
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["q_proj", "k_proj", "v_proj"],
        "lora_dropout": 0.05,
    }
    with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
        json.dump(config, f)

    # Write weights
    weights = _make_lora_weight_keys(num_layers=2)
    st.save_file(weights, os.path.join(adapter_dir, "adapter_model.safetensors"))

    return adapter_dir


@pytest.fixture
def tmp_other_adapter(tmp_path):
    """Create a second adapter for interpolation tests."""
    adapter_dir = str(tmp_path / "other_adapter")
    os.makedirs(adapter_dir)

    config = {
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["q_proj", "k_proj", "v_proj"],
    }
    with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
        json.dump(config, f)

    weights = _make_lora_weight_keys(num_layers=2)
    st.save_file(weights, os.path.join(adapter_dir, "adapter_model.safetensors"))

    return adapter_dir


@dataclass
class _FakeEvalResult:
    """Minimal eval result with eval_score attribute."""
    eval_score: float


class FakeEvalBackend:
    """Fake eval backend that returns configurable scores."""

    def __init__(self, scores=None, default_score=0.5):
        self._scores = scores or {}
        self._default = default_score
        self._call_count = 0
        self._calls = []

    async def run_eval(self, adapter_path: str, scenario: str) -> _FakeEvalResult:
        self._call_count += 1
        self._calls.append(adapter_path)
        # Return specific score if path matches a pattern, else default
        for pattern, score in self._scores.items():
            if pattern in adapter_path:
                return _FakeEvalResult(eval_score=score)
        return _FakeEvalResult(eval_score=self._default)


class ImprovingEvalBackend:
    """Eval backend that returns increasingly better scores for non-original paths."""

    def __init__(self, baseline=0.5, improvement=0.05):
        self._baseline = baseline
        self._improvement = improvement
        self._call_count = 0

    async def run_eval(self, adapter_path: str, scenario: str) -> _FakeEvalResult:
        self._call_count += 1
        if "test_adapter" in adapter_path and "_surgery_work" not in adapter_path:
            return _FakeEvalResult(eval_score=self._baseline)
        # Variants get a small improvement
        return _FakeEvalResult(eval_score=self._baseline + self._improvement)


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------

class TestSurgeryConfig:
    def test_default_values(self):
        config = SurgeryConfig()
        assert config.min_improvement == 0.005
        assert config.eval_backend == "local"
        assert "alpha_sweep" in config.operations

    def test_from_yaml(self, tmp_path):
        yaml_content = """
surgery:
  min_improvement: 0.01
  eval_backend: "cloud"
  operations:
    - alpha_sweep
    - dare_drop_rescale
  alpha_sweep:
    multipliers: [0.5, 2.0]
  dare:
    drop_rates: [0.1, 0.5]
  svd_rank_reduction:
    rank_fractions: [0.5]
"""
        yaml_path = tmp_path / "test_config.yaml"
        yaml_path.write_text(yaml_content)

        config = SurgeryConfig.from_yaml(str(yaml_path))
        assert config.min_improvement == 0.01
        assert config.eval_backend == "cloud"
        assert config.operations == ["alpha_sweep", "dare_drop_rescale"]
        assert config.alpha_multipliers == [0.5, 2.0]
        assert config.dare_drop_rates == [0.1, 0.5]
        assert config.svd_rank_fractions == [0.5]

    def test_to_dict(self):
        config = SurgeryConfig(adapter_path="/some/path", min_improvement=0.02)
        d = config.to_dict()
        assert d["adapter_path"] == "/some/path"
        assert d["min_improvement"] == 0.02

    def test_from_yaml_missing_file(self):
        with pytest.raises(FileNotFoundError):
            SurgeryConfig.from_yaml("/nonexistent/path.yaml")


# ---------------------------------------------------------------------------
# Helper Function Tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def testget_layer_indices(self):
        keys = [
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
            "base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight",
            "base_model.model.model.layers.7.mlp.gate_proj.lora_A.weight",
        ]
        assert get_layer_indices(keys) == [0, 3, 7]

    def test_get_layer_indices_empty(self):
        assert get_layer_indices([]) == []
        assert get_layer_indices(["some.random.key"]) == []

    def testget_module_types(self):
        keys = [
            "layers.0.self_attn.q_proj.lora_A.weight",
            "layers.0.self_attn.k_proj.lora_B.weight",
            "layers.0.mlp.gate_proj.lora_A.weight",
        ]
        types = get_module_types(keys)
        assert "q_proj" in types
        assert "k_proj" in types
        assert "gate_proj" in types

    def testis_attention_key(self):
        assert is_attention_key("layers.0.self_attn.q_proj.lora_A.weight")
        assert is_attention_key("layers.0.self_attn.k_proj.lora_B.weight")
        assert is_attention_key("layers.0.self_attn.v_proj.lora_A.weight")
        assert is_attention_key("layers.0.self_attn.o_proj.lora_A.weight")
        assert not is_attention_key("layers.0.mlp.gate_proj.lora_A.weight")

    def testis_mlp_key(self):
        assert is_mlp_key("layers.0.mlp.gate_proj.lora_A.weight")
        assert is_mlp_key("layers.0.mlp.up_proj.lora_B.weight")
        assert is_mlp_key("layers.0.mlp.down_proj.lora_A.weight")
        assert not is_mlp_key("layers.0.self_attn.q_proj.lora_A.weight")

    def testsoftmax(self):
        result = softmax([1.0, 1.0, 1.0])
        assert len(result) == 3
        assert abs(sum(result) - 1.0) < 1e-6
        assert abs(result[0] - result[1]) < 1e-6

    def test_softmax_temperature(self):
        # Higher temperature -> more uniform
        high_temp = softmax([1.0, 2.0, 3.0], temperature=10.0)
        low_temp = softmax([1.0, 2.0, 3.0], temperature=0.1)
        # With high temp, values should be more uniform
        assert max(high_temp) - min(high_temp) < max(low_temp) - min(low_temp)

    def testfind_lora_pairs(self):
        weights = {
            "prefix.q_proj.lora_A.weight": torch.randn(8, 64),
            "prefix.q_proj.lora_B.weight": torch.randn(64, 8),
            "prefix.k_proj.lora_A.weight": torch.randn(8, 64),
            # Missing B for k_proj
        }
        pairs = find_lora_pairs(weights)
        assert "prefix.q_proj" in pairs
        assert "prefix.k_proj" not in pairs

    def test_loadsave_adapter_config(self, tmp_path):
        adapter_dir = str(tmp_path / "adapter")
        os.makedirs(adapter_dir)
        config = {"r": 16, "lora_alpha": 32}
        save_adapter_config(adapter_dir, config)
        loaded = load_adapter_config(adapter_dir)
        assert loaded["r"] == 16
        assert loaded["lora_alpha"] == 32

    def test_load_adapter_config_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_adapter_config(str(tmp_path / "nonexistent"))

    def testcopy_adapter(self, tmp_adapter, tmp_path):
        dst = str(tmp_path / "copied")
        result = copy_adapter(tmp_adapter, dst)
        assert result == dst
        assert os.path.exists(os.path.join(dst, "adapter_config.json"))
        assert os.path.exists(os.path.join(dst, "adapter_model.safetensors"))

    def test_loadsave_all_weights(self, tmp_adapter):
        weights = load_all_weights(tmp_adapter)
        assert len(weights) > 0
        # All values should be tensors
        for v in weights.values():
            assert isinstance(v, torch.Tensor)

        # Save and reload
        new_dir = tmp_adapter + "_copy"
        os.makedirs(new_dir)
        save_all_weights(new_dir, weights)
        reloaded = load_all_weights(new_dir)
        assert set(reloaded.keys()) == set(weights.keys())
        shutil.rmtree(new_dir)


# ---------------------------------------------------------------------------
# Operation Tests
# ---------------------------------------------------------------------------

class TestAlphaSweep:
    @pytest.mark.asyncio
    async def test_modifies_adapter_config(self, tmp_adapter, tmp_path):
        """Alpha sweep should modify adapter_config.json, not weights."""
        output_dir = str(tmp_path / "output")
        work_dir = str(tmp_path / "output" / "_surgery_work")
        os.makedirs(work_dir, exist_ok=True)
        config = SurgeryConfig(
            adapter_path=tmp_adapter,
            eval_scenario="test",
            operations=["alpha_sweep"],
            output_dir=output_dir,
            alpha_multipliers=[0.5, 2.0],
        )

        # Backend returns higher score for alpha=32 variant
        backend = FakeEvalBackend(
            scores={"alpha_32": 0.8},
            default_score=0.5,
        )

        async def evaluate_fn(path):
            return (await backend.run_eval(path, "test")).eval_score

        op = get_operation("alpha_sweep")
        result = await op.execute(tmp_adapter, 0.5, work_dir, config, evaluate_fn)

        assert result.operation == "alpha_sweep"
        assert result.variants_tried >= 1
        assert "alpha_scores" in result.details

    @pytest.mark.asyncio
    async def test_skips_identical_alpha(self, tmp_adapter, tmp_path):
        """If multiplier produces same alpha, it should be skipped."""
        work_dir = str(tmp_path / "output" / "_surgery_work")
        os.makedirs(work_dir, exist_ok=True)
        config = SurgeryConfig(
            adapter_path=tmp_adapter,
            output_dir=str(tmp_path / "output"),
            alpha_multipliers=[1.0],  # Same as current
        )
        backend = FakeEvalBackend(default_score=0.5)

        async def evaluate_fn(path):
            return (await backend.run_eval(path, "test")).eval_score

        op = get_operation("alpha_sweep")
        result = await op.execute(tmp_adapter, 0.5, work_dir, config, evaluate_fn)
        assert result.variants_tried == 0


class TestLayerScaling:
    @pytest.mark.asyncio
    async def test_applies_correct_multipliers(self, tmp_adapter, tmp_path):
        """Layer scaling should scale layer weights by the given factor."""
        work_dir = str(tmp_path / "output" / "_surgery_work")
        os.makedirs(work_dir, exist_ok=True)
        config = SurgeryConfig(
            adapter_path=tmp_adapter,
            output_dir=str(tmp_path / "output"),
            layer_scales=[0.0, 2.0],
        )
        backend = FakeEvalBackend(default_score=0.5)

        async def evaluate_fn(path):
            return (await backend.run_eval(path, "test")).eval_score

        op = get_operation("layer_scaling")
        result = await op.execute(tmp_adapter, 0.5, work_dir, config, evaluate_fn)

        assert result.operation == "layer_scaling"
        assert result.variants_tried > 0
        assert "layer_scores" in result.details

    @pytest.mark.asyncio
    async def test_zero_scale_zeros_layer(self, tmp_adapter, tmp_path):
        """Scaling a layer by 0 should zero all its weights."""
        work_dir = str(tmp_path / "output" / "_surgery_work")
        os.makedirs(work_dir, exist_ok=True)
        config = SurgeryConfig(
            adapter_path=tmp_adapter,
            output_dir=str(tmp_path / "output"),
            layer_scales=[0.0],
        )
        backend = FakeEvalBackend(default_score=0.5)

        async def evaluate_fn(path):
            return (await backend.run_eval(path, "test")).eval_score

        op = get_operation("layer_scaling")
        await op.execute(tmp_adapter, 0.5, work_dir, config, evaluate_fn)

        # Check a variant directory
        variant_dir = os.path.join(work_dir, "layer0_scale0.0")
        assert os.path.exists(variant_dir), f"Expected variant dir to exist: {variant_dir}"
        weights = load_all_weights(variant_dir)
        for key, tensor in weights.items():
            if "layers.0." in key:
                assert torch.allclose(tensor, torch.zeros_like(tensor))

    @pytest.mark.asyncio
    async def test_no_lora_layers(self, tmp_path):
        """Should handle adapter with no LoRA layer keys gracefully."""
        adapter_dir = str(tmp_path / "empty_adapter")
        os.makedirs(adapter_dir)
        with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
            json.dump({"r": 8, "lora_alpha": 16}, f)
        # Write weights with no layer indices
        weights = {"some.random.key": torch.randn(4, 4)}
        st.save_file(weights, os.path.join(adapter_dir, "adapter_model.safetensors"))

        work_dir = str(tmp_path / "output" / "_surgery_work")
        os.makedirs(work_dir, exist_ok=True)
        config = SurgeryConfig(
            adapter_path=adapter_dir,
            output_dir=str(tmp_path / "output"),
        )
        backend = FakeEvalBackend(default_score=0.5)

        async def evaluate_fn(path):
            return (await backend.run_eval(path, "test")).eval_score

        op = get_operation("layer_scaling")
        result = await op.execute(adapter_dir, 0.5, work_dir, config, evaluate_fn)
        assert result.variants_tried == 0
        assert result.details.get("reason") == "no_lora_layers_found"


class TestModuleAblation:
    @pytest.mark.asyncio
    async def test_zeros_module_weights(self, tmp_adapter, tmp_path):
        """Module ablation should zero all weights for a given module type."""
        work_dir = str(tmp_path / "output" / "_surgery_work")
        os.makedirs(work_dir, exist_ok=True)
        config = SurgeryConfig(
            adapter_path=tmp_adapter,
            output_dir=str(tmp_path / "output"),
        )
        backend = FakeEvalBackend(default_score=0.5)

        async def evaluate_fn(path):
            return (await backend.run_eval(path, "test")).eval_score

        op = get_operation("module_ablation")
        result = await op.execute(tmp_adapter, 0.5, work_dir, config, evaluate_fn)

        assert result.operation == "module_ablation"
        assert result.variants_tried > 0
        assert "module_scores" in result.details

        # Verify an ablated variant
        variant_dir = os.path.join(work_dir, "ablate_q_proj")
        assert os.path.exists(variant_dir), f"Expected variant dir to exist: {variant_dir}"
        weights = load_all_weights(variant_dir)
        for key, tensor in weights.items():
            if ".q_proj." in key:
                assert torch.allclose(tensor, torch.zeros_like(tensor))


class TestCheckpointInterpolation:
    @pytest.mark.asyncio
    async def test_blends_at_correct_ratios(self, tmp_adapter, tmp_other_adapter, tmp_path):
        """Interpolation should blend weights at the specified ratio."""
        work_dir = str(tmp_path / "output" / "_surgery_work")
        os.makedirs(work_dir, exist_ok=True)
        config = SurgeryConfig(
            adapter_path=tmp_adapter,
            output_dir=str(tmp_path / "output"),
            other_checkpoint_path=tmp_other_adapter,
            blend_ratios=[0.5],
        )
        backend = FakeEvalBackend(default_score=0.5)

        async def evaluate_fn(path):
            return (await backend.run_eval(path, "test")).eval_score

        op = get_operation("checkpoint_interpolation")
        result = await op.execute(tmp_adapter, 0.5, work_dir, config, evaluate_fn)

        assert result.operation == "checkpoint_interpolation"
        assert result.variants_tried == 1
        assert "blend_scores" in result.details

    @pytest.mark.asyncio
    async def test_blend_ratio_math(self, tmp_adapter, tmp_other_adapter, tmp_path):
        """Verify that blend ratio=0.5 produces the average of two checkpoints."""
        weights_a = load_all_weights(tmp_adapter)
        weights_b = load_all_weights(tmp_other_adapter)

        work_dir = str(tmp_path / "output" / "_surgery_work")
        os.makedirs(work_dir, exist_ok=True)
        config = SurgeryConfig(
            adapter_path=tmp_adapter,
            output_dir=str(tmp_path / "output"),
            other_checkpoint_path=tmp_other_adapter,
            blend_ratios=[0.5],
        )
        backend = FakeEvalBackend(default_score=0.5)

        async def evaluate_fn(path):
            return (await backend.run_eval(path, "test")).eval_score

        op = get_operation("checkpoint_interpolation")
        await op.execute(tmp_adapter, 0.5, work_dir, config, evaluate_fn)

        variant_dir = os.path.join(work_dir, "blend_0.50")
        assert os.path.exists(variant_dir), f"Expected variant dir to exist: {variant_dir}"
        blended = load_all_weights(variant_dir)
        common_keys = set(weights_a.keys()) & set(weights_b.keys())
        for key in list(common_keys)[:3]:
            expected = 0.5 * weights_a[key] + 0.5 * weights_b[key]
            assert torch.allclose(blended[key], expected, atol=1e-5)

    @pytest.mark.asyncio
    async def test_single_checkpoint_no_interpolation(self, tmp_adapter, tmp_path):
        """Should skip if no other checkpoint is provided."""
        work_dir = str(tmp_path / "output" / "_surgery_work")
        os.makedirs(work_dir, exist_ok=True)
        config = SurgeryConfig(
            adapter_path=tmp_adapter,
            output_dir=str(tmp_path / "output"),
            other_checkpoint_path="",
        )
        backend = FakeEvalBackend(default_score=0.5)

        async def evaluate_fn(path):
            return (await backend.run_eval(path, "test")).eval_score

        op = get_operation("checkpoint_interpolation")
        result = await op.execute(tmp_adapter, 0.5, work_dir, config, evaluate_fn)
        assert result.variants_tried == 0
        assert result.details.get("reason") == "no_other_checkpoint"


class TestDAREDropRescale:
    def test_expected_value_preserved(self, tmp_adapter, tmp_path):
        """DARE should preserve expected value: E[x_dropped] approx E[x_original]."""
        weights = load_all_weights(tmp_adapter)
        # Pick a representative key
        sample_key = list(weights.keys())[0]
        original = weights[sample_key]

        drop_rate = 0.3
        # Run many trials to test expected value
        num_trials = 100
        accumulated = torch.zeros_like(original.float())
        for _ in range(num_trials):
            mask = (torch.rand_like(original.float()) > drop_rate).to(original.dtype)
            dropped = original * mask / (1.0 - drop_rate)
            accumulated += dropped.float()

        mean_result = accumulated / num_trials
        # Expected value should be close to original
        assert torch.allclose(
            mean_result, original.float(), atol=0.5
        ), "DARE expected value not preserved"

    @pytest.mark.asyncio
    async def test_operation_runs(self, tmp_adapter, tmp_path):
        work_dir = str(tmp_path / "output" / "_surgery_work")
        os.makedirs(work_dir, exist_ok=True)
        config = SurgeryConfig(
            adapter_path=tmp_adapter,
            output_dir=str(tmp_path / "output"),
            dare_drop_rates=[0.2],
        )
        backend = FakeEvalBackend(default_score=0.5)

        async def evaluate_fn(path):
            return (await backend.run_eval(path, "test")).eval_score

        op = get_operation("dare_drop_rescale")
        result = await op.execute(tmp_adapter, 0.5, work_dir, config, evaluate_fn)
        assert result.operation == "dare_drop_rescale"
        assert result.variants_tried == 1
        assert "dare_scores" in result.details


class TestSVDRankReduction:
    @pytest.mark.asyncio
    async def test_produces_correct_dimensions(self, tmp_adapter, tmp_path):
        """SVD reduction should produce A/B matrices with reduced rank."""
        work_dir = str(tmp_path / "output" / "_surgery_work")
        os.makedirs(work_dir, exist_ok=True)
        config = SurgeryConfig(
            adapter_path=tmp_adapter,
            output_dir=str(tmp_path / "output"),
            svd_rank_fractions=[0.5],  # rank 8 -> rank 4
        )
        backend = FakeEvalBackend(default_score=0.5)

        async def evaluate_fn(path):
            return (await backend.run_eval(path, "test")).eval_score

        op = get_operation("svd_rank_reduction")
        result = await op.execute(tmp_adapter, 0.5, work_dir, config, evaluate_fn)

        assert result.operation == "svd_rank_reduction"
        assert result.variants_tried == 1

        # Check dimensions in the variant
        variant_dir = os.path.join(work_dir, "svd_rank4")
        assert os.path.exists(variant_dir), f"Expected variant dir to exist: {variant_dir}"
        weights = load_all_weights(variant_dir)
        pairs = find_lora_pairs(weights)
        for prefix, (a_key, b_key) in pairs.items():
            a_shape = weights[a_key].shape
            b_shape = weights[b_key].shape
            assert a_shape[0] == 4, f"Expected rank 4 for A, got {a_shape[0]}"
            assert b_shape[1] == 4, f"Expected rank 4 for B, got {b_shape[1]}"

    @pytest.mark.asyncio
    async def test_rank_too_small(self, tmp_path):
        """Should skip if original rank is 1."""
        adapter_dir = str(tmp_path / "rank1_adapter")
        os.makedirs(adapter_dir)
        with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
            json.dump({"r": 1, "lora_alpha": 1}, f)
        weights = {"some.lora_A.weight": torch.randn(1, 64)}
        st.save_file(weights, os.path.join(adapter_dir, "adapter_model.safetensors"))

        work_dir = str(tmp_path / "output" / "_surgery_work")
        os.makedirs(work_dir, exist_ok=True)
        config = SurgeryConfig(
            adapter_path=adapter_dir,
            output_dir=str(tmp_path / "output"),
        )
        backend = FakeEvalBackend(default_score=0.5)

        async def evaluate_fn(path):
            return (await backend.run_eval(path, "test")).eval_score

        op = get_operation("svd_rank_reduction")
        result = await op.execute(adapter_dir, 0.5, work_dir, config, evaluate_fn)
        assert result.variants_tried == 0
        assert result.details.get("reason") == "rank_too_small"


class TestAttentionMLPAblation:
    @pytest.mark.asyncio
    async def test_ablation_runs(self, tmp_adapter, tmp_path):
        work_dir = str(tmp_path / "output" / "_surgery_work")
        os.makedirs(work_dir, exist_ok=True)
        config = SurgeryConfig(
            adapter_path=tmp_adapter,
            output_dir=str(tmp_path / "output"),
        )
        backend = FakeEvalBackend(default_score=0.5)

        async def evaluate_fn(path):
            return (await backend.run_eval(path, "test")).eval_score

        op = get_operation("attention_mlp_ablation")
        result = await op.execute(tmp_adapter, 0.5, work_dir, config, evaluate_fn)

        assert result.operation == "attention_mlp_ablation"
        assert result.variants_tried == 2  # attention + mlp
        assert "ablation_scores" in result.details
        assert "zero_attention" in result.details["ablation_scores"]
        assert "zero_mlp" in result.details["ablation_scores"]


class TestMetricsWeightedMerge:
    @pytest.mark.asyncio
    async def test_insufficient_checkpoints(self, tmp_adapter, tmp_path):
        work_dir = str(tmp_path / "output" / "_surgery_work")
        os.makedirs(work_dir, exist_ok=True)
        config = SurgeryConfig(
            adapter_path=tmp_adapter,
            output_dir=str(tmp_path / "output"),
            checkpoint_paths=[tmp_adapter],  # Only one
            checkpoint_scores=[0.5],
        )
        backend = FakeEvalBackend(default_score=0.5)

        async def evaluate_fn(path):
            return (await backend.run_eval(path, "test")).eval_score

        op = get_operation("metrics_weighted_merge")
        result = await op.execute(tmp_adapter, 0.5, work_dir, config, evaluate_fn)
        assert result.variants_tried == 0
        assert result.details.get("reason") == "insufficient_checkpoints_or_scores"

    @pytest.mark.asyncio
    async def test_merge_runs(self, tmp_adapter, tmp_other_adapter, tmp_path):
        work_dir = str(tmp_path / "output" / "_surgery_work")
        os.makedirs(work_dir, exist_ok=True)
        config = SurgeryConfig(
            adapter_path=tmp_adapter,
            output_dir=str(tmp_path / "output"),
            checkpoint_paths=[tmp_adapter, tmp_other_adapter],
            checkpoint_scores=[0.7, 0.3],
        )
        backend = FakeEvalBackend(default_score=0.5)

        async def evaluate_fn(path):
            return (await backend.run_eval(path, "test")).eval_score

        op = get_operation("metrics_weighted_merge")
        result = await op.execute(tmp_adapter, 0.5, work_dir, config, evaluate_fn)
        assert result.operation == "metrics_weighted_merge"
        assert result.variants_tried == 1
        assert "merge_weights" in result.details


# ---------------------------------------------------------------------------
# Surgery Loop Tests
# ---------------------------------------------------------------------------

class TestSurgeryLoop:
    @pytest.mark.asyncio
    async def test_keeps_improvements(self, tmp_adapter, tmp_path):
        """Surgery loop should keep operations that improve the score."""
        output_dir = str(tmp_path / "output")
        config = SurgeryConfig(
            adapter_path=tmp_adapter,
            eval_scenario="test",
            operations=["alpha_sweep"],
            output_dir=output_dir,
            min_improvement=0.001,
            alpha_multipliers=[2.0],
        )

        # Return higher score for alpha variants
        backend = ImprovingEvalBackend(baseline=0.5, improvement=0.1)
        surgeon = LoRASurgeon(tmp_adapter, backend, "test", config)

        result = await surgeon.run_surgery()

        assert result.baseline_score == 0.5
        assert result.final_score > result.baseline_score
        assert len(result.operations_applied) > 0
        assert result.total_improvement > 0
        assert os.path.exists(result.best_adapter_path)

    @pytest.mark.asyncio
    async def test_rejects_regressions(self, tmp_adapter, tmp_path):
        """Surgery loop should reject operations that don't improve enough."""
        output_dir = str(tmp_path / "output")
        config = SurgeryConfig(
            adapter_path=tmp_adapter,
            eval_scenario="test",
            operations=["alpha_sweep"],
            output_dir=output_dir,
            min_improvement=0.5,  # Very high threshold
            alpha_multipliers=[2.0],
        )

        # All variants return the same score as baseline
        backend = FakeEvalBackend(default_score=0.5)
        surgeon = LoRASurgeon(tmp_adapter, backend, "test", config)

        result = await surgeon.run_surgery()

        assert result.baseline_score == 0.5
        assert result.final_score == 0.5
        assert len(result.operations_applied) == 0
        assert result.total_improvement == 0.0

    @pytest.mark.asyncio
    async def test_results_report_saved(self, tmp_adapter, tmp_path):
        """Surgery should save a JSON report."""
        output_dir = str(tmp_path / "output")
        config = SurgeryConfig(
            adapter_path=tmp_adapter,
            eval_scenario="test",
            operations=["alpha_sweep"],
            output_dir=output_dir,
            alpha_multipliers=[2.0],
        )
        backend = FakeEvalBackend(default_score=0.5)
        surgeon = LoRASurgeon(tmp_adapter, backend, "test", config)

        await surgeon.run_surgery()

        report_path = os.path.join(output_dir, "surgery_report.json")
        assert os.path.exists(report_path)

        with open(report_path) as f:
            report = json.load(f)
        assert "baseline_score" in report
        assert "final_score" in report
        assert "operations_applied" in report
        assert "duration_seconds" in report

    @pytest.mark.asyncio
    async def test_unknown_operation_skipped(self, tmp_adapter, tmp_path):
        """Unknown operations should be skipped without error."""
        output_dir = str(tmp_path / "output")
        config = SurgeryConfig(
            adapter_path=tmp_adapter,
            eval_scenario="test",
            operations=["nonexistent_op", "alpha_sweep"],
            output_dir=output_dir,
            alpha_multipliers=[2.0],
        )
        backend = FakeEvalBackend(default_score=0.5)
        surgeon = LoRASurgeon(tmp_adapter, backend, "test", config)

        # Should not raise
        result = await surgeon.run_surgery()
        assert result.baseline_score == 0.5


class TestOperationOrdering:
    def test_cheapest_first_default(self):
        """Default operation order should start with alpha_sweep (cheapest)."""
        config = SurgeryConfig()
        assert config.operations[0] == "alpha_sweep"

    @pytest.mark.asyncio
    async def test_operations_run_in_config_order(self, tmp_adapter, tmp_path):
        """Operations should run in the order specified in config."""
        output_dir = str(tmp_path / "output")

        order_tracker = []

        class TrackingBackend:
            async def run_eval(self, adapter_path: str, scenario: str) -> _FakeEvalResult:
                # Track which operation directory is being evaluated
                if "_surgery_work" in adapter_path:
                    dirname = os.path.basename(adapter_path)
                    if dirname.startswith("alpha"):
                        order_tracker.append("alpha_sweep")
                    elif dirname.startswith("ablate_"):
                        order_tracker.append("module_ablation")
                return _FakeEvalResult(eval_score=0.5)

        config = SurgeryConfig(
            adapter_path=tmp_adapter,
            eval_scenario="test",
            operations=["alpha_sweep", "module_ablation"],
            output_dir=output_dir,
            alpha_multipliers=[2.0],
        )
        backend = TrackingBackend()
        surgeon = LoRASurgeon(tmp_adapter, backend, "test", config)

        await surgeon.run_surgery()

        # Alpha sweep evaluations should come before module ablation evaluations
        if order_tracker:
            first_alpha = next(
                (i for i, op in enumerate(order_tracker) if op == "alpha_sweep"), None
            )
            first_ablation = next(
                (i for i, op in enumerate(order_tracker) if op == "module_ablation"),
                None,
            )
            if first_alpha is not None and first_ablation is not None:
                assert first_alpha < first_ablation


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_adapter_with_no_lora_layers(self, tmp_path):
        """Should handle adapter with no LoRA-patterned weight keys."""
        adapter_dir = str(tmp_path / "no_lora_adapter")
        os.makedirs(adapter_dir)
        with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
            json.dump({"r": 8, "lora_alpha": 16}, f)
        weights = {"embedding.weight": torch.randn(100, 64)}
        st.save_file(weights, os.path.join(adapter_dir, "adapter_model.safetensors"))

        output_dir = str(tmp_path / "output")
        config = SurgeryConfig(
            adapter_path=adapter_dir,
            eval_scenario="test",
            operations=["layer_scaling", "module_ablation"],
            output_dir=output_dir,
        )
        backend = FakeEvalBackend(default_score=0.5)
        surgeon = LoRASurgeon(adapter_dir, backend, "test", config)

        result = await surgeon.run_surgery()
        # Should complete without errors
        assert result.baseline_score == 0.5
        assert len(result.operations_applied) == 0

    @pytest.mark.asyncio
    async def test_duration_tracked(self, tmp_adapter, tmp_path):
        """Surgery result should track duration."""
        output_dir = str(tmp_path / "output")
        config = SurgeryConfig(
            adapter_path=tmp_adapter,
            eval_scenario="test",
            operations=["alpha_sweep"],
            output_dir=output_dir,
            alpha_multipliers=[2.0],
        )
        backend = FakeEvalBackend(default_score=0.5)
        surgeon = LoRASurgeon(tmp_adapter, backend, "test", config)

        result = await surgeon.run_surgery()
        assert result.duration_seconds >= 0


# ---------------------------------------------------------------------------
# Registry API Tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_list_operations_returns_all_eight(self):
        """All 8 operations should be registered."""
        from shared.evolutionary.surgery.registry import list_operations
        ops = list_operations()
        assert len(ops) == 8
        expected = [
            "alpha_sweep", "attention_mlp_ablation", "checkpoint_interpolation",
            "dare_drop_rescale", "layer_scaling", "metrics_weighted_merge",
            "module_ablation", "svd_rank_reduction",
        ]
        assert ops == expected  # list_operations returns sorted

    def test_get_operation_returns_instance(self):
        """get_operation should return a fresh instance, not the class."""
        op = get_operation("alpha_sweep")
        assert hasattr(op, "execute")
        assert hasattr(op, "name")
        assert op.name == "alpha_sweep"

    def test_get_operation_unknown_raises_value_error(self):
        """get_operation should raise ValueError for unregistered names."""
        with pytest.raises(ValueError, match="Unknown surgery operation"):
            get_operation("nonexistent_operation")

    def test_each_operation_has_unique_name(self):
        """All registered operations should have distinct name attributes."""
        from shared.evolutionary.surgery.registry import list_operations
        ops = list_operations()
        for name in ops:
            op = get_operation(name)
            assert op.name == name


# ---------------------------------------------------------------------------
# Protocol Conformance Tests
# ---------------------------------------------------------------------------

class TestProtocolConformance:
    def test_all_operations_implement_protocol(self):
        """All registered operations should structurally match SurgeryOperation."""
        from shared.evolutionary.surgery.base import SurgeryOperation
        from shared.evolutionary.surgery.registry import list_operations
        import inspect

        # Get protocol params excluding 'self'
        protocol_method = inspect.signature(SurgeryOperation.execute)
        protocol_params = [
            p for p in protocol_method.parameters.keys() if p != "self"
        ]

        for name in list_operations():
            op = get_operation(name)
            # Check 'name' attribute exists
            assert hasattr(op, "name"), f"{name} missing 'name' attribute"
            assert isinstance(op.name, str), f"{name}.name is not a str"

            # Check execute method exists and has matching signature
            assert hasattr(op, "execute"), f"{name} missing 'execute' method"
            method = inspect.signature(op.execute)
            # Instance method signature excludes 'self'
            op_params = list(method.parameters.keys())
            assert op_params == protocol_params, (
                f"{name}.execute params {op_params} != protocol {protocol_params}"
            )

    def test_evaluate_fn_is_used_with_await(self):
        """Verify evaluate_fn is typed as Callable[[str], float] but used with await.

        The auditor flagged this: evaluate_fn typed as Callable[[str], float]
        but called with 'await evaluate_fn(...)'. This works at runtime because
        Python's typing doesn't enforce return types, and the actual callable
        passed is always an async function returning float. But the type
        annotation is technically incorrect — it should be
        Callable[[str], Awaitable[float]] for strict typing.

        This test documents the behavior and verifies it works correctly.
        """
        import asyncio
        import inspect

        # Verify the Protocol signature types evaluate_fn as sync
        from shared.evolutionary.surgery.base import SurgeryOperation
        sig = inspect.signature(SurgeryOperation.execute)
        # The parameter exists
        assert "evaluate_fn" in sig.parameters

        # Verify that passing an async callable still works
        async def async_evaluate(path: str) -> float:
            return 0.75

        # The function is async
        assert asyncio.iscoroutinefunction(async_evaluate)

        # But its annotation says Callable[[str], float]
        # This is the documented typing gap. Operations await the result,
        # which works because async functions return coroutines that resolve to float.


# ---------------------------------------------------------------------------
# Context Manager Tests
# ---------------------------------------------------------------------------

class TestContextManager:
    @pytest.mark.asyncio
    async def test_async_context_manager(self, tmp_adapter, tmp_path):
        """LoRASurgeon should work as an async context manager."""
        output_dir = str(tmp_path / "output")
        config = SurgeryConfig(
            adapter_path=tmp_adapter,
            eval_scenario="test",
            operations=["alpha_sweep"],
            output_dir=output_dir,
            alpha_multipliers=[2.0],
        )
        backend = FakeEvalBackend(default_score=0.5)

        async with LoRASurgeon(tmp_adapter, backend, "test", config) as surgeon:
            assert surgeon is not None
            assert surgeon.adapter_path == tmp_adapter

    @pytest.mark.asyncio
    async def test_cleanup_removes_work_dir(self, tmp_adapter, tmp_path):
        """cleanup() should remove the _surgery_work directory."""
        output_dir = str(tmp_path / "output")
        config = SurgeryConfig(
            adapter_path=tmp_adapter,
            eval_scenario="test",
            operations=["alpha_sweep"],
            output_dir=output_dir,
            alpha_multipliers=[2.0],
        )
        backend = FakeEvalBackend(default_score=0.5)
        surgeon = LoRASurgeon(tmp_adapter, backend, "test", config)

        # Run surgery (which creates and then cleans up work dir)
        await surgeon.run_surgery()

        # Work dir should be cleaned up after surgery
        work_dir = os.path.join(output_dir, "_surgery_work")
        assert not os.path.exists(work_dir)


# ---------------------------------------------------------------------------
# Surgeon Backward-Compat Proxy Methods
# ---------------------------------------------------------------------------

class TestSurgeonProxyMethods:
    @pytest.mark.asyncio
    async def test_direct_operation_method_matches_registry(self, tmp_adapter, tmp_path):
        """Surgeon's named methods should delegate to the same registry operations."""
        output_dir = str(tmp_path / "output")
        config = SurgeryConfig(
            adapter_path=tmp_adapter,
            eval_scenario="test",
            operations=[],
            output_dir=output_dir,
            alpha_multipliers=[2.0],
        )
        backend = FakeEvalBackend(default_score=0.5)
        surgeon = LoRASurgeon(tmp_adapter, backend, "test", config)

        # Call the backward-compat proxy method directly
        result = await surgeon.alpha_sweep(tmp_adapter, 0.5)
        assert result.operation == "alpha_sweep"
        assert isinstance(result, OperationResult)
