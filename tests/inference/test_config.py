"""Tests for shared/inference/config.py — config loading, profiles, merging."""

from __future__ import annotations

import pytest
from pathlib import Path

from shared.inference.config import (
    DoLaConfig,
    InferencePluginConfig,
    MinPConfig,
    RepetitionPenaltyConfig,
    VLLMHookConfig,
    InferenceOverrides,
    _deep_merge,
    _build_sub_config,
)


# ---------------------------------------------------------------------------
# DoLaConfig
# ---------------------------------------------------------------------------

class TestDoLaConfig:
    def test_defaults(self):
        cfg = DoLaConfig()
        assert cfg.enabled is False
        assert cfg.premature_layers == "high"
        assert cfg.mature_layer is None
        assert cfg.relative_top == 0.1
        assert cfg.jsd_threshold == 0.0

    def test_explicit_layers(self):
        cfg = DoLaConfig(premature_layers=[2, 4, 6, 8])
        assert cfg.premature_layers == [2, 4, 6, 8]


# ---------------------------------------------------------------------------
# InferencePluginConfig.from_dict
# ---------------------------------------------------------------------------

class TestFromDict:
    def test_empty_dict_returns_defaults(self):
        cfg = InferencePluginConfig.from_dict({})
        assert cfg.dola.enabled is False
        assert cfg.min_p.enabled is False

    def test_plugins_nested_structure(self):
        raw = {
            "plugins": {
                "dola": {"enabled": True, "premature_layers": "low"},
                "min_p": {"enabled": True, "threshold": 0.03},
            },
        }
        cfg = InferencePluginConfig.from_dict(raw)
        assert cfg.dola.enabled is True
        assert cfg.dola.premature_layers == "low"
        assert cfg.min_p.enabled is True
        assert cfg.min_p.threshold == 0.03

    def test_unknown_keys_are_ignored(self):
        raw = {
            "plugins": {
                "dola": {"enabled": True, "unknown_key": 42},
            },
        }
        cfg = InferencePluginConfig.from_dict(raw)
        assert cfg.dola.enabled is True

    def test_vllm_hook_at_top_level(self):
        raw = {
            "vllm_hook": {"enabled": True, "registry_port": 9999},
        }
        cfg = InferencePluginConfig.from_dict(raw)
        assert cfg.vllm_hook.enabled is True
        assert cfg.vllm_hook.registry_port == 9999

    def test_inference_overrides(self):
        raw = {
            "inference": {"temperature": 0.5, "seed": 42},
        }
        cfg = InferencePluginConfig.from_dict(raw)
        assert cfg.inference.temperature == 0.5
        assert cfg.inference.seed == 42
        assert cfg.inference.top_p is None


# ---------------------------------------------------------------------------
# InferencePluginConfig.from_yaml
# ---------------------------------------------------------------------------

class TestFromYaml:
    def test_load_default_yaml(self):
        cfg = InferencePluginConfig.from_yaml("configs/inference/default.yaml")
        assert cfg.dola.enabled is False
        assert cfg.dola.premature_layers == "high"
        assert cfg.dola.relative_top == 0.1
        assert cfg.vllm_hook.enabled is False

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            InferencePluginConfig.from_yaml("nonexistent.yaml")


# ---------------------------------------------------------------------------
# InferencePluginConfig.from_profile
# ---------------------------------------------------------------------------

class TestFromProfile:
    def test_factual_profile(self):
        cfg = InferencePluginConfig.from_profile("factual")
        assert cfg.dola.enabled is True
        assert cfg.dola.premature_layers == "high"
        assert cfg.dola.relative_top == 0.1

    def test_reasoning_profile(self):
        cfg = InferencePluginConfig.from_profile("reasoning")
        assert cfg.dola.enabled is True
        assert cfg.dola.premature_layers == "low"

    def test_creative_profile(self):
        cfg = InferencePluginConfig.from_profile("creative")
        assert cfg.dola.enabled is False
        assert cfg.min_p.enabled is True
        assert cfg.min_p.threshold == 0.02

    def test_extends_inherits_base_values(self):
        """Profile should inherit values from the base config it extends."""
        cfg = InferencePluginConfig.from_profile("factual")
        # vllm_hook is not overridden in factual profile — should get base value
        assert cfg.vllm_hook.registry_port == 9090

    def test_missing_profile_raises(self):
        with pytest.raises(FileNotFoundError):
            InferencePluginConfig.from_profile("nonexistent")


# ---------------------------------------------------------------------------
# needs_layer_hooks
# ---------------------------------------------------------------------------

class TestNeedsLayerHooks:
    def test_dola_enabled(self, factual_config):
        assert factual_config.needs_layer_hooks() is True

    def test_all_disabled(self, default_config):
        assert default_config.needs_layer_hooks() is False

    def test_logits_only(self, creative_config):
        assert creative_config.needs_layer_hooks() is False


# ---------------------------------------------------------------------------
# to_dict / from_dict round-trip
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_round_trip_preserves_values(self):
        original = InferencePluginConfig(
            dola=DoLaConfig(enabled=True, premature_layers=[2, 4, 6]),
            min_p=MinPConfig(enabled=True, threshold=0.03),
            repetition_penalty=RepetitionPenaltyConfig(enabled=True, penalty=1.5, window=128),
        )
        roundtripped = InferencePluginConfig.from_dict(original.to_dict())
        assert roundtripped.dola.enabled is True
        assert roundtripped.dola.premature_layers == [2, 4, 6]
        assert roundtripped.min_p.threshold == 0.03
        assert roundtripped.repetition_penalty.penalty == 1.5
        assert roundtripped.repetition_penalty.window == 128


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class TestDeepMerge:
    def test_shallow_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3}
        assert _deep_merge(base, override) == {"a": 1, "b": 3}

    def test_nested_merge(self):
        base = {"plugins": {"dola": {"enabled": False, "relative_top": 0.1}}}
        override = {"plugins": {"dola": {"enabled": True}}}
        result = _deep_merge(base, override)
        assert result["plugins"]["dola"]["enabled"] is True
        assert result["plugins"]["dola"]["relative_top"] == 0.1

    def test_does_not_mutate_base(self):
        base = {"a": {"b": 1}}
        override = {"a": {"b": 2}}
        _deep_merge(base, override)
        assert base["a"]["b"] == 1


class TestBuildSubConfig:
    def test_empty_dict_returns_defaults(self):
        cfg = _build_sub_config(DoLaConfig, {})
        assert cfg.enabled is False

    def test_unknown_keys_filtered(self):
        cfg = _build_sub_config(DoLaConfig, {"enabled": True, "bogus": 99})
        assert cfg.enabled is True
