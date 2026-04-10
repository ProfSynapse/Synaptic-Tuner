"""Tests for shared/inference/registry.py — plugin discovery and instantiation."""

from __future__ import annotations

import pytest

from shared.inference.config import (
    DoLaConfig,
    InferencePluginConfig,
    MinPConfig,
    RepetitionPenaltyConfig,
)
from shared.inference.registry import PluginRegistry


class TestPluginRegistry:
    def test_no_plugins_enabled(self, default_config):
        registry = PluginRegistry(default_config)
        assert registry.active_plugin_names == []
        assert registry.layer_hook_plugins == []
        assert registry.logits_plugins == []
        assert registry.needs_layer_hooks is False

    def test_dola_only(self):
        cfg = InferencePluginConfig(
            dola=DoLaConfig(enabled=True, premature_layers="high"),
        )
        registry = PluginRegistry(cfg)
        assert "dola" in registry.active_plugin_names
        assert len(registry.layer_hook_plugins) == 1
        assert registry.layer_hook_plugins[0].name == "dola"
        assert registry.needs_layer_hooks is True

    def test_logits_only_plugins(self):
        cfg = InferencePluginConfig(
            min_p=MinPConfig(enabled=True, threshold=0.05),
            repetition_penalty=RepetitionPenaltyConfig(enabled=True, penalty=1.2),
        )
        registry = PluginRegistry(cfg)
        assert "min_p" in registry.active_plugin_names
        assert "repetition_penalty" in registry.active_plugin_names
        assert len(registry.logits_plugins) == 2
        assert registry.needs_layer_hooks is False

    def test_mixed_plugins(self, factual_config):
        factual_config.repetition_penalty = RepetitionPenaltyConfig(enabled=True)
        registry = PluginRegistry(factual_config)
        assert "dola" in registry.active_plugin_names
        assert "repetition_penalty" in registry.active_plugin_names
        assert len(registry.layer_hook_plugins) == 1
        assert len(registry.logits_plugins) == 1
        assert registry.needs_layer_hooks is True

    def test_active_plugin_names_are_sorted(self):
        cfg = InferencePluginConfig(
            min_p=MinPConfig(enabled=True, threshold=0.05),
            dola=DoLaConfig(enabled=True),
            repetition_penalty=RepetitionPenaltyConfig(enabled=True),
        )
        registry = PluginRegistry(cfg)
        names = registry.active_plugin_names
        assert names == sorted(names)

    def test_from_profile_factual(self):
        cfg = InferencePluginConfig.from_profile("factual")
        registry = PluginRegistry(cfg)
        assert "dola" in registry.active_plugin_names
        assert registry.needs_layer_hooks is True

    def test_from_profile_creative(self):
        cfg = InferencePluginConfig.from_profile("creative")
        registry = PluginRegistry(cfg)
        assert "min_p" in registry.active_plugin_names
        assert registry.needs_layer_hooks is False
