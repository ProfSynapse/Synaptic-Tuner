"""Tests for shared/inference/loader.py — top-level config loader."""

from __future__ import annotations

from shared.inference.config import InferencePluginConfig
from shared.inference.loader import load_inference_config


class TestLoadInferenceConfig:
    def test_default_fallback(self):
        """With no args, loads configs/inference/default.yaml."""
        cfg = load_inference_config()
        assert isinstance(cfg, InferencePluginConfig)
        assert cfg.dola.enabled is False

    def test_direct_path(self):
        cfg = load_inference_config(config_path="configs/inference/default.yaml")
        assert isinstance(cfg, InferencePluginConfig)

    def test_named_profile(self):
        cfg = load_inference_config(profile="factual")
        assert cfg.dola.enabled is True

    def test_missing_path_returns_defaults(self):
        cfg = load_inference_config(config_path="nonexistent/path.yaml")
        assert isinstance(cfg, InferencePluginConfig)
        assert cfg.dola.enabled is False

    def test_missing_profile_returns_defaults(self):
        cfg = load_inference_config(profile="nonexistent_profile")
        assert isinstance(cfg, InferencePluginConfig)
        assert cfg.dola.enabled is False

    def test_missing_base_dir_returns_defaults(self):
        cfg = load_inference_config(base_dir="nonexistent/dir")
        assert isinstance(cfg, InferencePluginConfig)
        assert cfg.dola.enabled is False

    def test_config_path_takes_precedence_over_profile(self):
        """config_path should be used even if profile is also given."""
        cfg = load_inference_config(
            config_path="configs/inference/default.yaml",
            profile="factual",
        )
        # default.yaml has dola disabled
        assert cfg.dola.enabled is False
