"""Tests for PivotRL config loading and backward compatibility."""

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

CONFIGS_DIR = ROOT / "Trainers" / "grpo" / "configs"


def _load(name: str) -> dict:
    with open(CONFIGS_DIR / name, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class TestPivotConfig:

    def test_base_config_no_pivot(self):
        cfg = _load("config.yaml")
        assert "pivot" not in cfg or not cfg.get("pivot", {}).get("enabled", False)

    def test_pivot_config_enabled(self):
        cfg = _load("pivot_config.yaml")
        assert cfg["pivot"]["enabled"] is True

    def test_has_functional_equivalence_reward(self):
        cfg = _load("pivot_config.yaml")
        reward_names = [r["name"] for r in cfg["rewards"]["items"]]
        assert "functional_equivalence" in reward_names

    def test_pivot_config_defaults(self):
        cfg = _load("pivot_config.yaml")
        p = cfg["pivot"]
        assert p["profiling"]["n_rollouts"] == 8
        assert p["filtering"]["variance_threshold"] == pytest.approx(0.1)
        assert p["filtering"]["min_candidates"] == 50
        assert p["filtering"]["max_candidates"] is None
        assert p["filtering"]["mean_reward_range"] is None
        assert p["cache"]["enabled"] is True
