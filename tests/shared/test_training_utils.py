"""
Tests for shared/training_utils.py

Covers setup_wandb(), extract_previous_log_entries(), save_training_lineage(),
build_base_lineage(), and apply_tier_preset().
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from shared.training_utils import (
    apply_tier_preset,
    build_base_lineage,
    extract_previous_log_entries,
    save_training_lineage,
    setup_wandb,
)


# ---------------------------------------------------------------------------
# setup_wandb
# ---------------------------------------------------------------------------

class TestSetupWandb:
    def test_returns_false_without_api_key(self):
        """Should return False when WANDB_API_KEY not set."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("WANDB_API_KEY", None)
            assert setup_wandb() is False

    def test_returns_false_if_wandb_not_installed(self):
        """Should return False if wandb is not importable."""
        with patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}):
            with patch.dict("sys.modules", {"wandb": None}):
                assert setup_wandb() is False

    def test_returns_true_on_success(self):
        """Should return True when wandb login succeeds."""
        mock_wandb = MagicMock()
        with patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}):
            with patch.dict("sys.modules", {"wandb": mock_wandb}):
                result = setup_wandb()
                assert result is True
                mock_wandb.login.assert_called_once_with(
                    key="test-key", relogin=True, force=True
                )

    def test_returns_false_on_login_exception(self):
        """Should return False if wandb.login raises an exception."""
        mock_wandb = MagicMock()
        mock_wandb.login.side_effect = Exception("network error")
        with patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}):
            with patch.dict("sys.modules", {"wandb": mock_wandb}):
                assert setup_wandb() is False


# ---------------------------------------------------------------------------
# extract_previous_log_entries
# ---------------------------------------------------------------------------

class TestExtractPreviousLogEntries:
    def _create_checkpoint_structure(self, base_dir: Path, step: int, log_entries: list) -> str:
        """Create a realistic checkpoint directory structure with log file."""
        run_dir = base_dir / "output" / "20251114_135227"
        ckpt_dir = run_dir / "checkpoints" / f"checkpoint-{step}"
        logs_dir = run_dir / "logs"

        ckpt_dir.mkdir(parents=True)
        logs_dir.mkdir(parents=True)

        # Write log file
        log_file = logs_dir / "training_20251114_135227.jsonl"
        with open(log_file, "w") as f:
            for entry in log_entries:
                f.write(json.dumps(entry) + "\n")

        return str(ckpt_dir)

    def test_extracts_entries_up_to_step(self, tmp_path):
        """Should return entries up to and including the resume step."""
        entries = [
            {"step": 10, "loss": 2.5},
            {"step": 20, "loss": 2.0},
            {"step": 30, "loss": 1.8},
            {"step": 40, "loss": 1.5},
            {"step": 50, "loss": 1.3},
        ]
        ckpt_path = self._create_checkpoint_structure(tmp_path, 30, entries)
        result = extract_previous_log_entries(ckpt_path)
        assert len(result) == 3
        assert result[-1]["step"] == 30

    def test_returns_empty_for_bad_checkpoint_name(self, tmp_path):
        """Should return [] if checkpoint path has no step number."""
        ckpt_dir = tmp_path / "output" / "run1" / "checkpoints" / "bad-name"
        ckpt_dir.mkdir(parents=True)
        result = extract_previous_log_entries(str(ckpt_dir))
        assert result == []

    def test_returns_empty_when_no_logs_dir(self, tmp_path):
        """Should return [] if logs directory does not exist."""
        ckpt_dir = tmp_path / "output" / "run1" / "checkpoints" / "checkpoint-50"
        ckpt_dir.mkdir(parents=True)
        result = extract_previous_log_entries(str(ckpt_dir))
        assert result == []

    def test_returns_empty_when_no_log_files(self, tmp_path):
        """Should return [] if logs dir exists but is empty."""
        run_dir = tmp_path / "output" / "20251114_135227"
        ckpt_dir = run_dir / "checkpoints" / "checkpoint-50"
        logs_dir = run_dir / "logs"
        ckpt_dir.mkdir(parents=True)
        logs_dir.mkdir(parents=True)
        result = extract_previous_log_entries(str(ckpt_dir))
        assert result == []

    def test_uses_most_recent_log_file(self, tmp_path):
        """Should use the lexicographically last log file."""
        run_dir = tmp_path / "output" / "20251114_135227"
        ckpt_dir = run_dir / "checkpoints" / "checkpoint-10"
        logs_dir = run_dir / "logs"
        ckpt_dir.mkdir(parents=True)
        logs_dir.mkdir(parents=True)

        # Write two log files
        (logs_dir / "training_20251114_100000.jsonl").write_text(
            json.dumps({"step": 10, "loss": 9.9}) + "\n"
        )
        (logs_dir / "training_20251114_120000.jsonl").write_text(
            json.dumps({"step": 10, "loss": 1.1}) + "\n"
        )

        result = extract_previous_log_entries(str(ckpt_dir))
        assert len(result) == 1
        assert result[0]["loss"] == 1.1  # From the later log file

    def test_returns_all_entries_when_resume_step_beyond_log(self, tmp_path):
        """Should return all entries if resume step exceeds log."""
        entries = [
            {"step": 10, "loss": 2.0},
            {"step": 20, "loss": 1.5},
        ]
        ckpt_path = self._create_checkpoint_structure(tmp_path, 100, entries)
        result = extract_previous_log_entries(ckpt_path)
        assert len(result) == 2

    def test_handles_entries_without_step_key(self, tmp_path):
        """Entries without 'step' should default to step 0."""
        entries = [
            {"loss": 2.5},  # No step key, defaults to 0
            {"step": 20, "loss": 1.5},
        ]
        ckpt_path = self._create_checkpoint_structure(tmp_path, 10, entries)
        result = extract_previous_log_entries(ckpt_path)
        # Entry with step 0 is <= 10, entry with step 20 is > 10
        assert len(result) == 1


# ---------------------------------------------------------------------------
# save_training_lineage
# ---------------------------------------------------------------------------

class TestSaveTrainingLineage:
    def test_saves_lineage_json(self, tmp_path):
        """Should save lineage dict to training_lineage.json."""
        lineage = {
            "training_type": "SFT",
            "model": {"base_model": "test/model"},
        }
        with patch(
            "shared.training_capacity.build_capacity_feature_row", return_value=None
        ):
            result = save_training_lineage(lineage, tmp_path)

        assert result == tmp_path / "training_lineage.json"
        assert result.exists()

        with open(result) as f:
            saved = json.load(f)
        assert saved["training_type"] == "SFT"

    def test_saves_capacity_features_when_available(self, tmp_path):
        """Should save capacity_features.json when feature row is non-empty."""
        lineage = {"training_type": "KTO"}
        mock_features = {"gpu_count": 1, "vram_gb": 24}
        with patch(
            "shared.training_capacity.build_capacity_feature_row",
            return_value=mock_features,
        ):
            save_training_lineage(lineage, tmp_path)

        features_path = tmp_path / "capacity_features.json"
        assert features_path.exists()
        with open(features_path) as f:
            saved = json.load(f)
        assert saved["gpu_count"] == 1

    def test_skips_capacity_features_when_none(self, tmp_path):
        """Should not save capacity_features.json when build returns None."""
        lineage = {"training_type": "GRPO"}
        with patch(
            "shared.training_capacity.build_capacity_feature_row", return_value=None
        ):
            save_training_lineage(lineage, tmp_path)

        features_path = tmp_path / "capacity_features.json"
        assert not features_path.exists()


# ---------------------------------------------------------------------------
# build_base_lineage
# ---------------------------------------------------------------------------

class TestBuildBaseLineage:
    @dataclass
    class FakeTrainerState:
        global_step: int = 100
        epoch: float = 3.0
        log_history: list = None

        def __post_init__(self):
            if self.log_history is None:
                self.log_history = [
                    {"step": 50, "loss": 2.0},
                    {"step": 100, "loss": 1.2},
                ]

    def _mock_lineage_deps(self):
        """Create patch context for build_base_lineage's lazy imports.

        build_base_lineage imports torch and shared.training_capacity inside
        the function body. We mock at the source module level.
        """
        mock_torch = MagicMock()
        mock_hw = {"gpu_count": 1, "gpu_name": "RTX 3090"}
        mock_capacity = {"total_steps": 100}

        return (
            mock_torch,
            mock_hw,
            mock_capacity,
            patch("shared.training_capacity.capture_hardware_info", return_value=mock_hw),
            patch("shared.training_capacity.summarize_capacity_from_logs", return_value=mock_capacity),
        )

    def test_returns_complete_lineage_structure(self, tmp_path):
        """Should return dict with all expected top-level keys."""
        mock_torch, mock_hw, mock_capacity, hw_patch, cap_patch = self._mock_lineage_deps()
        trainer = MagicMock()
        trainer.state = self.FakeTrainerState()

        with hw_patch, cap_patch:
            result = build_base_lineage(
                training_type="SFT",
                model_info={"base_model": "test/model", "max_seq_length": 2048, "load_in_4bit": True, "dtype": "float16"},
                lora_info={"rank": 16, "alpha": 32, "dropout": 0.05, "target_modules": ["q_proj"], "bias": "none"},
                training_info={"batch_size": 4, "gradient_accumulation_steps": 2, "effective_batch_size": 8,
                               "learning_rate": 2e-4, "num_epochs": 3, "max_steps": -1,
                               "warmup_ratio": 0.05, "lr_scheduler": "cosine", "optimizer": "adamw",
                               "max_grad_norm": 1.0, "gradient_checkpointing": True, "fp16": False, "bf16": True, "seed": 42},
                dataset_info={"source": "test.jsonl", "train_examples": 1000, "eval_examples": 100},
                run_dir=tmp_path,
                trainer=trainer,
                training_time_seconds=3661.5,
            )

        assert result["training_type"] == "SFT"
        assert "timestamp" in result
        assert result["model"]["base_model"] == "test/model"
        assert result["lora"]["rank"] == 16
        assert result["training"]["batch_size"] == 4
        assert result["dataset"]["source"] == "test.jsonl"
        assert result["hardware"] == mock_hw
        assert result["capacity_profile"] == mock_capacity

    def test_extracts_final_loss_from_log_history(self, tmp_path):
        """Should extract final_loss from the last log_history entry with 'loss'."""
        _, _, _, hw_patch, cap_patch = self._mock_lineage_deps()
        trainer = MagicMock()
        trainer.state = self.FakeTrainerState()

        with hw_patch, cap_patch:
            result = build_base_lineage(
                training_type="SFT",
                model_info={}, lora_info={}, training_info={}, dataset_info={},
                run_dir=tmp_path, trainer=trainer,
            )

        assert result["results"]["final_loss"] == 1.2
        assert result["results"]["final_step"] == 100
        assert result["results"]["total_epochs"] == 3.0

    def test_formats_training_time(self, tmp_path):
        """Should format training_time_seconds into human-readable string."""
        _, _, _, hw_patch, cap_patch = self._mock_lineage_deps()
        trainer = MagicMock()
        trainer.state = None

        with hw_patch, cap_patch:
            result = build_base_lineage(
                training_type="KTO",
                model_info={}, lora_info={}, training_info={}, dataset_info={},
                run_dir=tmp_path, trainer=trainer,
                training_time_seconds=7384.0,
            )

        assert result["results"]["training_time_seconds"] == 7384.0
        assert result["results"]["training_time_formatted"] == "2h 3m 4s"

    def test_handles_trainer_without_state(self, tmp_path):
        """Should handle trainer.state being None gracefully."""
        _, _, _, hw_patch, cap_patch = self._mock_lineage_deps()
        trainer = MagicMock()
        trainer.state = None

        with hw_patch, cap_patch:
            result = build_base_lineage(
                training_type="GRPO",
                model_info={}, lora_info={}, training_info={}, dataset_info={},
                run_dir=tmp_path, trainer=trainer,
            )

        assert result["results"] == {}

    def test_handles_no_training_time(self, tmp_path):
        """Should not add training_time fields when None."""
        _, _, _, hw_patch, cap_patch = self._mock_lineage_deps()
        trainer = MagicMock()
        trainer.state = None

        with hw_patch, cap_patch:
            result = build_base_lineage(
                training_type="SFT",
                model_info={}, lora_info={}, training_info={}, dataset_info={},
                run_dir=tmp_path, trainer=trainer,
                training_time_seconds=None,
            )

        assert "training_time_seconds" not in result["results"]
        assert "training_time_formatted" not in result["results"]

    def test_sub_dict_api_copies_input_dicts(self, tmp_path):
        """Input dicts should be copied, not referenced, to prevent mutation."""
        _, _, _, hw_patch, cap_patch = self._mock_lineage_deps()
        trainer = MagicMock()
        trainer.state = None

        model_info = {"base_model": "test"}
        with hw_patch, cap_patch:
            result = build_base_lineage(
                training_type="SFT",
                model_info=model_info, lora_info={}, training_info={}, dataset_info={},
                run_dir=tmp_path, trainer=trainer,
            )

        # Mutating the result should not affect the original
        result["model"]["extra_key"] = "injected"
        assert "extra_key" not in model_info

    def test_handles_empty_log_history(self, tmp_path):
        """Should not set final_loss if log_history is empty."""
        _, _, _, hw_patch, cap_patch = self._mock_lineage_deps()
        trainer = MagicMock()

        @dataclass
        class EmptyState:
            global_step: int = 50
            epoch: float = 1.0
            log_history: list = None

            def __post_init__(self):
                self.log_history = []

        trainer.state = EmptyState()

        with hw_patch, cap_patch:
            result = build_base_lineage(
                training_type="SFT",
                model_info={}, lora_info={}, training_info={}, dataset_info={},
                run_dir=tmp_path, trainer=trainer,
            )

        assert "final_loss" not in result["results"]
        assert result["results"]["final_step"] == 50


# ---------------------------------------------------------------------------
# apply_tier_preset
# ---------------------------------------------------------------------------

class TestApplyTierPreset:
    @dataclass
    class FakeTrainingSection:
        learning_rate: float = 2e-4
        num_epochs: int = 3
        warmup_ratio: float = 0.05

    @dataclass
    class FakeConfig:
        training: Any = None

        def __post_init__(self):
            if self.training is None:
                self.training = TestApplyTierPreset.FakeTrainingSection()

    def _make_tier_yaml(self, tmp_path: Path, tier_name: str, content: dict) -> Path:
        """Create a tier YAML file in the expected directory structure."""
        import yaml

        tiers_dir = tmp_path / "tiers"
        tiers_dir.mkdir(parents=True, exist_ok=True)
        tier_path = tiers_dir / f"{tier_name}.yaml"
        with open(tier_path, "w") as f:
            yaml.dump(content, f)
        return tmp_path

    def test_applies_mapped_values(self, tmp_path):
        """Should set config attributes based on tier_config_map."""
        configs_dir = self._make_tier_yaml(
            tmp_path, "quick",
            {"learning_rate": 5e-4, "num_epochs": 1, "warmup_ratio": 0.1},
        )

        config = self.FakeConfig()
        tier_map = {
            "learning_rate": ("training", "learning_rate"),
            "num_epochs": ("training", "num_epochs"),
            "warmup_ratio": ("training", "warmup_ratio"),
        }
        args = MagicMock()
        args.max_steps = None

        result = apply_tier_preset(config, "quick", tier_map, args, configs_dir)

        assert config.training.learning_rate == 5e-4
        assert config.training.num_epochs == 1
        assert config.training.warmup_ratio == 0.1
        assert result["learning_rate"] == 5e-4

    def test_max_steps_set_via_args(self, tmp_path):
        """max_steps from tier should be set on args, not config."""
        configs_dir = self._make_tier_yaml(
            tmp_path, "quick", {"max_steps": 200}
        )

        config = self.FakeConfig()
        args = MagicMock()
        args.max_steps = None

        apply_tier_preset(config, "quick", {}, args, configs_dir)

        assert args.max_steps == 200

    def test_max_steps_not_overridden_if_set(self, tmp_path):
        """max_steps from tier should NOT override user-provided value."""
        configs_dir = self._make_tier_yaml(
            tmp_path, "quick", {"max_steps": 200}
        )

        config = self.FakeConfig()
        args = MagicMock()
        args.max_steps = 500  # User already set this

        apply_tier_preset(config, "quick", {}, args, configs_dir)

        assert args.max_steps == 500  # Preserved user value

    def test_unknown_keys_ignored(self, tmp_path):
        """Keys not in tier_config_map should be silently skipped."""
        configs_dir = self._make_tier_yaml(
            tmp_path, "quick",
            {"unknown_key": 42, "learning_rate": 1e-3},
        )

        config = self.FakeConfig()
        tier_map = {"learning_rate": ("training", "learning_rate")}
        args = MagicMock()
        args.max_steps = None

        apply_tier_preset(config, "quick", tier_map, args, configs_dir)

        assert config.training.learning_rate == 1e-3

    def test_raises_for_missing_tier_file(self, tmp_path):
        """Should raise FileNotFoundError for non-existent tier."""
        config = self.FakeConfig()
        args = MagicMock()
        args.max_steps = None

        with pytest.raises(FileNotFoundError, match="Tier config not found"):
            apply_tier_preset(config, "nonexistent", {}, args, tmp_path)

    def test_returns_parsed_tier_config(self, tmp_path):
        """Should return the parsed tier config dict for logging."""
        configs_dir = self._make_tier_yaml(
            tmp_path, "standard",
            {"learning_rate": 2e-4, "num_epochs": 3},
        )

        config = self.FakeConfig()
        tier_map = {"learning_rate": ("training", "learning_rate")}
        args = MagicMock()
        args.max_steps = None

        result = apply_tier_preset(config, "standard", tier_map, args, configs_dir)

        assert isinstance(result, dict)
        assert result["learning_rate"] == 2e-4
        assert result["num_epochs"] == 3
