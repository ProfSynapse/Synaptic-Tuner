"""Tests for CLI handlers: hardware_plan, surgery, flywheel, cloud_train, cloud_run.

Each handler gets a happy-path test (json_mode where possible to avoid interactive prompts)
and at least one error-path test.
"""

from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass

import pytest


# ---------------------------------------------------------------------------
# HardwarePlanHandler
# ---------------------------------------------------------------------------


class TestHardwarePlanHandler:
    def _make_handler(self, args):
        from tuner.handlers.hardware_plan_handler import HardwarePlanHandler
        return HardwarePlanHandler(args=args)

    def test_missing_spec_returns_error(self):
        args = Namespace(json=True, experiment_spec=None)
        handler = self._make_handler(args)
        code = handler.handle()
        assert code == 1

    def test_nonexistent_spec_file_returns_error(self, tmp_path):
        args = Namespace(json=True, experiment_spec=str(tmp_path / "nope.yaml"))
        handler = self._make_handler(args)
        code = handler.handle()
        assert code == 1

    def test_happy_path_json_mode(self, tmp_path):
        """With a valid spec and mocked planner, handle() returns 0 in JSON mode."""
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text("name: test\n", encoding="utf-8")

        mock_spec = MagicMock()
        mock_spec.name = "test-experiment"

        @dataclass
        class FakeRecommendation:
            flavor: str = "a10g-small"
            pretty_name: str = "A10G"
            price_hr: float = 1.10
            estimated_memory_gb: float = 20.0
            estimated_headroom_gb: float = 4.0
            estimated_hours: float = 2.0
            estimated_cost: float = 2.20
            recommended_batch_size: int = 4
            recommended_gradient_accumulation: int = 2

        @dataclass
        class FakePlan:
            recommendation: object = None

        fake_plan = FakePlan(recommendation=FakeRecommendation())
        fake_plans = {"training": fake_plan}

        args = Namespace(
            json=True,
            experiment_spec=str(spec_path),
            optimize_for="balanced",
            max_hourly_price=None,
        )
        handler = self._make_handler(args)

        with patch(
            "tuner.handlers.hardware_plan_handler.load_experiment_spec",
            return_value=mock_spec,
        ):
            with patch(
                "tuner.handlers.hardware_plan_handler.plan_experiment_hardware",
                return_value=fake_plans,
            ):
                with patch(
                    "tuner.handlers.hardware_plan_handler.format_stage_plan_json",
                    return_value={"flavor": "a10g-small"},
                ):
                    code = handler.handle()

        assert code == 0


# ---------------------------------------------------------------------------
# SurgeryHandler
# ---------------------------------------------------------------------------


class TestSurgeryHandler:
    def _make_handler(self, args):
        from tuner.handlers.surgery_handler import SurgeryHandler
        return SurgeryHandler(args=args)

    def test_missing_deps_returns_error(self):
        """When LoRA surgery deps are not installed, handle() returns 1."""
        args = Namespace(json=True, surgery_config=None, subcommand=None)
        handler = self._make_handler(args)

        with patch.dict("sys.modules", {"shared.evolutionary.lora_surgery": None}):
            with patch(
                "builtins.__import__",
                side_effect=ImportError("No module named 'safetensors'"),
            ):
                code = handler.handle()

        assert code == 1

    def test_missing_adapter_config_returns_error(self, tmp_path):
        """When adapter_config.json is missing, returns INVALID_ADAPTER error."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        # No adapter_config.json created

        args = Namespace(
            json=True,
            surgery_config=None,
            subcommand=str(adapter_dir),
            eval_scenario=None,
        )
        handler = self._make_handler(args)

        mock_config = MagicMock()
        mock_config.adapter_path = str(adapter_dir)
        mock_config.eval_scenario = ""
        mock_config.eval_backend = "local"
        mock_config.operations = ["scale"]
        mock_config.min_improvement = 0.01
        mock_config.output_dir = str(tmp_path / "out")
        mock_config.local_min_vram_gb = 8

        # LoRASurgeon and SurgeryConfig are imported lazily inside handle(),
        # so we mock them via the import mechanism
        mock_module = MagicMock()
        mock_module.SurgeryConfig.return_value = mock_config
        import sys
        with patch.dict(sys.modules, {"shared.evolutionary.lora_surgery": mock_module}):
            code = handler.handle()

        assert code == 1


# ---------------------------------------------------------------------------
# FlywheelHandler
# ---------------------------------------------------------------------------


class TestFlywheelHandler:
    def _make_handler(self, args):
        from tuner.handlers.flywheel_handler import FlywheelHandler
        return FlywheelHandler(args=args)

    def test_json_mode_no_subcommand_lists_subcommands(self, capsys):
        args = Namespace(json=True, subcommand=None)
        handler = self._make_handler(args)
        code = handler.handle()
        assert code == 0
        output = capsys.readouterr().out
        assert "subcommands" in output

    def test_unknown_subcommand_returns_error(self, capsys):
        args = Namespace(json=True, subcommand="bogus")
        handler = self._make_handler(args)
        code = handler.handle()
        assert code == 1
        output = capsys.readouterr().out
        assert "UNKNOWN_SUBCOMMAND" in output

    def test_configure_subcommand_json_mode(self, capsys):
        args = Namespace(json=True, subcommand="configure", flywheel_config=None)
        handler = self._make_handler(args)

        # FlywheelHandler._handle_configure calls asdict() which requires a real dataclass
        from shared.flywheel.config import FlywheelConfig
        real_config = FlywheelConfig()
        with patch.object(handler, "_load_config", return_value=real_config):
            code = handler.handle()

        assert code == 0

    def test_configure_load_failure(self, capsys):
        args = Namespace(json=True, subcommand="configure", flywheel_config=None)
        handler = self._make_handler(args)

        with patch.object(handler, "_load_config", side_effect=FileNotFoundError("missing")):
            code = handler.handle()

        assert code == 1

    def test_versions_empty_dir(self, tmp_path, capsys):
        args = Namespace(json=True, subcommand="versions", flywheel_config=None)
        handler = self._make_handler(args)

        mock_config = MagicMock()
        mock_config.datasets_dir = "flywheel_datasets"

        with patch.object(handler, "_load_config", return_value=mock_config):
            with patch.object(type(handler), "repo_root", new_callable=PropertyMock, return_value=tmp_path):
                code = handler.handle()

        assert code == 0

    def test_versions_with_data(self, tmp_path, capsys):
        args = Namespace(json=True, subcommand="versions", flywheel_config=None)
        handler = self._make_handler(args)

        mock_config = MagicMock()
        mock_config.datasets_dir = "flywheel_datasets"

        datasets_dir = tmp_path / "flywheel_datasets"
        v1 = datasets_dir / "v001"
        v1.mkdir(parents=True)
        (v1 / "train.jsonl").write_text('{"a":1}\n{"a":2}\n', encoding="utf-8")

        with patch.object(handler, "_load_config", return_value=mock_config):
            with patch.object(type(handler), "repo_root", new_callable=PropertyMock, return_value=tmp_path):
                code = handler.handle()

        assert code == 0
        output = capsys.readouterr().out
        assert "v001" in output


# ---------------------------------------------------------------------------
# CloudTrainHandler
# ---------------------------------------------------------------------------


class TestCloudTrainHandler:
    def _make_handler(self, args):
        from tuner.handlers.cloud_train_handler import CloudTrainHandler
        return CloudTrainHandler(args=args)

    def test_json_mode_returns_status(self, capsys):
        args = Namespace(json=True)
        handler = self._make_handler(args)

        with patch.object(handler, "_get_provider_status", return_value=[]):
            code = handler.handle()

        assert code == 0
        output = capsys.readouterr().out
        assert "command" in output
        assert "cloud" in output

    def test_apply_training_overrides_no_args(self):
        args = Namespace(json=True)
        handler = self._make_handler(args)
        mock_config = MagicMock()
        # Should not raise with no override attributes
        result = handler._apply_training_overrides(mock_config)
        assert result is mock_config

    def test_apply_training_overrides_sets_values(self):
        args = Namespace(
            json=True,
            train_model_name="new-model",
            train_dataset_name=None,
            train_dataset_file=None,
            train_batch_size=8,
            train_gradient_accumulation=None,
            train_learning_rate=None,
            train_num_epochs=None,
            train_max_steps=None,
            train_max_seq_length=None,
            train_load_in_4bit=None,
            train_lora_target_modules=None,
            train_gpu=None,
            train_timeout_hours=None,
            train_cloud_image=None,
            train_image_profile=None,
        )
        handler = self._make_handler(args)
        mock_config = MagicMock()
        mock_config.dataset_name = None
        mock_config.dataset_file = None
        handler._apply_training_overrides(mock_config)
        assert mock_config.model_name == "new-model"
        assert mock_config.batch_size == 8

    def test_apply_training_overrides_sets_seed(self):
        # --train-seed wins over the recipe-loaded config.seed, including seed=0.
        args = Namespace(json=True, train_seed=0)
        handler = self._make_handler(args)
        mock_config = MagicMock()
        mock_config.method = "sft"
        handler._apply_training_overrides(mock_config)
        assert mock_config.seed == 0

    def test_apply_training_overrides_sets_beta_for_dpo(self):
        args = Namespace(json=True, train_beta=0.5)
        handler = self._make_handler(args)
        mock_config = MagicMock()
        mock_config.method = "dpo"
        handler._apply_training_overrides(mock_config)
        assert mock_config.beta == 0.5

    def test_apply_training_overrides_skips_beta_for_sft(self):
        # beta is DPO/KTO-only; a --train-beta override must not touch an SFT config.
        args = Namespace(json=True, train_beta=0.5)
        handler = self._make_handler(args)
        mock_config = MagicMock()
        mock_config.method = "sft"
        handler._apply_training_overrides(mock_config)
        assert "beta" not in mock_config.__dict__

    def test_method_labels(self):
        args = Namespace(json=True)
        handler = self._make_handler(args)
        labels = handler._load_method_labels()
        assert "sft" in labels
        assert "kto" in labels


# ---------------------------------------------------------------------------
# CloudRunHandler
# ---------------------------------------------------------------------------


class TestCloudRunHandler:
    def _make_handler(self, args):
        from tuner.handlers.cloud_run_handler import CloudRunHandler
        return CloudRunHandler(args=args)

    def test_missing_job_config_json_mode(self, capsys):
        args = Namespace(json=True, job_config=None)
        handler = self._make_handler(args)

        with patch.object(handler, "_resolve_job_config_path", side_effect=Exception("JSON mode requires --job-config")):
            code = handler.handle()

        assert code == 1

    def test_safe_template_dict_raises_on_missing_key(self):
        from tuner.handlers.cloud_run_handler import _SafeTemplateDict
        d = _SafeTemplateDict({"a": "1"})
        assert d["a"] == "1"
        from tuner.core.exceptions import CloudProviderError
        with pytest.raises(CloudProviderError, match="Missing template variable"):
            _ = d["nonexistent"]

    def test_list_job_configs_empty(self, tmp_path):
        args = Namespace(json=True, job_config=None)
        handler = self._make_handler(args)
        with patch.object(handler, "_jobs_dir", return_value=tmp_path / "nonexistent"):
            assert handler._list_job_configs() == []

    def test_list_job_configs_finds_yaml(self, tmp_path):
        jobs_dir = tmp_path / "jobs"
        jobs_dir.mkdir()
        (jobs_dir / "eval.yaml").write_text("name: eval\n", encoding="utf-8")
        (jobs_dir / "train.yaml").write_text("name: train\n", encoding="utf-8")
        (jobs_dir / "readme.md").write_text("not yaml\n", encoding="utf-8")

        args = Namespace(json=True, job_config=None)
        handler = self._make_handler(args)
        with patch.object(handler, "_jobs_dir", return_value=jobs_dir):
            configs = handler._list_job_configs()

        assert len(configs) == 2
        assert all(p.suffix == ".yaml" for p in configs)

    def test_load_job_config_invalid_yaml(self, tmp_path):
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("- just a list\n", encoding="utf-8")
        args = Namespace(json=True)
        handler = self._make_handler(args)
        from tuner.core.exceptions import CloudProviderError
        with pytest.raises(CloudProviderError, match="must be a YAML object"):
            handler._load_job_config(bad_yaml)

    def test_render_value_nested(self):
        args = Namespace(json=True)
        handler = self._make_handler(args)
        variables = {"name": "test", "version": "1.0"}
        result = handler._render_value(
            {"key": "{name}-{version}", "nested": ["{name}"]},
            variables,
        )
        assert result == {"key": "test-1.0", "nested": ["test"]}

    def test_render_value_passthrough_non_string(self):
        args = Namespace(json=True)
        handler = self._make_handler(args)
        assert handler._render_value(42, {}) == 42
        assert handler._render_value(None, {}) is None
