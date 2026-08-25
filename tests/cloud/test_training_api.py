"""Contract tests for the provider-neutral public cloud training API."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from synaptic_tuner.api.v1 import (
    CloudSourceContract,
    CloudTrainingAPI,
    CloudTrainingRequest,
)
from tuner.core.config import CloudTrainingConfig
from tuner.project import ProjectContext


def _config(tmp_path: Path, *, provider: str = "modal") -> CloudTrainingConfig:
    return CloudTrainingConfig(
        method="sft",
        platform=provider,
        config_path=tmp_path / "config.yaml",
        trainer_dir=tmp_path,
        model_name="default/model",
        dataset_file="default.jsonl",
        dataset_name="default/dataset",
        epochs=3,
        batch_size=4,
        learning_rate=1e-4,
        provider=provider,
        gpu_type="L40S",
        timeout_hours=6,
        artifact_backend="modal_volume",
        artifact_identifier="artifacts",
        repo_url="https://github.com/example/repo.git",
        repo_branch="main",
        repo_commit="0123456789abcdef",
    )


def _source() -> CloudSourceContract:
    return CloudSourceContract(
        source_lock=SimpleNamespace(
            engine_source=SimpleNamespace(commit="0123456789abcdef")
        ),
        runtime_layout=object(),
        checkout_policy=object(),
    )


def test_recipe_maps_to_provider_neutral_request():
    recipe_path = (
        Path(__file__).parents[2]
        / "Trainers"
        / "recipes"
        / "modal_smollm2_1p7b_sft_smoke.yaml"
    )
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))

    request = CloudTrainingRequest.from_recipe(recipe)

    assert request.provider == "modal"
    assert request.method == "sft"
    assert request.model_name == "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    assert request.dataset_name == "professorsynapse/claudesidian-synthetic-dataset"
    assert request.dataset_file == "nonthinking_tools_sft_04.22.26.jsonl"
    assert request.training["max_steps"] == 5
    assert request.runtime == {"gpu_type": "L4", "timeout_hours": 1}


def test_modal_prepare_applies_request_without_hf_gate(tmp_path):
    context = ProjectContext.standalone(engine_root=tmp_path)
    backend = MagicMock()
    backend.validate_environment.return_value = (True, "")
    backend.get_available_methods.return_value = ["sft"]
    backend.load_config.return_value = _config(tmp_path)
    request = CloudTrainingRequest(
        provider="modal",
        method="sft",
        model_name="new/model",
        dataset_name="org/data",
        dataset_file="train.jsonl",
        training={
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 2e-4,
            "max_steps": 5,
            "max_seq_length": 512,
        },
        runtime={"gpu_type": "L4", "timeout_hours": 1},
    )

    with patch(
        "synaptic_tuner.api.v1.training.TrainingBackendRegistry.list",
        return_value=["modal"],
    ), patch(
        "synaptic_tuner.api.v1.training.TrainingBackendRegistry.get",
        return_value=backend,
    ), patch(
        "synaptic_tuner.api.v1.training.require_current_hf_source_submission_authorization",
        side_effect=AssertionError("Modal inherited the HF gate"),
    ):
        plan = CloudTrainingAPI(context).prepare(request, source=_source())

    config = plan._config
    assert plan.summary["source_commit"] == "0123456789abcdef"
    assert config.model_name == "new/model"
    assert config.dataset_name == "org/data"
    assert config.dataset_file == "train.jsonl"
    assert config.epochs == 1
    assert config.batch_size == 2
    assert config.max_steps == 5
    assert config.max_seq_length == 512
    assert config.gpu_type == "L4"
    assert config.timeout_hours == 1
    assert config.source_lock is plan.source.source_lock


def test_hf_gate_precedes_backend_construction_and_is_cached(tmp_path):
    context = ProjectContext.standalone(engine_root=tmp_path)
    backend = MagicMock()
    backend.get_available_methods.return_value = ["sft"]
    backend.load_config.return_value = _config(tmp_path, provider="hf_jobs")
    events = []

    with patch(
        "synaptic_tuner.api.v1.training.TrainingBackendRegistry.list",
        return_value=["hf_jobs"],
    ), patch(
        "synaptic_tuner.api.v1.training.require_current_hf_source_submission_authorization",
        side_effect=lambda **_: events.append("authorize"),
    ), patch(
        "synaptic_tuner.api.v1.training.TrainingBackendRegistry.get",
        side_effect=lambda *_args, **_kwargs: events.append("backend") or backend,
    ):
        api = CloudTrainingAPI(context)
        assert api.provider_methods("hf_jobs", validate_environment=False) == ["sft"]
        api.prepare(
            CloudTrainingRequest(provider="hf_jobs", method="sft"),
            source=_source(),
            validate_environment=False,
        )

    assert events == ["authorize", "backend", "backend"]


def test_submit_normalizes_result_and_refuses_duplicate(tmp_path):
    context = ProjectContext.standalone(engine_root=tmp_path)
    backend = MagicMock()
    backend.validate_environment.return_value = (True, "")
    backend.get_available_methods.return_value = ["sft"]
    backend.load_config.return_value = _config(tmp_path)
    backend.execute.return_value = 0

    with patch(
        "synaptic_tuner.api.v1.training.TrainingBackendRegistry.list",
        return_value=["modal"],
    ), patch(
        "synaptic_tuner.api.v1.training.TrainingBackendRegistry.get",
        return_value=backend,
    ):
        api = CloudTrainingAPI(context)
        plan = api.prepare(
            CloudTrainingRequest(provider="modal", method="sft"),
            source=_source(),
        )
        result = api.submit(plan)

    assert result.success
    assert result.provider == "modal"
    assert result.artifact_identifier == "artifacts"
    backend.execute.assert_called_once_with(plan._config, python_path="")
    with pytest.raises(RuntimeError, match="already been submitted"):
        api.submit(plan)


def test_cloud_run_routes_declarative_recipe_through_public_api(capsys):
    from argparse import Namespace

    from tuner.handlers.cloud_run_handler import CloudRunHandler

    recipe_path = (
        Path(__file__).parents[2]
        / "Trainers"
        / "recipes"
        / "modal_smollm2_1p7b_sft_smoke.yaml"
    )
    api = MagicMock()
    api.prepare.return_value = SimpleNamespace(
        summary={
            "provider": "modal",
            "method": "sft",
            "model": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "dataset_file": "nonthinking_tools_sft_04.22.26.jsonl",
            "gpu": "L4",
            "timeout_hours": 1,
            "source_commit": "0123456789abcdef",
            "artifact_identifier": "artifacts",
        }
    )
    handler = CloudRunHandler(
        args=Namespace(
            json=True,
            job_config=str(recipe_path),
            auto_confirm=False,
            gpu=None,
            timeout_hours=None,
        )
    )

    with patch(
        "tuner.handlers.cloud_run_handler.CloudTrainingAPI",
        return_value=api,
    ):
        assert handler.handle() == 0

    request = api.prepare.call_args.args[0]
    assert isinstance(request, CloudTrainingRequest)
    assert request.provider == "modal"
    assert request.training["max_steps"] == 5
    assert '"operation": "training"' in capsys.readouterr().out


def test_unknown_recipe_option_is_rejected(tmp_path):
    config = _config(tmp_path)
    request = CloudTrainingRequest(
        provider="modal",
        method="sft",
        training={"made_up_hyperparameter": 1},
    )
    with pytest.raises(ValueError, match="Unknown training option"):
        CloudTrainingAPI.apply_request(config, request)
