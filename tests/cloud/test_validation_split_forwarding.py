"""Validation split settings across the cloud training lanes.

The HF Jobs command builder and RunPod startup command are covered in
test_hf_jobs_backend.py / test_runpod_backend.py; this module covers the shared
flag builder, the cloud / cloud-pipeline CLI overrides, and the run-experiment
HF training stage (experiment spec dataset section).
"""

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from shared.experiment_tracking.experiment_spec import ExperimentSpec
from tuner.cli.parser import create_parser
from tuner.core.config import CloudTrainingConfig, validation_split_flags
from tuner.core.exceptions import ConfigurationError
from tuner.handlers.cloud_train_handler import CloudTrainHandler
from tuner.handlers.stages import hf_training_stage
from tuner.handlers.stages.hf_training_stage import HFTrainingStageRunner


def _config(method="sft", **overrides):
    config = CloudTrainingConfig(
        method=method,
        platform="hf_jobs",
        config_path=Path("/fake"),
        trainer_dir=Path("/fake"),
        model_name="test",
        dataset_file="test",
        epochs=1,
        batch_size=4,
        learning_rate=2e-4,
        provider="hf_jobs",
        artifact_identifier="bucket",
        repo_commit="abc12345def67890",
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


# -- shared flag builder --------------------------------------------------------

def test_flags_for_grouped_split():
    assert validation_split_flags(
        method="dpo", split_dataset=True, test_size=0.2, validation_group_key=" provenance.prompt_key "
    ) == ["--split-dataset", "--test-size", "0.2", "--validation-group-key", "provenance.prompt_key"]


def test_no_flags_without_a_split_request():
    # Trainer configs always carry test_size; on its own it is not a request.
    assert validation_split_flags(method="sft", split_dataset=None, test_size=0.1, validation_group_key=None) == []
    assert validation_split_flags(method="grpo", split_dataset=False, test_size=0.1, validation_group_key=None) == []


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"method": "sft", "split_dataset": None, "test_size": None, "validation_group_key": "k"}, "requires dataset.split_dataset"),
        ({"method": "sft", "split_dataset": False, "test_size": 0.1, "validation_group_key": "k"}, "requires dataset.split_dataset"),
        ({"method": "sft", "split_dataset": True, "test_size": 1.5, "validation_group_key": None}, "test_size"),
        ({"method": "kto", "split_dataset": True, "test_size": 0, "validation_group_key": None}, "test_size"),
        ({"method": "grpo", "split_dataset": True, "test_size": None, "validation_group_key": None}, "not supported"),
        ({"method": "sft", "split_dataset": True, "test_size": None, "validation_group_key": "  "}, "non-empty"),
    ],
)
def test_invalid_combinations_are_refused(kwargs, message):
    with pytest.raises(ConfigurationError, match=message):
        validation_split_flags(**kwargs)


# -- cloud / cloud-pipeline CLI overrides ------------------------------------------

def _apply(argv, config):
    args = create_parser().parse_args(["cloud-pipeline", *argv])
    args.json = True
    return CloudTrainHandler(args=args)._apply_training_overrides(config)


def test_cli_overrides_set_split_settings():
    config = _apply(
        ["--train-split-dataset", "--train-test-size", "0.15", "--train-validation-group-key", "scenario_id"],
        _config(method="kto"),
    )
    assert (config.split_dataset, config.test_size, config.validation_group_key) == (True, 0.15, "scenario_id")


def test_cli_can_disable_a_configured_split():
    config = _apply(["--train-no-split-dataset"], _config(split_dataset=True))
    assert config.split_dataset is False


def test_cli_defaults_leave_trainer_config_values():
    config = _apply([], _config(split_dataset=True, test_size=0.3, validation_group_key="metadata.scenario"))
    assert (config.split_dataset, config.test_size, config.validation_group_key) == (True, 0.3, "metadata.scenario")


def test_cli_group_key_without_split_fails_before_submission():
    with pytest.raises(ConfigurationError, match="requires dataset.split_dataset"):
        _apply(["--train-validation-group-key", "scenario_id"], _config())


# -- run-experiment HF training stage -------------------------------------------------

def _spec(dataset_extra):
    return ExperimentSpec.from_dict(
        {
            "experiment": {
                "name": "unit",
                "provider": "hf_jobs",
                "method": "sft",
                "dataset": {"source": "org/data", "file": "train.jsonl", **dataset_extra},
                "training": {"model_name": "org/model"},
            }
        }
    )


class _Submitted(Exception):
    pass


def _run_stage(monkeypatch, tmp_path, spec, loaded_config):
    captured = {}

    class FakeBackend:
        context = None

        def load_config(self, method):
            return loaded_config

        def prepare_source(self, config, run_id):
            captured["config"] = config
            raise _Submitted

    monkeypatch.setattr(hf_training_stage.TrainingBackendRegistry, "get", lambda *a, **k: FakeBackend())
    tracking = SimpleNamespace(
        project_context=None,
        base_dir=tmp_path,
        tracking_uri=lambda path: f"tracking://{Path(path).name}",
    )
    runner = HFTrainingStageRunner(repo_root=tmp_path, tracking_service=tracking)
    monkeypatch.setattr(runner, "_recover_existing_training", lambda **_: None)
    experiment = SimpleNamespace(experiment_id="exp-1", source_transport_state=None)
    with pytest.raises(_Submitted):
        runner.run(spec, experiment)
    return captured["config"]


def test_experiment_spec_dataset_accepts_split_settings():
    spec = _spec({"split_dataset": True, "test_size": 0.2, "validation_group_key": "metadata.scenario"})
    assert (spec.dataset.split_dataset, spec.dataset.test_size, spec.dataset.validation_group_key) == (
        True, 0.2, "metadata.scenario",
    )
    with pytest.raises(ValueError, match="Invalid experiment.dataset"):
        _spec({"validation_group": "typo"})


def test_hf_training_stage_forwards_spec_split_settings(monkeypatch, tmp_path):
    spec = _spec({"split_dataset": True, "test_size": 0.2, "validation_group_key": "metadata.scenario"})
    config = _run_stage(monkeypatch, tmp_path, spec, _config())
    assert (config.split_dataset, config.test_size, config.validation_group_key) == (True, 0.2, "metadata.scenario")


def test_hf_training_stage_keeps_trainer_config_when_spec_is_silent(monkeypatch, tmp_path):
    config = _run_stage(monkeypatch, tmp_path, _spec({}), _config(split_dataset=True, test_size=0.3))
    assert (config.split_dataset, config.test_size, config.validation_group_key) == (True, 0.3, None)


def test_hf_training_stage_refuses_group_key_without_split(monkeypatch, tmp_path):
    spec = _spec({"validation_group_key": "metadata.scenario"})
    with pytest.raises(ConfigurationError, match="requires dataset.split_dataset"):
        _run_stage(monkeypatch, tmp_path, spec, _config())
