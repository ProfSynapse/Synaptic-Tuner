"""Contracts preventing revival of the legacy Modal training backend."""

from pathlib import Path

import pytest
import yaml

from tuner.backends.registry import TrainingBackendRegistry
from tuner.handlers.cloud_train_handler import PROVIDER_INFO


ROOT = Path(__file__).resolve().parents[2]


def test_legacy_modal_training_files_are_absent():
    assert not (ROOT / "tuner/backends/training/cloud/modal_backend.py").exists()
    assert not (ROOT / "Trainers/cloud/train_modal.py").exists()


def test_legacy_cloud_surfaces_do_not_register_or_export_modal():
    import tuner.backends.training as training
    import tuner.backends.training.cloud as cloud

    assert TrainingBackendRegistry.list() == ["rtx", "mac", "hf_jobs", "runpod"]
    assert set(PROVIDER_INFO) == {"hf_jobs", "runpod"}
    assert "modal" not in cloud.AVAILABLE_BACKENDS
    assert "ModalBackend" not in training.__all__
    assert "ModalBackend" not in cloud.__all__
    with pytest.raises(AttributeError):
        getattr(training, "ModalBackend")
    with pytest.raises(AttributeError):
        getattr(cloud, "ModalBackend")


def test_legacy_cloud_config_has_no_modal_provider_data():
    config = yaml.safe_load(
        (ROOT / "Trainers/cloud/cloud_config.yaml").read_text(encoding="utf-8")
    )

    assert "modal" not in config["pricing"]
    assert "modal" not in config["cloud"]
    assert all("modal" not in tier for tier in config["gpu_tiers"].values())


def test_modal_is_not_a_legacy_training_backend_choice():
    with pytest.raises(ValueError, match="Unknown training backend: 'modal'"):
        TrainingBackendRegistry.get("modal", repo_root=ROOT)
