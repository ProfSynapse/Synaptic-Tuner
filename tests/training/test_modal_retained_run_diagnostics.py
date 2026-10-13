"""A failed local download exposes its exact retained run without retry authority."""
from argparse import Namespace
import json

import pytest

from synaptic_tuner.api.v1.results import TrainingRunRef
from tuner.handlers.modal_job_config_handler import ModalJobConfigHandler
from tuner.training import modal_standalone_runner as runner
from tuner.training.modal_run_retention import load_retained_modal_run
from tests.training.test_modal_standalone_runner import _setup


@pytest.mark.parametrize("typed_failure", [False, True])
def test_download_failure_projects_retained_run_without_private_material(
    tmp_path, monkeypatch, capsys, typed_failure
):
    plan, context, events = _setup(tmp_path, monkeypatch)

    def fail(*args, **kwargs):
        if typed_failure:
            raise runner.ModalStandalonePhaseUnavailable("RUN_ARTIFACT_DOWNLOAD")
        raise OSError("private-key /private/customer/file")

    monkeypatch.setattr(runner, "download_verified_modal_artifact", fail)
    handler = ModalJobConfigHandler(Namespace(
        modal_profile="fixture", modal_environment="main", json=True,
    ), context)
    assert handler._execute(plan) == 2
    output = capsys.readouterr()
    details = json.loads(output.out)["error"]["details"]
    run = TrainingRunRef(**details["run_ref"])
    retained = load_retained_modal_run(
        tmp_path / "state" / "synaptic-training" / run.run_id, expected_run=run,
    )
    assert details == {
        "phase": "RUN_ARTIFACT_DOWNLOAD", "failure_class": "UNAVAILABLE",
        "location": "modal_standalone_runner.artifact_download",
        "retry_authorized": False, "run_ref": run.to_dict(), "retention_ready": True,
    }
    for private in (*retained.signing_keys.values(), "private-key", "/private/customer"):
        assert private not in output.out + output.err
    assert events == ["bootstrap", "cpu", "stage", "spawn"]


def test_failed_retention_does_not_advertise_readback(tmp_path, monkeypatch, capsys):
    plan, context, events = _setup(tmp_path, monkeypatch)

    def fail(**kwargs):
        raise OSError("private-key")

    monkeypatch.setattr(runner, "retain_modal_run", fail)
    handler = ModalJobConfigHandler(Namespace(
        modal_profile="fixture", modal_environment="main", json=True,
    ), context)
    assert handler._execute(plan) == 2
    details = json.loads(capsys.readouterr().out)["error"]["details"]
    assert details["phase"] == "RUN_RETAIN_READ_INPUTS"
    assert "run_ref" not in details and "retention_ready" not in details
    assert details["retry_authorized"] is False
    assert events == ["bootstrap", "cpu"]


def test_failure_requires_an_exact_typed_retained_run():
    with pytest.raises(TypeError):
        runner.ModalStandalonePhaseUnavailable(
            "RUN_ARTIFACT_DOWNLOAD", retained_run={"run_id": "untrusted"},
        )
