"""CLI reporting for a verified run whose configured evaluation gate fails."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from synaptic_tuner.api.v1.results import TrainingRunRef
from tuner.handlers.modal_job_config_handler import ModalJobConfigHandler
from tuner.training import modal_standalone_runner as runner


def _handler():
    return ModalJobConfigHandler(Namespace(
        modal_profile="explicit", modal_environment="main", json=True,
    ))


@pytest.mark.parametrize("passed,expected_exit", [(True, 0), (False, 2)])
def test_completed_evaluation_reports_run_and_saved_paths(tmp_path, monkeypatch, capsys,
                                                          passed, expected_exit):
    artifact = tmp_path / "final_model.tar"
    artifact.write_bytes(b"verified model")
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text("{}", encoding="utf-8")
    result = runner.ModalStandaloneRunResultV1(
        TrainingRunRef("run-evaluated", "project"), (artifact,), 125, 200,
        evaluation, passed,
    )
    monkeypatch.setattr(runner, "run_modal_standalone_job", lambda **_kwargs: result)

    assert _handler()._execute(object()) == expected_exit
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is passed
    assert payload["data"]["run"] == result.run.to_dict()
    assert payload["data"]["verified_artifacts"] == [str(artifact)]
    assert payload["data"]["evaluation_path"] == str(evaluation)
    assert payload["data"]["evaluation_passed"] is passed


def test_legacy_training_only_result_remains_successful(monkeypatch, capsys):
    legacy = SimpleNamespace(
        run=TrainingRunRef("run-training-only", "project"),
        artifact_paths=(Path("/verified/model.tar"),),
        gpu_only_timeout_estimate_minor_units=125,
        maximum_cost_minor_units=200,
    )
    monkeypatch.setattr(runner, "run_modal_standalone_job", lambda **_kwargs: legacy)

    assert _handler()._execute(object()) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True
    assert payload["data"]["evaluation_path"] is None
    assert payload["data"]["evaluation_passed"] is None


def test_evaluation_read_failure_keeps_closed_error_boundary(monkeypatch, capsys):
    def reject(**_kwargs):
        raise runner.ModalStandalonePhaseUnavailable("RUN_EVALUATION_READ")

    monkeypatch.setattr(runner, "run_modal_standalone_job", reject)
    assert _handler()._execute(object()) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is False
    assert "data" not in payload
    assert payload["error"]["details"] == {
        "phase": "RUN_EVALUATION_READ", "failure_class": "UNAVAILABLE",
        "location": "modal_standalone_runner.evaluation_read",
        "retry_authorized": False,
    }


@pytest.mark.skipif(os.name != "posix", reason="standalone Modal host is POSIX-only")
def test_valid_failed_evaluation_retains_run_artifacts_and_one_submission(tmp_path, monkeypatch):
    from tests.training.test_modal_standalone_runner import _setup
    from tests.training.test_modal_post_training_compilation import _evaluation
    from tuner.execution.providers.modal.packaged_reader import ModalPackagedReader
    from tuner.runtime.post_training_eval import validate_evaluation_record
    from tuner.training.packaged_compilation import (
        compile_packaged_sft_workload, packaged_configuration_digest,
    )

    plan, context, events = _setup(tmp_path, monkeypatch)
    post_training = _evaluation()
    recipe = replace(plan.recipe, post_training=post_training)
    config = recipe.packaged_config(plan.prepared_identity)
    plan = replace(
        plan, recipe=recipe,
        workload_digest=compile_packaged_sft_workload(resolved_config=config).fingerprint,
        configuration_digest=packaged_configuration_digest(config),
    )
    record = {
        "schema_version": "synaptic-post-training-evaluation/v1",
        "status": "failed", "gate_passed": False,
        "failure_code": "startup_failed", "case_count": 1,
        "passed_count": 0, "pass_rate": 0.0, "min_pass_rate": 1.0,
        "cases": [],
        "bindings": {"workload_digest": plan.workload_digest,
                     "model_snapshot_digest": "a" * 64,
                     "adapter_digest": "b" * 64},
    }
    validate_evaluation_record(record, config=post_training)
    raw = json.dumps({"evaluation": record}, sort_keys=True, separators=(",", ":")).encode()
    read_calls = []
    def read_evaluation(self, *_args, **_kwargs):
        read_calls.append(True)
        return raw
    monkeypatch.setattr(ModalPackagedReader, "read_evaluation", read_evaluation)

    result = runner.run_modal_standalone_job(
        plan=plan, context=context, modal_profile="explicit", modal_environment="main",
    )
    assert result.evaluation_passed is False
    assert result.evaluation_path is not None
    assert result.evaluation_path.read_bytes() == raw
    assert len(result.artifact_paths) == 5
    assert read_calls == [True]
    assert events == ["bootstrap", "cpu", "stage", "spawn"]
