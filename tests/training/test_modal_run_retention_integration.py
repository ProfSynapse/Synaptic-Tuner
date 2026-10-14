"""Normal-run private retention; no cold public reader or live service claim."""

import base64
import json
import stat

import pytest

from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
from synaptic_tuner.api.v1.reference.repositories import DurableWorkflowStoreV1
from synaptic_tuner.api.v1.training_facade import TrainingAPI
from tuner.training import modal_run_retention as retention
from tuner.training import modal_standalone_runner as runner
from tests.training.test_modal_standalone_runner import _setup, ROOT
from tuner.cli.router import route_command
from argparse import Namespace


def test_normal_runner_retains_original_inputs_before_start_and_reopens_store(
    tmp_path, monkeypatch, capsys
):
    plan, context, events = _setup(tmp_path, monkeypatch)
    captured = {}
    original_compose = runner.compose_modal_packaged_reference_host
    original_start = TrainingAPI.start

    def compose(**kwargs):
        captured["options"] = kwargs
        host = original_compose(**kwargs)
        captured["host"] = host
        return host

    def start(self, planned, preflight):
        options = captured["options"]
        run = options["run"]
        root = tmp_path / "state" / "synaptic-training" / run.run_id
        retained = retention.load_retained_modal_run(root, expected_run=run)
        assert events == ["bootstrap", "cpu"]
        assert dict(retained.signing_keys) == options["secrets"]._generated
        assert retained.qualification_key == b"k" * 32
        assert (
            retained.inputs["prepared_request"]["canonical_json"]
            == options["prepared_request"].canonical_json
        )
        assert (
            retained.inputs["components"]["execution_source"]
            == options["components"].execution_source.to_dict()
        )
        raw_config = base64.b64decode(
            retained.inputs["components"]["resolved_config"]["base64"]
        )
        assert (
            raw_config == options["components"].resolved_config.canonical_json.encode()
        )
        assert retained.inputs["training_plan"] == planned.to_dict()
        assert retained.inputs["training_preflight"] == preflight.to_dict()
        assert stat.S_IMODE(root.stat().st_mode) == 0o700
        for name in (
            retention.KEYS_FILE,
            retention.INPUTS_FILE,
            retention.WORKFLOW_FILE,
        ):
            assert stat.S_IMODE((root / name).stat().st_mode) == 0o600
        captured.update(root=root, retained=retained, planned=planned)
        assert options["records"] is options["streams"]
        assert type(options["records"]) is SQLiteDurableStoreV1
        return original_start(self, planned, preflight)

    monkeypatch.setattr(runner, "compose_modal_packaged_reference_host", compose)
    monkeypatch.setattr(TrainingAPI, "start", start)
    monkeypatch.setattr(
        "tuner.training.modal_recipe.plan_modal_sft_recipe",
        lambda *args, **kwargs: plan,
    )
    assert (
        route_command(
            Namespace(
                command="train",
                job_config=str(
                    ROOT / "Trainers/recipes/qwen35_4b_32k_modal_prompt_completion.yaml"
                ),
                plan=False,
                quote=False,
                modal_profile="explicit",
                modal_environment="main",
                json=True,
            ),
            context=context,
        )
        == 0
    )
    output = capsys.readouterr()
    assert json.loads(output.out)["success"] is True
    retained = captured["retained"]
    for value in (
        *retained.signing_keys.values(),
        base64.b64encode(retained.qualification_key).decode(),
        "fixture-token",
    ):
        assert value not in output.out + output.err
        assert value not in repr(retained)
    reopened = retention.load_retained_modal_run(
        captured["root"], expected_run=retained.run
    )
    assert dict(reopened.signing_keys) == dict(retained.signing_keys)
    assert reopened.qualification_key == retained.qualification_key
    assert reopened.inputs == retained.inputs
    with pytest.raises(RuntimeError, match="closed"):
        captured["options"]["records"].read(partition="effects", key="test")
    workflow = captured["host"].composition.stores.workflow_store
    with SQLiteDurableStoreV1(captured["root"] / retention.WORKFLOW_FILE) as store:
        cold_workflow = DurableWorkflowStoreV1(
            store,
            store,
            foundation_authenticator=workflow._foundation_authenticator,
            assessment_authenticator=workflow._assessment_authenticator,
            observation_authenticator=workflow._observation_authenticator,
            artifact_verifier=workflow._artifact_verifier,
        )
        assert cold_workflow._validated == {}
        record = cold_workflow.get(retained.run)
        assert record is not None
        assert record.run == retained.run
        assert record.plan_fingerprint == captured["planned"].plan_fingerprint
    assert events == ["bootstrap", "cpu", "stage", "spawn"]


@pytest.mark.parametrize("failure", ["write", "fsync", "readback"])
@pytest.mark.parametrize("through_cli", [False, True])
def test_late_retention_failure_never_starts_training(
    tmp_path, monkeypatch, capsys, failure, through_cli
):
    plan, context, events = _setup(tmp_path, monkeypatch)
    original_write, original_read = retention._write, retention._read
    stores = []
    original_compose = runner.compose_modal_packaged_reference_host

    def compose(**kwargs):
        stores.append(kwargs["records"])
        return original_compose(**kwargs)

    monkeypatch.setattr(runner, "compose_modal_packaged_reference_host", compose)

    def fail(*args, **kwargs):
        raise OSError("private-fixture-secret")

    if failure == "write":
        monkeypatch.setattr(retention, "_write", fail)
    elif failure == "fsync":

        def write(*args, **kwargs):
            with monkeypatch.context() as patch:
                patch.setattr(retention.os, "fsync", fail)
                return original_write(*args, **kwargs)

        monkeypatch.setattr(retention, "_write", write)
    else:

        def read(root, name, maximum):
            if name == retention.KEYS_FILE:
                return b"wrong"
            return original_read(root, name, maximum)

        monkeypatch.setattr(retention, "_read", read)

    def forbidden(*args, **kwargs):
        pytest.fail("retention failure reached training.start")

    monkeypatch.setattr(TrainingAPI, "start", forbidden)
    if through_cli:
        from tuner.handlers.modal_job_config_handler import ModalJobConfigHandler
        handler = ModalJobConfigHandler(Namespace(
            modal_profile="explicit", modal_environment="main", json=True,
        ), context)
        assert handler._execute(plan) == 2
        output = capsys.readouterr()
        assert json.loads(output.out)["error"]["details"] == {
            "phase": "RUN_RETAIN_READ_INPUTS",
            "failure_class": "UNAVAILABLE",
            "location": "modal_standalone_runner.retain_read_inputs",
            "retry_authorized": False,
        }
        assert "private-fixture-secret" not in output.out + output.err
    else:
        with pytest.raises(runner.ModalStandalonePhaseUnavailable) as error:
            runner.run_modal_standalone_job(
                plan=plan,
                context=context,
                modal_profile="explicit",
                modal_environment="main",
            )
        assert error.value.phase == "RUN_RETAIN_READ_INPUTS"
        assert "private-fixture-secret" not in str(error.value)
    assert events == ["bootstrap", "cpu"]
    assert len(stores) == 1
    with pytest.raises(RuntimeError, match="closed"):
        stores[0].read(partition="effects", key="test")
    roots = list((tmp_path / "state" / "synaptic-training").glob("modal-*"))
    assert any(
        (root / retention.WORKFLOW_FILE).exists() for root in roots if root.is_dir()
    )


def test_fixture_source_identity_is_explicit_and_other_process_calls_forward(tmp_path, monkeypatch):
    from tuner.execution.providers.modal import runtime_build
    forwarded = []
    sentinel = object()
    real_run = runtime_build.subprocess.run
    def original(command, *args, **kwargs):
        if command[:3] == ["git", "-C", str(tmp_path)]:
            return real_run(command, *args, **kwargs)
        forwarded.append((command, args, kwargs))
        return sentinel
    monkeypatch.setattr(runtime_build.subprocess, "run", original)
    _setup(tmp_path, monkeypatch)
    engine_root = runtime_build.Path(runtime_build.__file__).resolve().parents[4]
    query = ["git", "-C", str(engine_root), "rev-parse", "HEAD"]
    assert runtime_build.subprocess.run(query).stdout == ("1" * 40 + "\n").encode()
    assert runtime_build.subprocess.run(["other"], timeout=7) is sentinel
    assert forwarded == [(["other"], (), {"timeout": 7})]
    alternate = ["git", "-C", "/unselected", "rev-parse", "HEAD"]
    assert runtime_build.subprocess.run(alternate, check=False) is sentinel
    assert forwarded[-1] == (alternate, (), {"check": False})
