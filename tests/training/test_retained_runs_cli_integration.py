"""Actual CLI/read API with fresh readers and reopened stores, same fake provider.

This is not a new-process or live-provider recovery qualification. Only the
launcher and current provider observations are explicit test-owned stand-ins.
"""

import base64
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import pytest

from tuner.cli.parser import create_parser
from tuner.cli.router import route_command
from tuner.execution.providers.modal.packaged_binding import ModalPackagedRuntimeFactsV1
from tuner.training import modal_standalone_runner as runner
from tuner.training import modal_run_retention as retention
from tests.training.test_modal_standalone_runner import _setup


@pytest.mark.parametrize("initially_verified", [False, True])
def test_terminal_cli_reads_after_local_training_inputs_are_removed(
    tmp_path, monkeypatch, capsys, initially_verified
):
    from tuner.training import modal_retained_reader as reader
    from synaptic_tuner.api.v1.training_facade import TrainingAPI
    from tuner.training.modal_host_effects import ModalPackagedHostEffectsV1

    plan, context, events = _setup(tmp_path, monkeypatch)
    if initially_verified:
        report = runner.run_modal_standalone_job(
            plan=plan,
            context=context,
            modal_profile="fixture",
            modal_environment="main",
        )
        run = report.run
    else:
        from synaptic_tuner.api.v1.runs_facade import RunsAPI
        from synaptic_tuner.api.v1.results import TrainingRunRef

        def stop_first_verification(*args, **kwargs):
            raise RuntimeError("fixture local verification interruption")

        with monkeypatch.context() as patch:
            patch.setattr(RunsAPI, "verify", stop_first_verification)
            with pytest.raises(runner.ModalStandalonePhaseUnavailable) as failure:
                runner.run_modal_standalone_job(
                    plan=plan,
                    context=context,
                    modal_profile="fixture",
                    modal_environment="main",
                )
            assert failure.value.phase == "RUN_VERIFY", failure.value.start_causes
        run = failure.value.retained_run
        assert type(run) is TrainingRunRef
    root = tmp_path / "state" / "synaptic-training" / run.run_id
    retained = retention.load_retained_modal_run(root, expected_run=run)
    client, scope = runner.open_modal_host_scope()
    facts = ModalPackagedRuntimeFactsV1.from_dict(retained.inputs["runtime_facts"])
    monkeypatch.setattr(reader, "validate_local_launcher", lambda: None)
    monkeypatch.setattr(
        reader, "open_modal_host_scope", lambda **kwargs: (client, scope)
    )
    monkeypatch.setattr(
        reader,
        "observe_modal_host_scope",
        lambda **kwargs: (
            scope.account_ref,
            scope.workspace_ref,
            scope.environment_ref,
            scope.client_ref,
        ),
    )
    monkeypatch.setattr(
        reader._CurrentPackagedDeploymentReader,
        "observe",
        lambda *args, **kwargs: facts,
    )

    # Only tmp-owned local inputs are removed. Remote fake state and exact
    # retained authority/catalogs remain explicit inputs to the fresh readers.
    for path in (tmp_path / "source", tmp_path / ".tracking", tmp_path / "Trainers"):
        if path.exists():
            shutil.rmtree(path)
    keep = {retention.KEYS_FILE, retention.INPUTS_FILE, retention.WORKFLOW_FILE}
    for path in root.iterdir():
        if path.name not in keep:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("terminal read attempted training/source authority")

    for name in (
        "prepare_modal_standalone_local",
        "_qualified_runtime",
        "stage_published_modal_dataset",
    ):
        monkeypatch.setattr(runner, name, forbidden)
    monkeypatch.setattr(TrainingAPI, "start", forbidden)
    monkeypatch.setattr(ModalPackagedHostEffectsV1, "authorize", forbidden)
    monkeypatch.setattr(ModalPackagedHostEffectsV1, "bind", forbidden)
    monkeypatch.setattr("tuner.training.modal_recipe.plan_modal_sft_recipe", forbidden)
    original_open = Path.open

    def no_deleted_source(path, *args, **kwargs):
        if any(
            path.is_relative_to(base)
            for base in (
                tmp_path / "source",
                tmp_path / ".tracking",
                tmp_path / "Trainers",
            )
        ):
            forbidden()
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_deleted_source)

    sdk = sys.modules["modal"]
    publications = tuple(sdk.Volume.publications)
    capsys.readouterr()
    outputs = []
    for action in ("show", "verify", "download"):
        arguments = [
            "runs",
            action,
            "--run",
            run.run_id,
            "--modal-profile",
            "fixture",
            "--modal-environment",
            "main",
            "--json",
        ]
        if action == "download":
            arguments += [
                "--artifact-role",
                "final_model",
                "--dest",
                str(tmp_path / "redownload"),
            ]
        expected_exit = 2 if initially_verified and action == "verify" else 0
        assert (
            route_command(create_parser().parse_args(arguments), context=context)
            == expected_exit
        )
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        if initially_verified and action == "verify":
            assert payload["success"] is False
            assert payload["error"]["code"] == "RUN_READ_UNAVAILABLE"
        else:
            assert (
                payload["success"] is True and payload["data"]["run"] == run.to_dict()
            )
        for secret in (
            *retained.signing_keys.values(),
            base64.b64encode(retained.qualification_key).decode(),
            "fixture-token",
        ):
            assert secret not in captured.out + captured.err
        outputs.append(payload.get("data"))
    assert outputs[0]["outcome"]["state"] == "succeeded"
    if not initially_verified:
        assert outputs[1]["verification"]["verified"] is True
    downloaded = Path(outputs[2]["path"]).read_bytes()
    assert sha256(downloaded).hexdigest() == sha256(b"data").hexdigest()
    assert sdk.Volume.publications == list(publications)
    assert events == ["bootstrap", "cpu", "stage", "spawn"]


def test_active_cli_observes_known_call_pending_then_completed_without_replay(
    tmp_path, monkeypatch, capsys
):
    from tuner.training import modal_retained_reader as reader
    from synaptic_tuner.api.v1.training_facade import TrainingAPI
    from tuner.training.modal_host_effects import ModalPackagedHostEffectsV1
    from tuner.training.modal_host_reader import ModalPackagedCoordinatorReaderV1
    from tuner.execution.providers.modal.facade import ModalFunctionCallState

    plan, context, events = _setup(tmp_path, monkeypatch)

    def stop_original_observation(*args, **kwargs):
        raise RuntimeError("fixture host observation interruption")

    with monkeypatch.context() as patch:
        patch.setattr(
            ModalPackagedCoordinatorReaderV1,
            "_poll_packaged_call",
            stop_original_observation,
        )
        with pytest.raises(runner.ModalStandalonePhaseUnavailable) as failure:
            runner.run_modal_standalone_job(
                plan=plan,
                context=context,
                modal_profile="fixture",
                modal_environment="main",
            )
        assert failure.value.phase == "RUN_CALL_OBSERVE"
    run = failure.value.retained_run
    assert run is not None
    root = tmp_path / "state" / "synaptic-training" / run.run_id
    retained = retention.load_retained_modal_run(root, expected_run=run)
    client, scope = runner.open_modal_host_scope()
    facts = ModalPackagedRuntimeFactsV1.from_dict(retained.inputs["runtime_facts"])
    monkeypatch.setattr(reader, "validate_local_launcher", lambda: None)
    monkeypatch.setattr(
        reader, "open_modal_host_scope", lambda **kwargs: (client, scope)
    )
    monkeypatch.setattr(
        reader,
        "observe_modal_host_scope",
        lambda **kwargs: (
            scope.account_ref,
            scope.workspace_ref,
            scope.environment_ref,
            scope.client_ref,
        ),
    )
    monkeypatch.setattr(
        reader._CurrentPackagedDeploymentReader,
        "observe",
        lambda *args, **kwargs: facts,
    )
    for path in (tmp_path / "source", tmp_path / ".tracking", tmp_path / "Trainers"):
        if path.exists():
            shutil.rmtree(path)
    for path in root.iterdir():
        if path.name not in {
            retention.KEYS_FILE,
            retention.INPUTS_FILE,
            retention.WORKFLOW_FILE,
        }:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("active observation attempted training authority")

    for name in (
        "prepare_modal_standalone_local",
        "_qualified_runtime",
        "stage_published_modal_dataset",
    ):
        monkeypatch.setattr(runner, name, forbidden)
    monkeypatch.setattr(TrainingAPI, "start", forbidden)
    monkeypatch.setattr(ModalPackagedHostEffectsV1, "authorize", forbidden)
    monkeypatch.setattr(ModalPackagedHostEffectsV1, "bind", forbidden)
    monkeypatch.setattr("tuner.training.modal_recipe.plan_modal_sft_recipe", forbidden)
    states = [ModalFunctionCallState.PENDING, ModalFunctionCallState.RETURNED]
    polls = []

    def poll(self, binding, call, **kwargs):
        assert call == "fc-owned"
        state = states.pop(0)
        polls.append((call, state))
        return state, None if state is ModalFunctionCallState.PENDING else "f" * 64

    monkeypatch.setattr(ModalPackagedCoordinatorReaderV1, "_poll_packaged_call", poll)
    capsys.readouterr()
    outputs = []
    for action in ("show", "observe", "show", "observe", "show"):
        arguments = [
            "runs",
            action,
            "--run",
            run.run_id,
            "--modal-profile",
            "fixture",
            "--modal-environment",
            "main",
            "--json",
        ]
        assert (
            route_command(create_parser().parse_args(arguments), context=context) == 0
        )
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["success"] is True
        assert payload["data"]["run"] == run.to_dict()
        assert all(
            value not in captured.out + captured.err
            for value in retained.signing_keys.values()
        )
        outputs.append(payload["data"]["outcome"]["state"])
    assert outputs == ["queued", "queued", "queued", "succeeded", "succeeded"]
    assert polls == [
        ("fc-owned", ModalFunctionCallState.PENDING),
        ("fc-owned", ModalFunctionCallState.RETURNED),
    ]
    assert states == []
    assert events == ["bootstrap", "cpu", "stage", "spawn"]
