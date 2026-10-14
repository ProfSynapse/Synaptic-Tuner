"""Configured public cancellation with original durable authority and no sources."""

from dataclasses import replace
import shutil
import sys

import pytest

from synaptic_tuner.api.v1.runs_facade import RunOperationError
from tuner.handlers.runs_handler import _CurrentCancellationGrant
from tuner.training import modal_retained_reader as reader
from tuner.training import modal_standalone_runner as runner
from tuner.training.modal_host_reader import (
    ModalPackagedCoordinatorReaderV1,
    ModalPackagedReadUnavailable,
)
from tests.training.test_modal_standalone_runner import _setup


def _start(tmp_path, monkeypatch, *, enabled=True):
    plan, context, events = _setup(tmp_path, monkeypatch)
    plan = replace(plan, recipe=replace(plan.recipe, allow_cancel_reconcile=enabled))
    captured = []
    compose = runner.compose_modal_packaged_reference_host

    def capture(**kwargs):
        if enabled:
            kwargs["cancellation_grants"] = _CurrentCancellationGrant(kwargs["run"])
        assert kwargs.get("allow_cancel_reconcile", False) is enabled
        host = compose(**kwargs)
        captured.append(host)
        return host

    monkeypatch.setattr(runner, "compose_modal_packaged_reference_host", capture)
    with monkeypatch.context() as interrupted:
        interrupted.setattr(
            ModalPackagedCoordinatorReaderV1,
            "_poll_packaged_call",
            lambda *a, **kw: (_ for _ in ()).throw(
                ModalPackagedReadUnavailable("modal_packaged_call_read_unavailable")
            ),
        )
        with pytest.raises(runner.ModalStandalonePhaseUnavailable) as stopped:
            runner.run_modal_standalone_job(
                plan=plan,
                context=context,
                modal_profile="fixture",
                modal_environment="main",
            )
    assert stopped.value.phase == "RUN_CALL_OBSERVE"
    run = stopped.value.retained_run
    root = tmp_path / "state/synaptic-training" / run.run_id
    retained = reader.load_retained_modal_run(root, expected_run=run)
    client, scope = runner.open_modal_host_scope()
    facts = reader.ModalPackagedRuntimeFactsV1.from_dict(
        retained.inputs["runtime_facts"]
    )
    monkeypatch.setattr(reader, "validate_local_launcher", lambda: None)
    monkeypatch.setattr(reader, "open_modal_host_scope", lambda **kw: (client, scope))
    monkeypatch.setattr(
        reader,
        "observe_modal_host_scope",
        lambda **kw: (
            scope.account_ref,
            scope.workspace_ref,
            scope.environment_ref,
            scope.client_ref,
        ),
    )
    monkeypatch.setattr(
        reader._CurrentPackagedDeploymentReader, "observe", lambda *a, **kw: facts
    )
    return captured[0], run, root, events, client


def _sdk_call(monkeypatch, client, *, lost=False):
    calls = []

    class Call:
        object_id = "fc-owned"
        is_hydrated = True

        def __init__(self):
            self.client = client

        def hydrate(self, supplied):
            assert supplied is client
            return self

        def cancel(self, *, terminate_containers):
            assert terminate_containers is True
            calls.append(self.object_id)
            if lost:
                raise RuntimeError("untrusted cancellation reply")

        def get(self, *, timeout):
            raise TimeoutError()

    class Calls:
        @staticmethod
        def from_id(reference, *, client):
            assert reference == "fc-owned"
            return Call()

    monkeypatch.setattr(sys.modules["modal"], "FunctionCall", Calls, raising=False)
    return calls


def _open(run, root, grants):
    return reader.open_retained_modal_lifecycle(
        root=root,
        run=run,
        modal_profile="fixture",
        modal_environment="main",
        sdk=sys.modules["modal"],
        cancellation_grants=grants,
    )


def test_configured_normal_host_and_cold_cancel_acknowledgment(tmp_path, monkeypatch):
    host, run, root, events, client = _start(tmp_path, monkeypatch)
    calls = _sdk_call(monkeypatch, client)
    # End the warm object and remove only local input/config/source files.
    for p in tmp_path.iterdir():
        if p.name == "state":
            continue
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    with _open(run, root, _CurrentCancellationGrant(run)) as view:
        assert not hasattr(view, "training")
        result = view.runs.cancel(run, "explicit operator stop")
        assert result.state.value == "cancel_requested"
    with _open(run, root, _CurrentCancellationGrant(run)) as view:
        assert (
            view.runs.cancel(run, "different reason").state.value == "cancel_requested"
        )
    assert calls == ["fc-owned"]
    assert events == ["bootstrap", "cpu", "stage", "spawn"]


def test_lost_reply_reconcile_never_reissues_cancel(tmp_path, monkeypatch):
    _, run, root, events, client = _start(tmp_path, monkeypatch)
    calls = _sdk_call(monkeypatch, client, lost=True)
    with _open(run, root, _CurrentCancellationGrant(run)) as view:
        assert view.runs.cancel(run, "stop").state.value == "reconcile_required"

    class Denied:
        def authorize(self, requirements):
            raise ValueError("current permission refused")

        def bind(self, *a, **kw):
            pytest.fail("lookup must not bind a new cancellation")

    with _open(run, root, Denied()) as view:
        with pytest.raises(RunOperationError):
            view.runs.reconcile(run)
    with _open(run, root, _CurrentCancellationGrant(run)) as view:
        assert view.runs.reconcile(run).state.value == "reconcile_required"
    assert calls == ["fc-owned"]
    assert events == ["bootstrap", "cpu", "stage", "spawn"]


def test_historical_default_does_not_gain_cancellation(tmp_path, monkeypatch):
    _, run, root, events, client = _start(tmp_path, monkeypatch, enabled=False)
    calls = _sdk_call(monkeypatch, client)
    with pytest.raises(reader.ModalRetainedReadUnavailable):
        with _open(run, root, _CurrentCancellationGrant(run)):
            pytest.fail("historical false descriptor admitted")
    with reader.open_retained_modal_runs(
        root=root,
        run=run,
        modal_profile="fixture",
        modal_environment="main",
        sdk=sys.modules["modal"],
    ) as view:
        with pytest.raises(RunOperationError):
            view.runs.cancel(run, "stop")
    assert not calls
    assert events == ["bootstrap", "cpu", "stage", "spawn"]


@pytest.mark.parametrize("lost", [False, True])
def test_actual_runs_handler_uses_current_exact_run_request(
    tmp_path, monkeypatch, capsys, lost
):
    from tuner.cli.parser import create_parser
    from tuner.cli.router import route_command
    from tuner.project.context import ProjectContext
    import json

    _, run, root, events, client = _start(tmp_path, monkeypatch)
    calls = _sdk_call(monkeypatch, client, lost=lost)
    context = ProjectContext.standalone(engine_root=tmp_path)
    capsys.readouterr()
    arguments = [
        "runs",
        "cancel",
        "--run",
        run.run_id,
        "--reason",
        "operator stop",
        "--modal-profile",
        "fixture",
        "--modal-environment",
        "main",
        "--json",
    ]
    assert route_command(create_parser().parse_args(arguments), context=context) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["data"]["outcome"]["state"] == (
        "reconcile_required" if lost else "cancel_requested"
    )
    if lost:
        arguments = [
            "runs",
            "reconcile",
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
        assert (
            json.loads(capsys.readouterr().out)["data"]["outcome"]["state"]
            == "reconcile_required"
        )
    assert calls == ["fc-owned"]
    assert events == ["bootstrap", "cpu", "stage", "spawn"]


@pytest.mark.parametrize(
    "arguments",
    [
        [
            "runs",
            "cancel",
            "--run",
            "exact",
            "--modal-profile",
            "fixture",
            "--modal-environment",
            "main",
        ],
        [
            "runs",
            "reconcile",
            "--run",
            "exact",
            "--reason",
            "stop",
            "--modal-profile",
            "fixture",
            "--modal-environment",
            "main",
        ],
    ],
)
def test_cancel_parser_requires_exact_request_before_sdk(arguments):
    from tuner.cli.parser import create_parser

    with pytest.raises(SystemExit):
        create_parser().parse_args(arguments)


def test_changed_original_authority_key_refuses_before_sdk(tmp_path, monkeypatch):
    key = "SYNAPTIC_TRAIN_AUTHORITY_KEY"
    _, run, root, events, client = _start(tmp_path, monkeypatch)
    retained = reader.load_retained_modal_run(root, expected_run=run)
    keys = dict(retained.signing_keys)
    keys[key] = "0" * 64
    changed = replace(retained, signing_keys=keys)
    monkeypatch.setattr(reader, "load_retained_modal_run", lambda *a, **kw: changed)
    monkeypatch.setattr(
        reader,
        "open_modal_host_scope",
        lambda **kw: pytest.fail("invalid keys reached SDK"),
    )
    with pytest.raises(reader.ModalRetainedReadUnavailable):
        with _open(run, root, _CurrentCancellationGrant(run)):
            pytest.fail("changed original keys admitted")
    assert events == ["bootstrap", "cpu", "stage", "spawn"]


def test_cancel_after_submit_expiry_preserves_original_authority(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from synaptic_tuner.api.v1.reference.training import (
        ReferenceAuthorizationError,
        _commitment_document,
    )

    _, run, root, events, client = _start(tmp_path, monkeypatch)
    calls = _sdk_call(monkeypatch, client)

    class LateClock:
        def now_epoch(self):
            return int(datetime.now(timezone.utc).timestamp()) + 901

        def now(self):
            return (
                datetime.fromtimestamp(self.now_epoch(), timezone.utc)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z")
            )

        now_iso = now

    clock = LateClock()
    with reader.open_retained_modal_lifecycle(
        root=root,
        run=run,
        modal_profile="fixture",
        modal_environment="main",
        sdk=sys.modules["modal"],
        cancellation_grants=_CurrentCancellationGrant(run),
        clock=clock,
    ) as view:
        workflow = view.runs._operations._workflows.get(run)
        authorization = view.runs._operations._operations._coordinator._authorization
        before = _commitment_document(
            authorization._commitment(workflow.preflight_digest)
        )
        with pytest.raises(ReferenceAuthorizationError):
            authorization.issue_effect_grant(
                workflow.submit.canonical_command_bytes,
                preflight_digest=workflow.preflight_digest,
                now_epoch=clock.now_epoch(),
            )
        assert (
            view.runs.cancel(run, "current operator stop").state.value
            == "cancel_requested"
        )
        after = view.runs._operations._workflows.get(run)
        assert after.submit == workflow.submit
        assert (
            _commitment_document(authorization._commitment(workflow.preflight_digest))
            == before
        )
    assert calls == ["fc-owned"]


def test_cold_transport_rejects_original_stage_and_submit(tmp_path, monkeypatch):
    from tuner.execution.providers.modal.packaged_effects import (
        ModalPackagedFoundationEffectExecutor,
    )
    from tuner.execution.foundation_v2.commands import parse_exact_command

    _, run, root, events, client = _start(tmp_path, monkeypatch)
    transports = []
    original = ModalPackagedFoundationEffectExecutor.__init__

    def capture(self, **kwargs):
        transports.append(kwargs["transport"])
        original(self, **kwargs)

    monkeypatch.setattr(ModalPackagedFoundationEffectExecutor, "__init__", capture)
    calls = _sdk_call(monkeypatch, client)
    with _open(run, root, _CurrentCancellationGrant(run)) as view:
        workflow = view.runs._operations._workflows.get(run)
        for intent in (workflow.stage, workflow.submit):
            command = parse_exact_command(intent.canonical_command_bytes)
            for operation in (transports[0].execute_once, transports[0].lookup_once):
                with pytest.raises(reader.ModalRetainedReadUnavailable):
                    operation(None, command)
    assert not calls
    assert events == ["bootstrap", "cpu", "stage", "spawn"]


def test_wrong_current_run_permission_never_dispatches(tmp_path, monkeypatch):
    from synaptic_tuner.api.v1.results import TrainingRunRef

    _, run, root, events, client = _start(tmp_path, monkeypatch)
    calls = _sdk_call(monkeypatch, client)
    wrong = TrainingRunRef("different-run", run.project_ref)
    with _open(run, root, _CurrentCancellationGrant(wrong)) as view:
        with pytest.raises(RunOperationError):
            view.runs.cancel(run, "stop")
        workflow = view.runs._operations._workflows.get(run)
        assert workflow.cancel is not None
        assert workflow.phase.value == "cancel_intent_recorded"
    assert not calls
    assert events == ["bootstrap", "cpu", "stage", "spawn"]


def test_recipe_optin_is_exact_bool_and_absent_plan_bytes_unchanged(tmp_path):
    from pathlib import Path
    import yaml
    from tuner.training.modal_recipe import load_modal_sft_recipe

    source = Path(runner.__file__).resolve().parents[2]
    raw = yaml.safe_load(
        (
            source / "Trainers/recipes/qwen35_4b_32k_modal_prompt_completion.yaml"
        ).read_text()
    )
    path = tmp_path / "recipe.yaml"
    raw["allow_cancel_reconcile"] = True
    path.write_text(yaml.safe_dump(raw))
    assert (
        load_modal_sft_recipe(
            path, profiles_root=source / "Trainers/runtime_profiles"
        ).allow_cancel_reconcile
        is True
    )
    raw["allow_cancel_reconcile"] = 1
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_modal_sft_recipe(path, profiles_root=source / "Trainers/runtime_profiles")
    from tuner.training.modal_host_composition import _modal_packaged_descriptor

    assert (
        _modal_packaged_descriptor(logs=False).to_dict()
        == _modal_packaged_descriptor(
            logs=False, allow_cancel_reconcile=False
        ).to_dict()
    )


def test_changed_known_call_refuses_before_sdk(tmp_path, monkeypatch):
    from tuner.training.modal_host_storage import ModalHostStorageV1

    _, run, root, events, client = _start(tmp_path, monkeypatch)
    original = ModalHostStorageV1.catalog

    def catalog(self, name, **kwargs):
        retained = original(self, name, **kwargs)
        if name == "packaged-calls-v1":

            class ChangedCall:
                def resolve(self, key):
                    return "fc-other"

                def publish_if_absent(self, *args):
                    pytest.fail("readback cannot repair a call")

            return ChangedCall()
        return retained

    monkeypatch.setattr(ModalHostStorageV1, "catalog", catalog)
    monkeypatch.setattr(
        reader,
        "open_modal_host_scope",
        lambda **kw: pytest.fail("changed target reached SDK"),
    )
    with pytest.raises(reader.ModalRetainedReadUnavailable):
        with _open(run, root, _CurrentCancellationGrant(run)):
            pytest.fail("changed target admitted")
    assert events == ["bootstrap", "cpu", "stage", "spawn"]
