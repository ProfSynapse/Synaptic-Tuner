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
