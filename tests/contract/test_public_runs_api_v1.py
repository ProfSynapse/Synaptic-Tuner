"""Contract tests for the sole canonical run-observation facade."""

from __future__ import annotations

import inspect
import subprocess
import sys
from types import MappingProxyType

import pytest

from synaptic_tuner.api import v1
from synaptic_tuner.api.v1 import (
    APIHost,
    RunArtifactRequest,
    RunListRequest,
    RunLogEntry,
    RunLogLevel,
    RunLogPage,
    RunLogsRequest,
    RunOutcome,
    RunPage,
    RunVerification,
    RunsAPI,
)
from synaptic_tuner.api.v1.host import HostPorts
from synaptic_tuner.api.v1.results import (
    TrainingRunRef,
    TrainingRunState,
    VerifiedArtifact,
)


def _outcome(run: TrainingRunRef) -> RunOutcome:
    return RunOutcome("synaptic-run-outcome/v1", run, TrainingRunState.RUNNING)


def test_root_runs_exports_are_the_canonical_facade_identities() -> None:
    from synaptic_tuner.api.v1 import runs_facade

    assert RunsAPI is runs_facade.RunsAPI
    assert RunOutcome is runs_facade.RunOutcome
    assert v1.TrainingRunRef is TrainingRunRef
    assert v1.TrainingRunState is TrainingRunState
    with pytest.raises(ModuleNotFoundError):
        __import__("synaptic_tuner.api.v1.runs")


def test_runs_api_has_only_the_accepted_verbs() -> None:
    verbs = {
        name for name, member in RunsAPI.__dict__.items()
        if not name.startswith("_") and inspect.isfunction(member)
    }
    assert verbs == {
        "list", "show", "outcome", "logs", "cancel", "reconcile",
        "verify", "reverify", "artifacts",
    }


def test_runs_facade_reconstructs_and_binds_every_callback_result() -> None:
    run = TrainingRunRef("run-1", "project-1")
    outcome = _outcome(run)
    listing = RunListRequest("project-1", limit=1)
    logs_request = RunLogsRequest(run, limit=1, maximum_bytes=4096)
    log = RunLogEntry(
        1, "2026-08-30T12:00:00Z", RunLogLevel.INFO, "running", "ok", 2,
    )
    artifact = VerifiedArtifact("model", "a" * 64, 2)

    class Stream:
        def __init__(self):
            self.run = run
            self.artifact = artifact
            self.maximum_bytes = 2

        def iter_bytes(self):
            yield b"ok"

    class Operations:
        def list(self, request):
            assert request is not listing
            return RunPage(request, (outcome,))

        def show(self, requested):
            assert requested is not run
            return outcome

        outcome = show

        def logs(self, request):
            assert request is not logs_request
            return RunLogPage(request, (log,), 2)

        def cancel(self, requested, reason):
            assert requested is not run
            return outcome

        def reconcile(self, requested):
            assert requested is not run
            return outcome

        def verify(self, requested):
            assert requested is not run
            return RunVerification(run, True, "2026-08-30T12:01:00Z")

        reverify = verify

        def artifacts(self, request):
            assert request.run is not run
            return Stream()

    api = RunsAPI(Operations())
    assert api.list(listing) == RunPage(listing, (outcome,))
    assert api.show(run) == outcome
    assert api.outcome(run) == outcome
    assert api.logs(logs_request) == RunLogPage(logs_request, (log,), 2)
    assert api.cancel(run, "operator request") == outcome
    assert api.reconcile(run) == outcome
    assert api.verify(run).run == run
    assert api.reverify(run).run == run
    assert b"".join(api.artifacts(RunArtifactRequest(run, "model", 2)).iter_bytes()) == b"ok"


def test_runs_facade_rejects_callback_identity_drift() -> None:
    requested = TrainingRunRef("run-1", "project-1")
    wrong = TrainingRunRef("run-2", "project-1")

    class Operations:
        def show(self, run):
            return _outcome(wrong)

    with pytest.raises(ValueError, match="bind"):
        RunsAPI(Operations()).show(requested)


@pytest.mark.parametrize(
    "verb",
    ["list", "show", "outcome", "logs", "cancel", "reconcile", "verify", "reverify", "artifacts"],
)
@pytest.mark.parametrize("raises", [False, True])
def test_runs_facade_rejects_presented_input_mutation_on_return_and_raise(verb, raises) -> None:
    run = TrainingRunRef("run-1", "project-1")
    listing = RunListRequest("project-1", limit=1)
    logs = RunLogsRequest(run, limit=1, maximum_bytes=4096)
    artifact = RunArtifactRequest(run, "model", 2)

    class Operations:
        def __getattr__(self, name):
            def callback(value, *extra):
                target = value.run if name in {"logs", "artifacts"} else value
                field = "project_ref" if name == "list" else "run_id"
                object.__setattr__(target, field, "changed")
                if raises:
                    raise RuntimeError("collaborator detail")
                return object()
            return callback

    api = RunsAPI(Operations())
    invocation = {
        "list": lambda: api.list(listing),
        "show": lambda: api.show(run),
        "outcome": lambda: api.outcome(run),
        "logs": lambda: api.logs(logs),
        "cancel": lambda: api.cancel(run, "operator request"),
        "reconcile": lambda: api.reconcile(run),
        "verify": lambda: api.verify(run),
        "reverify": lambda: api.reverify(run),
        "artifacts": lambda: api.artifacts(artifact),
    }[verb]
    with pytest.raises(ValueError, match="input changed") as captured:
        invocation()
    if raises:
        pending = [captured.value]
        seen = set()
        while pending:
            error = pending.pop()
            if id(error) in seen:
                continue
            seen.add(id(error))
            assert type(error) is not RuntimeError
            assert "collaborator detail" not in str(error)
            pending.extend(item for item in (error.__cause__, error.__context__) if item is not None)


def test_canonical_run_and_artifact_contracts_reject_nonexact_inputs() -> None:
    class Text(str):
        pass
    class RunSubclass(TrainingRunRef):
        pass
    class DictSubclass(dict):
        pass

    with pytest.raises(TypeError):
        TrainingRunRef(Text("run-1"), "project-1")
    with pytest.raises(TypeError, match="exact object"):
        TrainingRunRef.from_dict(MappingProxyType({"run_id": "run-1", "project_ref": "project-1"}))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact object"):
        VerifiedArtifact.from_dict(DictSubclass({"role": "model", "sha256": "a" * 64, "size_bytes": 1}))
    with pytest.raises(TypeError, match="exact integer"):
        VerifiedArtifact("model", "a" * 64, 1.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact TrainingRunRef"):
        RunsAPI(object()).show(RunSubclass("run-1", "project-1"))


def test_logs_and_verification_results_detach_nested_run_identities() -> None:
    run = TrainingRunRef("run-1", "project-1")
    logs_request = RunLogsRequest(run, limit=1, maximum_bytes=4096)
    callback_logs = RunLogPage(logs_request, (), 0)
    callback_verification = RunVerification(run, True, "2026-08-30T12:01:00Z")

    class Operations:
        def logs(self, supplied):
            return callback_logs

        def verify(self, supplied):
            return callback_verification

        reverify = verify

    api = RunsAPI(Operations())
    public_logs = api.logs(logs_request)
    public_verify = api.verify(run)
    public_reverify = api.reverify(run)
    assert public_logs.request.run is not callback_logs.request.run
    assert public_verify.run is not callback_verification.run
    assert public_reverify.run is not callback_verification.run

    object.__setattr__(callback_logs.request.run, "run_id", "mutated")
    object.__setattr__(callback_verification.run, "run_id", "mutated")
    assert public_logs.request.run == run
    assert public_verify.run == run
    assert public_reverify.run == run


@pytest.mark.parametrize("container", [MappingProxyType, dict])
def test_run_parsers_require_exact_builtin_objects(container) -> None:
    document = _outcome(TrainingRunRef("run-1", "project-1")).to_dict()
    hostile = container(document)
    if container is dict:
        class DictSubclass(dict):
            pass
        hostile = DictSubclass(document)
    with pytest.raises(TypeError, match="exact object"):
        RunOutcome.from_dict(hostile)  # type: ignore[arg-type]


def test_run_parsers_reject_hostile_field_name_subclasses_without_callbacks() -> None:
    class Field(str):
        calls = 0

        def __hash__(self):
            type(self).calls += 1
            if type(self).calls > 1:
                raise RuntimeError("secret callback")
            return str.__hash__(self)

    document = _outcome(TrainingRunRef("run-1", "project-1")).to_dict()
    value = dict(document)
    original = value.pop("state")
    dict.__setitem__(value, Field("state"), original)
    with pytest.raises(TypeError, match="field names") as captured:
        RunOutcome.from_dict(value)
    assert captured.value.__cause__ is None
    assert Field.calls == 1


def test_api_host_uses_the_canonical_runs_facade() -> None:
    ports = HostPorts(
        lifecycle=object(), runs=object(), grants=object(), secrets=object(),
        evidence_replay=object(), authenticator=object(),
        clock=lambda: "2026-08-30T12:00:00Z", git_remote=object(),
        modal_reads=object(), training_resolver=object(),
    )
    host = APIHost(object(), ports)
    assert type(host.runs) is RunsAPI


def test_public_v1_import_does_not_load_provider_or_database_modules() -> None:
    code = """
import sys
import synaptic_tuner.api.v1
for prefix in ('huggingface_hub', 'modal', 'runpod', 'sqlite3', 'tuner'):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules), prefix
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
