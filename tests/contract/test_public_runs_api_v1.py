"""Contract tests for run identities, lifecycle ports, and import boundaries."""

from __future__ import annotations

import inspect
import subprocess
import sys

import pytest

from synaptic_tuner.api.v1 import (
    APIHost,
    ArtifactPage,
    ArtifactsRequest,
    CancelResult,
    LogPage,
    ReconcileRequest,
    RunCancelRequest,
    RunListRequest,
    RunLogsRequest,
    RunPage,
    RunRef,
    RunState,
    RunStatus,
    RunVerification,
    RunsAPI,
    VerifyRequest,
)
from synaptic_tuner.api.v1.host import HostPorts
from synaptic_tuner.api.v1.execution import RunRef as ExecutionRunRef


def test_run_ref_has_one_public_identity() -> None:
    assert RunRef is ExecutionRunRef


def test_runs_api_has_only_the_accepted_verbs() -> None:
    verbs = {
        name
        for name, member in RunsAPI.__dict__.items()
        if not name.startswith("_") and inspect.isfunction(member)
    }
    assert verbs == {"list", "show", "logs", "cancel", "reconcile", "verify", "artifacts"}


def test_runs_facade_preserves_engine_run_ref_across_every_verb() -> None:
    run = RunRef("run-1", "project-1")
    status = RunStatus(run, RunState.RUNNING, "2026-08-25T12:00:00Z")

    class Repository:
        def list(self, request):
            assert request.project_ref == run.project_ref
            return RunPage((status,))

        def show(self, requested):
            assert requested is run
            return status

        def logs(self, request):
            assert request.run is run
            return LogPage(run, ())

        def cancel(self, request):
            assert request.run is run
            return CancelResult(run, RunState.CANCEL_REQUESTED, True)

        def reconcile(self, request):
            assert request.run is run
            return status

        def verify(self, request):
            assert request.run is run
            from synaptic_tuner.api.v1 import ArtifactState

            return RunVerification(run, ArtifactState.PENDING, "2026-08-25T12:01:00Z")

        def artifacts(self, request):
            assert request.run is run
            return ArtifactPage(run, ())

    api = RunsAPI(Repository())
    assert api.list(RunListRequest("project-1")).runs == (status,)
    assert api.show(run) is status
    assert api.logs(RunLogsRequest(run)).run is run
    assert api.cancel(RunCancelRequest(run, "operator request")).run is run
    assert api.reconcile(ReconcileRequest(run)).run is run
    assert api.verify(VerifyRequest(run)).run is run
    assert api.artifacts(ArtifactsRequest(run)).run is run


def test_runs_facade_rejects_repository_identity_drift() -> None:
    requested = RunRef("run-1", "project-1")
    wrong = RunRef("provider-job-99", "project-1")

    class Repository:
        def show(self, run):
            return RunStatus(wrong, RunState.RUNNING, "2026-08-25T12:00:00Z")

    with pytest.raises(ValueError, match="requested run"):
        RunsAPI(Repository()).show(requested)


def test_api_host_keeps_persistence_separate_from_run_operations() -> None:
    class Training:
        pass

    class RunOperations:
        pass

    lifecycle = object()
    runs = RunOperations()
    ports = HostPorts(
        lifecycle=lifecycle,
        runs=runs,
        grants=object(),
        secrets=object(),
        evidence_replay=object(),
        authenticator=object(),
        clock=lambda: "2026-08-25T12:00:00Z",
        git_remote=object(),
        modal_reads=object(),
        training_resolver=object(),
    )

    host = APIHost(Training(), ports)

    assert host.ports.lifecycle is lifecycle
    assert host.runs._operations is runs


def test_public_v1_import_does_not_load_provider_or_database_modules() -> None:
    code = """
import sys
import synaptic_tuner.api.v1
for prefix in ('huggingface_hub', 'modal', 'runpod', 'sqlite3'):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules), prefix
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
