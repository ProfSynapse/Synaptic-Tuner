"""Generic terminal-run CLI delegation; provider-free retained reader double."""

from contextlib import contextmanager
from hashlib import sha256
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.runs_facade import RunOutcome, RunVerification
from synaptic_tuner.api.v1.results import TrainingRunState
from tuner.cli.parser import create_parser
from tuner.cli.router import route_command
from tuner.project import ProjectContext

ARGS = [
    "--run",
    "modal-test",
    "--modal-profile",
    "named",
    "--modal-environment",
    "main",
    "--json",
]


@pytest.mark.parametrize(
    "arguments",
    [
        ["runs"],
        ["runs", "cancel", *ARGS],
        ["runs", "show", "--run", "latest"],
        ["runs", "show", *ARGS, "--job-config", "recipe.yaml"],
        ["runs", "show", *ARGS, "--dest", "output"],
        ["runs", "download", *ARGS],
        ["runs", "show", *ARGS, "--fresh-attempt"],
        ["runs", "show", *ARGS, "--run", "../escape"],
        ["train", "--artifact-role", "final_model"],
    ],
)
def test_rejects_incomplete_or_execution_options(arguments):
    with pytest.raises(SystemExit) as error:
        create_parser().parse_args(arguments)
    assert error.value.code == 2


def setup_reader(tmp_path, monkeypatch, *, verified=True, fail=False):
    from tuner.training import modal_local_preparation

    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    calls = []
    monkeypatch.setattr(
        modal_local_preparation,
        "training_state_root",
        lambda *, create: (calls.append(("root", create)), root)[1],
    )
    run = TrainingRunRef("modal-test", "standalone-training")

    class Runs:
        def show(self, supplied):
            assert supplied == run
            calls.append("show")
            return RunOutcome(
                "synaptic-run-outcome/v1", run, TrainingRunState.SUCCEEDED
            )

        def verify(self, supplied):
            assert supplied == run
            calls.append("verify")
            return RunVerification(run, verified, "2026-10-13T00:00:00Z")

        def artifacts(self, request):
            assert request.run == run and request.role == "final_model"
            calls.append("artifacts")
            from synaptic_tuner.api.v1.results import VerifiedArtifact

            artifact = VerifiedArtifact("final_model", sha256(b"data").hexdigest(), 4)
            return SimpleNamespace(
                artifact=artifact, iter_bytes=lambda: iter([b"data"])
            )

    module = ModuleType("tuner.training.modal_retained_reader")

    @contextmanager
    def open_reader(**kwargs):
        assert kwargs["root"] == root / run.run_id
        assert kwargs["run"] == run
        assert kwargs["modal_profile"] == "named"
        assert kwargs["modal_environment"] == "main"
        calls.append("open")
        if fail:
            raise RuntimeError("private-key-provider-exception")
        try:
            yield SimpleNamespace(run=run, runs=Runs())
        finally:
            calls.append("close")

    module.open_retained_modal_runs = open_reader
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, "modal", ModuleType("modal"))
    return calls


@pytest.mark.parametrize("action", ["show", "verify", "download"])
def test_routes_only_public_reads_and_closes_reader(
    tmp_path, monkeypatch, capsys, action
):
    calls = setup_reader(tmp_path, monkeypatch)
    command = ["runs", action, *ARGS]
    if action == "download":
        command += [
            "--artifact-role",
            "final_model",
            "--dest",
            str(tmp_path / "download"),
        ]
    args = create_parser().parse_args(command)
    assert (
        route_command(args, context=ProjectContext.standalone(engine_root=tmp_path))
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True
    assert payload["data"]["run"]["run_id"] == "modal-test"
    assert calls == [
        ("root", False),
        "open",
        "artifacts" if action == "download" else action,
        "close",
    ]
    if action == "download":
        from pathlib import Path

        assert Path(payload["data"]["path"]).read_bytes() == b"data"


def test_failed_verification_returns_nonzero(tmp_path, monkeypatch, capsys):
    setup_reader(tmp_path, monkeypatch, verified=False)
    assert (
        route_command(
            create_parser().parse_args(["runs", "verify", *ARGS]),
            context=ProjectContext.standalone(engine_root=tmp_path),
        )
        == 2
    )
    assert json.loads(capsys.readouterr().out)["success"] is False


def test_reader_failure_has_only_closed_message(tmp_path, monkeypatch, capsys):
    setup_reader(tmp_path, monkeypatch, fail=True)
    assert (
        route_command(
            create_parser().parse_args(["runs", "show", *ARGS]),
            context=ProjectContext.standalone(engine_root=tmp_path),
        )
        == 2
    )
    output = capsys.readouterr()
    assert json.loads(output.out)["error"]["code"] == "RUN_READ_UNAVAILABLE"
    assert "private-key-provider-exception" not in output.out + output.err


def test_fresh_help_does_not_import_provider_or_create_processes(tmp_path):
    from pathlib import Path
    import subprocess

    root = Path(__file__).resolve().parents[2]
    code = r"""
import builtins, socket, subprocess, sys
sys.path.insert(0, sys.argv[1])
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'modal', 'torch', 'sklearn', 'pandas'}:
        raise AssertionError('optional provider imported for help')
    return original(name, *args, **kwargs)
def forbidden(*args, **kwargs):
    raise AssertionError('help attempted process/network')
builtins.__import__ = guarded
subprocess.Popen = forbidden
socket.socket = forbidden
from tuner.cli.parser import create_parser
create_parser().parse_args(['runs', '--help'])
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(root)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "--artifact-role" in result.stdout
