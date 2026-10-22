"""Real CPU guardian and SQLite checks for the private qualification consumer."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import tempfile

import pytest

from examples.training_qualification import native_cpu_process as cpu
from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
from tuner.runtime.native_mlx_material import canonical


@pytest.fixture
def selection(monkeypatch):
    previous = os.umask(0o077)
    try:
        with tempfile.TemporaryDirectory(prefix="native-cpu-process-", dir=Path.home()) as name:
            root = Path(name)
            # The local shared Python ancestors are group-writable and correctly
            # fail NativeTree. Keep the real-process test executable private;
            # the reviewed Mac config instead selects its actual regular Python.
            real_python = sys.executable
            executable = root / "python"
            executable.write_text("#!/bin/sh\nexec " + shlex.quote(real_python) + ' "$@"\n')
            executable.chmod(0o700)
            monkeypatch.setattr(sys, "executable", str(executable))
            source = root / "source"
            for member in (cpu._GUARDIAN, cpu._WORKER, cpu._REQUEST,
                           "examples/training_qualification/native_cpu_process.py"):
                destination = source / member
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((cpu.ENGINE_ROOT / member).read_bytes())
            subprocess.run(["git", "init", "-q", "-b", "feat/cpu-probe", str(source)], check=True)
            for command in (
                ("config", "user.email", "cpu-probe@example.invalid"),
                ("config", "user.name", "CPU Probe"),
                ("remote", "add", "origin", "https://github.com/ProfSynapse/Synaptic-Tuner.git"),
                ("add", "."), ("commit", "-qm", "fixed CPU probe"),
            ):
                subprocess.run(["git", "-C", str(source), *command], check=True)
            commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
            monkeypatch.setattr(cpu, "ENGINE_ROOT", source)
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            config = {
                "schema": "synaptic-native-cpu-process-config/v1",
                "engine_root": str(source), "engine_commit": commit,
                "engine_ref": "refs/heads/feat/cpu-probe",
                "engine_remote": "https://github.com/ProfSynapse/Synaptic-Tuner.git",
                "launcher": sys.executable, "executable": sys.executable,
                "executable_sha256": digest(Path(sys.executable)),
                "python_version": platform.python_version(), "platform": platform.system(),
                "machine": platform.machine(),
                "guardian_sha256": digest(source / cpu._GUARDIAN),
                "worker_sha256": digest(source / cpu._WORKER),
                "request_sha256": digest(source / cpu._REQUEST),
                "state_root": str(root / "state"), "attempt": "cpu-one",
                "limits": {
                    "timeout_seconds": 3, "termination_grace_seconds": 1,
                    "maximum_output_bytes": 4096, "maximum_file_bytes": 1024 * 1024,
                },
            }
            config_path = root / "config.json"
            config_path.write_bytes(canonical(config))
            yield config_path, config, source, root / "state"
    finally:
        os.umask(previous)


def test_real_short_worker_exit_is_retained_and_cannot_replay(selection):
    config_path, _, _, state = selection
    inspected = cpu.inspect(str(config_path))
    assert inspected["status"] == "REVIEW_REQUIRED" and not state.exists()
    result = cpu.run(str(config_path))
    assert result["status"] == "NORMAL_EXIT_OBSERVED"
    assert result["outcome"]["reason"] == "worker_exited"
    assert result["outcome"]["exit_code"] == 0
    assert (state / "worker-result.json").read_bytes() == cpu._RESULT
    assert json.loads((state / "qualification-result.json").read_bytes()) == result
    with SQLiteDurableStoreV1(state / "workflow.sqlite3", create=False) as records:
        for stage in ("claim", "spawn", "outcome"):
            assert records.read(partition="effects", key=f"native-mlx/cpu-one/{stage}") is not None
        assert records.read(partition="effects", key="native-mlx/cpu-one/diagnostic") is None
    before = (state / "qualification-result.json").read_bytes()
    with pytest.raises(ValueError, match="already used"):
        cpu.run(str(config_path))
    assert (state / "qualification-result.json").read_bytes() == before


def test_dirty_source_and_existing_state_reject_before_claim(selection):
    config_path, _, source, state = selection
    worker = source / cpu._WORKER
    worker.write_bytes(worker.read_bytes() + b"# drift\n")
    with pytest.raises(ValueError, match="source selection differs"):
        cpu.inspect(str(config_path))
    assert not state.exists()
    subprocess.run(["git", "-C", str(source), "restore", "--", cpu._WORKER], check=True)
    state.mkdir(mode=0o700)
    (state / "preserved").write_bytes(b"existing")
    with pytest.raises(ValueError, match="already used"):
        cpu.run(str(config_path))
    assert (state / "preserved").read_bytes() == b"existing"


def test_guardian_failure_retains_closed_diagnostic_and_consumed_claim(selection):
    config_path, config, source, state = selection
    guardian = source / cpu._GUARDIAN
    guardian.write_text("raise SystemExit(123)\n")
    subprocess.run(["git", "-C", str(source), "add", cpu._GUARDIAN], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-qm", "fixed guardian failure"], check=True)
    config["engine_commit"] = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True,
    ).strip()
    config["guardian_sha256"] = hashlib.sha256(guardian.read_bytes()).hexdigest()
    config_path.write_bytes(canonical(config))
    with pytest.raises(ValueError, match="guardian outcome unavailable"):
        cpu.run(str(config_path))
    assert state.exists() and not (state / "qualification-result.json").exists()
    with SQLiteDurableStoreV1(state / "workflow.sqlite3", create=False) as records:
        for stage in ("claim", "spawn", "diagnostic"):
            assert records.read(partition="effects", key=f"native-mlx/cpu-one/{stage}") is not None
        assert records.read(partition="effects", key="native-mlx/cpu-one/outcome") is None
        raw = records.read(partition="effects", key="native-mlx/cpu-one/diagnostic").canonical
        assert json.loads(raw)["value"] == {"category": "guardian_exit", "exit_code": 123}
        assert b"source" not in raw and b"python" not in raw
    with pytest.raises(ValueError, match="already used"):
        cpu.run(str(config_path))


def test_cli_rejection_is_closed_and_does_not_create_state(selection, capsys):
    config_path, _, source, state = selection
    (source / cpu._WORKER).write_text("changed")
    assert cpu.main(["run", str(config_path)]) == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out)["status"] == "REJECTED_OR_UNCERTAIN"
    assert str(source) not in captured.out and not state.exists()
