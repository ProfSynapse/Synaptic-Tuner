from __future__ import annotations

from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

import tuner.runtime.dispatch as dispatch_module
from tuner.cloud.runtime_layout import CloudRuntimeLayout, RuntimeMount
from tuner.runtime.dispatch import (
    DispatchSpec,
    EngineDispatcher,
    ProcessResult,
    SubprocessRunner,
    build_dispatch_invocation,
)
from tuner.training.methods.sft import compile_sft_workload
from tests.training.test_sft_compilation import _config, _execution_source


def _layout(tmp_path: Path) -> CloudRuntimeLayout:
    sources = tmp_path / "sources"
    writable = tmp_path / "writable"
    engine = sources / "engine"
    project = sources / "project"
    engine.mkdir(parents=True)
    entrypoint = engine / "Trainers" / "sft" / "runtime_v1.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("# staged fixture\n", encoding="utf-8")
    project.mkdir()
    mounts = tuple(
        RuntimeMount(name, writable / name, PurePosixPath("/scratch/task") / name, False)
        for name in ("artifacts", "state", "tracking", "cache", "tmp")
    )
    return CloudRuntimeLayout(
        engine=RuntimeMount(
            "engine", engine, PurePosixPath("/opt/product/modules/trainer-engine"), True
        ),
        project=RuntimeMount(
            "project", project, PurePosixPath("/srv/tenant/project"), True
        ),
        writable=mounts,
    )


def test_dispatch_uses_layout_roots_and_a_writable_process_cwd(tmp_path: Path) -> None:
    workload = compile_sft_workload(
        resolved_config=_config(), execution_source=_execution_source()
    )
    spec = DispatchSpec(
        workload=workload,
        layout=_layout(tmp_path),
        entrypoint=PurePosixPath(workload.entrypoint),
    )
    invocation = build_dispatch_invocation(spec)

    assert invocation.argv[1] == (
        "/opt/product/modules/trainer-engine/Trainers/sft/runtime_v1.py"
    )
    assert invocation.cwd == PurePosixPath("/scratch/task/tmp")
    assert invocation.cwd not in {
        PurePosixPath("/opt/product/modules/trainer-engine"),
        PurePosixPath("/srv/tenant/project"),
    }
    assert invocation.environment_map["SYNAPTIC_PROJECT_ROOT"] == "/srv/tenant/project"
    assert invocation.environment_map["PYTHONPATH"] == (
        "/opt/product/modules/trainer-engine"
    )
    assert invocation.environment_map["PYTHONNOUSERSITE"] == "1"
    assert invocation.environment_map["PYTHONSAFEPATH"] == "1"
    assert invocation.stdin == workload.canonical_bytes


def test_dispatcher_passes_one_deterministic_invocation_to_runner(tmp_path: Path) -> None:
    workload = compile_sft_workload(
        resolved_config=_config(), execution_source=_execution_source()
    )
    spec = DispatchSpec(workload, _layout(tmp_path), PurePosixPath(workload.entrypoint))
    calls = []

    class Runner:
        def run(self, invocation):
            calls.append(invocation)
            return ProcessResult(0, "ok", "")

    result = EngineDispatcher(Runner()).dispatch(spec)
    assert result.exit_code == 0
    assert calls == [build_dispatch_invocation(spec)]


def test_dispatch_rejects_redirected_staged_entrypoint(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    entrypoint = layout.engine.source / "Trainers" / "sft" / "runtime_v1.py"
    target = layout.engine.source / "actual.py"
    target.write_text("# actual\n", encoding="utf-8")
    entrypoint.unlink()
    try:
        entrypoint.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")
    workload = compile_sft_workload(
        resolved_config=_config(), execution_source=_execution_source()
    )
    with pytest.raises(ValueError, match="redirected"):
        build_dispatch_invocation(
            DispatchSpec(workload, layout, PurePosixPath(workload.entrypoint))
        )


def test_subprocess_runner_removes_hostile_python_environment(
    tmp_path: Path, monkeypatch
) -> None:
    workload = compile_sft_workload(
        resolved_config=_config(), execution_source=_execution_source()
    )
    invocation = build_dispatch_invocation(
        DispatchSpec(workload, _layout(tmp_path), PurePosixPath(workload.entrypoint))
    )
    observed = {}

    def fake_run(*args, **kwargs):
        observed.update(kwargs["env"])
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(dispatch_module.subprocess, "run", fake_run)
    SubprocessRunner(
        base_environment={
            "PATH": "fixture",
            "PYTHONPATH": "hostile",
            "PYTHONHOME": "hostile-home",
            "PYTHONUSERBASE": "hostile-user",
        }
    ).run(invocation)

    assert observed["PYTHONPATH"] == "/opt/product/modules/trainer-engine"
    assert observed["PYTHONNOUSERSITE"] == "1"
    assert observed["PYTHONSAFEPATH"] == "1"
    assert "PYTHONHOME" not in observed
    assert "PYTHONUSERBASE" not in observed
