"""Actual isolated owned control path with explicit CPU/image-library substitutes."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tuner.runtime import packaged_grpo_execution as runtime
from tuner.runtime import packaged_sft_execution as shared
from tuner.runtime.releases import parse_packaged_runtime_release
from tests.runtime.test_packaged_grpo_native import material

_BOOTSTRAP = """
import json,sys
from pathlib import Path
source=Path(sys.argv[1]); fixture=Path(sys.argv[2])
sys.path[:0]=[str(fixture),str(source)]
import grpo_child_fixture
from tuner.runtime import packaged_grpo_child as child
assert Path(child.__file__).absolute().parents[2] == source
grp_args=sys.argv[3:]
grpo_child_fixture.install(source)
sys.argv=['tuner.runtime.packaged_grpo_child',*grp_args]
classified=child._classified_failure
def diagnose(phase,error):
    locations=[]; trace=error.__traceback__
    while trace is not None:
        locations.append([Path(trace.tb_frame.f_code.co_filename).name,trace.tb_lineno])
        trace=trace.tb_next
    print(json.dumps({'phase':phase,'class':type(error).__name__,'locations':locations}),file=sys.stderr)
    return classified(phase,error)
child._classified_failure=diagnose
result=child.main()
payload=json.loads(Path(grp_args[1]).read_bytes())
paths=sorted({str(Path(module.__file__).absolute().relative_to(source))
 for module in tuple(sys.modules.values()) if getattr(module,'__file__',None)
 and Path(module.__file__).is_file() and Path(module.__file__).absolute().is_relative_to(source)
 and not Path(module.__file__).absolute().is_relative_to(source/'tests')})
Path(payload['paths']['tracking'],'cpu-owned-imports.json').write_text(json.dumps(paths))
raise SystemExit(result)
"""


def _ports(monkeypatch, bootstrap=_BOOTSTRAP):
    from tuner.runtime import packaged_training_worker as worker
    monkeypatch.setattr(worker, "admit_packaged_training_release", lambda raw, **kwargs:
                        parse_packaged_runtime_release(json.loads(raw)))
    monkeypatch.setattr(shared, "_inspect_parent_release", lambda *args, **kwargs: None)
    original = subprocess.Popen
    commands = []
    source = Path(runtime.__file__).absolute().parents[2]
    fixture = Path(__file__).absolute().parent
    def spawn(argv, **kwargs):
        assert tuple(argv[:5]) == (sys.executable, "-I", "-B", "-m", "tuner.runtime.packaged_grpo_child")
        commands.append(tuple(argv))
        # Test-only isolated bootstrap points to actual source/installed module
        # origin; no production Python is copied or found through checkout cwd.
        actual = (sys.executable, "-I", "-B", "-c", bootstrap, str(source), str(fixture), *argv[5:])
        return original(actual, **kwargs)
    monkeypatch.setattr(shared, "subprocess", SimpleNamespace(Popen=spawn, DEVNULL=subprocess.DEVNULL,
                                                              TimeoutExpired=subprocess.TimeoutExpired))
    return commands


def _prepare(model, cache):
    from Trainers.sft import runtime_v1 as utils
    snapshot = utils._model_snapshot_path(model, cache)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_bytes(b"{}")
    return snapshot


def test_real_isolated_child_core_and_native_five_roles(tmp_path, monkeypatch):
    admitted, compiled = material(tmp_path)
    commands = _ports(monkeypatch)
    try:
        result = runtime.execute_admitted_packaged_grpo(admitted, model_preparer=_prepare)
        assert len(commands) == 1
        assert result.workload_fingerprint == compiled.fingerprint
        assert [entry["role"] for entry in result.artifacts] == list(runtime.ROLES)
        for entry in result.artifacts:
            raw = (admitted.paths.artifacts / entry["path"]).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
        metric = json.loads((admitted.paths.artifacts / "training_metrics.json").read_bytes())
        assert metric["global_step"] == 3
        assert len(metric["log_history"]) == 3
        assert metric["train_metrics"]["train_loss"] == 1 / 3
        imported = json.loads((admitted.paths.tracking / "cpu-owned-imports.json").read_bytes())
        assert "Trainers/grpo/src/model_loader.py" in imported
        assert "shared/model_loading/merge.py" in imported
        from tuner.runtime.offline_grpo_worker import GRPO_OFFLINE_MEMBERS
        assert set(imported) <= set(GRPO_OFFLINE_MEMBERS)
        with pytest.raises(runtime.PackagedGRPOExecutionError):
            runtime.execute_admitted_packaged_grpo(admitted, model_preparer=lambda *args: pytest.fail("replay"))
        assert len(commands) == 1
    finally:
        admitted.close()


def test_changed_input_never_prepares_or_starts(tmp_path, monkeypatch):
    admitted, _ = material(tmp_path)
    commands = _ports(monkeypatch)
    admitted.paths.prepared_input.write_bytes(b"changed")
    try:
        with pytest.raises(runtime.PackagedGRPOExecutionError):
            runtime.execute_admitted_packaged_grpo(admitted, model_preparer=lambda *args: pytest.fail("prepare"))
        assert not commands
    finally:
        admitted.close()


@pytest.mark.parametrize("mutation", ("valid", "declared_hash", "unselected_wheel_python"))
def test_actual_selected_whole_wheel_guard_and_declared_source_filter(mutation):
    source = Path(runtime.__file__).absolute().parents[2]
    fixture = Path(__file__).absolute().parent
    script = (
        "import sys; from pathlib import Path; "
        "sys.path[:0]=[sys.argv[2],sys.argv[1]]; "
        "import grpo_child_fixture; "
        "grpo_child_fixture.check_installed_guard(Path(sys.argv[1]),sys.argv[3])"
    )
    result = subprocess.run((sys.executable, "-I", "-B", "-c", script, str(source), str(fixture), mutation),
                            capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")


def test_child_close_failure_attempts_other_descriptors_resources_and_guard(tmp_path, monkeypatch):
    injected = '''
import os
payload_for_fault=json.loads(Path(grp_args[1]).read_bytes())
original_close=os.close
failed_fd=payload_for_fault['dataset_fd']
closed=[]
def fail_one_close(fd):
    closed.append(fd)
    original_close(fd)
    if fd == failed_fd:
        raise OSError('test_close_failure')
os.close=fail_one_close
original_resources=child.shared._close_resources
resource_paths=[]
def observe_resources(resources):
    resource_paths.extend(str(getattr(value,'path',getattr(value,'snapshot',''))) for value in resources)
    return original_resources(resources)
child.shared._close_resources=observe_resources
result=child.main()
os.close=original_close
assert result != 0
assert payload_for_fault['metric_fd'] in closed
assert len(resource_paths) >= 3
assert not any(type(item).__name__ == '_OwnedImportGuard' for item in sys.meta_path)
Path(payload_for_fault['paths']['tracking'],'cleanup-control.json').write_text(json.dumps({'closed':True,'resources':len(resource_paths)}))
'''
    bootstrap = _BOOTSTRAP.replace("result=child.main()", injected)
    admitted, _ = material(tmp_path)
    commands = _ports(monkeypatch, bootstrap)
    try:
        with pytest.raises(runtime.PackagedGRPOExecutionError) as caught:
            runtime.execute_admitted_packaged_grpo(admitted, model_preparer=_prepare)
        assert caught.value.stage == "TRAINER"
        assert len(commands) == 1
        assert json.loads((admitted.paths.tracking / "cleanup-control.json").read_bytes())["closed"]
        assert not (admitted.paths.state / "packaged-terminal.json").exists()
        assert not list(admitted.paths.artifacts.iterdir())
    finally:
        admitted.close()


def test_parent_close_failure_attempts_custody_and_admitted_cleanup(tmp_path, monkeypatch):
    admitted, compiled = material(tmp_path)
    from Trainers.sft import runtime_v1 as utils

    # Stop after both IPC descriptors have been allocated, before process spawn.
    monkeypatch.setattr(runtime, "_admit_contracts", lambda *args: compiled)
    monkeypatch.setattr(shared, "_inspect_parent_release", lambda *args, **kwargs: None)
    original_close = runtime.os.close
    descriptor = None
    original_seal = utils._sealed_prepared_dataset
    def seal(*args, **kwargs):
        nonlocal descriptor
        result = original_seal(*args, **kwargs)
        descriptor = result[0]
        return result
    monkeypatch.setattr(utils, "_sealed_prepared_dataset", seal)
    calls = []
    def close(fd):
        calls.append(fd)
        original_close(fd)
        if fd == descriptor:
            raise OSError("test_close_failure")
    monkeypatch.setattr(runtime.os, "close", close)
    def reject(*args, **kwargs):
        raise ValueError("test_no_spawn")
    monkeypatch.setattr(runtime, "_invocation_spec", reject)
    try:
        with pytest.raises(runtime.PackagedGRPOExecutionError) as caught:
            runtime.execute_admitted_packaged_grpo(admitted, model_preparer=_prepare)
        assert caught.value.stage == "CLEANUP_UNCERTAIN"
        assert descriptor in calls
        assert len(set(calls)) >= 3
        assert admitted._used
        assert all(directory.fd is None for directory in admitted._directories)
    finally:
        monkeypatch.setattr(runtime.os, "close", original_close)
        admitted.close()
