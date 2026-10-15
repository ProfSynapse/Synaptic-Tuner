"""Real isolated GRPO failures and their closed, non-authorizing diagnostics."""
import sys
from types import ModuleType

import pytest

from tuner.runtime import packaged_grpo_execution as execution
from tuner.runtime import packaged_sft_execution as shared
from tests.runtime.test_packaged_grpo_child import _BOOTSTRAP, _ports, _prepare
from tests.runtime.test_packaged_grpo_native import material


@pytest.mark.parametrize("error,stage", [
    ("RuntimeError", "RUNTIME"), ("ValueError", "VALUE"),
    ("ImportError", "IMPORT"), ("TimeoutError", "TIMEOUT"),
    ("LookupError", "OTHER"),
])
def test_actual_isolated_model_failure_reaches_parent(tmp_path, monkeypatch, error, stage):
    injected = f'''
import unsloth
def rejected_model(**kwargs):
    raise {error}('PRIVATE_EXCEPTION_SENTINEL')
unsloth.FastLanguageModel.from_pretrained = rejected_model
'''
    bootstrap = _BOOTSTRAP.replace("grpo_child_fixture.install(source)",
                                   "grpo_child_fixture.install(source)\n" + injected)
    admitted, _ = material(tmp_path)
    commands = _ports(monkeypatch, bootstrap)
    with pytest.raises(execution.PackagedGRPOExecutionError) as caught:
        execution.execute_admitted_packaged_grpo(admitted, model_preparer=_prepare)
    expected = f"TRAINER_CHILD_EXEC_{stage}_MODEL_LIBRARY_LOAD"
    assert caught.value.stage == expected
    assert "PRIVATE_EXCEPTION_SENTINEL" not in str(caught.value)
    assert len(commands) == 1 and admitted._used
    assert not list(admitted.paths.artifacts.iterdir())
    assert not (admitted.paths.state / "packaged-terminal.json").exists()


def test_shared_oom_identity_is_required_not_class_name(monkeypatch):
    extension, cuda = ModuleType("torch._C"), ModuleType("torch.cuda")
    class RealOOM(RuntimeError):
        pass
    class OutOfMemoryError(RuntimeError):
        pass
    extension.OutOfMemoryError = cuda.OutOfMemoryError = RealOOM
    monkeypatch.setitem(sys.modules, "torch._C", extension)
    monkeypatch.setitem(sys.modules, "torch.cuda", cuda)
    for error, expected in [(RealOOM("private"), "TORCH_OOM"),
                            (OutOfMemoryError("private"), "OTHER")]:
        failure = execution._classified_grpo_failure("EXEC", error, "TRAIN_CALL")
        assert execution.GRPO_CHILD_EXIT_STAGES[failure.exit_code] == f"TRAINER_CHILD_EXEC_{expected}_TRAIN_CALL"
    cuda.OutOfMemoryError = OutOfMemoryError
    failure = execution._classified_grpo_failure("EXEC", RealOOM(), "TRAIN_CALL")
    assert "TORCH_OOM" not in execution.GRPO_CHILD_EXIT_STAGES[failure.exit_code]


@pytest.mark.parametrize("exit_code", [1, 255, -9])
def test_unrecognized_exit_stays_inconclusive(tmp_path, monkeypatch, exit_code):
    admitted, _ = material(tmp_path)
    _ports(monkeypatch)
    monkeypatch.setattr(shared, "_run_child_process", lambda *args, **kwargs: exit_code)
    with pytest.raises(execution.PackagedGRPOExecutionError) as caught:
        execution.execute_admitted_packaged_grpo(admitted, model_preparer=_prepare)
    assert caught.value.stage == "TRAINER"


def test_supervisor_timeout_is_separate_from_child_report(tmp_path, monkeypatch):
    admitted, _ = material(tmp_path)
    _ports(monkeypatch)
    def expired(*args, **kwargs):
        raise shared.subprocess.TimeoutExpired("private", 1)
    monkeypatch.setattr(shared, "_run_child_process", expired)
    with pytest.raises(execution.PackagedGRPOExecutionError) as caught:
        execution.execute_admitted_packaged_grpo(admitted, model_preparer=_prepare)
    assert caught.value.stage == "CHILD_DEADLINE"


def test_wire_tables_are_exact_and_exit_status_fits_byte():
    from tuner.execution.providers.modal import packaged_worker as worker
    from tuner.training import modal_host_reader as reader
    from scripts import inspect_modal_packaged_call as inspector
    expected = frozenset(
        "TRAINER_CHILD_" + phase + "_" + category
        for phase in ("TRANSPORT", "RELEASE", "INPUT", "IMPORT", "EXEC", "POSTCHECK")
        for category in ("OS", "IMPORT", "VALUE", "SYSTEM_EXIT", "OTHER")
    ) | frozenset("TRAINER_CHILD_EXEC_" + category for category in (
        "RUNTIME", "TYPE", "ATTRIBUTE", "KEY", "MEMORY", "TORCH_OOM", "TIMEOUT",
    )) | frozenset(
        "TRAINER_CHILD_EXEC_" + category + "_" + phase
        for phase in ("INPUT_ADMISSION", "REWARDS", "DEPENDENCIES", "CONFIG",
                      "MODEL_LIBRARY_LOAD", "LORA_ATTACH", "DATA_PREP", "TRAINER_SETUP",
                      "TRAIN_CALL", "MEASUREMENT_BEGIN", "MEASUREMENT_END",
                      "MEASUREMENT_RECORD", "SAVE", "POST_SAVE")
        for category in ("OS", "IMPORT", "VALUE", "SYSTEM_EXIT", "OTHER", "RUNTIME",
                         "TYPE", "ATTRIBUTE", "KEY", "MEMORY", "TORCH_OOM", "TIMEOUT")
    )
    assert execution.GRPO_CHILD_FAILURE_STAGES == expected
    assert worker.GRPO_CHILD_FAILURE_STAGES == reader.GRPO_CHILD_FAILURE_STAGES == inspector.GRPO_CHILD_FAILURE_STAGES == expected
    assert all(type(code) is int and 0 < code < 256 for code in execution.GRPO_CHILD_EXIT_STAGES)
    assert len(set(execution.GRPO_CHILD_EXIT_STAGES.values())) == len(execution.GRPO_CHILD_EXIT_STAGES)
    for stage in expected:
        issued = worker.packaged_worker_failure("GRPO_" + stage)
        assert issued["failure_stage"] == "GRPO_" + stage
        assert issued["failure_stage"] in reader._FIXED_WORKER_FAILURE_STAGES
        assert issued["failure_stage"] in inspector._WORKER_FAILURE_STAGES
    assert worker.packaged_worker_failure("GRPO_TRAINER_CHILD_EXEC_OTHER_PRIVATE")["failure_stage"] == "SFT_UNKNOWN"
