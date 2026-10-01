from pathlib import Path
import importlib.util
import sys

import pytest

from tuner.runtime import packaged_sft_child as child
from tuner.runtime.packaged_sft_execution import _digest


def test_zero_argument_child_fails_closed(capsys):
    assert child.main() == 42
    assert "PACKAGED_SFT_CHILD_REJECTED" in capsys.readouterr().err


@pytest.mark.parametrize("phase", ("TRANSPORT", "RELEASE", "INPUT", "IMPORT", "EXEC", "POSTCHECK"))
@pytest.mark.parametrize("fault,category", (
    (OSError, "OS"), (ImportError, "IMPORT"), (ValueError, "VALUE"),
    (SystemExit, "SYSTEM_EXIT"), (KeyboardInterrupt, "OTHER"),
    (RuntimeError, "RUNTIME"), (TypeError, "TYPE"), (AttributeError, "ATTRIBUTE"),
    (KeyError, "KEY"), (MemoryError, "MEMORY"),
))
def test_packaged_child_phase_and_exception_are_fixed_without_failure_text(monkeypatch, capsys, phase, fault, category):
    from tuner.runtime.packaged_sft_execution import CHILD_EXIT_STAGES
    monkeypatch.setattr(sys, "argv", ["child"])
    def fail(_argv, current_phase):
        current_phase[0] = phase
        raise fault("PRIVATE_SENTINEL")
    monkeypatch.setattr(child, "_run_packaged_child", fail)
    exit_code = child.main()
    effective_category = category if phase == "EXEC" or category in child._FAILURE_CATEGORIES else "OTHER"
    assert CHILD_EXIT_STAGES[exit_code] == "TRAINER_CHILD_" + phase + "_" + effective_category
    assert capsys.readouterr().err == "PACKAGED_SFT_CHILD_REJECTED\n"


def test_packaged_child_keeps_clean_system_exit_zero(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["child"])
    monkeypatch.setattr(child, "_run_packaged_child", lambda _argv, _phase: (_ for _ in ()).throw(SystemExit(0)))
    assert child.main() == 0
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("failure", (
    child._ChildFailure(0), child._ChildFailure(True), child._ChildFailure(75),
    type("SpoofedFailure", (child._ChildFailure,), {})(40),
))
def test_spoofed_child_failure_cannot_return_success_or_unreserved_code(monkeypatch, capsys, failure):
    monkeypatch.setattr(sys, "argv", ["child"])
    monkeypatch.setattr(child, "run_packaged_child", lambda: (_ for _ in ()).throw(failure))
    assert child.main() == 2
    assert capsys.readouterr().err == "PACKAGED_SFT_CHILD_REJECTED\n"


def test_trainer_postcheck_failure_has_its_own_closed_phase_and_cleanup(monkeypatch, capsys):
    from types import SimpleNamespace
    from tuner.runtime import packaged_sft_execution as seam
    monkeypatch.setattr(sys, "argv", ["child"])
    checks = []
    def check():
        checks.append(True)
        if len(checks) == 2:
            raise OSError("PRIVATE_SENTINEL")
    private = SimpleNamespace(check=check, close=lambda: None)
    monkeypatch.setattr(seam, "_retain_private_snapshot", lambda *_: private)
    def run(_argv, _phase):
        child._run_private_trainer(compile("pass", "trainer-test", "exec"), Path("/installed/trainer.py"), {}, None, {}, classify=True)
    monkeypatch.setattr(child, "_run_packaged_child", run)
    assert seam.CHILD_EXIT_STAGES[child.main()] == "TRAINER_CHILD_POSTCHECK_OS"
    assert checks == [True, True]
    assert capsys.readouterr().err == "PACKAGED_SFT_CHILD_REJECTED\n"


@pytest.mark.parametrize("milestone", child._EXEC_RUNTIME_MILESTONES)
def test_runtime_milestone_maps_exact_runtime_error_without_text(monkeypatch, milestone):
    from types import SimpleNamespace
    from tuner.runtime import packaged_sft_execution as seam
    checks = []
    private = SimpleNamespace(check=lambda: checks.append(True), close=lambda: checks.append("close"))
    monkeypatch.setattr(seam, "_retain_private_snapshot", lambda *_: private)
    source = "__packaged_runtime_phase__(" + repr(milestone) + "); raise RuntimeError('PRIVATE_SENTINEL')"
    with pytest.raises(child._ChildFailure) as caught:
        child._run_private_trainer(compile(source, "trainer-test", "exec"), Path("/installed/trainer.py"), {}, None, {}, classify=True)
    assert caught.value.exit_code == 80 + child._EXEC_RUNTIME_MILESTONES.index(milestone)
    assert checks == [True, True, "close"]
    assert "PRIVATE_SENTINEL" not in str(caught.value)


@pytest.mark.parametrize("marker", ("UNKNOWN", 17, None))
def test_unknown_runtime_milestone_keeps_generic_runtime_code(monkeypatch, marker):
    from types import SimpleNamespace
    from tuner.runtime import packaged_sft_execution as seam
    private = SimpleNamespace(check=lambda: None, close=lambda: None)
    monkeypatch.setattr(seam, "_retain_private_snapshot", lambda *_: private)
    source = "__packaged_runtime_phase__(" + repr(marker) + "); raise RuntimeError('PRIVATE_SENTINEL')"
    with pytest.raises(child._ChildFailure) as caught:
        child._run_private_trainer(compile(source, "trainer-test", "exec"), Path("/installed/trainer.py"), {}, None, {}, classify=True)
    assert caught.value.exit_code == 70


def test_trainer_bootstrap_markers_precede_each_import_boundary():
    source = (Path(__file__).resolve().parents[2] / "Trainers/sft/train_sft.py").read_text(encoding="utf-8")
    for milestone, target in (
        ("BOOTSTRAP_ENV", "from shared.env_bootstrap import"),
        ("TORCH_IMPORT", "import torch  # noqa: E402"),
        ("UNSLOTH_IMPORT", "from unsloth import is_bfloat16_supported"),
        ("TRAINER_IMPORT", "from transformers import Trainer"),
    ):
        assert source.index('_mark_packaged_runtime_phase("' + milestone + '")') < source.index(target)
    assert source.count('_mark_packaged_runtime_phase("TRAIN_CALL")') == 2
    assert '_mark_packaged_runtime_phase("TRAIN_CALL")\n            trainer.train(' in source


@pytest.mark.parametrize("milestone", child._EXEC_RUNTIME_MILESTONES)
def test_unknown_library_error_keeps_callback_milestone_and_cleanup(monkeypatch, milestone):
    from types import SimpleNamespace
    from tuner.runtime import packaged_sft_execution as seam
    checks = []
    private = SimpleNamespace(check=lambda: checks.append(True), close=lambda: checks.append("close"))
    monkeypatch.setattr(seam, "_retain_private_snapshot", lambda *_: private)
    source = ("class LibraryError(RuntimeError):\n"
              "    def __str__(self): raise AssertionError('PRIVATE_SENTINEL')\n"
              "    def __getattribute__(self, name): raise AssertionError('PRIVATE_SENTINEL')\n"
              "__packaged_runtime_phase__(" + repr(milestone) + ")\n"
              "raise LibraryError('PRIVATE_SENTINEL')")
    with pytest.raises(child._ChildFailure) as caught:
        child._run_private_trainer(compile(source, "trainer-test", "exec"), Path("/installed/trainer.py"), {}, None, {}, classify=True)
    assert caught.value.exit_code == 100 + child._EXEC_RUNTIME_MILESTONES.index(milestone)
    assert checks == [True, True, "close"]


@pytest.mark.parametrize("milestone", (None, "UNKNOWN", 17, *child._EXEC_RUNTIME_MILESTONES))
def test_torch_oom_alias_identity_and_subclasses_without_diagnostic_import(monkeypatch, milestone):
    from types import ModuleType
    # Pinned torch/cuda/__init__.py aliases torch._C.OutOfMemoryError;
    # Exceptions.cpp creates that exception with RuntimeError as its base.
    extension, cuda = ModuleType("torch._C"), ModuleType("torch.cuda")
    oom = type("OutOfMemoryError", (RuntimeError,), {})
    extension.OutOfMemoryError = cuda.OutOfMemoryError = oom
    monkeypatch.setitem(sys.modules, "torch._C", extension)
    monkeypatch.setitem(sys.modules, "torch.cuda", cuda)
    expected = (120 + child._EXEC_RUNTIME_MILESTONES.index(milestone)
                if type(milestone) is str and milestone in child._EXEC_RUNTIME_MILESTONES else 76)
    for error in (oom("PRIVATE_SENTINEL"), type("DerivedOOM", (oom,), {})("PRIVATE_SENTINEL")):
        assert child._classified_failure("EXEC", error, milestone).exit_code == expected
    spoof = type("OutOfMemoryError", (RuntimeError,), {"__module__": "torch"})
    assert child._classified_failure("EXEC", spoof(), None).exit_code == 64
    cuda.OutOfMemoryError = spoof
    assert child._classified_failure("EXEC", oom(), None).exit_code == 64
    monkeypatch.delitem(sys.modules, "torch.cuda")
    assert child._classified_failure("EXEC", oom(), None).exit_code == 64
    assert child._classified_failure("IMPORT", oom(), "TRAIN_CALL").exit_code == 59


def test_real_loaded_torch_oom_identity_without_gpu_allocation():
    torch = pytest.importorskip("torch")
    assert torch.cuda.OutOfMemoryError is torch._C.OutOfMemoryError
    assert issubclass(torch.cuda.OutOfMemoryError, RuntimeError)
    assert child._classified_failure("EXEC", torch.cuda.OutOfMemoryError("PRIVATE_SENTINEL"), "TRAIN_CALL").exit_code == 130


@pytest.mark.parametrize("marker", ("UNKNOWN", 17, None))
def test_unknown_library_error_invalid_callback_clears_previous_milestone(monkeypatch, marker):
    from types import SimpleNamespace
    from tuner.runtime import packaged_sft_execution as seam
    monkeypatch.setattr(seam, "_retain_private_snapshot", lambda *_: SimpleNamespace(check=lambda: None, close=lambda: None))
    source = ("__packaged_runtime_phase__('TRAIN_CALL')\n"
              "__packaged_runtime_phase__(" + repr(marker) + ")\n"
              "class LibraryError(RuntimeError): pass\nraise LibraryError('PRIVATE_SENTINEL')")
    with pytest.raises(child._ChildFailure) as caught:
        child._run_private_trainer(compile(source, "trainer-test", "exec"), Path("/installed/trainer.py"), {}, None, {}, classify=True)
    assert caught.value.exit_code == 64


def test_new_stages_survive_worker_consumer_and_inspector_closed_contracts(monkeypatch):
    import pickle
    from types import SimpleNamespace
    from tuner.runtime.packaged_sft_execution import CHILD_FAILURE_STAGES
    from tuner.execution.providers.modal.packaged_worker import PACKAGED_WORKER_FAILURE_STAGES, packaged_worker_failure
    from tuner.training import modal_standalone_runner as consumer
    from scripts import inspect_modal_packaged_call as inspector
    expected = {"SFT_" + stage for stage in CHILD_FAILURE_STAGES}
    assert expected <= PACKAGED_WORKER_FAILURE_STAGES
    assert expected <= set(inspector._WORKER_FAILURE_STAGES)
    monkeypatch.setattr(inspector, "_pinned_python", lambda: True)
    api = SimpleNamespace(DATA_FORMAT_PICKLE=1)
    for stage in expected:
        result = packaged_worker_failure(stage)
        assert result["failure_stage"] == stage
        assert consumer.ModalStandalonePhaseUnavailable("RUN_WORKER_" + stage).phase == "RUN_WORKER_" + stage
        output = SimpleNamespace(data_format=1, result=SimpleNamespace(data_blob_id="", data=pickle.dumps(result)))
        assert inspector._classify_fixed_failure(output, api, pickle.dumps) == "WORKER_" + stage
        result["failure_stage"] = stage + "_PRIVATE_SENTINEL"
        output.result.data = pickle.dumps(result)
        assert inspector._classify_fixed_failure(output, api, pickle.dumps) == "PROVIDER_SUCCESS_UNKNOWN"


def test_direct_trainer_without_packaged_hook_preserves_existing_execution(monkeypatch):
    from types import SimpleNamespace
    from tuner.runtime import packaged_sft_execution as seam
    checks = []
    private = SimpleNamespace(check=lambda: checks.append(True), close=lambda: checks.append("close"))
    monkeypatch.setattr(seam, "_retain_private_snapshot", lambda *_: private)
    source = "assert '__packaged_runtime_phase__' not in globals()"
    child._run_private_trainer(compile(source, "trainer-test", "exec"), Path("/installed/trainer.py"), {}, None, {}, classify=False)
    assert checks == [True, True, "close"]


def test_unowned_import_directory_is_rejected(tmp_path, monkeypatch):
    owned = tmp_path / "owned.py"
    owned.write_text("x = 1")
    unowned = tmp_path / "hostile" / "configs.py"
    unowned.parent.mkdir()
    unowned.write_text("raise Exception('must not execute')")
    spec = importlib.util.spec_from_file_location("configs", unowned)
    monkeypatch.setattr(child.importlib.machinery.PathFinder, "find_spec", lambda *args: spec)
    guard = child._OwnedImportGuard({owned: _digest(owned.read_bytes())})
    with pytest.raises(ImportError, match="PACKAGED_CHILD_IMPORT_REJECTED"):
        guard.find_spec("configs")


def test_owned_source_loader_rejects_substitution(tmp_path):
    owned = tmp_path / "owned.py"
    owned.write_text("x = 1")
    loader = child._OwnedSourceLoader("owned", str(owned), _digest(owned.read_bytes()))
    owned.write_text("x = 2")
    with pytest.raises(ValueError, match="PACKAGED_CHILD_IMPORT_REJECTED"):
        loader.get_code("owned")


def test_child_forbids_ambient_dotenv_discovery():
    with pytest.raises(ImportError, match="PACKAGED_CHILD_DOTENV_REJECTED"):
        child._OwnedImportGuard({}).find_spec("dotenv")


@pytest.mark.parametrize("fault", [ValueError, SystemExit, KeyboardInterrupt, lambda _: child._ChildFailure(40)])
def test_local_diagnostic_dispatch_is_explicit_and_closed(monkeypatch, capsys, fault):
    monkeypatch.setattr(sys, "argv", ["child", "--qualify-local", "transport", "digest"])
    def reject(args):
        assert args == sys.argv[1:]
        raise fault("PRIVATE_SENTINEL")
    monkeypatch.setattr(child, "run_local_cpu_child", reject)
    monkeypatch.setattr(child, "run_packaged_child", lambda: pytest.fail("production path entered"))
    assert child.main() == 2
    assert capsys.readouterr().err == "PACKAGED_SFT_CHILD_REJECTED\n"


def test_diagnostic_environment_never_inherits_credentials(monkeypatch):
    from types import SimpleNamespace
    from tuner.runtime.packaged_training_worker import local_cpu_environment
    monkeypatch.setenv("HF_TOKEN", "private")
    monkeypatch.setenv("PYTHONPATH", "unowned")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "all")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    env = local_cpu_environment(SimpleNamespace(python_executable="/opt/python/bin/python"))
    assert env["PATH"] == "/opt/python/bin:/usr/local/bin:/usr/bin:/bin"
    assert "HF_TOKEN" not in env and "PYTHONPATH" not in env and "LD_LIBRARY_PATH" not in env
    assert env["NVIDIA_VISIBLE_DEVICES"] == "void" and env["CUDA_VISIBLE_DEVICES"] == ""


@pytest.mark.parametrize("names", [["null", "zero", "fd"], ["nvidia0"], ["nvidiactl"], ["nvidia-caps"], ["dri"], ["kfd"], ["null", "null"], [str(n) for n in range(4097)]])
def test_gpu_device_namespace_is_bounded_descriptor_relative_and_closed(monkeypatch, names):
    import stat
    from types import SimpleNamespace
    from contextlib import nullcontext
    calls = []
    info = SimpleNamespace(st_mode=stat.S_IFDIR | 0o755, st_dev=1, st_ino=2)
    def open_dev(path, flags):
        assert path == "/dev" and flags == 7
        calls.append("open-nofollow")
        return 42
    def scan(fd):
        assert fd == 42
        return nullcontext(iter(SimpleNamespace(name=name) for name in names))
    def named(path, *, follow_symlinks):
        assert path == "/dev" and follow_symlinks is False
        return info
    fake = SimpleNamespace(name="posix", O_RDONLY=1, O_DIRECTORY=2, O_NOFOLLOW=4,
        open=open_dev, fstat=lambda fd: info, stat=named, scandir=scan, close=lambda fd: calls.append("close"))
    monkeypatch.setattr(child, "os", fake)
    if names == ["null", "zero", "fd"]:
        child._check_no_gpu_devices()
    else:
        with pytest.raises(ValueError): child._check_no_gpu_devices()
    assert calls == ["open-nofollow", "close"]


@pytest.mark.parametrize("key,value", [("NVIDIA_VISIBLE_DEVICES", "all"), ("CUDA_VISIBLE_DEVICES", "0"), ("HF_TOKEN", "secret"), ("PYTHONPATH", "unowned")])
def test_local_cpu_child_rejects_visibility_override_and_unexpected_environment(monkeypatch, key, value):
    from types import SimpleNamespace
    from tuner.runtime.packaged_training_worker import local_cpu_environment
    release = SimpleNamespace(python_executable="/opt/python/bin/python")
    environment = local_cpu_environment(release)
    monkeypatch.setattr(child, "os", SimpleNamespace(environ=environment))
    child._check_local_cpu_environment(release)
    environment[key] = value
    with pytest.raises(ValueError): child._check_local_cpu_environment(release)


def test_gpu_device_namespace_rejects_linked_dev_root(monkeypatch):
    import stat
    from types import SimpleNamespace
    fake = SimpleNamespace(name="posix", O_RDONLY=1, O_DIRECTORY=2, O_NOFOLLOW=4,
        open=lambda *a: 42, fstat=lambda fd: SimpleNamespace(st_dev=1, st_ino=2),
        stat=lambda *a, **k: SimpleNamespace(st_mode=stat.S_IFLNK | 0o777, st_dev=1, st_ino=2),
        close=lambda fd: None, scandir=lambda fd: pytest.fail("must reject before scanning"))
    monkeypatch.setattr(child, "os", fake)
    with pytest.raises(ValueError): child._check_no_gpu_devices()


def test_gpu_device_namespace_rejects_root_replaced_during_scan(monkeypatch):
    import stat
    from types import SimpleNamespace
    from contextlib import nullcontext
    events = []
    original = SimpleNamespace(st_mode=stat.S_IFDIR | 0o755, st_dev=1, st_ino=2)
    replacement = SimpleNamespace(st_mode=stat.S_IFDIR | 0o755, st_dev=1, st_ino=3)
    def named(path, *, follow_symlinks):
        assert path == "/dev" and follow_symlinks is False
        events.append("stat")
        return original if events.count("stat") == 1 else replacement
    def scan(fd):
        assert fd == 42
        events.append("scan")
        return nullcontext(iter([SimpleNamespace(name="null")]))
    fake = SimpleNamespace(name="posix", O_RDONLY=1, O_DIRECTORY=2, O_NOFOLLOW=4,
        open=lambda *a: 42, fstat=lambda fd: original, stat=named, scandir=scan,
        close=lambda fd: events.append("close"))
    monkeypatch.setattr(child, "os", fake)
    with pytest.raises(ValueError): child._check_no_gpu_devices()
    assert events == ["stat", "scan", "stat", "close"]


@pytest.mark.parametrize("introduced", [None, "nvidia0", "dri", "kfd"])
def test_real_diagnostic_dispatch_checks_devices_before_and_after_compile(tmp_path, monkeypatch, capsys, introduced):
    """Portable boundary doubles; dispatch, device checks and source compilation are real."""
    import stat
    import json
    from types import SimpleNamespace
    from contextlib import nullcontext
    from tuner.runtime import packaged_training_worker as worker
    from tuner.runtime import packaged_sft_execution as seam
    from tuner.runtime.releases import PackagedTrainingRuntimeReleaseV1, PACKAGED_RUNTIME_RELEASE_SCHEMA
    events = []
    release = SimpleNamespace(manifest_digest="a" * 64, python_executable="/opt/python/bin/python",
                              canonical_bytes=lambda: b"release")
    monkeypatch.setattr(PackagedTrainingRuntimeReleaseV1, "from_dict", lambda _: release)
    monkeypatch.setattr(worker, "admit_packaged_training_release", lambda *a, **k: release)
    trainer = tmp_path / "trainer.py"
    trainer.write_text("raise AssertionError('trainer execution forbidden')\n")
    digest = _digest(trainer.read_bytes())
    monkeypatch.setattr(seam, "_inspect_release", lambda _: trainer)
    monkeypatch.setattr(child, "_installed_import_guard", lambda *a: child._OwnedImportGuard({trainer: digest}))
    root = Path("/tmp/qualification-fixture")
    transport = root / "transport.json"
    payload = {"schema_version": worker.LOCAL_CPU_PROTOCOL,
        "release": {"schema_version": PACKAGED_RUNTIME_RELEASE_SCHEMA}, "input_fd": 3,
        "root": str(root), "root_identity": [1, 2], "member": {"path": "fixture.json",
        "size_bytes": len(worker.LOCAL_CPU_MODEL), "sha256": _digest(worker.LOCAL_CPU_MODEL), "device": 1, "inode": 4}}
    raw = seam._canonical(payload)
    read = child.stable_read
    monkeypatch.setattr(child, "stable_read", lambda path, *args: raw if path == transport else read(path, *args))
    monkeypatch.setattr(child, "sys", SimpleNamespace(argv=["child", "--qualify-local", str(transport), _digest(raw)],
        flags=SimpleNamespace(isolated=1), stdout=sys.stdout, stderr=sys.stderr))
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(F_SEAL_SEAL=1, F_SEAL_SHRINK=2,
        F_SEAL_GROW=4, F_SEAL_WRITE=8, F_GET_SEALS=9, fcntl=lambda *a: 15))
    monkeypatch.setattr(seam, "_HeldDirectory", lambda path: SimpleNamespace(fd=100, identity=(1, 2), check=lambda: None, close=lambda: None))
    monkeypatch.setattr(seam, "_HeldModelFile", lambda *a: SimpleNamespace(check=lambda: None, close=lambda: None))
    root_info = SimpleNamespace(st_mode=stat.S_IFDIR | 0o700, st_dev=1, st_ino=2, st_uid=50)
    dev_info = SimpleNamespace(st_mode=stat.S_IFDIR | 0o755, st_dev=1, st_ino=5)
    def scan(fd):
        assert fd == 200
        events.append("check")
        names = ["null"] + ([introduced] if introduced and "compile" in events else [])
        return nullcontext(iter(SimpleNamespace(name=name) for name in names))
    def open_dev(path, flags):
        assert path == "/dev" and flags == 7
        return 200
    def named(path, *, follow_symlinks):
        assert path == "/dev" and follow_symlinks is False
        return dev_info
    fake = SimpleNamespace(name="posix", environ=worker.local_cpu_environment(release), getuid=lambda: 50,
        pread=lambda *a: worker.LOCAL_CPU_DATA, O_RDONLY=1, O_DIRECTORY=2, O_NOFOLLOW=4,
        open=open_dev, fstat=lambda fd: root_info if fd == 100 else dev_info,
        stat=named, scandir=scan, close=lambda fd: None)
    monkeypatch.setattr(child, "os", fake)
    compile_source = child._OwnedSourceLoader.get_code
    def compile_and_introduce_device(loader, fullname):
        code = compile_source(loader, fullname)
        events.append("compile")
        return code
    monkeypatch.setattr(child._OwnedSourceLoader, "get_code", compile_and_introduce_device)
    assert child.main() == (2 if introduced else 0)
    assert events == ["check", "compile", "check"]
    captured = capsys.readouterr()
    if introduced:
        assert captured.out == ""
        assert captured.err == "PACKAGED_SFT_CHILD_REJECTED\n"
    else:
        assert json.loads(captured.out)["gpu_device_namespace_absent"] is True
        assert captured.err == ""
    assert list(tmp_path.iterdir()) == [trainer]  # No diagnostic evidence or training files written.


@pytest.mark.parametrize("fault", [RuntimeError, SystemExit, KeyboardInterrupt])
def test_diagnostic_harness_closes_failure_text(monkeypatch, capsys, fault):
    import tuner.runtime.packaged_training_worker as worker
    def reject(_): raise fault("PRIVATE_SENTINEL")
    monkeypatch.setattr(worker, "qualify_installed_child", reject)
    assert worker.local_cpu_main({}) == 2
    assert capsys.readouterr().err == "PACKAGED_LOCAL_CPU_REJECTED\n"


@pytest.mark.skipif(sys.platform != "linux", reason="native Linux sealed-descriptor diagnostic protocol")
@pytest.mark.parametrize("mutation", [None, "environment", "transport", "snapshot", "source", "unsealed", "nvidia_env", "cuda_env"])
def test_native_local_child_diagnostic_authenticates_without_executing_ml(monkeypatch, capsys, mutation):
    import os
    import tempfile
    import fcntl
    from types import SimpleNamespace
    from tuner.runtime import packaged_training_worker as worker
    from tuner.runtime import packaged_sft_execution as seam
    from tuner.runtime.releases import PackagedTrainingRuntimeReleaseV1, PACKAGED_RUNTIME_RELEASE_SCHEMA
    release = SimpleNamespace(manifest_digest="a" * 64, python_executable=sys.executable,
                              canonical_bytes=lambda: b"release")
    monkeypatch.setattr(PackagedTrainingRuntimeReleaseV1, "from_dict", lambda _: release)
    monkeypatch.setattr(worker, "admit_packaged_training_release", lambda *a, **k: release)
    monkeypatch.setattr(sys, "flags", SimpleNamespace(isolated=1, optimize=0))
    monkeypatch.setattr(os, "environ", worker.local_cpu_environment(release))
    if mutation == "environment": os.environ["HF_TOKEN"] = "PRIVATE_SENTINEL"
    if mutation == "nvidia_env": os.environ["NVIDIA_VISIBLE_DEVICES"] = "all"
    if mutation == "cuda_env": os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    monkeypatch.setattr(child, "_check_no_gpu_devices", lambda: None)
    fd = os.memfd_create("qualification-test", os.MFD_ALLOW_SEALING)
    try:
        os.write(fd, worker.LOCAL_CPU_DATA)
        if mutation != "unsealed":
            fcntl.fcntl(fd, fcntl.F_ADD_SEALS, fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE)
        with tempfile.TemporaryDirectory(prefix="qualification-", dir="/tmp") as temporary:
            root = Path(temporary)
            trainer = root / "trainer.py"
            trainer.write_text("raise AssertionError('ML MUST NOT EXECUTE')\n")
            digest = _digest(trainer.read_bytes())
            monkeypatch.setattr(seam, "_inspect_release", lambda _: trainer)
            monkeypatch.setattr(child, "_installed_import_guard", lambda *a: child._OwnedImportGuard({trainer: digest}))
            fixture = root / "fixture.json"
            fixture.write_bytes(worker.LOCAL_CPU_MODEL if mutation != "snapshot" else b"substitution")
            fixture.chmod(0o400)
            info = fixture.stat()
            payload = {"schema_version": worker.LOCAL_CPU_PROTOCOL,
                "release": {"schema_version": PACKAGED_RUNTIME_RELEASE_SCHEMA}, "input_fd": fd,
                "root": str(root), "root_identity": [root.stat().st_dev, root.stat().st_ino],
                "member": {"path": "fixture.json", "size_bytes": len(worker.LOCAL_CPU_MODEL),
                    "sha256": _digest(worker.LOCAL_CPU_MODEL), "device": info.st_dev, "inode": info.st_ino}}
            raw = seam._canonical(payload)
            transport = root / "transport.json"
            transport.write_bytes(raw if mutation != "transport" else raw + b" ")
            if mutation == "source": trainer.write_text("raise AssertionError('replacement')\n")
            args = ["--qualify-local", str(transport), _digest(raw)]
            if mutation:
                with pytest.raises((ValueError, OSError)): child.run_local_cpu_child(args)
            else:
                assert child.run_local_cpu_child(args) == 0
                import json
                result = json.loads(capsys.readouterr().out)
                assert result["training_executed"] is False
                assert not (root / "packaged-terminal.json").exists()
    finally:
        os.close(fd)
