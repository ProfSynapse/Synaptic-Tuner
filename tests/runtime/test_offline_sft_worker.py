from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tuner.runtime import offline_sft_worker
from tuner.runtime.offline_sft_worker import (
    OfflineSFTWorkerError,
    closure_digest,
    load_offline_sft_worker_closure,
)


_ROOT = Path(__file__).parents[2]
_MANIFEST = (
    _ROOT / "tuner" / "runtime" / "manifests" / "offline-sft-worker-v1.json"
)


def _stage(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    document = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    engine = (tmp_path / "engine").resolve()
    for member in document["members"]:
        relative = Path(*member["path"].split("/"))
        destination = engine / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_ROOT / relative, destination)
    control = (tmp_path / "control").resolve()
    control.mkdir()
    manifest = control / "offline-sft-worker-v1.json"
    shutil.copy2(_MANIFEST, manifest)
    return engine, manifest, document


def test_exact_selective_closure_loads(tmp_path: Path) -> None:
    engine, manifest, document = _stage(tmp_path)

    closure = load_offline_sft_worker_closure(
        manifest,
        expected_digest=document["closure_digest"],
        engine_root=engine,
    )

    assert len(closure.members) == 66
    assert closure.payload_bytes == document["payload_bytes"]
    assert closure.closure_digest == closure_digest(document)


def test_closure_rejects_changed_and_extra_members(tmp_path: Path) -> None:
    engine, manifest, document = _stage(tmp_path)
    target = engine / "shared" / "env_bootstrap.py"
    target.write_bytes(target.read_bytes() + b"\n")

    with pytest.raises(OfflineSFTWorkerError, match="does not match closure"):
        load_offline_sft_worker_closure(
            manifest,
            expected_digest=document["closure_digest"],
            engine_root=engine,
        )

    shutil.copy2(_ROOT / "shared" / "env_bootstrap.py", target)
    extra = engine / "shared" / "ambient.py"
    extra.write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(OfflineSFTWorkerError, match="exactly match"):
        load_offline_sft_worker_closure(
            manifest,
            expected_digest=document["closure_digest"],
            engine_root=engine,
        )


def test_closure_rejects_wrong_expected_digest(tmp_path: Path) -> None:
    engine, manifest, _ = _stage(tmp_path)

    with pytest.raises(OfflineSFTWorkerError, match="digest does not match"):
        load_offline_sft_worker_closure(
            manifest,
            expected_digest="0" * 64,
            engine_root=engine,
        )


def test_owned_import_guard_rejects_module_outside_closure(tmp_path: Path) -> None:
    engine, manifest, document = _stage(tmp_path)
    closure = load_offline_sft_worker_closure(
        manifest,
        expected_digest=document["closure_digest"],
        engine_root=engine,
    )
    guard = offline_sft_worker._OwnedModuleFinder(closure, engine)

    with pytest.raises(ModuleNotFoundError, match="outside"):
        guard.find_spec("shared.evolutionary", [str(engine / "shared")])
    with pytest.raises(ModuleNotFoundError, match="outside"):
        guard.find_spec("SynthChat", None)


def test_fixed_trainer_arguments_reject_optional_features() -> None:
    with pytest.raises(OfflineSFTWorkerError, match="feature is unavailable"):
        offline_sft_worker._validate_trainer_arguments(
            ["--evolutionary-enabled", "--max-steps", "1", "--no-load-in-4bit"]
        )


def test_isolated_bootstrap_rejects_optional_feature_without_traceback(
    tmp_path: Path,
) -> None:
    engine, manifest, document = _stage(tmp_path)
    environment = {
        **os.environ,
        "SYNAPTIC_ENGINE_ROOT": str(engine),
        "SYNAPTIC_WORKER_CLOSURE_MANIFEST": str(manifest),
        "SYNAPTIC_WORKER_CLOSURE_DIGEST": document["closure_digest"],
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
    }

    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            str(engine / "tuner" / "runtime" / "offline_sft_worker.py"),
            "--",
            "--evolutionary-enabled",
            "--max-steps",
            "1",
            "--no-load-in-4bit",
        ],
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert completed.stderr.strip() == "OFFLINE_SFT_WORKER_REJECTED"


_WORKER = "tuner/runtime/offline_sft_worker.py"


def _chmod_every_member(
    engine: Path, document: dict[str, object], mode: int
) -> None:
    for member in document["members"]:
        os.chmod(engine / Path(*member["path"].split("/")), mode)


def test_closure_authenticates_members_carrying_a_synthesized_execute_bit(
    tmp_path: Path,
) -> None:
    """T1 (section 23.4): git_mode 100644 presented as 0o744 authenticates.

    DrvFs with ``metadata`` stores a real POSIX mode only for Linux-written
    files and synthesizes 0744 for every Windows-written one, so on the
    mandated bind all 66 members carry the owner-execute bit while every one
    of them records 100644. Modes are set explicitly rather than inherited
    from the source tree so the fixture is deterministic on ext4.
    """

    engine, manifest, document = _stage(tmp_path)
    assert {member["git_mode"] for member in document["members"]} == {"100644"}
    _chmod_every_member(engine, document, 0o744)

    closure = load_offline_sft_worker_closure(
        manifest,
        expected_digest=document["closure_digest"],
        engine_root=engine,
    )

    assert len(closure.members) == len(document["members"])


def test_closure_rejects_a_longer_member_carrying_the_execute_bit(
    tmp_path: Path,
) -> None:
    """T2 (section 23.4): the size_bytes branch still rejects."""

    engine, manifest, document = _stage(tmp_path)
    target = engine / "shared" / "env_bootstrap.py"
    original = target.read_bytes()
    target.write_bytes(original + b"\n")
    os.chmod(target, 0o744)

    with pytest.raises(OfflineSFTWorkerError, match="does not match closure"):
        load_offline_sft_worker_closure(
            manifest,
            expected_digest=document["closure_digest"],
            engine_root=engine,
        )


def test_closure_rejects_an_edited_member_of_equal_length(
    tmp_path: Path,
) -> None:
    """T3 (section 23.4): the sha256 branch still rejects at equal length."""

    engine, manifest, document = _stage(tmp_path)
    target = engine / "shared" / "env_bootstrap.py"
    original = target.read_bytes()
    edited = original[:-1] + b" "
    assert len(edited) == len(original) and edited != original
    target.write_bytes(edited)
    os.chmod(target, 0o744)

    with pytest.raises(OfflineSFTWorkerError, match="does not match closure"):
        load_offline_sft_worker_closure(
            manifest,
            expected_digest=document["closure_digest"],
            engine_root=engine,
        )


def test_closure_authenticates_a_member_recording_100755_without_the_bit(
    tmp_path: Path,
) -> None:
    """T4 (section 23.4): the mode is compared in NEITHER direction.

    T1 covers a member recorded 100644 whose file is executable. This covers
    the opposite pairing, and it is the test that fails if the comparison is
    relaxed to tolerate only the DrvFs direction instead of being deleted.
    No member of the real closure records 100755, so the manifest is
    synthesized and its digest re-derived; the parser enforces
    recorded == observed == expected.
    """

    engine, manifest, document = _stage(tmp_path)
    for member in document["members"]:
        if member["path"] == _WORKER:
            member["git_mode"] = "100755"
            break
    else:  # pragma: no cover - the worker module is always a member
        raise AssertionError(f"{_WORKER} is not a closure member")
    document["closure_digest"] = closure_digest(document)
    manifest.write_bytes(
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    os.chmod(engine / Path(*_WORKER.split("/")), 0o644)

    closure = load_offline_sft_worker_closure(
        manifest,
        expected_digest=document["closure_digest"],
        engine_root=engine,
    )

    assert any(member.git_mode == "100755" for member in closure.members)
