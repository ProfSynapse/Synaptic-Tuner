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
