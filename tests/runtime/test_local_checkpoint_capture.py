"""Trainer-local checkpoint records are complete, bounded, and non-authorizing."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from Trainers.sft import runtime_v1 as runtime

FINGERPRINT = "a" * 64


def _saved_step(root, step=10):
    checkpoint = root / "checkpoints" / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    for name in ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (checkpoint / name).write_bytes(name.encode("ascii"))
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "epoch": 0.5}), encoding="utf-8"
    )
    return checkpoint


def test_local_checkpoint_record_round_trip_and_mutation_rejection(tmp_path):
    checkpoint = _saved_step(tmp_path)
    record_path = runtime.record_local_checkpoint(tmp_path, 10, workload_fingerprint=FINGERPRINT)
    record = runtime.verify_local_checkpoint_record(tmp_path, 10, workload_fingerprint=FINGERPRINT)
    assert record_path.read_bytes() == runtime._canonical_json(record)
    assert record["step"] == 10
    assert {item["name"] for item in record["members"]} == {
        "adapter_model.safetensors", "optimizer.pt", "scheduler.pt",
        "rng_state.pth", "trainer_state.json",
    }
    with pytest.raises(FileExistsError):
        runtime.record_local_checkpoint(tmp_path, 10, workload_fingerprint=FINGERPRINT)
    (checkpoint / "optimizer.pt").write_bytes(b"changed")
    with pytest.raises(runtime.RuntimeV1Error, match="does not bind"):
        runtime.verify_local_checkpoint_record(tmp_path, 10, workload_fingerprint=FINGERPRINT)


def test_local_checkpoint_rejects_incomplete_and_oversize_before_record(tmp_path, monkeypatch):
    checkpoint = _saved_step(tmp_path)
    (checkpoint / "rng_state.pth").unlink()
    with pytest.raises(runtime.RuntimeV1Error, match="missing full trainer state"):
        runtime.record_local_checkpoint(tmp_path, 10, workload_fingerprint=FINGERPRINT)
    assert not (tmp_path / "checkpoint-catalog").exists()
    (checkpoint / "rng_state.pth").write_bytes(b"rng")
    monkeypatch.setattr(runtime, "_MAX_LOCAL_CHECKPOINT_MEMBER_BYTES", 3)
    with pytest.raises(runtime.RuntimeV1Error, match="bounded regular"):
        runtime.record_local_checkpoint(tmp_path, 10, workload_fingerprint=FINGERPRINT)


def test_local_checkpoint_rejects_symlink_and_wrong_step(tmp_path):
    checkpoint = _saved_step(tmp_path)
    (checkpoint / "trainer_state.json").write_text('{"global_step":9}', encoding="utf-8")
    with pytest.raises(runtime.RuntimeV1Error, match="step differs"):
        runtime.record_local_checkpoint(tmp_path, 10, workload_fingerprint=FINGERPRINT)
    (checkpoint / "trainer_state.json").write_text('{"global_step":10}', encoding="utf-8")
    (checkpoint / "optimizer.pt").unlink()
    try:
        (checkpoint / "optimizer.pt").symlink_to(checkpoint / "scheduler.pt")
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation unavailable")
    with pytest.raises(runtime.RuntimeV1Error):
        runtime.record_local_checkpoint(tmp_path, 10, workload_fingerprint=FINGERPRINT)


def test_local_progress_is_finite_exclusive_and_canonical(tmp_path):
    path = runtime.record_local_progress(
        tmp_path, workload_fingerprint=FINGERPRINT,
        step=5, max_steps=44, epoch=0.25, loss=2.5, learning_rate=1e-4
    )
    assert runtime.verify_local_progress_record(tmp_path, 5, workload_fingerprint=FINGERPRINT)["loss"] == 2.5
    assert path.read_bytes() == runtime._canonical_json(
        runtime.verify_local_progress_record(tmp_path, 5, workload_fingerprint=FINGERPRINT)
    )
    with pytest.raises(FileExistsError):
        runtime.record_local_progress(tmp_path, workload_fingerprint=FINGERPRINT,
                                      step=5, max_steps=44, epoch=0.25, loss=2.5)
    with pytest.raises(runtime.RuntimeV1Error, match="invalid"):
        runtime.record_local_progress(tmp_path, workload_fingerprint=FINGERPRINT,
                                      step=6, max_steps=44, epoch=0.3, loss=float("nan"))
    path.write_bytes(b'{"step":5}')
    with pytest.raises(runtime.RuntimeV1Error, match="invalid"):
        runtime.verify_local_progress_record(tmp_path, 5, workload_fingerprint=FINGERPRINT)


def test_actual_sft_callback_records_completed_save_and_loss_without_heavy_imports(tmp_path):
    """Exercise the callback source without importing the GPU training stack."""
    source = (Path(__file__).resolve().parents[2] / "Trainers/sft/train_sft.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    callback = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == "RuntimeV1LocalStateCallback")
    namespace = {"TrainerCallback": object, "Path": Path}
    exec(compile(ast.Module(body=[callback], type_ignores=[]), "trainer-callback", "exec"), namespace)
    instance = namespace["RuntimeV1LocalStateCallback"](tmp_path, FINGERPRINT)
    _saved_step(tmp_path)
    state = SimpleNamespace(is_world_process_zero=True, global_step=10,
                            max_steps=44, epoch=0.5)
    instance.on_log(None, state, None, logs={"loss": 2.0, "learning_rate": 1e-4})
    instance.on_save(None, state, None)
    assert runtime.verify_local_progress_record(tmp_path, 10, workload_fingerprint=FINGERPRINT)["loss"] == 2.0
    assert runtime.verify_local_checkpoint_record(tmp_path, 10, workload_fingerprint=FINGERPRINT)["step"] == 10
