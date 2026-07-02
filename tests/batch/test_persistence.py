"""Persistence + resume + sync-hook contract tests (no model, CPU-only, fast).

Location: tests/batch/test_persistence.py

Covers the durability core of the batch feature independent of any engine:
atomic JSONL append, checkpoint config-hash gating, resume skip-by-id,
config-mismatch refusal, id sanitization, and the sync hook (fires with the
right env vars; a failing sync-cmd never fails the run).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tuner.batch.persistence import (  # noqa: E402
    ConfigMismatchError,
    JsonlAppender,
    RunCheckpoint,
    compute_config_hash,
    read_jsonl_ids,
    sanitize_id,
)
from tuner.batch.sync_hook import SyncHook  # noqa: E402


def test_config_hash_is_key_order_independent():
    a = compute_config_hash({"model": "m", "seed": 1, "engine": "hf-batched"})
    b = compute_config_hash({"engine": "hf-batched", "seed": 1, "model": "m"})
    assert a == b
    assert a != compute_config_hash({"model": "m", "seed": 2, "engine": "hf-batched"})


def test_jsonl_appender_writes_complete_lines(tmp_path):
    path = tmp_path / "out.jsonl"
    app = JsonlAppender(path)
    app.append({"id": "a", "v": 1})
    app.append_many([{"id": "b", "v": 2}, {"id": "c", "v": 3}])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert [json.loads(l)["id"] for l in lines] == ["a", "b", "c"]
    assert read_jsonl_ids(path) == ["a", "b", "c"]


def test_read_jsonl_ids_tolerates_torn_trailing_line(tmp_path):
    path = tmp_path / "out.jsonl"
    path.write_text('{"id": "a"}\n{"id": "b"}\n{"id": "c", "partia', encoding="utf-8")
    assert read_jsonl_ids(path) == ["a", "b"]


def test_sanitize_id_is_safe_and_collision_resistant():
    assert sanitize_id("plain-id_1") == "plain-id_1"
    a = sanitize_id("a/b:c")
    b = sanitize_id("a_b_c")
    # Both sanitize to something filesystem-safe, but must NOT collide.
    assert "/" not in a and ":" not in a
    assert a != b


def test_checkpoint_resume_skips_done_ids(tmp_path):
    config = {"model": "m", "engine": "hf-batched", "seed": 1}
    cp = RunCheckpoint.load_or_create(tmp_path, config, resume=False)
    cp.mark_done(["a", "b"])

    # Re-load with resume: the done set persists and skips are by id.
    cp2 = RunCheckpoint.load_or_create(tmp_path, config, resume=True)
    assert cp2.is_done("a") and cp2.is_done("b")
    assert not cp2.is_done("c")


def test_checkpoint_reconciles_index_ids_not_in_checkpoint(tmp_path):
    """A row in the JSONL index but not the checkpoint (crash between writes)
    still counts as done after reconciliation."""
    config = {"model": "m"}
    cp = RunCheckpoint.load_or_create(
        tmp_path, config, resume=False, index_ids=["x", "y"]
    )
    assert cp.is_done("x") and cp.is_done("y")


def test_resume_refuses_on_config_mismatch(tmp_path):
    RunCheckpoint.load_or_create(tmp_path, {"model": "m", "seed": 1}, resume=False)
    with pytest.raises(ConfigMismatchError) as exc:
        RunCheckpoint.load_or_create(tmp_path, {"model": "m", "seed": 2}, resume=True)
    assert "different configuration" in str(exc.value)


def test_fresh_run_into_existing_dir_refuses_without_resume(tmp_path):
    RunCheckpoint.load_or_create(tmp_path, {"model": "m"}, resume=False)
    with pytest.raises(ConfigMismatchError) as exc:
        RunCheckpoint.load_or_create(tmp_path, {"model": "m"}, resume=False)
    assert "--resume" in str(exc.value)


def test_sync_hook_fires_with_env_vars(tmp_path):
    marker = tmp_path / "synced.txt"
    # Portable: write the two env vars to a marker file.
    cmd = (
        f"python -c \"import os,pathlib; "
        f"pathlib.Path(r'{marker}').write_text("
        f"os.environ['TUNER_SYNC_DIR']+'|'+os.environ['TUNER_SYNC_REASON'])\""
    )
    hook = SyncHook(tmp_path, cmd, sync_every=2)
    hook.note_rows(1)
    assert not marker.exists()  # threshold not reached yet
    hook.note_rows(1)  # now 2 -> fires periodic
    assert marker.exists()
    content = marker.read_text()
    assert str(tmp_path) in content
    assert content.endswith("|periodic")

    hook.final()
    assert marker.read_text().endswith("|final")


def test_sync_hook_failure_does_not_raise(tmp_path):
    warnings = []
    hook = SyncHook(
        tmp_path, "exit 7", sync_every=1, warn=lambda m: warnings.append(m)
    )
    hook.note_rows(1)  # command exits non-zero
    hook.final()
    assert warnings  # warned, but did not raise
    assert any("exited 7" in w or "7" in w for w in warnings)


def test_sync_hook_disabled_when_no_cmd(tmp_path):
    hook = SyncHook(tmp_path, None, sync_every=1)
    assert not hook.enabled
    hook.note_rows(5)  # no-op
    hook.final()  # no-op
