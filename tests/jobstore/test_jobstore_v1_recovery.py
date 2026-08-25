"""Two-lane cold-open and explicit WAL-recovery gates."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from synaptic_tuner._next_api_v1 import _store as store_module
from synaptic_tuner._next_api_v1._store import JobStore, RecoveryRequired

from .test_jobstore_v1 import Harness, NOW, SequentialIds, _start, harness


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cold_reopen_succeeds_but_ordinary_sidecar_requires_recovery_without_mutation(
    harness: Harness,
) -> None:
    run = _start(harness).run
    cold = JobStore(harness.store.location, clock=lambda: NOW, id_factory=SequentialIds())
    assert cold.status(harness.access, run).run == run

    wal = Path(str(harness.store_path) + "-wal")
    wal.write_bytes(b"retained-sidecar-evidence")
    before = {harness.store_path: _hash(harness.store_path), wal: _hash(wal)}
    with pytest.raises(RecoveryRequired, match="explicit recovery"):
        JobStore(harness.store.location, clock=lambda: NOW, id_factory=SequentialIds())
    assert {path: _hash(path) for path in before} == before


def test_explicit_private_recovery_refuses_and_preserves_genuine_retained_wal(
    harness: Harness, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _start(harness).run
    wal = Path(str(harness.store_path) + "-wal")
    shm = Path(str(harness.store_path) + "-shm")
    script = (
        "import os,sqlite3,sys;"
        "c=sqlite3.connect(sys.argv[1]);"
        "c.execute('PRAGMA journal_mode=WAL');"
        "c.execute('PRAGMA wal_autocheckpoint=0');"
        "c.execute('INSERT INTO cursors(token_digest,run_id,after_sequence,created_at) "
        "VALUES (?,?,?,?)',('" + "e" * 64
        + "',sys.argv[2],7,'2026-08-25T16:00:00Z'));"
        "c.commit();os._exit(0)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(harness.store_path), run.run_id],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        pytest.skip(
            "isolated SQLite subprocess could not construct retained WAL: "
            + completed.stderr.strip()
        )
    if not wal.exists() or wal.stat().st_size == 0:
        pytest.skip("platform SQLite checkpointed the WAL during abrupt subprocess exit")

    journal = Path(str(harness.store_path) + "-journal")
    evidence_paths = (harness.store_path, wal, shm, journal)

    def evidence() -> dict[Path, tuple[bool, int | None, str | None]]:
        return {
            path: (
                path.exists(),
                path.stat().st_size if path.exists() else None,
                _hash(path) if path.exists() else None,
            )
            for path in evidence_paths
        }

    evidence_before = evidence()
    with pytest.raises(RecoveryRequired):
        JobStore(harness.store.location, clock=lambda: NOW, id_factory=SequentialIds())
    assert evidence() == evidence_before

    connect_calls = 0

    def forbidden_connect(*_args, **_kwargs):
        nonlocal connect_calls
        connect_calls += 1
        raise AssertionError("refusal-only recovery must not open SQLite")

    monkeypatch.setattr(store_module.sqlite3, "connect", forbidden_connect)
    with pytest.raises(RecoveryRequired, match="offline snapshot/replay"):
        JobStore._recover_sidecars(harness.store.location)
    assert connect_calls == 0
    assert evidence() == evidence_before