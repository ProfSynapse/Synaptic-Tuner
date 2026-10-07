"""Exact source-tree and closure staging mechanics for Modal workers."""

from __future__ import annotations

import hashlib
from pathlib import Path

from tuner.project.execution_source import ExecutionSourceV1
from tuner.runtime.offline_sft_worker import (
    OFFLINE_SFT_MANIFEST_NAME, load_offline_sft_worker_closure,
    parse_offline_sft_worker_manifest,
)

from .mounted_io import read_regular
from .worker_ports import ModalRemotePhaseError


def read_locked_closure_manifest(source: ExecutionSourceV1) -> bytes:
    return (
        Path(source.roots["engine"]) / "tuner" / "runtime" / "manifests"
        / OFFLINE_SFT_MANIFEST_NAME
    ).read_bytes()


def write_runtime_closure_manifest(path: str, payload: bytes) -> None:
    runtime_manifest = Path(path)
    if not runtime_manifest.is_absolute() or runtime_manifest.resolve() != runtime_manifest:
        raise ModalRemotePhaseError(124, "worker_control_path_noncanonical")
    runtime_manifest.parent.mkdir(parents=True, exist_ok=True)
    if runtime_manifest.parent.resolve(strict=True) != runtime_manifest.parent:
        raise ModalRemotePhaseError(124, "worker_control_path_noncanonical")
    with runtime_manifest.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    if runtime_manifest.read_bytes() != payload:
        raise OSError("manifest round trip mismatch")


def stage_runtime_worker(source: ExecutionSourceV1, manifest_path: str, payload: bytes) -> None:
    engine = Path(source.roots["engine"])
    manifest = parse_offline_sft_worker_manifest(
        payload, source_ref="verified-engine-closure", manifest_path=Path(manifest_path),
    )
    if engine.resolve(strict=True) != engine:
        raise ModalRemotePhaseError(124, "worker_source_path_noncanonical")
    if Path(manifest_path).resolve(strict=True) != Path(manifest_path):
        raise ModalRemotePhaseError(124, "worker_control_path_noncanonical")
    retained = engine.with_name(engine.name + "-source")
    retained.mkdir(exist_ok=False)
    checkout = retained / "checkout"
    try:
        engine.rename(checkout)
        engine.mkdir(exist_ok=False)
    except FileExistsError:
        raise
    except OSError:
        raise ModalRemotePhaseError(124, "worker_source_retain_failed") from None
    try:
        for member in manifest.closure.members:
            contents = read_regular(checkout, checkout / member.path, member.size_bytes)
            if len(contents) != member.size_bytes or hashlib.sha256(contents).hexdigest() != member.sha256:
                raise ValueError("worker source member differs from locked closure")
            destination = engine / member.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(contents)
            destination.chmod(0o755 if member.git_mode == "100755" else 0o644)
    except FileExistsError:
        raise
    except Exception:
        raise ModalRemotePhaseError(124, "worker_source_copy_failed") from None
    try:
        load_offline_sft_worker_closure(
            Path(manifest_path), expected_digest=manifest.closure.closure_digest,
            engine_root=engine,
        )
    except Exception:
        raise ModalRemotePhaseError(124, "worker_closure_rejected") from None


__all__: list[str] = []
