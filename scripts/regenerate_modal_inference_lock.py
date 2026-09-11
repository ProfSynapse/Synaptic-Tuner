"""Refresh content hashes in the existing Modal inference runtime locks.

This is deliberately a hash-only maintainer.  It never creates either lock,
discovers source files, changes the reviewed inventory, or updates runtime and
dependency pins.  A bare invocation checks; ``--write`` refreshes only source
sizes/digests, closure totals/digest, and the containing inventory entry.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
import stat
import tempfile

RUNTIME_RELATIVE = "tuner/execution/providers/modal/inference-runtime.lock.json"
CLOSURE_RELATIVE = "tuner/execution/providers/modal/inference-worker-closure.json"
DEPENDENCY_RELATIVE = "tuner/execution/providers/modal/inference-dependencies.lock"
SOURCE_MEMBERS = (
    "Evaluator/__init__.py",
    "Evaluator/base_client.py",
    "Evaluator/chat_session.py",
    "Evaluator/config.py",
    "Evaluator/enums.py",
    "Evaluator/openai_compat_client.py",
    "Evaluator/owned_process.py",
    "Evaluator/protocols.py",
    "Evaluator/verified_vllm_chat.py",
    "Evaluator/vllm_client.py",
    "Evaluator/vllm_runtime.py",
    "synaptic_tuner/__init__.py",
    "synaptic_tuner/_version.py",
    "synaptic_tuner/api/__init__.py",
    "synaptic_tuner/api/v1/__init__.py",
    "synaptic_tuner/api/v1/_contract.py",
    "synaptic_tuner/api/v1/_timestamps.py",
    "synaptic_tuner/api/v1/planning.py",
    "synaptic_tuner/api/v1/providers.py",
    "synaptic_tuner/api/v1/results.py",
    "synaptic_tuner/api/v1/runs_facade.py",
    "synaptic_tuner/api/v1/training_facade.py",
    "synaptic_tuner/api/v1/training_input.py",
    "tuner/__init__.py",
    "tuner/cloud/__init__.py",
    "tuner/cloud/runtime_layout.py",
    "tuner/execution/__init__.py",
    "tuner/execution/_effect_executor.py",
    "tuner/execution/broker.py",
    "tuner/execution/contracts.py",
    "tuner/execution/coordinator_v1/__init__.py",
    "tuner/execution/coordinator_v1/coordinator.py",
    "tuner/execution/coordinator_v1/cursors.py",
    "tuner/execution/coordinator_v1/foundation.py",
    "tuner/execution/coordinator_v1/model.py",
    "tuner/execution/coordinator_v1/ports.py",
    "tuner/execution/coordinator_v1/state_machine.py",
    "tuner/execution/coordinator_v1/stores.py",
    "tuner/execution/evidence.py",
    "tuner/execution/foundation_v2/__init__.py",
    "tuner/execution/foundation_v2/authority.py",
    "tuner/execution/foundation_v2/canonical.py",
    "tuner/execution/foundation_v2/commands.py",
    "tuner/execution/foundation_v2/executors.py",
    "tuner/execution/foundation_v2/identities.py",
    "tuner/execution/foundation_v2/observations.py",
    "tuner/execution/foundation_v2/operations.py",
    "tuner/execution/foundation_v2/preparation.py",
    "tuner/execution/foundation_v2/receipts.py",
    "tuner/execution/foundation_v2/references.py",
    "tuner/execution/foundation_v2/repository.py",
    "tuner/execution/lifecycle.py",
    "tuner/execution/operation.py",
    "tuner/execution/providers/__init__.py",
    "tuner/execution/providers/contracts.py",
    "tuner/execution/providers/modal/__init__.py",
    "tuner/execution/providers/modal/binding.py",
    "tuner/execution/providers/modal/config.py",
    "tuner/execution/providers/modal/contracts.py",
    "tuner/execution/providers/modal/control.py",
    "tuner/execution/providers/modal/coordinator_adapter.py",
    "tuner/execution/providers/modal/coordinator_binding.py",
    "tuner/execution/providers/modal/coordinator_bundle.py",
    "tuner/execution/providers/modal/coordinator_dispatch.py",
    "tuner/execution/providers/modal/coordinator_launch.py",
    "tuner/execution/providers/modal/coordinator_preflight.py",
    "tuner/execution/providers/modal/coordinator_reader.py",
    "tuner/execution/providers/modal/coordinator_staging.py",
    "tuner/execution/providers/modal/coordinator_submit_preparation.py",
    "tuner/execution/providers/modal/coordinator_wire.py",
    "tuner/execution/providers/modal/deployment_identity.py",
    "tuner/execution/providers/modal/deployment_v1.py",
    "tuner/execution/providers/modal/facade.py",
    "tuner/execution/providers/modal/inference_artifacts.py",
    "tuner/execution/providers/modal/inference_binding.py",
    "tuner/execution/providers/modal/inference_bootstrap.py",
    "tuner/execution/providers/modal/inference_channel.py",
    "tuner/execution/providers/modal/inference_commands.py",
    "tuner/execution/providers/modal/inference_entrypoint.py",
    "tuner/execution/providers/modal/inference_model.py",
    "tuner/execution/providers/modal/inference_preparation.py",
    "tuner/execution/providers/modal/inference_runtime.py",
    "tuner/execution/providers/modal/inference_wire.py",
    "tuner/execution/providers/modal/inference_worker.py",
    "tuner/execution/providers/modal/inference_workload.py",
    "tuner/execution/providers/modal/manifest.py",
    "tuner/execution/providers/modal/model_snapshot.py",
    "tuner/execution/providers/modal/mounted_io.py",
    "tuner/execution/providers/modal/resolution.py",
    "tuner/execution/providers/modal/runtime.py",
    "tuner/execution/providers/modal/worker_ports.py",
    "tuner/execution/registry.py",
    "tuner/execution/service.py",
    "tuner/inference/__init__.py",
    "tuner/inference/retrieved_model.py",
    "tuner/inference/run_chat.py",
    "tuner/inference/serving_target.py",
    "tuner/project/__init__.py",
    "tuner/project/config_layers.py",
    "tuner/project/context.py",
    "tuner/project/errors.py",
    "tuner/project/execution_source.py",
    "tuner/project/git_verification.py",
    "tuner/project/manifest.py",
    "tuner/project/path_refs.py",
    "tuner/project/secrets.py",
    "tuner/project/source_bundle.py",
    "tuner/runtime/__init__.py",
    "tuner/runtime/artifacts.py",
    "tuner/runtime/dispatch.py",
    "tuner/runtime/offline_sft_worker.py",
    "tuner/runtime/verification.py",
    "tuner/training/__init__.py",
    "tuner/training/contracts.py",
    "tuner/training/coordinator_material.py",
    "tuner/training/methods/__init__.py",
    "tuner/training/methods/sft.py",
    "tuner/training/recipes.py",
)


class MaintenanceFault(RuntimeError):
    pass


def _repo_root() -> Path:
    source = Path(__file__)
    if source.is_symlink():
        raise MaintenanceFault("SCRIPT_IDENTITY_INVALID")
    script = source.resolve(strict=True)
    root = script.parents[1]
    if (
        script.parent.name != "scripts"
        or script.name != "regenerate_modal_inference_lock.py"
    ):
        raise MaintenanceFault("SCRIPT_IDENTITY_INVALID")
    return root


REPO_ROOT = _repo_root()


def _safe_reader():
    path = (
        Path(__file__)
        .resolve(strict=True)
        .with_name("regenerate_modal_runtime_lock.py")
    )
    spec = importlib.util.spec_from_file_location("_modal_runtime_lock_reader", path)
    if spec is None or spec.loader is None:
        raise MaintenanceFault("SAFE_READER_UNAVAILABLE")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._safe_regular_bytes, module.LockRegenerationError


_safe_regular_bytes, _SafeReadError = _safe_reader()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _read(root: Path, relative: str, maximum: int = 64 * 1024 * 1024) -> bytes:
    try:
        return _safe_regular_bytes(root, relative, maximum=maximum)[0]
    except _SafeReadError as exc:
        raise MaintenanceFault(str(exc)) from exc


def _member(root: Path, path: str) -> dict[str, object]:
    payload = _read(root, path)
    return {
        "path": path,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _load(payload: bytes) -> dict[str, object]:
    value = json.loads(payload.decode("utf-8"))
    if type(value) is not dict or payload != _canonical(value):
        raise MaintenanceFault("DOCUMENT_INVALID")
    return value


def _entries(value: object, expected_paths: tuple[str, ...]) -> None:
    if type(value) is not list or len(value) != len(expected_paths):
        raise MaintenanceFault("INVENTORY_INVALID")
    for raw, expected in zip(value, expected_paths, strict=True):
        if (
            type(raw) is not dict
            or set(raw) != {"path", "size_bytes", "sha256"}
            or raw["path"] != expected
            or type(raw["size_bytes"]) is not int
            or not 0 <= raw["size_bytes"] <= 64 * 1024 * 1024
            or type(raw["sha256"]) is not str
            or len(raw["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in raw["sha256"])
        ):
            raise MaintenanceFault("INVENTORY_ENTRY_INVALID")


def _production_manifest(root: Path, expected: bytes) -> None:
    """Apply production parsing in this single-process maintenance CLI.

    The root hook is restored unconditionally. This tool must not be embedded
    in a serving process or invoked concurrently with serving callbacks.
    """
    from tuner.execution.providers.modal import inference_runtime

    original = inference_runtime._runtime_root
    try:
        inference_runtime._runtime_root = lambda: root
        payload, _ = inference_runtime._manifest()
    finally:
        inference_runtime._runtime_root = original
    if payload != expected:
        raise MaintenanceFault("PRODUCTION_MANIFEST_MISMATCH")


def refresh(root: Path) -> tuple[bytes, bytes, bytes, bytes]:
    """Return current/runtime and refreshed/runtime+closure bytes."""
    runtime_before = _read(root, RUNTIME_RELATIVE, 1024 * 1024)
    closure_before = _read(root, CLOSURE_RELATIVE, 1024 * 1024)
    dependency = _read(root, DEPENDENCY_RELATIVE)
    runtime = _load(runtime_before)
    closure = _load(closure_before)
    try:
        _production_manifest(root, runtime_before)
    except Exception as exc:
        raise MaintenanceFault("EXISTING_RUNTIME_REJECTED") from exc
    expected_runtime_fields = {
        "schema_version",
        "base_registry_reference",
        "sdk_version",
        "python",
        "dependency_lock_path",
        "worker_closure_manifest_path",
        "distributions",
        "source_inventory",
    }
    expected_closure_fields = {
        "schema_version",
        "entrypoint",
        "member_count",
        "payload_bytes",
        "members",
        "closure_digest",
    }
    if (
        set(runtime) != expected_runtime_fields
        or set(closure) != expected_closure_fields
    ):
        raise MaintenanceFault("SCHEMA_INVALID")
    if (
        runtime["schema_version"] != "synaptic-modal-inference-runtime-lock/v1"
        or runtime["dependency_lock_path"] != DEPENDENCY_RELATIVE
        or runtime["worker_closure_manifest_path"] != CLOSURE_RELATIVE
        or closure["schema_version"] != "synaptic-modal-inference-worker-closure/v1"
        or closure["entrypoint"]
        != "tuner/execution/providers/modal/inference_bootstrap.py"
    ):
        raise MaintenanceFault("SCHEMA_INVALID")
    expected_inventory = tuple(
        sorted((*SOURCE_MEMBERS, DEPENDENCY_RELATIVE, CLOSURE_RELATIVE))
    )
    _entries(closure.get("members"), SOURCE_MEMBERS)
    _entries(runtime.get("source_inventory"), expected_inventory)
    if (
        SOURCE_MEMBERS != tuple(sorted(SOURCE_MEMBERS))
        or len(set(SOURCE_MEMBERS)) != 118
        or closure.get("member_count") != 118
        or type(closure.get("payload_bytes")) is not int
        or type(closure.get("closure_digest")) is not str
    ):
        raise MaintenanceFault("INVENTORY_INVALID")

    from tuner.execution.providers.modal.inference_runtime import _worker_closure

    try:
        _worker_closure(
            closure_before,
            runtime["source_inventory"],
            CLOSURE_RELATIVE,
            DEPENDENCY_RELATIVE,
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise MaintenanceFault("EXISTING_LOCK_REJECTED") from exc
    members = [_member(root, path) for path in SOURCE_MEMBERS]
    closure_new = dict(closure)
    closure_new["members"] = members
    closure_new["payload_bytes"] = sum(item["size_bytes"] for item in members)
    unsigned = dict(closure_new)
    unsigned.pop("closure_digest")
    closure_new["closure_digest"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
    closure_after = _canonical(closure_new)
    inventory_by_path = {
        item["path"]: dict(item) for item in runtime["source_inventory"]
    }
    for item in members:
        inventory_by_path[item["path"]] = dict(item)
    inventory_by_path[DEPENDENCY_RELATIVE] = {
        "path": DEPENDENCY_RELATIVE,
        "size_bytes": len(dependency),
        "sha256": hashlib.sha256(dependency).hexdigest(),
    }
    inventory_by_path[CLOSURE_RELATIVE] = {
        "path": CLOSURE_RELATIVE,
        "size_bytes": len(closure_after),
        "sha256": hashlib.sha256(closure_after).hexdigest(),
    }
    runtime_new = dict(runtime)
    runtime_new["source_inventory"] = [
        inventory_by_path[path] for path in expected_inventory
    ]
    runtime_after = _canonical(runtime_new)

    # Reuse the production closure validator after construction.  Its expected
    # inventory contract also proves the containing inventory and closure agree.
    if (
        _worker_closure(
            closure_after,
            runtime_new["source_inventory"],
            CLOSURE_RELATIVE,
            DEPENDENCY_RELATIVE,
        )
        != closure_new["closure_digest"]
    ):
        raise MaintenanceFault("PRODUCTION_VALIDATION_FAILED")
    return runtime_before, closure_before, runtime_after, closure_after


def _stage(path: Path, payload: bytes, mode: int) -> str:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        return temporary
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main(argv: list[str] | None = None, *, root: Path | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    repo = REPO_ROOT if root is None else root
    if root is None and str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        rb, cb, ra, ca = refresh(repo)
        if rb == ra and cb == ca:
            print(
                json.dumps({"status": "CURRENT", "member_count": 118}, sort_keys=True)
            )
            return 0
        if not args.write:
            print(
                "Modal inference locks are STALE; re-run with --write.", file=sys.stderr
            )
            return 3
        closure_path = repo / CLOSURE_RELATIVE
        runtime_path = repo / RUNTIME_RELATIVE
        # Re-read every source and both locks before mutation. The two atomic
        # replacements are deliberately closure-first: interruption leaves a
        # fail-closed mismatch, but this is not a transactional two-file write.
        if refresh(repo) != (rb, cb, ra, ca):
            raise MaintenanceFault("INPUT_CHANGED_BEFORE_WRITE")
        ct = None
        rt = None
        try:
            ct = _stage(closure_path, ca, stat.S_IMODE(closure_path.lstat().st_mode))
            rt = _stage(runtime_path, ra, stat.S_IMODE(runtime_path.lstat().st_mode))
            if (
                _read(repo, CLOSURE_RELATIVE, 1024 * 1024) != cb
                or _read(repo, RUNTIME_RELATIVE, 1024 * 1024) != rb
            ):
                raise MaintenanceFault("LOCK_CHANGED_BEFORE_REPLACE")
            os.replace(ct, closure_path)
            ct = None
            os.replace(rt, runtime_path)
            rt = None
        finally:
            for temporary in (ct, rt):
                if temporary is not None:
                    try:
                        os.unlink(temporary)
                    except FileNotFoundError:
                        pass
        if (
            _read(repo, CLOSURE_RELATIVE, 1024 * 1024) != ca
            or _read(repo, RUNTIME_RELATIVE, 1024 * 1024) != ra
        ):
            raise MaintenanceFault("VERIFY_AFTER_WRITE_FAILED")
    except (
        MaintenanceFault,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        UnicodeError,
    ) as exc:
        print(f"Modal inference lock maintenance failed: {exc}", file=sys.stderr)
        return 125
    print(json.dumps({"status": "REFRESHED", "member_count": 118}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
