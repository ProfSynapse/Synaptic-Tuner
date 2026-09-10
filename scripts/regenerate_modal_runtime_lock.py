"""Check or refresh only source hashes in the packaged Modal runtime lock.

The default is read-only. ``--write`` atomically replaces the lock after every
declared source has been safely read and the lock has been rechecked for races.
This tool performs no provider, SDK, network, package, or image operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile


LOCK_RELATIVE = "tuner/execution/providers/modal/modal-runtime-v1.lock.json"
LOCKED_FILES = {
    "bootstrap__synaptic_tuner____init___py": "synaptic_tuner/__init__.py",
    "bootstrap__synaptic_tuner___version_py": "synaptic_tuner/_version.py",
    "bootstrap__synaptic_tuner__api____init___py": "synaptic_tuner/api/__init__.py",
    "bootstrap__synaptic_tuner__api__v1____init___py": "synaptic_tuner/api/v1/__init__.py",
    "bootstrap__synaptic_tuner__api__v1___contract_py": "synaptic_tuner/api/v1/_contract.py",
    "bootstrap__synaptic_tuner__api__v1___timestamps_py": "synaptic_tuner/api/v1/_timestamps.py",
    "bootstrap__synaptic_tuner__api__v1__context_py": "synaptic_tuner/api/v1/context.py",
    "bootstrap__synaptic_tuner__api__v1__execution_py": "synaptic_tuner/api/v1/execution.py",
    "bootstrap__synaptic_tuner__api__v1__planning_py": "synaptic_tuner/api/v1/planning.py",
    "bootstrap__synaptic_tuner__api__v1__providers_py": "synaptic_tuner/api/v1/providers.py",
    "bootstrap__synaptic_tuner__api__v1__results_py": "synaptic_tuner/api/v1/results.py",
    "bootstrap__synaptic_tuner__api__v1__runs_facade_py": "synaptic_tuner/api/v1/runs_facade.py",
    "bootstrap__synaptic_tuner__api__v1__sources_py": "synaptic_tuner/api/v1/sources.py",
    "bootstrap__synaptic_tuner__api__v1__training_facade_py": "synaptic_tuner/api/v1/training_facade.py",
    "bootstrap__synaptic_tuner__api__v1__training_input_py": "synaptic_tuner/api/v1/training_input.py",
    "bootstrap__tuner____init___py": "tuner/__init__.py",
    "bootstrap__tuner__cloud____init___py": "tuner/cloud/__init__.py",
    "bootstrap__tuner__cloud__runtime_layout_py": "tuner/cloud/runtime_layout.py",
    "bootstrap__tuner__execution____init___py": "tuner/execution/__init__.py",
    "bootstrap__tuner__execution___effect_executor_py": "tuner/execution/_effect_executor.py",
    "bootstrap__tuner__execution__broker_py": "tuner/execution/broker.py",
    "bootstrap__tuner__execution__contracts_py": "tuner/execution/contracts.py",
    "bootstrap__tuner__execution__coordinator_v1____init___py": "tuner/execution/coordinator_v1/__init__.py",
    "bootstrap__tuner__execution__coordinator_v1__coordinator_py": "tuner/execution/coordinator_v1/coordinator.py",
    "bootstrap__tuner__execution__coordinator_v1__cursors_py": "tuner/execution/coordinator_v1/cursors.py",
    "bootstrap__tuner__execution__coordinator_v1__foundation_py": "tuner/execution/coordinator_v1/foundation.py",
    "bootstrap__tuner__execution__coordinator_v1__model_py": "tuner/execution/coordinator_v1/model.py",
    "bootstrap__tuner__execution__coordinator_v1__ports_py": "tuner/execution/coordinator_v1/ports.py",
    "bootstrap__tuner__execution__coordinator_v1__state_machine_py": "tuner/execution/coordinator_v1/state_machine.py",
    "bootstrap__tuner__execution__evidence_py": "tuner/execution/evidence.py",
    "bootstrap__tuner__execution__foundation_v2____init___py": "tuner/execution/foundation_v2/__init__.py",
    "bootstrap__tuner__execution__foundation_v2__authority_py": "tuner/execution/foundation_v2/authority.py",
    "bootstrap__tuner__execution__foundation_v2__canonical_py": "tuner/execution/foundation_v2/canonical.py",
    "bootstrap__tuner__execution__foundation_v2__commands_py": "tuner/execution/foundation_v2/commands.py",
    "bootstrap__tuner__execution__foundation_v2__executors_py": "tuner/execution/foundation_v2/executors.py",
    "bootstrap__tuner__execution__foundation_v2__identities_py": "tuner/execution/foundation_v2/identities.py",
    "bootstrap__tuner__execution__foundation_v2__observations_py": "tuner/execution/foundation_v2/observations.py",
    "bootstrap__tuner__execution__foundation_v2__operations_py": "tuner/execution/foundation_v2/operations.py",
    "bootstrap__tuner__execution__foundation_v2__preparation_py": "tuner/execution/foundation_v2/preparation.py",
    "bootstrap__tuner__execution__foundation_v2__receipts_py": "tuner/execution/foundation_v2/receipts.py",
    "bootstrap__tuner__execution__foundation_v2__references_py": "tuner/execution/foundation_v2/references.py",
    "bootstrap__tuner__execution__foundation_v2__repository_py": "tuner/execution/foundation_v2/repository.py",
    "bootstrap__tuner__execution__lifecycle_py": "tuner/execution/lifecycle.py",
    "bootstrap__tuner__execution__operation_py": "tuner/execution/operation.py",
    "bootstrap__tuner__execution__providers____init___py": "tuner/execution/providers/__init__.py",
    "bootstrap__tuner__execution__providers__contracts_py": "tuner/execution/providers/contracts.py",
    "bootstrap__tuner__execution__providers__modal____init___py": "tuner/execution/providers/modal/__init__.py",
    "bootstrap__tuner__execution__providers__modal__binding_py": "tuner/execution/providers/modal/binding.py",
    "bootstrap__tuner__execution__providers__modal__config_py": "tuner/execution/providers/modal/config.py",
    "bootstrap__tuner__execution__providers__modal__contracts_py": "tuner/execution/providers/modal/contracts.py",
    "bootstrap__tuner__execution__providers__modal__control_py": "tuner/execution/providers/modal/control.py",
    "bootstrap__tuner__execution__providers__modal__coordinator_adapter_py": "tuner/execution/providers/modal/coordinator_adapter.py",
    "bootstrap__tuner__execution__providers__modal__coordinator_binding_py": "tuner/execution/providers/modal/coordinator_binding.py",
    "bootstrap__tuner__execution__providers__modal__coordinator_bundle_py": "tuner/execution/providers/modal/coordinator_bundle.py",
    "bootstrap__tuner__execution__providers__modal__coordinator_dispatch_py": "tuner/execution/providers/modal/coordinator_dispatch.py",
    "bootstrap__tuner__execution__providers__modal__coordinator_logs_py": "tuner/execution/providers/modal/coordinator_logs.py",
    "bootstrap__tuner__execution__providers__modal__coordinator_producer_py": "tuner/execution/providers/modal/coordinator_producer.py",
    "bootstrap__tuner__execution__providers__modal__coordinator_wire_py": "tuner/execution/providers/modal/coordinator_wire.py",
    "bootstrap__tuner__execution__providers__modal__coordinator_worker_py": "tuner/execution/providers/modal/coordinator_worker.py",
    "bootstrap__tuner__execution__providers__modal__deployment_identity_py": "tuner/execution/providers/modal/deployment_identity.py",
    "bootstrap__tuner__execution__providers__modal__deployment_v1_py": "tuner/execution/providers/modal/deployment_v1.py",
    "bootstrap__tuner__execution__providers__modal__facade_py": "tuner/execution/providers/modal/facade.py",
    "bootstrap__tuner__execution__providers__modal__manifest_py": "tuner/execution/providers/modal/manifest.py",
    "bootstrap__tuner__execution__providers__modal__resolution_py": "tuner/execution/providers/modal/resolution.py",
    "bootstrap__tuner__execution__registry_py": "tuner/execution/registry.py",
    "bootstrap__tuner__execution__service_py": "tuner/execution/service.py",
    "bootstrap__tuner__project____init___py": "tuner/project/__init__.py",
    "bootstrap__tuner__project__config_layers_py": "tuner/project/config_layers.py",
    "bootstrap__tuner__project__context_py": "tuner/project/context.py",
    "bootstrap__tuner__project__errors_py": "tuner/project/errors.py",
    "bootstrap__tuner__project__execution_source_py": "tuner/project/execution_source.py",
    "bootstrap__tuner__project__git_verification_py": "tuner/project/git_verification.py",
    "bootstrap__tuner__project__manifest_py": "tuner/project/manifest.py",
    "bootstrap__tuner__project__path_refs_py": "tuner/project/path_refs.py",
    "bootstrap__tuner__project__secrets_py": "tuner/project/secrets.py",
    "bootstrap__tuner__project__source_bundle_py": "tuner/project/source_bundle.py",
    "bootstrap__tuner__runtime____init___py": "tuner/runtime/__init__.py",
    "bootstrap__tuner__runtime__artifacts_py": "tuner/runtime/artifacts.py",
    "bootstrap__tuner__runtime__dispatch_py": "tuner/runtime/dispatch.py",
    "bootstrap__tuner__runtime__manifests__offline-sft-worker-v1_json": "tuner/runtime/manifests/offline-sft-worker-v1.json",
    "bootstrap__tuner__runtime__offline_sft_worker_py": "tuner/runtime/offline_sft_worker.py",
    "bootstrap__tuner__training____init___py": "tuner/training/__init__.py",
    "bootstrap__tuner__training__contracts_py": "tuner/training/contracts.py",
    "bootstrap__tuner__training__coordinator_material_py": "tuner/training/coordinator_material.py",
    "bootstrap__tuner__training__methods____init___py": "tuner/training/methods/__init__.py",
    "bootstrap__tuner__training__methods__sft_py": "tuner/training/methods/sft.py",
    "bootstrap__tuner__training__recipes_py": "tuner/training/recipes.py",
    "bootstrap__tuner__training__resolution_py": "tuner/training/resolution.py",
    "bootstrap__tuner__training__service_py": "tuner/training/service.py",
    "dependency_lock": "requirements/modal-launcher-v1.lock",
    "deployment_wrapper": "tuner/execution/providers/modal/coordinator_deployment.py",
    "modal_mounted_io": "tuner/execution/providers/modal/mounted_io.py",
    "modal_runtime": "tuner/execution/providers/modal/runtime.py",
    "modal_worker_ports": "tuner/execution/providers/modal/worker_ports.py",
    "modal_worker_source": "tuner/execution/providers/modal/worker_source.py",
    "model_preparation": "tuner/execution/providers/modal/model_snapshot.py",
    "sft_runtime": "Trainers/sft/runtime_v1.py",
}
MAX_LOCK_BYTES = 256 * 1024
MAX_SOURCE_BYTES = 16 * 1024 * 1024


class LockRegenerationError(RuntimeError):
    pass


_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def _reject_reparse(path: Path) -> os.stat_result:
    observed = path.lstat()
    if getattr(observed, "st_file_attributes", 0) & _REPARSE_POINT:
        raise LockRegenerationError("REPARSE_POINT_REFUSED")
    return observed


def _repo_root() -> Path:
    source = Path(__file__)
    if source.is_symlink():
        raise LockRegenerationError("SCRIPT_IDENTITY_INVALID")
    script = source.resolve(strict=True)
    root = script.parents[1]
    if (script.parent.name != "scripts" or script.name != Path(__file__).name
            or (root / "scripts" / script.name).resolve(strict=True) != script):
        raise LockRegenerationError("SCRIPT_IDENTITY_INVALID")
    return root


def _safe_regular_bytes(root: Path, relative: str, *, maximum: int) -> tuple[bytes, os.stat_result]:
    pure = PurePosixPath(relative)
    if (not relative or pure.is_absolute() or "\\" in relative
            or any(part in {"", ".", ".."} for part in pure.parts)):
        raise LockRegenerationError("PATH_INVALID")
    path = root.joinpath(*pure.parts)
    try:
        # Refuse Windows reparse points explicitly as well as POSIX symlinks.
        # Check every retained pathname component; resolve() alone is not a
        # sufficient junction/reparse-point policy on Windows.
        _reject_reparse(root)
        cursor = root
        for part in pure.parts[:-1]:
            cursor = cursor / part
            parent = _reject_reparse(cursor)
            if not stat.S_ISDIR(parent.st_mode):
                raise LockRegenerationError("PATH_PARENT_INVALID")
        before = _reject_reparse(path)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise LockRegenerationError("NOT_REGULAR_FILE")
        if path.resolve(strict=True) != path.absolute():
            raise LockRegenerationError("PATH_ESCAPES_ROOT")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise LockRegenerationError("FILE_CHANGED_DURING_READ")
            chunks: list[bytes] = []
            size = 0
            while True:
                chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - size))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > maximum:
                    raise LockRegenerationError("FILE_EXCEEDS_BOUND")
            after = os.fstat(descriptor)
            if ((after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                    != (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)):
                raise LockRegenerationError("FILE_CHANGED_DURING_READ")
            return b"".join(chunks), before
        finally:
            os.close(descriptor)
    except FileNotFoundError:
        raise LockRegenerationError("FILE_MISSING") from None
    except OSError as exc:
        raise LockRegenerationError("FILE_UNREADABLE") from exc


def _canonical(document: dict[str, object]) -> bytes:
    return json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"


def _runtime_lock_type():
    repository = str(_repo_root())
    inserted = repository not in sys.path
    if inserted:
        sys.path.insert(0, repository)
    try:
        from tuner.execution.providers.modal.config import ModalRuntimeLockV1
        return ModalRuntimeLockV1
    finally:
        if inserted:
            sys.path.remove(repository)


def _updated(root: Path, current: bytes) -> bytes:
    if len(current) > MAX_LOCK_BYTES:
        raise LockRegenerationError("LOCK_EXCEEDS_BOUND")
    try:
        document = json.loads(current.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise LockRegenerationError("LOCK_INVALID") from exc
    if type(document) is not dict or _canonical(document) != current:
        raise LockRegenerationError("LOCK_NOT_CANONICAL")
    try:
        # Apply runtime composition's SDK-free structural/policy validation.
        # Preserve non-hash fields without independently approving their pins.
        lock_type = _runtime_lock_type()
        document = lock_type(document).to_dict()
    except (TypeError, ValueError) as exc:
        raise LockRegenerationError("LOCK_POLICY_INVALID") from exc
    locked = document["locked_files"]
    if type(locked) is not dict or set(locked) != set(LOCKED_FILES):
        raise LockRegenerationError("LOCKED_FILE_INVENTORY_INVALID")

    hashes: dict[str, str] = {}
    for name, expected_path in LOCKED_FILES.items():
        entry = locked[name]
        if type(entry) is not dict or set(entry) != {"path", "sha256"}:
            raise LockRegenerationError("LOCKED_FILE_ENTRY_INVALID")
        if entry["path"] != expected_path:
            raise LockRegenerationError("LOCKED_FILE_PATH_CHANGED")
        if (type(entry["sha256"]) is not str or len(entry["sha256"]) != 64
                or any(character not in "0123456789abcdef" for character in entry["sha256"])):
            raise LockRegenerationError("LOCKED_FILE_HASH_INVALID")
        payload, _ = _safe_regular_bytes(root, expected_path, maximum=MAX_SOURCE_BYTES)
        hashes[name] = hashlib.sha256(payload).hexdigest()

    refreshed = json.loads(json.dumps(document))
    for name, digest in hashes.items():
        refreshed["locked_files"][name]["sha256"] = digest
    return _canonical(refreshed)


def regenerate(root: Path, *, write: bool = False) -> int:
    supplied_root = root.absolute()
    root_identity = _reject_reparse(supplied_root)
    if stat.S_ISLNK(root_identity.st_mode) or not stat.S_ISDIR(root_identity.st_mode):
        raise LockRegenerationError("ROOT_IDENTITY_INVALID")
    root = root.resolve(strict=True)
    lock_path = root / LOCK_RELATIVE
    current, identity = _safe_regular_bytes(root, LOCK_RELATIVE, maximum=MAX_LOCK_BYTES)
    refreshed = _updated(root, current)
    if refreshed == current:
        print(json.dumps({"status": "CURRENT", "locked_file_count": len(LOCKED_FILES)}, sort_keys=True))
        return 0
    if not write:
        print("Modal runtime lock is STALE; re-run with --write.", file=sys.stderr)
        return 3

    # All sources were validated above. These pathname/identity checks narrow
    # ordinary local maintenance races; they are not a hostile-volume CAS or
    # retained-dirfd guarantee.
    observed, observed_identity = _safe_regular_bytes(root, LOCK_RELATIVE, maximum=MAX_LOCK_BYTES)
    if (observed != current or (observed_identity.st_dev, observed_identity.st_ino,
            observed_identity.st_mtime_ns) != (identity.st_dev, identity.st_ino, identity.st_mtime_ns)):
        raise LockRegenerationError("LOCK_CHANGED_BEFORE_WRITE")
    temporary_name = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(prefix=".modal-runtime-lock.", dir=lock_path.parent)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(refreshed)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_name, stat.S_IMODE(identity.st_mode))
        latest, latest_identity = _safe_regular_bytes(root, LOCK_RELATIVE, maximum=MAX_LOCK_BYTES)
        if latest != current or (latest_identity.st_dev, latest_identity.st_ino,
                latest_identity.st_mtime_ns) != (identity.st_dev, identity.st_ino, identity.st_mtime_ns):
            raise LockRegenerationError("LOCK_CHANGED_BEFORE_REPLACE")
        os.replace(temporary_name, lock_path)
        temporary_name = None
        if _safe_regular_bytes(root, LOCK_RELATIVE, maximum=MAX_LOCK_BYTES)[0] != refreshed:
            raise LockRegenerationError("LOCK_VERIFY_FAILED")
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    print(json.dumps({"status": "REFRESHED", "locked_file_count": len(LOCKED_FILES)}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Atomically refresh stale SHA-256 values.")
    args = parser.parse_args(argv)
    try:
        return regenerate(_repo_root(), write=args.write)
    except (LockRegenerationError, OSError) as exc:
        print(f"Modal runtime lock regeneration failed: {exc}", file=sys.stderr)
        return 125


if __name__ == "__main__":
    raise SystemExit(main())
