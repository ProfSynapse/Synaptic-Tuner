"""Propose or exclusively initialize the three Modal inference lock resources.

This is a bounded, offline, one-time initializer.  It consumes an explicitly
accepted candidate capture and the reviewed additive lock.  The existing
regenerator remains the only maintenance path after initialization.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys

_MAX_EVIDENCE_BYTES = 1024 * 1024
_MAX_ADDITIVE_BYTES = 1024 * 1024
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_REGISTRY = re.compile(r"^[^\s:@]+(?:/[^\s:@]+)+@sha256:[0-9a-f]{64}$")
_DIST_SEPARATORS = re.compile(r"[-_.]+")
_ADDITIVE_RELATIVE = "requirements/modal-inference-additions.lock"
_ADDITIVE_NAMES = frozenset(
    {
        "grpclib",
        "h2",
        "hpack",
        "hyperframe",
        "modal",
        "synchronicity",
        "toml",
        "types-certifi",
        "types-toml",
    }
)
_ISOLATED_PYTHON = "/opt/synaptic-inference/bin/python"
_SDK_VERSION = "1.5.4"


class InitializationFault(RuntimeError):
    """Closed initialization failure."""


def _load_sibling(name: str):
    path = Path(__file__).resolve(strict=True).with_name(name)
    spec = importlib.util.spec_from_file_location("_modal_lock_" + name, path)
    if spec is None or spec.loader is None:
        raise InitializationFault("HELPER_UNAVAILABLE")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_maintenance = _load_sibling("regenerate_modal_inference_lock.py")
_runtime_reader = _load_sibling("regenerate_modal_runtime_lock.py")
_capture = _load_sibling("capture_modal_inference_runtime.py")
SOURCE_MEMBERS = _maintenance.SOURCE_MEMBERS
RUNTIME_RELATIVE = _maintenance.RUNTIME_RELATIVE
CLOSURE_RELATIVE = _maintenance.CLOSURE_RELATIVE
DEPENDENCY_RELATIVE = _maintenance.DEPENDENCY_RELATIVE


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


def _relative(value: str) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise InitializationFault("PATH_INVALID")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise InitializationFault("PATH_INVALID")
    return value


def _text(value: object, maximum: int, code: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value.encode("utf-8")) > maximum
        or "\0" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise InitializationFault(code)
    return value


def _read(root: Path, relative: str, maximum: int) -> bytes:
    try:
        return _runtime_reader._safe_regular_bytes(
            root, _relative(relative), maximum=maximum
        )[0]
    except Exception as exc:
        raise InitializationFault("INPUT_READ_FAILED") from exc


def _document(payload: bytes, code: str) -> dict[str, object]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
        )
    except (UnicodeError, ValueError, TypeError, OverflowError) as exc:
        raise InitializationFault(code) from exc
    if type(value) is not dict or payload != _canonical(value):
        raise InitializationFault(code)
    return value


def _pairs(values: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in values:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _member(root: Path, relative: str) -> dict[str, object]:
    payload = _read(root, relative, 64 * 1024 * 1024)
    return {
        "path": relative,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _candidate(
    value: dict[str, object], accepted_digest: str, base_reference: str
) -> tuple[dict[str, str], dict[str, object]]:
    expected = {
        "candidate",
        "engine_wheel_name",
        "engine_wheel_sha256",
        "inspection_script_sha256",
        "modal_additions_sha256",
        "operator_selection_only",
        "provider_image_id",
        "python_preparation",
        "sandbox_id",
        "schema_version",
    }
    if (
        set(value) != expected
        or value["schema_version"] != "synaptic-modal-inference-runtime-capture/v1"
    ):
        raise InitializationFault("EVIDENCE_SCHEMA_INVALID")
    if value["operator_selection_only"] is not True:
        raise InitializationFault("EVIDENCE_STATUS_INVALID")
    candidate = value["candidate"]
    candidate_fields = {
        "distributions",
        "operator_selection",
        "python",
        "requirements",
        "schema_version",
        "status",
    }
    if type(candidate) is not dict or set(candidate) != candidate_fields:
        raise InitializationFault("EVIDENCE_SCHEMA_INVALID")
    if (
        candidate["schema_version"]
        != "synaptic-modal-inference-runtime-inspection-candidate/v1"
        or candidate["status"] != "CANDIDATE_ONLY"
    ):
        raise InitializationFault("EVIDENCE_STATUS_INVALID")
    selection = candidate["operator_selection"]
    if type(selection) is not dict or set(selection) != {"image", "source_commit"}:
        raise InitializationFault("EVIDENCE_SCHEMA_INVALID")
    if selection["image"] != base_reference:
        raise InitializationFault("BASE_IMAGE_MISMATCH")
    source_commit = _text(selection["source_commit"], 40, "EVIDENCE_SCHEMA_INVALID")
    try:
        _capture._parse_candidate(
            _canonical(candidate), image=base_reference, source_commit=source_commit
        )
    except Exception as exc:
        raise InitializationFault("EVIDENCE_CANDIDATE_INVALID") from exc
    python = candidate["python"]
    if (
        type(python) is not dict
        or set(python)
        != {"implementation", "version", "executable", "executable_sha256"}
        or python["implementation"] != "cpython"
        or type(python["executable_sha256"]) is not str
        or _DIGEST.fullmatch(python["executable_sha256"]) is None
    ):
        raise InitializationFault("PYTHON_IDENTITY_INVALID")
    if (
        _text(python["version"], 64, "PYTHON_IDENTITY_INVALID") != "3.12.13"
        or _text(python["executable"], 4096, "PYTHON_IDENTITY_INVALID")
        != _ISOLATED_PYTHON
    ):
        raise InitializationFault("PYTHON_IDENTITY_INVALID")
    requirements = candidate["requirements"]
    distributions = candidate["distributions"]
    if type(requirements) is not dict or set(requirements) != {"modal", "vllm"}:
        raise InitializationFault("DISTRIBUTIONS_INVALID")
    if type(distributions) is not dict or not distributions or len(distributions) > 512:
        raise InitializationFault("DISTRIBUTIONS_INVALID")
    checked: dict[str, str] = {}
    for name, version in distributions.items():
        if (
            type(name) is not str
            or type(version) is not str
            or not name
            or not version
            or len(name.encode()) > 128
            or len(version.encode()) > 256
            or name != _DIST_SEPARATORS.sub("-", name).lower()
        ):
            raise InitializationFault("DISTRIBUTIONS_INVALID")
        checked[name] = version
    if list(checked) != sorted(checked) or not {
        "modal",
        "vllm",
        "torch",
        "transformers",
        "tokenizers",
        "safetensors",
    }.issubset(checked):
        raise InitializationFault("DISTRIBUTIONS_INVALID")
    for name in ("modal", "vllm"):
        item = requirements[name]
        if (
            type(item) is not dict
            or set(item) != {"present", "version"}
            or item["present"] is not True
            or item["version"] != checked[name]
        ):
            raise InitializationFault("DISTRIBUTIONS_INVALID")
    if checked["modal"] != _SDK_VERSION:
        raise InitializationFault("DISTRIBUTIONS_INVALID")
    for field in (
        "engine_wheel_sha256",
        "inspection_script_sha256",
        "modal_additions_sha256",
    ):
        if type(value[field]) is not str or _DIGEST.fullmatch(value[field]) is None:
            raise InitializationFault("EVIDENCE_SCHEMA_INVALID")
    python_preparation = value["python_preparation"]
    if (
        type(python_preparation) is not dict
        or set(python_preparation) != {"executable", "qualification", "script_sha256"}
        or python_preparation["executable"] != _ISOLATED_PYTHON
        or python_preparation["qualification"] != "CANDIDATE_ONLY"
        or type(python_preparation["script_sha256"]) is not str
        or _DIGEST.fullmatch(python_preparation["script_sha256"]) is None
    ):
        raise InitializationFault("EVIDENCE_SCHEMA_INVALID")
    if (
        _text(value["engine_wheel_name"], 256, "EVIDENCE_SCHEMA_INVALID").endswith(
            ".whl"
        )
        is False
        or type(value["provider_image_id"]) is not str
        or not re.fullmatch(r"im-[A-Za-z0-9]{1,64}", value["provider_image_id"])
        or type(value["sandbox_id"]) is not str
        or not re.fullmatch(r"sb-[A-Za-z0-9]{1,64}", value["sandbox_id"])
        or not re.fullmatch(r"[0-9a-f]{40}", source_commit)
    ):
        raise InitializationFault("EVIDENCE_SCHEMA_INVALID")
    provenance = {
        "accepted_evidence_sha256": accepted_digest,
        "candidate_provider_image_id": value["provider_image_id"],
        "candidate_source_commit": source_commit,
        "candidate_modal_additions_sha256": value["modal_additions_sha256"],
        "engine_wheel_name": value["engine_wheel_name"],
        "engine_wheel_sha256": value["engine_wheel_sha256"],
        "inspection_script_sha256": value["inspection_script_sha256"],
        "status": "ACCEPTED_AS_STARTING_PINS_ONLY",
    }
    return checked, {
        "python": python,
        "sdk_version": checked["modal"],
        "provenance": provenance,
    }


def proposal(
    root: Path,
    *,
    accepted_evidence: str,
    accepted_evidence_sha256: str,
    base_registry_reference: str,
    additive_lock_sha256: str,
) -> dict[str, bytes]:
    for relative in (DEPENDENCY_RELATIVE, CLOSURE_RELATIVE, RUNTIME_RELATIVE):
        _target_absent(root / relative)
    if _DIGEST.fullmatch(accepted_evidence_sha256) is None:
        raise InitializationFault("ACCEPTED_EVIDENCE_DIGEST_INVALID")
    if _REGISTRY.fullmatch(base_registry_reference) is None:
        raise InitializationFault("BASE_IMAGE_INVALID")
    if (
        SOURCE_MEMBERS != tuple(sorted(SOURCE_MEMBERS))
        or len(SOURCE_MEMBERS) != 118
        or len(set(SOURCE_MEMBERS)) != 118
    ):
        raise InitializationFault("FIXED_INVENTORY_INVALID")
    evidence_bytes = _read(root, accepted_evidence, _MAX_EVIDENCE_BYTES)
    if hashlib.sha256(evidence_bytes).hexdigest() != accepted_evidence_sha256:
        raise InitializationFault("ACCEPTED_EVIDENCE_MISMATCH")
    evidence = _document(evidence_bytes, "EVIDENCE_DOCUMENT_INVALID")
    distributions, selected = _candidate(
        evidence, accepted_evidence_sha256, base_registry_reference
    )
    selected["provenance"]["accepted_evidence_path"] = _relative(accepted_evidence)
    selected["provenance"]["python_preparation_script_sha256"] = evidence[
        "python_preparation"
    ]["script_sha256"]
    additions = _read(root, _ADDITIVE_RELATIVE, _MAX_ADDITIVE_BYTES)
    additions_digest = hashlib.sha256(additions).hexdigest()
    if (
        type(additive_lock_sha256) is not str
        or _DIGEST.fullmatch(additive_lock_sha256) is None
        or additions_digest != additive_lock_sha256
        or additions_digest != evidence["modal_additions_sha256"]
    ):
        raise InitializationFault("ADDITIVE_LOCK_MISMATCH")
    selected_lines = [
        line
        for line in additions.decode("utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    if len(selected_lines) != 9:
        raise InitializationFault("ADDITIVE_LOCK_INVALID")
    addition_names: set[str] = set()
    for line in selected_lines:
        if "==" not in line or " --hash=sha256:" not in line:
            raise InitializationFault("ADDITIVE_LOCK_INVALID")
        name, remainder = line.split("==", 1)
        version, digest = remainder.split(" --hash=sha256:", 1)
        normalized = _DIST_SEPARATORS.sub("-", name).lower()
        if (
            normalized in addition_names
            or normalized not in distributions
            or distributions[normalized] != version
            or _DIGEST.fullmatch(digest) is None
        ):
            raise InitializationFault("ADDITIVE_LOCK_INVALID")
        addition_names.add(normalized)
    if addition_names != _ADDITIVE_NAMES:
        raise InitializationFault("ADDITIVE_LOCK_INVALID")
    dependency = _canonical(
        {
            "additive_lock": {
                "bytes_base64": base64.b64encode(additions).decode("ascii"),
                "path": _ADDITIVE_RELATIVE,
                "sha256": additions_digest,
                "size_bytes": len(additions),
            },
            "base_registry_reference": base_registry_reference,
            "distributions": distributions,
            "provenance": selected["provenance"],
            "schema_version": "synaptic-modal-inference-dependency-provenance/v1",
        }
    )
    members = [_member(root, relative) for relative in SOURCE_MEMBERS]
    unsigned = {
        "schema_version": "synaptic-modal-inference-worker-closure/v1",
        "entrypoint": "tuner/execution/providers/modal/inference_bootstrap.py",
        "member_count": len(members),
        "payload_bytes": sum(item["size_bytes"] for item in members),
        "members": members,
    }
    closure_document = dict(unsigned)
    closure_document["closure_digest"] = hashlib.sha256(
        _canonical(unsigned)
    ).hexdigest()
    closure = _canonical(closure_document)
    extras = {
        DEPENDENCY_RELATIVE: {
            "path": DEPENDENCY_RELATIVE,
            "size_bytes": len(dependency),
            "sha256": hashlib.sha256(dependency).hexdigest(),
        },
        CLOSURE_RELATIVE: {
            "path": CLOSURE_RELATIVE,
            "size_bytes": len(closure),
            "sha256": hashlib.sha256(closure).hexdigest(),
        },
    }
    inventory = sorted([*members, *extras.values()], key=lambda item: item["path"])
    runtime = _canonical(
        {
            "base_registry_reference": base_registry_reference,
            "dependency_lock_path": DEPENDENCY_RELATIVE,
            "distributions": distributions,
            "python": selected["python"],
            "schema_version": "synaptic-modal-inference-runtime-lock/v1",
            "sdk_version": selected["sdk_version"],
            "source_inventory": inventory,
            "worker_closure_manifest_path": CLOSURE_RELATIVE,
        }
    )
    try:
        from tuner.execution.providers.modal import inference_runtime
        from tuner.execution.providers.modal.inference_runtime import _worker_closure

        if (
            _worker_closure(closure, inventory, CLOSURE_RELATIVE, DEPENDENCY_RELATIVE)
            != closure_document["closure_digest"]
        ):
            raise ValueError("closure mismatch")
        original_root = inference_runtime._runtime_root
        original_reader = inference_runtime.read_regular
        try:
            inference_runtime._runtime_root = lambda: root
            inference_runtime.read_regular = (
                lambda selected_root, path, maximum: runtime
            )
            parsed, _ = inference_runtime._manifest()
        finally:
            inference_runtime._runtime_root = original_root
            inference_runtime.read_regular = original_reader
        if parsed != runtime:
            raise ValueError("manifest mismatch")
    except Exception as exc:
        raise InitializationFault("CROSS_RESOURCE_VALIDATION_FAILED") from exc
    return {
        DEPENDENCY_RELATIVE: dependency,
        CLOSURE_RELATIVE: closure,
        RUNTIME_RELATIVE: runtime,
    }


def _target_absent(path: Path) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    raise InitializationFault("TARGET_EXISTS")


def initialize(root: Path, **values: str) -> dict[str, bytes]:
    for relative in (DEPENDENCY_RELATIVE, CLOSURE_RELATIVE, RUNTIME_RELATIVE):
        _target_absent(root / relative)
    proposed = proposal(root, **values)
    # Recheck every accepted input and source immediately before writing. This
    # is trusted local maintenance, not a hostile-filesystem transaction.
    if proposal(root, **values) != proposed:
        raise InitializationFault("INPUT_CHANGED_BEFORE_WRITE")
    created: list[tuple[Path, tuple[int, int]]] = []
    try:
        for relative in (DEPENDENCY_RELATIVE, CLOSURE_RELATIVE, RUNTIME_RELATIVE):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                0o644,
            )
            try:
                info = os.fstat(descriptor)
                created.append((path, (info.st_dev, info.st_ino)))
                payload = proposed[relative]
                with os.fdopen(descriptor, "wb", closefd=False) as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except BaseException:
        for path, identity in reversed(created):
            try:
                info = path.lstat()
                if (info.st_dev, info.st_ino) == identity:
                    path.unlink()
            except FileNotFoundError:
                pass
        raise
    return proposed


def main(argv: list[str] | None = None, *, root: Path | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-evidence", required=True)
    parser.add_argument("--accepted-evidence-sha256", required=True)
    parser.add_argument("--base-registry-reference", required=True)
    parser.add_argument("--additive-lock-sha256", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve(strict=True).parents[1] if root is None else root
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    values = {
        "accepted_evidence": args.accepted_evidence,
        "accepted_evidence_sha256": args.accepted_evidence_sha256,
        "base_registry_reference": args.base_registry_reference,
        "additive_lock_sha256": args.additive_lock_sha256,
    }
    try:
        result = initialize(repo, **values) if args.write else proposal(repo, **values)
    except (
        InitializationFault,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        UnicodeError,
    ) as exc:
        code = str(exc) if type(exc) is InitializationFault else "INITIALIZATION_FAILED"
        print("Modal inference lock initialization failed: " + code, file=sys.stderr)
        return 125
    print(
        json.dumps(
            {
                "member_count": 118,
                "status": "INITIALIZED" if args.write else "PROPOSED",
                "resource_sha256": {
                    path: hashlib.sha256(payload).hexdigest()
                    for path, payload in sorted(result.items())
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
