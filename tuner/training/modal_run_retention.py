"""Private, create-only custody for new standalone Modal runs.

The loader checks private custody, byte commitments and selected run/scope
identity only; future composition must apply the existing typed parsers and
material/digest checks before reads. This is retention, not execution/source
restoration. No SDK handles, provider credentials or sealed log proof are stored.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType

from synaptic_tuner.api.v1.results import TrainingRunRef
from tuner.runtime.packaged_worker_closure import stable_read
from tuner.training.modal_local_preparation import private_directory

KEY_NAMES = frozenset({"SYNAPTIC_TRAIN_AUTHORITY_KEY", "SYNAPTIC_TRAIN_READER_KEY"})
KEYS_FILE = "authority-keys.json"
INPUTS_FILE = "readback-inputs.json"
WORKFLOW_FILE = "workflow.sqlite3"
_KEYS_SCHEMA = "synaptic-private-modal-run-keys/v1"
_INPUTS_SCHEMA = "synaptic-private-modal-run-inputs/v1"
_MAX_INPUTS = 8 * 1024 * 1024
_FIELDS = frozenset({
    "prepared_request", "components", "workload", "runtime_release", "runtime_facts",
    "deployment_facts", "client_binding", "volume_names", "modal_profile",
    "modal_environment", "profile_digest", "quote_digest", "maximum_cost_minor_units",
    "attempt_journal", "workflow_file", "engine_root", "cpu_receipt",
    "training_plan", "training_preflight",
})


class ModalRunRetentionUnavailable(RuntimeError):
    def __init__(self):
        super().__init__("modal_run_retention_unavailable")


@dataclass(frozen=True, slots=True)
class RetainedModalRunV1:
    run: TrainingRunRef
    signing_keys: object = field(repr=False)
    qualification_key: bytes = field(repr=False)
    inputs: object = field(repr=False)


def validate_retention_root(root: Path) -> Path:
    try:
        return private_directory(root, create=False)
    except Exception:
        pass
    raise ModalRunRetentionUnavailable()


def retained_bytes(raw: bytes) -> dict:
    if type(raw) is not bytes or not raw or len(raw) > 1024 * 1024:
        raise ModalRunRetentionUnavailable()
    return {"base64": base64.b64encode(raw).decode("ascii"),
            "sha256": hashlib.sha256(raw).hexdigest()}


def _decode_bytes(value):
    if type(value) is not dict or set(value) != {"base64", "sha256"}:
        raise ValueError
    raw = base64.b64decode(value["base64"], validate=True)
    if (not 0 < len(raw) <= 1024 * 1024 or base64.b64encode(raw).decode("ascii") != value["base64"]
            or hashlib.sha256(raw).hexdigest() != value["sha256"]):
        raise ValueError
    return raw


def _encode(document, maximum):
    raw = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                     allow_nan=False).encode("utf-8")
    if not 0 < len(raw) <= maximum:
        raise ValueError
    return raw


def _decode(raw, maximum):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError
            result[key] = value
        return result
    document = json.loads(raw, object_pairs_hook=unique,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError()))
    if type(document) is not dict or _encode(document, maximum) != raw:
        raise ValueError
    return document


def _validate_inputs(inputs, run):
    if type(inputs) is not dict or set(inputs) != _FIELDS:
        raise ValueError
    if inputs["workflow_file"] != WORKFLOW_FILE:
        raise ValueError
    for key in ("profile_digest", "quote_digest"):
        if type(inputs[key]) is not str or re.fullmatch("[0-9a-f]{64}", inputs[key]) is None:
            raise ValueError
    if type(inputs["maximum_cost_minor_units"]) is not int or inputs["maximum_cost_minor_units"] <= 0:
        raise ValueError
    for key in ("attempt_journal", "engine_root"):
        if type(inputs[key]) is not str or not Path(inputs[key]).is_absolute():
            raise ValueError
    for key in ("modal_profile", "modal_environment"):
        if type(inputs[key]) is not str or not inputs[key]:
            raise ValueError
    components = inputs["components"]
    if type(components) is not dict or set(components) != {
        "execution_source", "execution_context", "resolved_config", "runtime", "resources", "artifact_policy",
    }:
        raise ValueError
    if components["execution_source"]["run_ref"] != run.run_id:
        raise ValueError
    if inputs["prepared_request"]["project_ref"] != run.project_ref:
        raise ValueError
    scope = inputs["client_binding"]
    if (type(scope) is not dict or set(scope) != {
        "account_ref", "workspace_ref", "environment_ref", "client_ref", "sdk_version",
    } or inputs["runtime_facts"]["scope"] != scope
            or inputs["deployment_facts"]["scope"] != scope
            or scope["environment_ref"] != inputs["modal_environment"]):
        raise ValueError
    _decode_bytes(inputs["workload"])
    _decode_bytes(components["execution_context"])
    _decode_bytes(components["resolved_config"])


def _parent(root):
    validate_retention_root(root)
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    return descriptor, os.fstat(descriptor)


def _check_root(root, descriptor, original):
    current, named = os.fstat(descriptor), root.lstat()
    identity = original.st_dev, original.st_ino
    if ((current.st_dev, current.st_ino) != identity or (named.st_dev, named.st_ino) != identity
            or current.st_uid != os.geteuid() or stat.S_IMODE(current.st_mode) != 0o700):
        raise ValueError
    validate_retention_root(root)


def _write(root, name, raw):
    parent, parent_info = _parent(root)
    descriptor = None
    try:
        descriptor = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=parent)
        original = os.fstat(descriptor)
        remaining = memoryview(raw)
        while remaining:
            count = os.write(descriptor, remaining)
            if count <= 0:
                raise ValueError
            remaining = remaining[count:]
        os.fsync(descriptor)
        os.fsync(parent)
        _check_root(root, parent, parent_info)
        named = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if (named.st_dev, named.st_ino) != (original.st_dev, original.st_ino):
            raise ValueError
        if _read(root, name, len(raw)) != raw:
            raise ValueError
        _check_root(root, parent, parent_info)
        named = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if (named.st_dev, named.st_ino) != (original.st_dev, original.st_ino):
            raise ValueError
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)


def _read(root, name, maximum):
    parent, parent_info = _parent(root)
    try:
        before = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                or stat.S_IMODE(before.st_mode) != 0o600 or before.st_nlink != 1):
            raise ValueError
        raw = stable_read(root / name, maximum)
        after = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mode, before.st_uid, before.st_nlink) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mode, after.st_uid, after.st_nlink):
            raise ValueError
        _check_root(root, parent, parent_info)
        return raw
    finally:
        os.close(parent)


def retain_modal_run(*, root, run, signing_keys, qualification_key, inputs):
    try:
        if type(run) is not TrainingRunRef or type(qualification_key) is not bytes or len(qualification_key) != 32:
            raise ValueError
        keys = dict(signing_keys)
        if (set(keys) != KEY_NAMES or any(type(value) is not str or
                re.fullmatch("[0-9a-f]{64}", value) is None for value in keys.values())):
            raise ValueError
        _validate_inputs(inputs, run)
        raw = _encode({"schema_version": _INPUTS_SCHEMA, "run": run.to_dict(), "inputs": inputs}, _MAX_INPUTS)
        key_raw = _encode({
            "schema_version": _KEYS_SCHEMA, "run": run.to_dict(), "signing_keys": keys,
            "qualification_key": base64.b64encode(qualification_key).decode("ascii"),
            "inputs_sha256": hashlib.sha256(raw).hexdigest(),
        }, 4096)
        validate_retention_root(root)
        _write(root, KEYS_FILE, key_raw)
        _write(root, INPUTS_FILE, raw)
        load_retained_modal_run(root, expected_run=run)
        return
    except Exception:
        pass
    raise ModalRunRetentionUnavailable()


def load_retained_modal_run(root, *, expected_run):
    try:
        if type(expected_run) is not TrainingRunRef:
            raise ValueError
        key_doc = _decode(_read(root, KEYS_FILE, 4096), 4096)
        raw = _read(root, INPUTS_FILE, _MAX_INPUTS)
        document = _decode(raw, _MAX_INPUTS)
        if (set(key_doc) != {"schema_version", "run", "signing_keys", "qualification_key", "inputs_sha256"}
                or key_doc["schema_version"] != _KEYS_SCHEMA
                or set(document) != {"schema_version", "run", "inputs"}
                or document["schema_version"] != _INPUTS_SCHEMA
                or document["run"] != expected_run.to_dict() or key_doc["run"] != document["run"]
                or key_doc["inputs_sha256"] != hashlib.sha256(raw).hexdigest()):
            raise ValueError
        keys = key_doc["signing_keys"]
        qualification = base64.b64decode(key_doc["qualification_key"], validate=True)
        if (type(keys) is not dict or set(keys) != KEY_NAMES
                or any(type(v) is not str or re.fullmatch("[0-9a-f]{64}", v) is None for v in keys.values())
                or len(qualification) != 32
                or base64.b64encode(qualification).decode("ascii") != key_doc["qualification_key"]):
            raise ValueError
        _validate_inputs(document["inputs"], expected_run)
        return RetainedModalRunV1(expected_run, MappingProxyType(keys), qualification,
                                 MappingProxyType(document["inputs"]))
    except Exception:
        pass
    raise ModalRunRetentionUnavailable()
