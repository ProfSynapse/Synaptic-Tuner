"""Immutable fixed GRPO parent/child boundary, distinct from SFT outer five."""
from __future__ import annotations

import hashlib
import re
import importlib.resources
from pathlib import Path
from types import SimpleNamespace

from .packaged_worker_closure import stable_read
from .offline_sft_worker import _strict_document, _canonical_json

GRPO_OUTER_SCHEMA = "synaptic-packaged-grpo-worker-closure/v1"
GRPO_OUTER_RESOURCE = "manifests/packaged-grpo-worker-v1.json"
GRPO_OUTER_MEMBERS = tuple(sorted((
    "tuner/runtime/packaged_grpo_child.py", "tuner/runtime/packaged_grpo_constructor_child.py",
    "tuner/runtime/packaged_grpo_execution.py",
    "tuner/runtime/packaged_grpo_verification.py", "tuner/runtime/packaged_grpo_worker_closure.py",
    "tuner/runtime/packaged_sft_child.py", "tuner/runtime/packaged_sft_execution.py",
    "tuner/runtime/packaged_training_worker.py", "tuner/runtime/packaged_worker_closure.py",
    "tuner/runtime/releases.py", "tuner/training/packaged_grpo_compilation.py",
    "tuner/runtime/offline_grpo_worker.py", "tuner/runtime/offline_sft_worker.py",
    "tuner/runtime/manifests/offline-grpo-worker-v1.json",
)))


def parse_packaged_grpo_worker_closure(raw, *, verify_sources=True, root=None):
    if type(raw) is not bytes or not 0 < len(raw) <= 1024 * 1024 or type(verify_sources) is not bool:
        raise ValueError("PACKAGED_GRPO_CLOSURE_REJECTED")
    value = _strict_document(raw)
    if (set(value) != {"schema_version", "members", "closure_digest"}
            or value["schema_version"] != GRPO_OUTER_SCHEMA
            or _canonical_json(value) + b"\n" != raw
            or type(value["members"]) is not list
            or tuple(item.get("path") if type(item) is dict else None for item in value["members"])
                != GRPO_OUTER_MEMBERS):
        raise ValueError("PACKAGED_GRPO_CLOSURE_REJECTED")
    unsigned = {"schema_version": GRPO_OUTER_SCHEMA, "members": value["members"]}
    digest = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    if value["closure_digest"] != digest:
        raise ValueError("PACKAGED_GRPO_CLOSURE_REJECTED")
    root = Path(__file__).absolute().parents[2] if root is None else Path(root)
    for member in value["members"]:
        if (set(member) != {"path", "size_bytes", "sha256"}
                or type(member["size_bytes"]) is not int or not 0 < member["size_bytes"] <= 1024 * 1024
                or type(member["sha256"]) is not str or re.fullmatch(r"[0-9a-f]{64}", member["sha256"]) is None):
            raise ValueError("PACKAGED_GRPO_CLOSURE_REJECTED")
        if verify_sources:
            source = stable_read(root / member["path"], member["size_bytes"])
            if len(source) != member["size_bytes"] or hashlib.sha256(source).hexdigest() != member["sha256"]:
                raise ValueError("PACKAGED_GRPO_CLOSURE_REJECTED")
    return SimpleNamespace(digest=digest, members=tuple(value["members"]))



def load_packaged_grpo_worker_closure():
    resource = importlib.resources.files("tuner.runtime").joinpath(GRPO_OUTER_RESOURCE)
    if not isinstance(resource, Path):
        raise ValueError("PACKAGED_GRPO_CLOSURE_REJECTED")
    return parse_packaged_grpo_worker_closure(stable_read(resource, 1024 * 1024))


def captured_grpo_outer_document(root):
    root = Path(root)
    members = []
    for name in GRPO_OUTER_MEMBERS:
        raw = stable_read(root / name, 1024 * 1024)
        members.append({"path": name, "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    document = {"schema_version": GRPO_OUTER_SCHEMA, "members": members}
    document["closure_digest"] = hashlib.sha256(_canonical_json(document)).hexdigest()
    parse_packaged_grpo_worker_closure(_canonical_json(document) + b"\n", verify_sources=False)
    return document
