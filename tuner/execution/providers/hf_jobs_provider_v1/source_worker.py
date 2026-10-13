"""Ordinary Jobs parent: authenticated private source, anonymous model, native output.

No provider SDK calls, protected smoke policy, or Modal image augmentation.
The explicitly passed HF job secret stays in the privileged parent; the child
receives only its reset admitted environment.
"""

import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

from .bootstrap import _read, _write


ROLES = frozenset({"final_model", "tokenizer", "training_lineage", "training_metrics", "workload_record"})
PER_ARTIFACT = 192 * 1024 * 1024
ALL_ARTIFACTS = 256 * 1024 * 1024


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _runtime(inventory, source):
    runtime = inventory["runtime"]
    if (runtime["python_implementation"] != platform.python_implementation()
            or runtime["python_version"] != platform.python_version()
            or runtime["os"] != platform.system()
            or runtime["architecture"] != platform.machine()
            or runtime["libc"] != " ".join(platform.libc_ver())
            or os.path.realpath(sys.executable) != os.path.realpath(source["runtime"]["interpreter"]["executable"])):
        raise ValueError("HF_RUNTIME_REJECTED")
    executable = Path(sys.executable).resolve(strict=True)
    with executable.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != source["runtime"]["interpreter"]["executable_digest"]:
        raise ValueError("HF_RUNTIME_REJECTED")
    observed, expected = {}, {}
    for row in inventory["distributions"]:
        name = re.sub(r"[-_.]+", "-", row["name"]).lower()
        if name in expected:
            raise ValueError("HF_RUNTIME_REJECTED")
        expected[name] = row["version"]
    for index, distribution in enumerate(importlib.metadata.distributions(), 1):
        if index > 4096:
            raise ValueError("HF_RUNTIME_REJECTED")
        name = re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower()
        if name in observed:
            raise ValueError("HF_RUNTIME_REJECTED")
        observed[name] = distribution.version
    if observed != expected:
        raise ValueError("HF_RUNTIME_REJECTED")


def execute(document):
    """Run exactly the admitted offline worker once; retain partial outputs on failure."""
    if type(document) is not dict or set(document) != {
        "material_digest", "workload_fingerprint", "source_members", "input", "dispatch",
        "source", "runtime_inventory_sha256", "model", "timeout_seconds",
    }:
        raise ValueError("HF_WORKER_INPUT")
    timeout = document["timeout_seconds"]
    if type(timeout) is not int or not 1 <= timeout <= 86400:
        raise ValueError("HF_WORKER_INPUT")
    deadline = time.monotonic() + timeout
    source = document["source"]
    roots = source["runtime"]["roots"]
    if roots != {name: "/workspace/private/" + ("" if name in {"engine", "project"} else "run/") + name for name in
                 ("engine", "project", "artifacts", "state", "tracking", "cache", "tmp")}:
        raise ValueError("HF_WORKER_INPUT")
    private = Path("/workspace/private")
    private.mkdir(mode=0o700)
    (private / "run").mkdir(mode=0o700)
    for root in (*roots.values(), "/workspace/private/control", "/workspace/private/model-persistent"):
        Path(root).mkdir(mode=0o700)
    records = document["source_members"]
    if type(records) is not list or not 1 <= len(records) <= 256:
        raise ValueError("HF_WORKER_INPUT")
    total, seen = 0, set()
    for row in records:
        name, size = row["path"], row["size_bytes"]
        if name in seen or type(size) is not int or not 0 <= size <= 64 * 1024 * 1024:
            raise ValueError("HF_WORKER_INPUT")
        seen.add(name)
        total += size
        if total > 64 * 1024 * 1024 or not name.startswith(("engine/", "control/")):
            raise ValueError("HF_WORKER_INPUT")
        raw = _read("/source", name, size)
        if _sha(raw) != row["sha256"]:
            raise ValueError("HF_SOURCE_REJECTED")
        _write(private, name, raw)
    inventory = json.loads(_read(private, "control/runtime-inventory.json", next(
        row["size_bytes"] for row in records if row["path"] == "control/runtime-inventory.json")))
    if _sha(_read(private, "control/runtime-inventory.json", next(
        row["size_bytes"] for row in records if row["path"] == "control/runtime-inventory.json"))) != document["runtime_inventory_sha256"]:
        raise ValueError("HF_RUNTIME_REJECTED")
    _runtime(inventory, source)
    identity = document["input"]
    raw = _read("/artifacts", "input/" + identity["content_digest"] + "/private-dataset.jsonl",
                identity["size_bytes"])
    if _sha(raw) != identity["content_digest"]:
        raise ValueError("HF_INPUT_REJECTED")
    _write(private, "run/state/prepared-inputs/" + identity["content_digest"] + "/private-dataset.jsonl", raw)
    from tuner.execution.providers.modal.model_snapshot import prepare_model_snapshot
    # Never expose the admitted job secret to this anonymous public-model API.
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        prepare_model_snapshot(model_ref=document["model"]["ref"], revision=document["model"]["revision"],
            token=False, persistent_root=private / "model-persistent", destination_root=private / "run/cache",
            scratch_root=private / "run/tmp")
    if time.monotonic() >= deadline:
        raise ValueError("HF_WORKER_DEADLINE")
    dispatch = document["dispatch"]
    environment = dict(dispatch["environment"])
    if any(re.search(r"(?:^|_)(?:token|secret|password|api_key|credential)(?:$|_)", key, re.I)
           for key in environment):
        raise ValueError("HF_WORKER_INPUT")
    with open(os.devnull, "wb") as sink:
        result = subprocess.run(dispatch["argv"], cwd=dispatch["cwd"], env=environment,
            input=__import__("base64").b64decode(dispatch["stdin_base64"], validate=True),
            stdout=sink, stderr=sink, timeout=max(0.001, deadline - time.monotonic()), check=False)
    if result.returncode != 0:
        raise ValueError("HF_TRAINER_FAILED")
    inventory_size = os.stat(private / "run/state/runtime-v1-inventory.json", follow_symlinks=False).st_size
    if not 0 < inventory_size <= 65536:
        raise ValueError("HF_ARTIFACT_REJECTED")
    inventory = json.loads(_read(private, "run/state/runtime-v1-inventory.json", inventory_size))
    if (set(inventory) != {"schema_version", "workload_fingerprint", "artifacts"}
            or inventory["schema_version"] != "synaptic-artifact-inventory/v1"
            or inventory["workload_fingerprint"] != document["workload_fingerprint"]):
        raise ValueError("HF_ARTIFACT_REJECTED")
    members, total, roles, names = [], 0, set(), set()
    for row in inventory["artifacts"]:
        role, name, size, digest = row["role"], row["path"], row["size"], row["sha256"]
        if (role not in ROLES or role in roles or name in names or "/" in name or "\\" in name
                or name in {".", ".."} or type(size) is not int or not 0 < size <= PER_ARTIFACT):
            raise ValueError("HF_ARTIFACT_REJECTED")
        roles.add(role)
        names.add(name)
        total += size
        if total > ALL_ARTIFACTS or time.monotonic() >= deadline:
            raise ValueError("HF_ARTIFACT_REJECTED")
        raw = _read(private, "run/artifacts/" + name, size)
        if _sha(raw) != digest:
            raise ValueError("HF_ARTIFACT_REJECTED")
        _write("/artifacts", "artifacts/" + name, raw, private=False)
        if _sha(_read("/artifacts", "artifacts/" + name, size)) != digest:
            raise ValueError("HF_ARTIFACT_REJECTED")
        members.append({"role": role, "path": "artifacts/" + name, "size_bytes": size, "sha256": digest})
    if roles != ROLES:
        raise ValueError("HF_ARTIFACT_REJECTED")
    terminal = {"schema_version": "synaptic-hf-native-completion/v1", "status": "completed",
        "material_digest": document["material_digest"], "workload_fingerprint": document["workload_fingerprint"],
        "job_id": os.environ.get("JOB_ID"), "artifacts": sorted(members, key=lambda row: row["role"])}
    _write("/artifacts", "completion.json", json.dumps(terminal, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode(), private=False)
    return 0
