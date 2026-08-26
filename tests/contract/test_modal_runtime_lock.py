from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema


ROOT = Path(__file__).resolve().parents[2]


def test_modal_runtime_lock_schema_and_file_digests_are_exact():
    lock_path = ROOT / "tuner/execution/providers/modal/modal-runtime-v1.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    schema = json.loads((ROOT / "schemas/synaptic-modal-runtime-lock-v1.schema.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(lock)
    assert lock_path.read_bytes() == json.dumps(lock,sort_keys=True,indent=2).encode("utf-8") + b"\n"
    for member in lock["locked_files"].values():
        assert hashlib.sha256((ROOT / member["path"]).read_bytes()).hexdigest() == member["sha256"]
    assert lock["registry_reference"].endswith("@sha256:5266c57be21059bfb407d80dc2f448868a5c2e2dbe7b2aa27780f48b48cbec39")
