from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import jsonschema
import pytest


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


def test_modal_launcher_lock_checkout_preserves_hash_bound_lf_bytes(tmp_path):
    source = ROOT / "requirements/modal-launcher-v1.lock"
    expected = source.read_bytes()
    assert b"\r" not in expected

    repository = tmp_path / "checkout"
    repository.mkdir()

    def git(*arguments: str) -> None:
        subprocess.run(
            ["git", *arguments], cwd=repository, check=True,
            text=True, capture_output=True,
        )

    git("init", "--quiet")
    git("config", "core.autocrlf", "true")
    git("config", "user.name", "Modal lock test")
    git("config", "user.email", "modal-lock-test@example.invalid")
    (repository / ".gitattributes").write_bytes(
        (ROOT / ".gitattributes").read_bytes()
    )
    lock = repository / "requirements/modal-launcher-v1.lock"
    lock.parent.mkdir()
    lock.write_bytes(expected)
    control = repository / "native-checkout.txt"
    control.write_bytes(b"one\ntwo\n")
    git("add", ".gitattributes", "requirements/modal-launcher-v1.lock", "native-checkout.txt")
    git("commit", "--quiet", "-m", "fixture")
    lock.unlink()
    control.unlink()
    git("checkout", "--quiet", "--", "requirements/modal-launcher-v1.lock", "native-checkout.txt")

    assert control.read_bytes() == b"one\r\ntwo\r\n"
    assert lock.read_bytes() == expected


@pytest.mark.parametrize("member", ["modal_worker_ports", "modal_worker_source"])
def test_extracted_worker_source_cannot_disappear_from_lock(member):
    from tuner.execution.providers.modal.config import ModalRuntimeLockV1

    document = ModalRuntimeLockV1.packaged().to_dict()
    del document["locked_files"][member]
    schema = json.loads((ROOT / "schemas/synaptic-modal-runtime-lock-v1.schema.json").read_text(encoding="utf-8"))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(document)
    with pytest.raises(ValueError):
        ModalRuntimeLockV1(document)
