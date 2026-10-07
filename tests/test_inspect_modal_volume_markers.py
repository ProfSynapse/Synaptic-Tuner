"""Provider-free checks for retained-claim marker inspection."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "inspect_modal_volume_markers.py"
_SPEC = importlib.util.spec_from_file_location("inspect_modal_volume_markers", _SCRIPT)
inspection = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(inspection)


def _records(private: Path):
    names = [f"probe-{number:032x}.bin" for number in (1, 2, 3)]
    digests = [hashlib.sha256(bytes([number]) * 32).hexdigest()
               for number in (1, 2, 3)]
    ids = [f"vo-Test{number}" for number in (1, 2, 3)]
    claim = {
        "schema_version": inspection._SCHEMA, "app": "test-probe-app",
        "environment": "test", "image_id": "im-Test", "mount_parent": "/mnt",
        "mount_paths": ["/mnt/control", "/mnt/artifacts", "/mnt/model-cache"],
        "inspect_links": False, "verify_mapping": True,
        "marker_names": names, "marker_sha256": digests,
        "volume_names": ["test-control", "test-artifacts", "test-model-cache"],
    }
    claim["selection_sha256"] = hashlib.sha256(json.dumps(
        claim, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    receipt = {
        "schema_version": inspection._SCHEMA,
        "selection_sha256": claim["selection_sha256"],
        "marker_names": names, "marker_sha256": digests,
        "volume_ids": ids,
        "marker_upload": "accepted_create_only_no_host_readback",
    }
    for filename, value in (("claim.json", claim), ("marker-receipt.json", receipt)):
        target = private / filename
        target.write_bytes((json.dumps(value, sort_keys=True,
                                       separators=(",", ":")) + "\n").encode())
        target.chmod(0o600)
    return ids, names, digests


@pytest.mark.skipif(os.name != "posix", reason="private POSIX retained records")
def test_authenticates_exact_receipt_tuple_without_writes(tmp_path, monkeypatch):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    monkeypatch.setattr(inspection, "_private_directory", lambda path: os.open(
        path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    ))
    ids, names, digests = _records(private)
    before = {entry.name: entry.read_bytes() for entry in private.iterdir()}
    inspection.authenticate_marker(private, ids[1], names[1], digests[1])
    for triple in ((ids[0], names[1], digests[1]),
                   (ids[1], names[1], digests[0]),
                   (ids[1], "other", digests[1])):
        with pytest.raises(inspection.InspectionUnavailable, match="CLAIM_INVALID"):
            inspection.authenticate_marker(private, *triple)
    assert {entry.name: entry.read_bytes() for entry in private.iterdir()} == before


@pytest.mark.skipif(os.name != "posix", reason="private POSIX retained records")
@pytest.mark.parametrize("change", ["selection", "receipt", "extra", "permissions"])
def test_rejects_tampered_or_exposed_records(tmp_path, monkeypatch, change):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    monkeypatch.setattr(inspection, "_private_directory", lambda path: os.open(
        path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    ))
    ids, names, digests = _records(private)
    if change == "permissions":
        (private / "claim.json").chmod(0o644)
    else:
        target = private / ("claim.json" if change in {"selection", "extra"}
                            else "marker-receipt.json")
        value = json.loads(target.read_bytes())
        if change == "selection":
            value["image_id"] = "im-Tampered"
        elif change == "receipt":
            value["volume_ids"][0] = "vo-Tampered"
        else:
            value["extra"] = "unbound"
        target.write_bytes((json.dumps(value, sort_keys=True,
                                       separators=(",", ":")) + "\n").encode())
    with pytest.raises(inspection.InspectionUnavailable, match="CLAIM_INVALID"):
        inspection.authenticate_marker(private, ids[0], names[0], digests[0])


def test_inspection_uses_only_exact_32_byte_digest_read():
    calls = []
    class Reader:
        async def list_prefix(self, **_):
            raise AssertionError("listing must remain unqualified")

        async def read_exact(self, **kwargs):
            calls.append(("read", kwargs))
            return b"opaque"  # Never returned by the inspection.

    path = "probe-" + "1" * 32 + ".bin"
    digest = "a" * 64
    result = asyncio.run(inspection.inspect(
        Reader(), volume_id="vo-Test", path=path, digest=digest,
    ))
    assert result == "MARKER_MATCH"
    assert calls == [
        ("read", {"volume_id": "vo-Test", "path": path, "expected_size": 32,
                  "expected_sha256": digest, "max_bytes": 32}),
    ]


def test_read_failure_is_closed():
    class Reader:
        async def read_exact(self, **_):
            raise inspection.BoundedVolumeReadError("modal_volume_download_failed")

    assert asyncio.run(inspection.inspect(
        Reader(), volume_id="vo-Test", path="marker", digest="a" * 64,
    )) == "PROVIDER_READ_UNAVAILABLE"


def test_invalid_claim_stops_before_provider_import(tmp_path, monkeypatch, capsys):
    imported = []
    original = __import__("builtins").__import__

    def guarded(name, *args, **kwargs):
        if name == "modal" or name.startswith("modal."):
            imported.append(name)
            raise AssertionError("provider imported")
        return original(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded)
    code = inspection.main([
        "--claim-dir", str(tmp_path / "missing"),
        "--volume-id", "vo-Test", "--path", "probe-" + "1" * 32 + ".bin",
        "--sha256", "a" * 64,
        "--modal-profile", "test-profile",
    ])
    assert code == 1
    assert imported == []
    assert json.loads(capsys.readouterr().out) == {
        "schema_version": inspection._OUTPUT_SCHEMA,
        "authority": "DIAGNOSTIC_ONLY", "result": "CLAIM_INVALID",
    }
