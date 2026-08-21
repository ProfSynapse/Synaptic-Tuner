from __future__ import annotations

import json
import sys
from types import ModuleType

import pytest

from tuner.cloud.hf_training_smoke_artifacts import EXPECTED_PATHS
import tuner.cloud.hf_training_smoke_download_child as child
from tuner.cloud.hf_training_smoke_download_child import _inventory


class _BucketFile:
    __dataclass_fields__ = {
        name: object() for name in ("type", "path", "size", "xet_hash", "mtime", "uploaded_at")
    }
    def __init__(self, **kwargs):
        self.type = kwargs.pop("type")
        self.path = kwargs.pop("path")
        self.size = kwargs.pop("size")
        self.xet_hash = kwargs.pop("xetHash")
        if kwargs:
            raise TypeError("unexpected fields")


def _install_bucket_file(monkeypatch) -> None:
    package = ModuleType("huggingface_hub")
    api = ModuleType("huggingface_hub.hf_api")
    api.BucketFile = _BucketFile
    monkeypatch.setitem(sys.modules, "huggingface_hub", package)
    monkeypatch.setitem(sys.modules, "huggingface_hub.hf_api", api)


def _value():
    return [
        {"path": path, "bytes": index + 1, "provider_xet_hash": f"{index + 1:064x}"}
        for index, path in enumerate(sorted(EXPECTED_PATHS))
    ]


def test_inventory_builds_exact_prelisted_remote_local_pairs(tmp_path, monkeypatch) -> None:
    _install_bucket_file(monkeypatch)
    destination = tmp_path / "output"
    destination.mkdir()
    inventory = tmp_path / "inventory.json"
    inventory.write_bytes(json.dumps(_value(), ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii"))
    pairs = _inventory(inventory, destination, "training/slot")
    assert len(pairs) == 15
    assert [remote.path for remote, _local in pairs] == [
        f"training/slot/{path}" for path in sorted(EXPECTED_PATHS)
    ]
    assert [local.relative_to(destination).as_posix() for _remote, local in pairs] == sorted(EXPECTED_PATHS)


@pytest.mark.parametrize(
    "raw",
    [
        b'[{"path":"x","path":"y","bytes":1,"provider_xet_hash":null}]',
        b'[{"bytes":NaN,"path":"x","provider_xet_hash":null}]',
        b' [ ]',
        b'[]\n',
    ],
)
def test_inventory_rejects_hostile_or_noncanonical_json(tmp_path, monkeypatch, raw: bytes) -> None:
    _install_bucket_file(monkeypatch)
    destination = tmp_path / "output"
    destination.mkdir()
    inventory = tmp_path / "inventory.json"
    inventory.write_bytes(raw)
    with pytest.raises(Exception):
        _inventory(inventory, destination, "training/slot")


def test_baseexception_escapes_without_status_envelope(monkeypatch, capfd) -> None:
    class Parser:
        def parse_args(self, argv):
            raise KeyboardInterrupt

    monkeypatch.setattr(child, "_parser", lambda: Parser())
    with pytest.raises(KeyboardInterrupt):
        child.run([])
    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""
