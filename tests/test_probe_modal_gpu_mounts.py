"""Provider-free checks for the private GPU mount diagnostic."""

from __future__ import annotations

import importlib.util
import asyncio
from contextlib import nullcontext
import os
from pathlib import Path
import pickle
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probe_modal_gpu_mounts.py"
_SPEC = importlib.util.spec_from_file_location("probe_modal_gpu_mounts", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)


@pytest.mark.skipif(os.name != "posix", reason="Linux descriptor semantics")
def test_exclusive_claim_blocks_replay(tmp_path: Path) -> None:
    claim = {"schema_version": probe._SCHEMA, "selection_sha256": "a" * 64}
    root = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        probe._exclusive_record(root, "claim.json", claim)
        assert (tmp_path / "claim.json").read_text().strip()
        with pytest.raises(probe.ProbeUnavailable, match="CLAIM_ALREADY_CONSUMED"):
            probe._exclusive_record(root, "claim.json", claim)
    finally:
        os.close(root)


def test_raw_result_requires_exact_serialized_closed_payload() -> None:
    api = SimpleNamespace(
        DATA_FORMAT_PICKLE=1,
        GenericResult=SimpleNamespace(GENERIC_STATUS_SUCCESS=1),
    )
    values = ("LINK", "DIRECTORY", "ABSENT")
    payload = pickle.dumps({"schema_version": probe._SCHEMA, "roots": values})
    output = SimpleNamespace(
        data_format=1,
        result=SimpleNamespace(status=1, data_blob_id="", data=payload),
    )
    assert probe._classify_raw(output, api, pickle.dumps) == {
        "control": "LINK", "artifacts": "DIRECTORY", "model_cache": "ABSENT",
    }
    output.result.data = pickle.dumps({
        "schema_version": probe._SCHEMA, "roots": values,
        "extra": "untrusted",
    })
    assert probe._classify_raw(output, api, pickle.dumps) == "OUTPUT_UNCLASSIFIED"
    output.result.data_blob_id = "blob"
    assert probe._classify_raw(output, api, pickle.dumps) == "OUTPUT_UNCLASSIFIED"


def test_image_read_requires_exact_returned_id() -> None:
    class Request:
        def __init__(self, *, image_id):
            self.image_id = image_id

    class Response:
        def __init__(self, image_id):
            self.image_id = image_id

    class Stub:
        returned = "im-EXACT"

        async def ImageFromId(self, request, *, retry, timeout):
            assert request.image_id == "im-EXACT"
            assert retry is None and timeout == 15
            return Response(self.returned)

    api = SimpleNamespace(ImageFromIdRequest=Request, ImageFromIdResponse=Response)
    stub = Stub()
    client = SimpleNamespace(stub=stub)
    assert asyncio.run(probe._read_image_identity(client, "im-EXACT", api)) is True
    stub.returned = "im-OTHER"
    assert asyncio.run(probe._read_image_identity(client, "im-EXACT", api)) is False


@pytest.mark.skipif(os.name != "posix", reason="Linux descriptor semantics")
def test_remote_probe_never_reads_symlink_target(monkeypatch) -> None:
    import os
    import stat

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(os, "open", lambda *args, **kwargs: 17)
    monkeypatch.setattr(os, "close", lambda descriptor: None)

    def fake_stat(leaf, *, dir_fd=None, follow_symlinks=True):
        calls.append((leaf, "follow" if follow_symlinks else "nofollow"))
        if leaf == "control":
            return SimpleNamespace(st_mode=stat.S_IFLNK)
        if leaf == "artifacts":
            return SimpleNamespace(st_mode=stat.S_IFREG)
        raise FileNotFoundError

    monkeypatch.setattr(os, "stat", fake_stat)
    assert probe._remote_probe() == {
        "schema_version": probe._SCHEMA,
        "roots": ("LINK", "OTHER", "ABSENT"),
    }
    assert calls == [
        ("control", "nofollow"),
        ("artifacts", "nofollow"),
        ("model-cache", "nofollow"),
    ]


@pytest.mark.parametrize("hydrate_after_deploy", [True, False])
def test_execute_creates_exact_resources_then_spawns_once(
    monkeypatch, hydrate_after_deploy: bool,
) -> None:
    events: list[tuple[str, object]] = []
    ids = iter(("vo-C", "vo-A", "vo-M"))

    class Volume:
        is_hydrated = True

        def __init__(self):
            self.object_id = next(ids)

        def hydrate(self, client):
            events.append(("hydrate_volume", self.object_id))

        @staticmethod
        def from_name(name, **kwargs):
            assert kwargs["create_if_missing"] is False
            return Volume()

        objects = SimpleNamespace(create=lambda name, **kwargs: (
            events.append(("create_volume", (name, kwargs["allow_existing"])))
        ))

    class Image:
        latest = None

        def __init__(self, identity):
            self.object_id = identity
            self.is_hydrated = False
            Image.latest = self

        @staticmethod
        def from_id(identity, **kwargs):
            events.append(("image_handle", identity))
            return Image(identity)

    class Function:
        def spawn(self):
            events.append(("spawn", None))
            return SimpleNamespace(object_id="fc-EXACT")

    class App:
        def __init__(self, name, **kwargs):
            events.append(("construct_app", name))

        def function(self, **kwargs):
            assert kwargs["gpu"] == "L40S"
            assert kwargs["volumes"].keys() == set(probe._MOUNTS)
            assert kwargs["secrets"] == []
            assert kwargs["block_network"] is True
            assert kwargs["restrict_modal_access"] is True
            assert kwargs["retries"] == 0
            assert kwargs["serialized"] is True
            events.append(("configure_function", None))
            return lambda fn: Function()

        def deploy(self, **kwargs):
            assert Image.latest is not None
            Image.latest.is_hydrated = hydrate_after_deploy
            events.append(("deploy", None))

    sdk = SimpleNamespace(App=App, Volume=Volume, Image=Image)
    modal = ModuleType("modal")
    modal.__path__ = []
    modal_utils = ModuleType("modal._utils")
    modal_utils.__path__ = []
    async_utils = ModuleType("modal._utils.async_utils")
    async_utils.synchronizer = SimpleNamespace(
        create_blocking=lambda fn: (
            (lambda *args: True) if fn is probe._read_app_absence
            else (lambda *args: True) if fn is probe._read_image_identity
            else (lambda *args: {"control": "LINK"})
        ),
    )
    serialization = ModuleType("modal._serialization")
    serialization.serialize = pickle.dumps
    proto = ModuleType("modal_proto")
    proto.api_pb2 = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "modal", modal)
    monkeypatch.setitem(sys.modules, "modal._utils", modal_utils)
    monkeypatch.setitem(sys.modules, "modal._utils.async_utils", async_utils)
    monkeypatch.setitem(sys.modules, "modal._serialization", serialization)
    monkeypatch.setitem(sys.modules, "modal_proto", proto)
    monkeypatch.setattr(probe, "_deadline", lambda seconds: nullcontext())
    monkeypatch.setattr(probe, "_exclusive_record", lambda *args: events.append(("record_call", None)))
    args = SimpleNamespace(
        app="probe-test", environment="default", image_id="im-EXACT",
        volume_names=("probe-control", "probe-artifacts", "probe-cache"), claim_fd=17,
    )
    if hydrate_after_deploy:
        assert probe.execute(args, sdk, object()) == {"control": "LINK"}
    else:
        with pytest.raises(probe.ProbeUnavailable, match="IMAGE_IDENTITY_INVALID"):
            probe.execute(args, sdk, object())
        assert [name for name, _ in events].count("spawn") == 0
        return
    assert [name for name, _ in events].count("spawn") == 1
    assert [name for name, _ in events].index("image_handle") < [name for name, _ in events].index("deploy")
    assert [value[1] for name, value in events if name == "create_volume"] == [False] * 3
    assert [name for name, _ in events].index("deploy") < [name for name, _ in events].index("spawn")
    assert [name for name, _ in events].index("spawn") < [name for name, _ in events].index("record_call")


def test_main_claims_generated_selection_before_provider_use(monkeypatch, capsys) -> None:
    events = []
    fake_modal = ModuleType("modal")
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setattr(probe, "_require_host", lambda: None)
    monkeypatch.setattr(probe, "_private_directory", lambda path: 71)
    monkeypatch.setattr(probe.os, "close", lambda descriptor: None)
    monkeypatch.setattr(probe.secrets, "token_hex", lambda count: "a" * 32)
    monkeypatch.setattr(probe, "_exclusive_record", lambda *args: events.append(("claim", args[2])))

    def fake_client(sdk, profile):
        events.append(("client", profile))
        return object()

    def fake_execute(args, sdk, client):
        events.append(("execute", args.app))
        assert args.volume_names == (
            "probe-control-" + "a" * 32,
            "probe-artifacts-" + "a" * 32,
            "probe-model-cache-" + "a" * 32,
        )
        return {"control": "LINK"}

    monkeypatch.setattr(probe, "_client", fake_client)
    monkeypatch.setattr(probe, "execute", fake_execute)
    exit_code = probe.main([
        "--claim-dir", "/private/probe", "--name-prefix", "probe",
        "--environment", "default", "--modal-profile", "test",
        "--image-id", "im-EXACT",
    ])
    assert exit_code == 0
    assert [name for name, _ in events] == ["claim", "client", "execute"]
    assert events[0][1]["app"] == "probe-probe-" + "a" * 32
    assert "DIAGNOSTIC_ONLY" in capsys.readouterr().out
