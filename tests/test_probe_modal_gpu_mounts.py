"""Provider-free checks for the private GPU mount diagnostic."""

from __future__ import annotations

import importlib.util
import asyncio
import hashlib
import itertools
import json
from contextlib import contextmanager, nullcontext
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
    output.result.data = pickle.dumps({
        "schema_version": probe._SCHEMA,
        "mapping": ("MATCH", "MARKER_MISSING", "ROOT_UNSAFE"),
    })
    assert probe._classify_raw(output, api, pickle.dumps, False, True) == {
        "control": "MATCH", "artifacts": "MARKER_MISSING",
        "model_cache": "ROOT_UNSAFE",
    }
    assert probe._classify_raw(output, api, pickle.dumps, True) == "OUTPUT_UNCLASSIFIED"
    output.result.data_blob_id = "blob"
    assert probe._classify_raw(output, api, pickle.dumps) == "OUTPUT_UNCLASSIFIED"
    output.result.data_blob_id = ""
    output.result.data = pickle.dumps({
        "schema_version": probe._SCHEMA,
        "links": ("ABS_TARGET_OWNER_ROOT", "REL_TRAVERSAL", "NOT_LINK"),
    })
    assert probe._classify_raw(output, api, pickle.dumps, True) == {
        "control": "ABS_TARGET_OWNER_ROOT",
        "artifacts": "REL_TRAVERSAL",
        "model_cache": "NOT_LINK",
    }
    assert probe._classify_raw(output, api, pickle.dumps) == "OUTPUT_UNCLASSIFIED"


def test_link_result_inventory_fits_inline_pickle_cap() -> None:
    assert max(len(pickle.dumps({
        "schema_version": probe._SCHEMA, "links": values,
    })) for values in itertools.product(probe._LINK_CATEGORIES, repeat=3)) <= 512


def test_mapping_result_inventory_fits_inline_pickle_cap() -> None:
    assert max(len(pickle.dumps({
        "schema_version": probe._SCHEMA, "mapping": values,
    })) for values in itertools.product(probe._MAPPING_CATEGORIES, repeat=3)) <= 512


def test_mapping_target_category_never_emits_path() -> None:
    identity = "vo-Exact123"
    assert probe._mapping_target_category("/__modal/volumes/" + identity, identity) == "MATCH_MODAL_ID"
    assert probe._mapping_target_category("/__modal/volumes/" + identity + "/data", identity) == "MATCH_MODAL_ID_DESCENDANT"
    assert probe._mapping_target_category("/__modal/volumes/" + identity + "-other", identity) == "MATCH"
    assert probe._mapping_target_category("/other/private/path", identity) == "MATCH"
    assert probe._mapping_target_category("/other/private/path", "bad") == "UNAVAILABLE"


def test_prepare_marker_submits_only_create_only_upload(monkeypatch) -> None:
    monkeypatch.setattr(probe, "_deadline", lambda seconds: nullcontext())
    name = "probe-" + "a" * 32 + ".bin"
    raw = b"x" * 32
    events = []

    class Volume:
        stored = None

        @contextmanager
        def batch_upload(self, *, force):
            assert force is False
            events.append("upload")
            yield self

        def put_file(self, spool, path, *, mode):
            assert path == name and mode == 0o600
            assert spool.seekable()
            self.stored = spool.read()

        def read_file(self, path):
            pytest.fail("host read is not bounded by the Modal SDK")

        def iterdir(self, path, *, recursive):
            pytest.fail("host listing is not bounded by the Modal SDK")

    volume = Volume()
    assert probe._prepare_marker(volume, name, raw) is True
    assert events == ["upload"]
    assert volume.stored == raw


@pytest.mark.skipif(os.name != "posix", reason="Linux descriptor semantics")
def test_mapping_probe_reads_only_role_markers(tmp_path: Path, monkeypatch) -> None:
    names = tuple("probe-" + f"{index:032x}" + ".bin" for index in range(3))
    raw = tuple(bytes([index + 1]) * 32 for index in range(3))
    digests = tuple(hashlib.sha256(value).hexdigest() for value in raw)
    for index, leaf in enumerate(probe._MOUNT_LEAVES):
        target = tmp_path / f"target-{index}"
        target.mkdir(mode=0o700)
        (target / names[index]).write_bytes(raw[index])
        (target / "other-data").write_text("must remain unread")
        (tmp_path / leaf).symlink_to(target.name, target_is_directory=True)
    monkeypatch.setattr(probe, "_MOUNT_PARENTS", (str(tmp_path),))
    original_open = os.open
    def checked_open(path, flags, *args, **kwargs):
        assert path != "other-data"
        assert flags & os.O_NOFOLLOW
        if path.startswith("probe-"):
            assert flags & os.O_NONBLOCK
        return original_open(path, flags, *args, **kwargs)
    monkeypatch.setattr(os, "open", checked_open)
    monkeypatch.setattr(os, "listdir", lambda *args, **kwargs: pytest.fail("listed contents"))
    monkeypatch.setattr(os, "scandir", lambda *args, **kwargs: pytest.fail("scanned contents"))
    result = probe._remote_probe(str(tmp_path), False, True, names, digests)
    assert result == {"schema_version": probe._SCHEMA, "mapping": ("MATCH",) * 3}
    switched = probe._remote_probe(str(tmp_path), False, True, names, digests[::-1])
    assert switched["mapping"] == ("MISMATCH", "MATCH", "MISMATCH")
    (tmp_path / "control").unlink()
    (tmp_path / "control").symlink_to("target-1", target_is_directory=True)
    remapped = probe._remote_probe(str(tmp_path), False, True, names, digests)
    assert remapped["mapping"] == ("MARKER_MISSING", "MATCH", "MATCH")


def test_repeated_dynamic_link_labels_match_closed_pickle_inventory() -> None:
    dynamic = tuple("".join(("ABS_", "TARGET_OWNER_ROOT")) for _ in range(3))
    assert dynamic[0] is not dynamic[1]
    canonical = tuple(probe._canonical_link_category(value) for value in dynamic)
    expected = (probe._LINK_CATEGORIES[probe._LINK_CATEGORIES.index(
        "ABS_TARGET_OWNER_ROOT")],) * 3
    assert canonical == expected
    assert canonical[0] is canonical[1]
    assert pickle.dumps({"schema_version": probe._SCHEMA, "links": canonical}) == \
        pickle.dumps({"schema_version": probe._SCHEMA, "links": expected})


@pytest.mark.skipif(os.name != "posix", reason="Linux descriptor semantics")
def test_remote_repeated_link_result_matches_closed_raw_inventory(
    tmp_path: Path, monkeypatch,
) -> None:
    (tmp_path / "target").mkdir(mode=0o700)
    for leaf in probe._MOUNT_LEAVES:
        (tmp_path / leaf).symlink_to("target", target_is_directory=True)
    monkeypatch.setattr(probe, "_MOUNT_PARENTS", (str(tmp_path),))
    result = probe._remote_probe(str(tmp_path), True)
    assert result["links"] == ("REL_TARGET_OWNER_SELF",) * 3
    api = SimpleNamespace(GenericResult=SimpleNamespace(GENERIC_STATUS_SUCCESS=1),
                          DATA_FORMAT_PICKLE=2)
    output = SimpleNamespace(
        result=SimpleNamespace(status=1, data_blob_id="", data=pickle.dumps(result)),
        data_format=2,
    )
    assert probe._classify_raw(output, api, pickle.dumps, True) == {
        "control": "REL_TARGET_OWNER_SELF",
        "artifacts": "REL_TARGET_OWNER_SELF",
        "model_cache": "REL_TARGET_OWNER_SELF",
    }


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
@pytest.mark.parametrize("mount_parent", ["/mnt", "/workspace"])
def test_remote_probe_never_reads_symlink_target(monkeypatch, mount_parent) -> None:
    import os
    import stat

    calls: list[tuple[str, str]] = []
    opens = []
    def fake_open(*args, **kwargs):
        opens.append((args, kwargs))
        return 17
    monkeypatch.setattr(os, "open", fake_open)
    monkeypatch.setattr(os, "close", lambda descriptor: None)

    def fake_stat(leaf, *, dir_fd=None, follow_symlinks=True):
        calls.append((leaf, "follow" if follow_symlinks else "nofollow"))
        if leaf == "control":
            return SimpleNamespace(st_mode=stat.S_IFLNK)
        if leaf == "artifacts":
            return SimpleNamespace(st_mode=stat.S_IFREG)
        raise FileNotFoundError

    monkeypatch.setattr(os, "stat", fake_stat)
    assert probe._remote_probe(mount_parent) == {
        "schema_version": probe._SCHEMA,
        "roots": ("LINK", "OTHER", "ABSENT"),
    }
    assert opens == [((mount_parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW), {})]
    assert calls == [
        ("control", "nofollow"),
        ("artifacts", "nofollow"),
        ("model-cache", "nofollow"),
    ]


@pytest.mark.parametrize(
    ("hydrate_after_deploy", "deploy_fails", "mutate_volume_after_deploy"),
    [(True, False, False), (False, False, False),
     (True, True, False), (True, False, True)],
)
@pytest.mark.parametrize("mount_parent", ["/mnt", "/workspace"])
@pytest.mark.parametrize(
    ("inspect_links", "verify_mapping"),
    [(False, False), (True, False), (False, True)],
)
def test_execute_creates_exact_resources_then_spawns_once(
    monkeypatch, hydrate_after_deploy: bool, deploy_fails: bool,
    mutate_volume_after_deploy: bool, mount_parent: str,
    inspect_links: bool, verify_mapping: bool,
) -> None:
    events: list[tuple[str, object]] = []
    ids = iter(("vo-C", "vo-A", "vo-M"))

    class Volume:
        is_hydrated = True
        instances = []

        def __init__(self):
            self.object_id = next(ids)
            Volume.instances.append(self)

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
        def spawn(self, parent, inspect, verify, names, digests, ids):
            events.append(("spawn", (parent, inspect, verify, names, digests, ids)))
            return SimpleNamespace(object_id="fc-EXACT")

    class App:
        def __init__(self, name, **kwargs):
            events.append(("construct_app", name))

        def function(self, **kwargs):
            assert kwargs["name"] == "mount_probe"
            assert kwargs["gpu"] == "L40S"
            assert kwargs["volumes"].keys() == set(probe._mount_paths(mount_parent))
            assert kwargs["secrets"] == []
            assert kwargs["block_network"] is True
            assert kwargs["restrict_modal_access"] is True
            assert kwargs["retries"] == 0
            assert kwargs["serialized"] is False
            assert kwargs["include_source"] is True
            events.append(("configure_function", None))
            return lambda fn: Function()

        def deploy(self, **kwargs):
            if deploy_fails:
                raise ValueError("secret provider payload")
            assert Image.latest is not None
            Image.latest.is_hydrated = hydrate_after_deploy
            if mutate_volume_after_deploy:
                Volume.instances[0].object_id = "vo-Changed"
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
    records = {}
    def fake_record(fd, leaf, value):
        records[leaf] = value
        events.append(("record", leaf))
    monkeypatch.setattr(probe, "_exclusive_record", fake_record)
    monkeypatch.setattr(probe, "_prepare_marker", lambda volume, name, raw: (
        events.append(("marker", volume.object_id)) or True
    ))
    marker_names = tuple("probe-" + f"{index:032x}" + ".bin" for index in range(3))
    marker_bytes = tuple(bytes([index + 1]) * 32 for index in range(3))
    marker_digests = tuple(hashlib.sha256(value).hexdigest() for value in marker_bytes)
    args = SimpleNamespace(
        app="probe-test", environment="default", image_id="im-EXACT",
        volume_names=("probe-control", "probe-artifacts", "probe-cache"), claim_fd=17,
        mount_parent=mount_parent, inspect_links=inspect_links,
        verify_mapping=verify_mapping,
        marker_names=marker_names if verify_mapping else (),
        marker_digests=marker_digests if verify_mapping else (),
        marker_bytes=marker_bytes if verify_mapping else (),
        selection_sha256="a" * 64,
    )
    if deploy_fails:
        with pytest.raises(probe.ProbeUnavailable, match="APP_DEPLOY_UNAVAILABLE") as caught:
            probe.execute(args, sdk, object())
        assert caught.value.failure_class == "ValueError"
        assert [name for name, _ in events].count("spawn") == 0
        return
    if mutate_volume_after_deploy:
        with pytest.raises(probe.ProbeUnavailable, match="VOLUME_IDENTITY_INVALID"):
            probe.execute(args, sdk, object())
        assert [name for name, _ in events].count("spawn") == 0
        return
    if hydrate_after_deploy:
        assert probe.execute(args, sdk, object()) == {"control": "LINK"}
    else:
        with pytest.raises(probe.ProbeUnavailable, match="POSTDEPLOY_IMAGE_INVALID"):
            probe.execute(args, sdk, object())
        assert [name for name, _ in events].count("spawn") == 0
        return
    assert [name for name, _ in events].count("spawn") == 1
    expected_ids = ("vo-C", "vo-A", "vo-M") if verify_mapping else ()
    assert ("spawn", (mount_parent, inspect_links, verify_mapping,
                      args.marker_names, args.marker_digests, expected_ids)) in events
    assert [name for name, _ in events].index("image_handle") < [name for name, _ in events].index("deploy")
    assert [value[1] for name, value in events if name == "create_volume"] == [False] * 3
    assert [name for name, _ in events].index("deploy") < [name for name, _ in events].index("spawn")
    assert events.index(("spawn", (mount_parent, inspect_links, verify_mapping,
                                   args.marker_names, args.marker_digests, expected_ids))) < \
        events.index(("record", "call.json"))
    if verify_mapping:
        assert [value for name, value in events if name == "marker"] == ["vo-C", "vo-A", "vo-M"]
        assert events.index(("record", "marker-receipt.json")) < \
            [name for name, _ in events].index("deploy")
        assert records["marker-receipt.json"] == {
            "schema_version": probe._SCHEMA,
            "selection_sha256": "a" * 64,
            "marker_names": marker_names,
            "marker_sha256": marker_digests,
            "volume_ids": ("vo-C", "vo-A", "vo-M"),
            "marker_upload": "accepted_create_only_no_host_readback",
        }


@pytest.mark.parametrize("mount_parent", ["/mnt", "/workspace"])
@pytest.mark.parametrize("inspect_links", [False, True])
def test_main_claims_generated_selection_before_provider_use(monkeypatch, capsys,
                                                              mount_parent,
                                                              inspect_links) -> None:
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
        "--image-id", "im-EXACT", "--mount-parent", mount_parent,
        *(["--inspect-links"] if inspect_links else []),
    ])
    assert exit_code == 0
    assert [name for name, _ in events] == ["claim", "client", "execute"]
    assert events[0][1]["app"] == "probe-probe-" + "a" * 32
    assert events[0][1]["mount_parent"] == mount_parent
    assert events[0][1]["mount_paths"] == probe._mount_paths(mount_parent)
    assert events[0][1]["inspect_links"] is inspect_links
    assert events[0][1]["verify_mapping"] is False
    assert events[0][1]["marker_names"] == ()
    assert events[0][1]["marker_sha256"] == ()
    claim = dict(events[0][1])
    digest = claim.pop("selection_sha256")
    assert digest == hashlib.sha256(
        json.dumps(claim, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert "DIAGNOSTIC_ONLY" in capsys.readouterr().out


def test_mapping_claim_binds_distinct_markers_before_provider_use(monkeypatch, capsys) -> None:
    events = []
    fake_modal = ModuleType("modal")
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setattr(probe, "_require_host", lambda: None)
    monkeypatch.setattr(probe, "_private_directory", lambda path: 71)
    monkeypatch.setattr(probe.os, "close", lambda descriptor: None)
    names = iter(("a" * 32, "b" * 32, "c" * 32, "d" * 32))
    monkeypatch.setattr(probe.secrets, "token_hex", lambda count: next(names))
    marker_bytes = tuple(bytes([index + 1]) * 32 for index in range(3))
    values = iter(marker_bytes)
    monkeypatch.setattr(probe.secrets, "token_bytes", lambda count: next(values))
    monkeypatch.setattr(probe, "_exclusive_record", lambda *args: events.append(("claim", args[2])))
    monkeypatch.setattr(probe, "_client", lambda sdk, profile: events.append(("client", profile)) or object())

    def fake_execute(args, sdk, client):
        events.append(("execute", args.app))
        assert args.marker_bytes == marker_bytes
        return {"control": "MATCH"}

    monkeypatch.setattr(probe, "execute", fake_execute)
    assert probe.main([
        "--claim-dir", "/private/probe", "--name-prefix", "probe",
        "--environment", "default", "--modal-profile", "test",
        "--image-id", "im-EXACT", "--verify-mapping",
    ]) == 0
    assert [name for name, _ in events] == ["claim", "client", "execute"]
    claim = dict(events[0][1])
    assert claim["verify_mapping"] is True
    assert claim["inspect_links"] is False
    assert claim["marker_names"] == tuple(
        "probe-" + char * 32 + ".bin" for char in "abc"
    )
    assert claim["marker_sha256"] == tuple(hashlib.sha256(value).hexdigest()
                                            for value in marker_bytes)
    assert not any(value.hex() in json.dumps(claim) for value in marker_bytes)
    digest = claim.pop("selection_sha256")
    assert digest == hashlib.sha256(json.dumps(
        claim, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    assert '"result":{"control":"MATCH"}' in capsys.readouterr().out


def test_mapping_mode_rejects_combination_before_claim(monkeypatch, capsys) -> None:
    monkeypatch.setattr(probe, "_require_host", lambda: None)
    monkeypatch.setattr(probe, "_private_directory", lambda path: 71)
    monkeypatch.setattr(probe.os, "close", lambda descriptor: None)
    monkeypatch.setattr(probe, "_exclusive_record", lambda *args: pytest.fail("claim written"))
    assert probe.main([
        "--claim-dir", "/private/probe", "--name-prefix", "probe",
        "--environment", "default", "--modal-profile", "test",
        "--image-id", "im-EXACT", "--verify-mapping", "--inspect-links",
    ]) == 1
    assert '"result":"INPUT_INVALID"' in capsys.readouterr().out


def test_mount_parent_rejects_unlisted_path() -> None:
    with pytest.raises(probe.ProbeUnavailable, match="INPUT_INVALID"):
        probe._mount_paths("/root")
    with pytest.raises(probe.ProbeUnavailable, match="INPUT_INVALID"):
        probe._mount_paths("/workspace/../mnt")


@pytest.mark.parametrize("inspect_links", [False, True])
def test_remote_probe_rejects_unlisted_parent_before_io(monkeypatch, inspect_links) -> None:
    monkeypatch.setattr(probe.os, "open", lambda *args, **kwargs: pytest.fail("unexpected open"))
    assert probe._remote_probe("/root", inspect_links) == {
        "schema_version": probe._SCHEMA,
        "links" if inspect_links else "roots":
            ("UNAVAILABLE", "UNAVAILABLE", "UNAVAILABLE"),
    }


@pytest.mark.skipif(os.name != "posix", reason="Linux descriptor semantics")
def test_link_metadata_opens_only_directory_roots(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    (target / "secret.txt").write_text("never read this")
    (tmp_path / "control").symlink_to("target", target_is_directory=True)
    original_open = os.open
    opened = []

    def checked_open(path, flags, *args, **kwargs):
        opened.append(path)
        assert path != "secret.txt"
        assert flags & os.O_NOFOLLOW
        return original_open(path, flags, *args, **kwargs)

    parent = original_open(tmp_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        monkeypatch.setattr(os, "open", checked_open)
        monkeypatch.setattr(os, "listdir", lambda *args, **kwargs: pytest.fail("listed contents"))
        monkeypatch.setattr(os, "scandir", lambda *args, **kwargs: pytest.fail("scanned contents"))
        assert probe._inspect_root_link(parent, "control") == "REL_TARGET_OWNER_SELF"
        assert opened == ["target"]
    finally:
        os.close(parent)


@pytest.mark.skipif(os.name != "posix", reason="Linux descriptor semantics")
def test_link_metadata_stops_on_parent_traversal(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "control").symlink_to("../elsewhere", target_is_directory=True)
    parent = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        monkeypatch.setattr(os, "open", lambda *args, **kwargs: pytest.fail("opened target"))
        assert probe._inspect_root_link(parent, "control") == "REL_TRAVERSAL"
    finally:
        os.close(parent)


def test_provider_stage_and_output_do_not_expose_exception_text(monkeypatch, capsys) -> None:
    InvalidError = type("InvalidError", (Exception,), {"__module__": "modal.exception"})
    sdk_frame = {"__name__": "modal._functions", "InvalidError": InvalidError}

    with pytest.raises(probe.ProbeUnavailable) as caught:
        with probe._provider_stage("APP_DEPLOY_UNAVAILABLE"):
            exec(compile(
                "\n" * 1147 +
                "raise InvalidError('volumes mounts unsupported private provider detail')\n",
                "/private/secret/modal/_functions.py", "exec",
            ), sdk_frame)
    assert str(caught.value) == "APP_DEPLOY_UNAVAILABLE"
    assert caught.value.failure_class == "InvalidError"
    assert caught.value.failure_origin == "_functions.py:1148"
    assert caught.value.failure_topics == ("VOLUME", "MOUNT")
    assert caught.value.failure_action == "UNSUPPORTED"

    untrusted_frame = {"__name__": "untrusted.runner"}
    exec(compile(
        "def fail():\n    raise ValueError('another secret')\n",
        "/private/secret/modal/runner.py", "exec",
    ), untrusted_frame)
    with pytest.raises(ValueError) as untrusted:
        untrusted_frame["fail"]()
    assert probe._failure_origin(untrusted.value) is None

    fake_modal = ModuleType("modal")
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setattr(probe, "_require_host", lambda: None)
    monkeypatch.setattr(probe, "_private_directory", lambda path: 71)
    monkeypatch.setattr(probe.os, "close", lambda descriptor: None)
    monkeypatch.setattr(probe, "_exclusive_record", lambda *args: None)
    monkeypatch.setattr(probe, "_client", lambda sdk, profile: object())
    monkeypatch.setattr(probe, "execute", lambda *args: (_ for _ in ()).throw(caught.value))
    assert probe.main([
        "--claim-dir", "/private/probe", "--name-prefix", "probe",
        "--environment", "default", "--modal-profile", "test",
        "--image-id", "im-EXACT",
    ]) == 1
    output = capsys.readouterr().out
    assert '"result":"APP_DEPLOY_UNAVAILABLE"' in output
    assert '"failure_class":"InvalidError"' in output
    assert '"failure_origin":"_functions.py:1148"' in output
    assert '"message_topic_hints":["VOLUME","MOUNT"]' in output
    assert '"message_action_hint":"UNSUPPORTED"' in output
    assert "private provider detail" not in output
    assert "/private/secret" not in output


def test_function_create_topics_absent_for_unmatched_or_other_stage() -> None:
    InvalidError = type("InvalidError", (Exception,), {"__module__": "modal.exception"})
    sdk_frame = {"__name__": "modal._functions", "InvalidError": InvalidError}
    source = "\n" * 1147 + "raise InvalidError('opaque provider value')\n"
    with pytest.raises(probe.ProbeUnavailable) as caught:
        with probe._provider_stage("APP_DEPLOY_UNAVAILABLE"):
            exec(compile(source, "/private/modal/_functions.py", "exec"), sdk_frame)
    assert caught.value.failure_origin == "_functions.py:1148"
    assert caught.value.failure_topics == ()
    assert caught.value.failure_action is None

    source = "\n" * 1147 + "raise InvalidError('volumes unsupported')\n"
    with pytest.raises(probe.ProbeUnavailable) as wrong_stage:
        with probe._provider_stage("FUNCTION_CONSTRUCT_UNAVAILABLE"):
            exec(compile(source, "/private/modal/_functions.py", "exec"), sdk_frame)
    assert wrong_stage.value.failure_topics == ()
    assert wrong_stage.value.failure_action is None

    with pytest.raises(probe.ProbeUnavailable) as wrong_origin:
        with probe._provider_stage("APP_DEPLOY_UNAVAILABLE"):
            exec(compile("\n" + source, "/private/modal/_functions.py", "exec"), sdk_frame)
    assert wrong_origin.value.failure_origin == "_functions.py:1149"
    assert wrong_origin.value.failure_topics == ()
    assert wrong_origin.value.failure_action is None
