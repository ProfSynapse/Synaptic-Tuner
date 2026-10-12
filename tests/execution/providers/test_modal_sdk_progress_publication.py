"""Actual bound root with SDK-shaped create-only complete-byte publication."""
from contextlib import contextmanager
from threading import Event, Lock
from types import SimpleNamespace
import time

import pytest

from tuner.execution.providers.modal.packaged_progress import (
    ModalSDKProgressPublication, SDKProgressPublicationError,
)
from tests import test_modal_volume_root_binding as roots


class MemoryVolume:
    def __init__(self, binding, client):
        self.object_id, self.client = binding.volume_id, client
        self.is_hydrated = True
        self._metadata = SimpleNamespace(version=1)
        self.files, self.modes, self.calls = {}, {}, []
        self.lock = Lock()
        self.started, self.release = None, None
        self.error = None

    @contextmanager
    def batch_upload(self, force=False):
        assert force is False
        self.calls.append(force)
        pending = []
        class Upload:
            def put_file(self, source, path, mode):
                assert mode == 0o600
                pending.append((path.lstrip("/"), source.read(), mode))
        yield Upload()
        if self.started is not None:
            self.started.set()
            assert self.release.wait(3)
        if self.error:
            raise self.error
        with self.lock:
            if any(path in self.files for path, _, _ in pending):
                raise FileExistsError("provider detail never exported")
            for path, raw, mode in pending:
                self.files[path], self.modes[path] = raw, mode

    def read_file(self, path):
        raw = self.files[path.lstrip("/")]
        yield raw[:1]
        yield raw[1:]

    def iterdir(self, prefix, recursive=False):
        assert recursive is False
        for path, raw in self.files.items():
            if path.rsplit("/", 1)[0] == prefix.lstrip("/"):
                yield SimpleNamespace(path=path, size=len(raw), type=1)


@pytest.fixture
def bound(monkeypatch):
    with contextmanager(roots.layout.__wrapped__)(monkeypatch) as layout:
        with roots._bind(layout) as binding:
            yield binding, layout


def publisher(binding, volume, client, *, deadline=None):
    return ModalSDKProgressPublication(binding=binding, volume=volume, client=client,
        prefix="operations/probe/progress", deadline=deadline)


def test_create_only_complete_bytes_collision_and_remote_inventory(bound):
    binding, layout = bound
    client = object()
    volume = MemoryVolume(binding, client)
    first = publisher(binding, volume, client)
    path = "operations/probe/progress/final/000.json"
    first.publish_exclusive(path, b'{"value":1}')
    assert first.read_final(path) == b'{"value":1}'
    assert volume.modes[path] == 0o600
    assert first.final_inventory("operations/probe/progress/final", 2) == ((path, 11),)
    # The mounted namespace is allowed to be stale; it is not final authority.
    assert not (layout[2] / path).exists()
    with pytest.raises(SDKProgressPublicationError) as replay:
        first.publish_exclusive(path, b'{"value":2}')
    assert replay.value.code == "SDK_PUBLISH_REPLAY" and len(volume.calls) == 1
    # A separately admitted one-shot collision exercises the server guard.
    with pytest.raises(SDKProgressPublicationError) as collision:
        publisher(binding, volume, client).publish_exclusive(path, b'{"value":2}')
    assert collision.value.code == "SDK_PUBLISH_COLLISION"
    assert first.read_final(path) == b'{"value":1}'


def test_timeout_retains_uncertain_attempt_and_never_replays(bound):
    binding, _ = bound
    client = object()
    volume = MemoryVolume(binding, client)
    volume.started, volume.release = Event(), Event()
    pub = publisher(binding, volume, client, deadline=time.monotonic() + 0.05)
    path = "operations/probe/progress/final/000.json"
    try:
        with pytest.raises(SDKProgressPublicationError) as timeout:
            pub.publish_exclusive(path, b'{"value":1}')
        assert timeout.value.code == "SDK_PUBLISH_DEADLINE"
        assert volume.started.is_set() and path not in volume.files
        with pytest.raises(SDKProgressPublicationError) as replay:
            pub.publish_exclusive(path, b'{"value":1}')
        assert replay.value.code == "SDK_PUBLISH_REPLAY" and len(volume.calls) == 1
    finally:
        volume.release.set()
    for _ in range(100):
        if path in volume.files:
            break
        time.sleep(0.001)
    assert volume.files[path] == b'{"value":1}'


@pytest.mark.parametrize("path,raw", [("../escape", b"{}"), ("outside/leaf", b"{}"),
    ("operations/probe/progress/final/000.json", b"not canonical"),
    ("operations/probe/progress/final/000.json", b" " * 8193)])
def test_invalid_input_rejects_before_sdk_effect(bound, path, raw):
    binding, _ = bound
    client = object()
    volume = MemoryVolume(binding, client)
    with pytest.raises(SDKProgressPublicationError):
        publisher(binding, volume, client).publish_exclusive(path, raw)
    assert volume.calls == []


@pytest.mark.parametrize("mutation", ["client", "id", "version", "root"])
def test_identity_substitution_rejects_before_effect(bound, mutation):
    binding, layout = bound
    client = object()
    volume = MemoryVolume(binding, client)
    pub = publisher(binding, volume, client)
    if mutation == "client": volume.client = object()
    if mutation == "id": volume.object_id = "vo-Other"
    if mutation == "version": volume._metadata.version = 2
    if mutation == "root": layout[0].unlink()
    with pytest.raises(SDKProgressPublicationError):
        pub.publish_exclusive("operations/probe/progress/final/000.json", b"{}")
    assert volume.calls == []
