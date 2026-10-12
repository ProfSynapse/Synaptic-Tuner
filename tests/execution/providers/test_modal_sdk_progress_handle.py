"""Pinned real SDK synchronization regression; no client connection or RPC."""
from contextlib import contextmanager
from importlib.metadata import version
import hashlib

import pytest

from tuner.execution.providers.modal.packaged_progress import (
    ModalSDKProgressPublication, SDKProgressPublicationError,
)
from tests import test_modal_volume_root_binding as roots


@pytest.mark.parametrize("volume_version", [0, 1, 2])
def test_real_synchronized_volume_metadata_accessor(monkeypatch, volume_version):
    modal = pytest.importorskip("modal")
    from modal_proto import api_pb2
    assert version("modal") == "1.5.4"
    # Construction and controlled hydration perform no connection or RPC.
    client = modal.Client("http://127.0.0.1:1", api_pb2.CLIENT_TYPE_CLIENT, None)
    volume = modal.Volume._new_hydrated("vo-Control", client,
        api_pb2.VolumeMetadata(version=volume_version))
    assert volume.is_hydrated is True
    assert volume.client is client
    assert getattr(volume, "_metadata", None) is None
    assert volume._get_metadata().version == volume_version
    with contextmanager(roots.layout.__wrapped__)(monkeypatch) as layout:
        with roots._bind(layout) as binding:
            if volume_version == 2:
                with pytest.raises(SDKProgressPublicationError) as error:
                    ModalSDKProgressPublication(binding=binding, volume=volume, client=client,
                        prefix="operations/probe/progress", deadline=None)
                assert error.value.code == "SDK_PUBLISH_IDENTITY"
            else:
                publisher = ModalSDKProgressPublication(binding=binding, volume=volume, client=client,
                    prefix="operations/probe/progress", deadline=None)
                publisher._check()
                other_client = modal.Client("http://127.0.0.1:2", api_pb2.CLIENT_TYPE_CLIENT, None)
                with pytest.raises(SDKProgressPublicationError) as wrong_client:
                    ModalSDKProgressPublication(binding=binding, volume=volume, client=other_client,
                        prefix="operations/probe/progress", deadline=None)
                assert wrong_client.value.code == "SDK_PUBLISH_IDENTITY"
                other_volume = modal.Volume._new_hydrated("vo-Other", client,
                    api_pb2.VolumeMetadata(version=volume_version))
                with pytest.raises(SDKProgressPublicationError) as wrong_id:
                    ModalSDKProgressPublication(binding=binding, volume=other_volume, client=client,
                        prefix="operations/probe/progress", deadline=None)
                assert wrong_id.value.code == "SDK_PUBLISH_IDENTITY"


def test_real_container_sdk_upload_collision_and_readback_requires_http(monkeypatch):
    modal = pytest.importorskip("modal")
    from modal_proto import api_pb2
    from google.protobuf.empty_pb2 import Empty
    from modal._utils.async_utils import synchronizer
    from modal._utils.http_utils import ClientSessionRegistry
    assert version("modal") == "1.5.4"
    objects, files, trace = {}, {}, []
    class Stub:
        async def MountPutFile(self, request, **kwargs):
            trace.append("MOUNT_INLINE" if request.data else "MOUNT_HASH")
            if request.data:
                assert len(request.data) <= 8192
                assert hashlib.sha256(request.data).hexdigest() == request.sha256_hex
                objects[request.sha256_hex] = bytes(request.data)
            return api_pb2.MountPutFileResponse(exists=request.sha256_hex in objects)
        async def VolumePutFiles(self, request, **kwargs):
            trace.append("CREATE_ONLY")
            assert request.volume_id == "vo-Control"
            assert request.disallow_overwrite_existing_files is True
            assert len(request.files) == 1 and request.files[0].mode == 0o600
            leaf = request.files[0]
            if leaf.filename in files:
                raise modal.exception.AlreadyExistsError("controlled collision")
            files[leaf.filename] = objects[leaf.sha256_hex]
            return Empty()
        async def VolumeGetFile2(self, request, **kwargs):
            trace.append("READ_URLS")
            assert request.volume_id == "vo-Control" and request.path in files
            return api_pb2.VolumeGetFile2Response(get_urls=["https://unused.invalid/inline-leaf"],
                size=len(files[request.path]))
    client = modal.Client("http://127.0.0.1:1", api_pb2.CLIENT_TYPE_CONTAINER, None)
    synchronizer._translate_in(client)._stub = Stub()
    volume = modal.Volume._new_hydrated("vo-Control", client, api_pb2.VolumeMetadata(version=1))
    class BlockedHTTP:
        def get(self, url):
            assert url == "https://unused.invalid/inline-leaf"
            trace.append("HTTP_DISABLED")
            raise RuntimeError("controlled disabled HTTP")
    monkeypatch.setattr(ClientSessionRegistry, "get_session", lambda: BlockedHTTP())
    with contextmanager(roots.layout.__wrapped__)(monkeypatch) as layout:
        with roots._bind(layout) as binding:
            def publication():
                return ModalSDKProgressPublication(binding=binding, volume=volume, client=client,
                    prefix="operations/probe/progress", deadline=None)
            path = "operations/probe/progress/final/000.json"
            publication().publish_exclusive(path, b'{"value":1}')
            with pytest.raises(SDKProgressPublicationError) as collision:
                publication().publish_exclusive(path, b'{"value":2}')
            assert collision.value.code == "SDK_PUBLISH_COLLISION"
            assert files["/" + path] == b'{"value":1}'
            assert trace == ["MOUNT_HASH", "MOUNT_INLINE", "CREATE_ONLY"] * 2
            # Even an inline-sized v1 leaf takes the URL/HTTP download path.
            # This controlled refusal reproduces the optional read boundary,
            # not the live provider's undisclosed exception or cause.
            with pytest.raises(RuntimeError, match="controlled disabled HTTP"):
                tuple(volume.read_file("/" + path))
            assert trace[6] == "READ_URLS" and "HTTP_DISABLED" in trace[7:]
