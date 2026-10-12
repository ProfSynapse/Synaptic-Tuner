"""Pinned real SDK synchronization regression; no client connection or RPC."""
from contextlib import contextmanager
from importlib.metadata import version

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
