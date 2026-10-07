"""Credential-free real-SDK surface check; no client or request is created."""
from __future__ import annotations

import pytest

from tuner.execution.providers.modal.model_snapshot import _bind_hub_api


def test_installed_hub_supports_the_preparation_contract():
    hub = pytest.importorskip("huggingface_hub")
    _bind_hub_api(hub.HfApi, hub.snapshot_download)
    from huggingface_hub.hf_api import ModelInfo

    info = ModelInfo(id="fixture/model", sha="c" * 40, siblings=[{
        "rfilename": "model.safetensors", "size": 1, "blobId": "a" * 40,
        "lfs": {"sha256": "b" * 64, "size": 1, "pointerSize": 1},
    }])
    item = info.siblings[0]
    assert item.size == 1 and item.blob_id == "a" * 40
    assert item.lfs.sha256 == "b" * 64


def test_incompatible_hub_fails_without_calling_it():
    class API:
        def __init__(self, *, endpoint, token):
            pytest.fail("signature probe constructed a client")

        def model_info(self, repo_id):
            pytest.fail("signature probe made a request")

    with pytest.raises(TypeError):
        _bind_hub_api(API, lambda: pytest.fail("signature probe downloaded"))
