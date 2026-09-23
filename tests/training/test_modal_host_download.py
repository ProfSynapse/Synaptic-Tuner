from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path

import pytest

from synaptic_tuner.api.v1.results import TrainingRunRef, VerifiedArtifact
from tuner.training.modal_host_download import (
    ModalArtifactDownloadUnavailable, download_verified_modal_artifact,
)


pytestmark = pytest.mark.skipif(os.name != "posix", reason="first host is POSIX-only")


class _Stream:
    def __init__(self, chunks):
        self.artifact = VerifiedArtifact("final_model", sha256(b"adapter").hexdigest(), 7)
        self.chunks = chunks

    def iter_bytes(self):
        yield from self.chunks


class _Runs:
    def __init__(self, chunks):
        self.chunks = chunks

    def artifacts(self, request):
        assert request.role == "final_model"
        return _Stream(self.chunks)


def _download(tmp_path: Path, chunks):
    return download_verified_modal_artifact(
        _Runs(chunks), TrainingRunRef("run", "project"),
        role="final_model", output_root=(tmp_path / "private").resolve(),
    )


def test_download_publishes_exact_authenticated_bytes_once(tmp_path):
    result = _download(tmp_path, (b"ad", b"apter"))
    assert result.read_bytes() == b"adapter"
    assert result.stat().st_mode & 0o077 == 0
    with pytest.raises(ModalArtifactDownloadUnavailable, match="modal_artifact_download_failed"):
        _download(tmp_path, (b"adapter",))
    assert result.read_bytes() == b"adapter"


def test_download_does_not_publish_truncated_or_provider_exception(tmp_path):
    with pytest.raises(ModalArtifactDownloadUnavailable, match="modal_artifact_download_failed"):
        _download(tmp_path, (b"ad",))
    assert not (tmp_path / "private" / "final_model.artifact").exists()

    def broken():
        yield b"ad"
        raise RuntimeError("private-token-and-provider-path")

    with pytest.raises(ModalArtifactDownloadUnavailable) as error:
        _download(tmp_path, broken())
    assert "private-token" not in str(error.value)
    assert not (tmp_path / "private" / "final_model.artifact").exists()


def test_download_rejects_symlink_root(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    (tmp_path / "private").symlink_to(target, target_is_directory=True)
    with pytest.raises(ModalArtifactDownloadUnavailable):
        _download(tmp_path, (b"adapter",))
    assert not (target / "final_model.artifact").exists()
