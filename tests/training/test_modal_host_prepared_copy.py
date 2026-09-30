from __future__ import annotations

import os

import pytest

from tests.dataset_prep.test_context_messages_v2 import _config
from tuner.dataset_prep import DatasetPublicationUncertainV1, prepare_dataset_v2
from tuner.training.modal_host_prepared_copy import (
    ModalPreparedCopyUnavailable, stage_published_modal_dataset,
)
from tuner.dataset_prep.publication import snapshot_prepared_dataset_v2


pytestmark = pytest.mark.skipif(os.name != "posix", reason="first host is POSIX-only")


def test_private_copy_preserves_exact_publication_and_is_one_use(tmp_path):
    _, _, _, config = _config(tmp_path / "source")
    published_root = tmp_path / "published"
    try:
        publication = prepare_dataset_v2(config, published_root)
    except DatasetPublicationUncertainV1:
        publication = prepare_dataset_v2(config, published_root)
    identity = publication.semantic_identity
    private = (tmp_path / "native-private").resolve()
    first = stage_published_modal_dataset(
        publication.path / "dataset.jsonl", private_root=private,
        expected_dataset_digest=identity.dataset_digest,
        expected_content_digest=identity.dataset_sha256,
    )
    copied, _ = snapshot_prepared_dataset_v2(first.parent)
    assert copied.semantic_identity == identity
    assert first.parent.parent == private
    assert first.parent.stat().st_mode & 0o077 == 0
    with pytest.raises(ModalPreparedCopyUnavailable):
        stage_published_modal_dataset(
            publication.path, private_root=private,
            expected_dataset_digest=identity.dataset_digest,
            expected_content_digest=identity.dataset_sha256,
        )


def test_private_copy_rejects_changed_identity_without_path_leak(tmp_path):
    _, _, _, config = _config(tmp_path / "source")
    publication = prepare_dataset_v2(config, tmp_path / "published")
    with pytest.raises(ModalPreparedCopyUnavailable) as error:
        stage_published_modal_dataset(
            publication.path, private_root=(tmp_path / "native-private").resolve(),
            expected_dataset_digest="0" * 64,
            expected_content_digest="0" * 64,
        )
    assert "published" not in str(error.value)
