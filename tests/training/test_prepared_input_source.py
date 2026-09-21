from __future__ import annotations

from pathlib import Path

import pytest

from tests.dataset_prep.test_context_messages_v2 import _config
from tuner.dataset_prep import (
    DatasetPublicationUncertainV1,
    DatasetPrepValidationError,
    LocalPreparedTrainingInputSource,
    ROW_SCHEMA_VERSION_V2,
    prepare_dataset_v2,
)
from tuner.training.contracts import PreparedTrainingInputIdentity


def _prepared(tmp_path: Path):
    _bundle, _ids, _documents, config = _config(tmp_path / "source")
    try:
        return prepare_dataset_v2(config, tmp_path / "published")
    except DatasetPublicationUncertainV1:
        return prepare_dataset_v2(config, tmp_path / "published")


def test_local_prepared_source_reopens_reverifies_and_issues_one_use_lease(tmp_path):
    prepared = _prepared(tmp_path)
    semantic = prepared.semantic_identity
    identity = PreparedTrainingInputIdentity(
        f"prepared://sha256/{semantic.dataset_digest}",
        semantic.dataset_digest,
        semantic.dataset_sha256,
        semantic.dataset_bytes,
        ROW_SCHEMA_VERSION_V2,
    )
    source = LocalPreparedTrainingInputSource(prepared.path.resolve(), identity)
    lease = source.open_lease()
    stream = lease.take_stream()
    assert stream.read() == (prepared.path / "dataset.jsonl").read_bytes()
    with pytest.raises(ValueError, match="already consumed"):
        lease.take_stream()
    lease.close()


def test_local_prepared_source_rejects_mutation_between_resolution_and_stage(tmp_path):
    prepared = _prepared(tmp_path)
    semantic = prepared.semantic_identity
    identity = PreparedTrainingInputIdentity(
        f"prepared://sha256/{semantic.dataset_digest}",
        semantic.dataset_digest,
        semantic.dataset_sha256,
        semantic.dataset_bytes,
        ROW_SCHEMA_VERSION_V2,
    )
    source = LocalPreparedTrainingInputSource(prepared.path.resolve(), identity)
    dataset = prepared.path / "dataset.jsonl"
    dataset.chmod(0o600)
    with dataset.open("ab") as stream:
        stream.write(b"mutated\n")
    with pytest.raises(DatasetPrepValidationError):
        source.open_lease()
