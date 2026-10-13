from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from synaptic_tuner.api.v1.training_facade import TrainingAPI
from synaptic_tuner.api.v1.training_input import TrainingInputV1
from synaptic_tuner.api.v1.training_sources import LocalTrainingInputPathV1, TrainingPreparationConfigV1
from tests.contract.test_public_training_input_v1 import _document
from tests.dataset_prep.test_conversation_messages import build, fixture
from tuner.dataset_prep.conversation_messages import (
    MAX_EXECUTABLE_BYTES, NORMALIZER_REF, ROW_SCHEMA, ConversationNormalizerConfigV1,
    publish_conversation_dataset,
)
from tuner.training.conversation_preparation import (
    ConversationPreparedInputFormatVerifierV1, ConversationTrainingInputNormalizerV1,
)
from tuner.training.input_preparation import (
    PosixPrivatePreparedRootAuthorityV1, TrainingInputPreparationServiceV1,
)


def test_real_training_api_prepare_and_retained_bytes_after_raw_source_removal(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    document, dataset, metadata = fixture()
    (source / "raw.jsonl").write_bytes(dataset)
    (source / "run.json").write_bytes(metadata)
    config = ConversationNormalizerConfigV1.from_dict(document)
    root = tmp_path / "prepared"
    verifier = ConversationPreparedInputFormatVerifierV1()
    service = TrainingInputPreparationServiceV1(
        prepared_root=root, normalizers={NORMALIZER_REF: ConversationTrainingInputNormalizerV1(allowed_source_roots=(source,))},
        format_verifiers={ROW_SCHEMA: verifier}, root_authority=PosixPrivatePreparedRootAuthorityV1(),
    )
    api = TrainingAPI(service, clock=SimpleNamespace(now=lambda: "2026-10-13T00:00:00Z"))
    result = api.prepare(LocalTrainingInputPathV1(source), TrainingPreparationConfigV1(
        "request-a", "project-a", TrainingInputV1.from_dict(_document()).canonical_json(), config,
    ))
    identity = result.prepared.identity
    assert identity == result.retained_source.identity
    assert TrainingInputV1.from_json(result.prepared.request.canonical_json).dataset.ref == identity.ref
    observed = verifier.verify(root / ("dataset-" + identity.revision))
    assert observed.content_digest == identity.content_digest
    (source / "raw.jsonl").unlink()
    (source / "run.json").unlink()
    source.rmdir()
    lease = result.retained_source.open_lease()
    stream = lease.take_stream()
    assert stream.read() == observed.snapshot
    with pytest.raises(ValueError):
        lease.take_stream()
    lease.close()
    public = json.dumps(result.prepared.to_dict())
    assert str(source) not in public and "question a" not in public and "private_note" not in public
    # Registration is preparation only: existing compiler/release format remains unavailable.
    assert identity.format == ROW_SCHEMA


def test_publication_can_exceed_executable_bound_without_claiming_execution(tmp_path):
    document, dataset, metadata = fixture()
    rows = [json.loads(line) for line in dataset.splitlines()]
    # Three individually bounded rows, with two splits, exceeding 2MiB as a set.
    rows.append(json.loads(json.dumps(rows[0])))
    rows[2]["group"] = "c"
    for index, row in enumerate(rows):
        row["conversations"][-1]["content"] = str(index) + "x" * (750 * 1024)
    from tuner.dataset_prep.conversation_messages import canonical, sha
    dataset = b"".join(canonical(row) + b"\n" for row in rows)
    document["source"]["dataset"].update(sha256=sha(dataset), size_bytes=len(dataset))
    document["source"]["rows"] = 3
    document["source"]["metadata"] = None
    prepared = build(document, dataset, None)
    assert len(prepared.dataset_raw) > MAX_EXECUTABLE_BYTES
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    verified = publish_conversation_dataset(root, prepared)
    with pytest.raises(ValueError, match="2MiB"):
        ConversationPreparedInputFormatVerifierV1().verify(verified.path)


def test_old_expected_identity_rejects_rewritten_provenance(tmp_path):
    from tests.dataset_prep.test_conversation_messages import rehash
    from tuner.dataset_prep.conversation_messages import verify_bytes
    prepared = build()
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    verified = publish_conversation_dataset(root, prepared)
    verifier = ConversationPreparedInputFormatVerifierV1()
    old = verifier.verify(verified.path)
    from synaptic_tuner.api.v1.training_sources import PreparedTrainingInputIdentity
    identity = PreparedTrainingInputIdentity("prepared://sha256/" + old.semantic_digest, old.semantic_digest,
                                            old.content_digest, old.size_bytes, ROW_SCHEMA)
    source = verifier.source(verified.path, identity)
    manifest = json.loads(prepared.manifest_raw)
    rows = [json.loads(line) for line in prepared.dataset_raw.splitlines()]
    manifest["provenance"][0]["source_row_sha256"] = "1" * 64
    altered_manifest, altered_rows = rehash(manifest, rows)
    # This is a new self-consistent preparer assertion, not source-origin proof.
    new_name = json.loads(altered_manifest)["dataset_id"]
    changed = verify_bytes(tmp_path / new_name, altered_manifest, altered_rows)
    assert changed.semantic_identity != verified.semantic_identity
    for name, payload in (("manifest.json", altered_manifest), ("dataset.jsonl", altered_rows)):
        (verified.path / name).chmod(0o600)
        (verified.path / name).write_bytes(payload)
    with pytest.raises(ValueError):
        source.open_lease()
