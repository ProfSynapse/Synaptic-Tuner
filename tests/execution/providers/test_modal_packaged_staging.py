"""Provider-free qualification of packaged prepared-input staging."""

from __future__ import annotations

from io import BytesIO
import hashlib

import pytest

from synaptic_tuner.api.v1._contract import PreparedTrainingInputIdentity
from tuner.execution.providers.modal.facade import ModalFacadeError
from tuner.execution.providers.modal.contracts import (
    operation_path,
    provider_entry_identity,
)
from tuner.execution.providers.modal.packaged_staging import (
    ModalPackagedInputStager,
    ModalPackagedPreparedInputDescriptor,
    ModalPackagedStageReceipt,
    prepare_modal_packaged_stage,
)
from tuner.runtime.releases import PackagedExecutionBindingV1
from tuner.training.contracts import RetainedTrainingInputStreamLease

from tests.execution.providers.test_modal_packaged_binding import _release_and_execution
from tests.execution.providers.test_modal_sdk154_adapter import (
    Entry,
    FakeVolume,
    make_facade,
)


def _case(payload: bytes = b"prepared-input"):
    release, facts, provider, original, _ = _release_and_execution()
    content_digest = hashlib.sha256(payload).hexdigest()
    identity = PreparedTrainingInputIdentity(
        original.prepared_input_ref, original.prepared_input_revision,
        content_digest, len(payload), original.prepared_input_format,
    )
    execution = PackagedExecutionBindingV1.build(
        run_ref=original.run_ref, runtime_release=release,
        provider_runtime_binding=provider,
        prepared_input_ref=identity.ref, prepared_input_revision=identity.revision,
        prepared_input_content_digest=identity.content_digest,
        prepared_input_size_bytes=identity.size_bytes,
        prepared_input_format=identity.format,
        workload_digest=original.workload_digest,
        configuration_digest=original.configuration_digest,
        artifact_policy_digest=original.artifact_policy_digest,
    )
    facade, _ = make_facade()
    stream = BytesIO(payload)
    lease = RetainedTrainingInputStreamLease(identity, stream)
    material = prepare_modal_packaged_stage(
        execution, lease, stage_effect_id="stage-effect", artifact_volume_id="av",
    )
    return facade, execution, material, stream, payload


def test_descriptor_is_effect_scoped_and_contains_no_host_locator() -> None:
    _, execution, material, _, _ = _case()
    descriptor = material.descriptor
    assert descriptor.relative_path == (
        "operations/stage-effect/input/prepared/"
        f"{execution.prepared_input_content_digest}/payload.bin"
    )
    encoded = descriptor.canonical_bytes.lower()
    for forbidden in (b"file://", b"http://", b"https://", b"upload", b"\\"):
        assert forbidden not in encoded


def test_stage_uploads_once_then_streams_exact_readback_and_lists_exact_member(
    monkeypatch,
) -> None:
    facade, _, material, stream, payload = _case()
    volume = FakeVolume.registry["artifact-name"]
    events: list[str] = []
    original_iterdir = volume.iterdir
    original_upload = volume.batch_upload
    original_read = volume.read_file

    def iterdir(*args, **kwargs):
        events.append("list")
        yield from original_iterdir(*args, **kwargs)

    def upload(*args, **kwargs):
        events.append("upload")
        return original_upload(*args, **kwargs)

    def read(*args, **kwargs):
        events.append("readback")
        yield from original_read(*args, **kwargs)

    monkeypatch.setattr(volume, "iterdir", iterdir)
    monkeypatch.setattr(volume, "batch_upload", upload)
    monkeypatch.setattr(volume, "read_file", read)
    receipt = ModalPackagedInputStager(facade).stage_once(material)
    assert type(receipt) is ModalPackagedStageReceipt
    assert receipt.path == material.descriptor.relative_path
    assert receipt.size_bytes == len(payload)
    assert receipt.content_digest == hashlib.sha256(payload).hexdigest()
    assert volume.files == {material.descriptor.relative_path: payload}
    assert events == ["list", "upload", "readback", "list"]
    assert stream.closed


def test_collision_fails_before_upload_without_overwrite(monkeypatch) -> None:
    facade, _, material, stream, _ = _case()
    volume = FakeVolume.registry["artifact-name"]
    volume.files[material.descriptor.relative_path] = b"attacker-content"
    uploads: list[object] = []
    monkeypatch.setattr(
        volume, "batch_upload",
        lambda **kwargs: uploads.append(kwargs) or (_ for _ in ()).throw(AssertionError()),
    )
    with pytest.raises(ModalFacadeError, match="collision"):
        ModalPackagedInputStager(facade).stage_once(material)
    assert uploads == []
    assert volume.files[material.descriptor.relative_path] == b"attacker-content"
    # Rejection precedes lease transfer, allowing only the caller's explicit
    # same-effect recovery policy to decide whether it may still be consumed.
    assert not stream.closed


@pytest.mark.parametrize("mode", ("short", "long", "nonbytes"))
def test_stream_mismatch_fails_closed_and_consumes_lease(mode: str) -> None:
    facade, execution, material, original_stream, payload = _case()
    identity = material.descriptor.identity
    if mode == "short":
        hostile = BytesIO(payload[:-1])
    elif mode == "long":
        hostile = BytesIO(payload + b"x")
    else:
        class NonBytes:
            closed = False
            def read(self, _size=-1): return "not-bytes"
            def close(self): self.closed = True
        hostile = NonBytes()
    original_stream.close()
    lease = RetainedTrainingInputStreamLease(identity, hostile)
    rebound = prepare_modal_packaged_stage(
        execution, lease, stage_effect_id="stage-effect", artifact_volume_id="av",
    )
    with pytest.raises(ModalFacadeError, match="write_failed"):
        ModalPackagedInputStager(facade).stage_once(rebound)
    assert hostile.closed


def test_readback_mismatch_never_returns_receipt(monkeypatch) -> None:
    facade, _, material, _, _ = _case()
    volume = FakeVolume.registry["artifact-name"]
    original_upload = volume.batch_upload

    class CorruptAfterUpload:
        def __enter__(self):
            return self.inner.__enter__()
        def __exit__(self, kind, value, trace):
            result = self.inner.__exit__(kind, value, trace)
            if value is None:
                volume.files[material.descriptor.relative_path] = b"corrupt"
            return result
        def __init__(self, inner): self.inner = inner

    monkeypatch.setattr(
        volume, "batch_upload",
        lambda **kwargs: CorruptAfterUpload(original_upload(**kwargs)),
    )
    with pytest.raises(ModalFacadeError, match="read"):
        ModalPackagedInputStager(facade).stage_once(material)


def test_descriptor_and_receipt_reject_cross_effect_or_volume_substitution() -> None:
    _, execution, material, _, _ = _case()
    descriptor = material.descriptor
    with pytest.raises(ValueError, match="stage scoped"):
        ModalPackagedPreparedInputDescriptor(
            descriptor.identity, "other-effect", descriptor.artifact_volume_id,
            descriptor.relative_path,
        )
    receipt = ModalPackagedStageReceipt(
        descriptor.stage_effect_id, execution.binding_digest,
        descriptor.artifact_volume_id, descriptor.relative_path,
        descriptor.identity.size_bytes, descriptor.identity.content_digest,
        provider_entry_identity(
            descriptor.artifact_volume_id, descriptor.relative_path,
            descriptor.identity.size_bytes,
        ),
    )
    changed = receipt.to_dict()
    changed["schema_version"] = "other"
    with pytest.raises(ValueError, match="invalid"):
        ModalPackagedStageReceipt.from_dict(changed)


@pytest.mark.parametrize(
    "hostile_path",
    (
        r"C:\Users\operator\prepared.jsonl",
        "/tmp/prepared.jsonl",
        "../prepared.jsonl",
        "operations/other-effect/input/prepared/" + "a" * 64 + "/payload.bin",
    ),
)
def test_receipt_rejects_host_paths_and_cross_effect_provider_paths(
    hostile_path: str,
) -> None:
    _, execution, material, _, _ = _case()
    descriptor = material.descriptor
    entry = provider_entry_identity(
        descriptor.artifact_volume_id, descriptor.relative_path,
        descriptor.identity.size_bytes,
    )
    with pytest.raises(ValueError, match="provider path|effect scoped"):
        ModalPackagedStageReceipt(
            descriptor.stage_effect_id, execution.binding_digest,
            descriptor.artifact_volume_id, hostile_path,
            descriptor.identity.size_bytes, descriptor.identity.content_digest,
            entry,
        )


def test_receipt_rejects_arbitrary_provider_entry_identity() -> None:
    _, execution, material, _, _ = _case()
    descriptor = material.descriptor
    with pytest.raises(ValueError, match="provider entry identity"):
        ModalPackagedStageReceipt(
            descriptor.stage_effect_id, execution.binding_digest,
            descriptor.artifact_volume_id, descriptor.relative_path,
            descriptor.identity.size_bytes, descriptor.identity.content_digest,
            "attacker-entry",
        )


def test_receipt_provider_identity_is_exactly_volume_path_and_size_bound() -> None:
    facade, _, material, _, _ = _case()
    receipt = ModalPackagedInputStager(facade).stage_once(material)
    assert receipt.path == operation_path(
        receipt.stage_effect_id, "input", "prepared", receipt.content_digest,
        "payload.bin",
    )
    assert receipt.provider_entry_id == provider_entry_identity(
        receipt.artifact_volume_id, receipt.path, receipt.size_bytes,
    )
