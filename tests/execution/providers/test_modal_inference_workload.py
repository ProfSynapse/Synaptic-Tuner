"""Provider-free acceptance tests for authenticated Modal workload identity."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from tuner.execution.foundation_v2.canonical import (
    canonical_bytes,
    parse_canonical_object,
)
from tuner.execution.providers.modal.contracts import ArtifactRole
from tuner.execution.providers.modal.inference_workload import (
    ModalInferenceWorkloadBinding,
    ModalInferenceWorkloadError,
    bind_modal_inference_workload,
)
from tests.execution.providers import modal_coordinator_fixtures as shared_cases
from tests.execution.providers import test_modal_coordinator_bundle as bundle_cases
from tests.execution.providers import test_modal_coordinator_producer as producer_cases
from tests.execution.providers import (
    test_modal_coordinator_read_transport as read_cases,
)
from tests.execution.providers import (
    test_modal_inference_source_integration as source_cases,
)
from tests.training.test_sft_compilation import _config
from tuner.execution.providers.modal import inference_workload as workload_module


def _case(monkeypatch, *, equal_revisions=True):
    model = _config().to_dict()["model"]
    revision = model["revision"]
    original_bundle = bundle_cases._fixture

    def equal_revision_bundle(*args, **kwargs):
        extra = dict(kwargs.pop("config_extra", {}) or {})
        extra["model"] = dict(model) | {"tokenizer_revision": revision}
        return original_bundle(*args, config_extra=extra, **kwargs)

    if equal_revisions:
        monkeypatch.setattr(shared_cases, "_fixture", equal_revision_bundle)
    original_inventory = producer_cases._inventory

    def exact_inventory(invocation, *, mutate=None):
        inventory, files = original_inventory(invocation, mutate=mutate)
        document = parse_canonical_object(inventory, name="artifact inventory")
        record = next(
            item
            for item in document["artifacts"]
            if item["role"] == ArtifactRole.WORKLOAD_RECORD.value
        )
        files[record["path"]] = invocation.workload
        record["size"] = len(invocation.workload)
        record["sha256"] = hashlib.sha256(invocation.workload).hexdigest()
        return canonical_bytes(document), files

    monkeypatch.setattr(producer_cases, "_inventory", exact_inventory)
    monkeypatch.setattr(read_cases, "_inventory", exact_inventory)
    values = source_cases._case(monkeypatch)
    binder, runs, run = values[:3]
    source = binder.bind(runs, run)
    transport = binder._reader._transport
    return source, transport, values


def _bind(source, transport, **overrides):
    values = {
        "launch_source": transport._launch_source,
        "foundation_authenticator": transport._foundation,
        "assessment_authenticator": transport._assessments,
        "binding_authority": transport._authority,
        "stage_verifier": transport._stage,
        "launch_verifier": transport._launch,
        "recipes": transport._recipes,
        "bounds": transport._bounds,
    }
    values.update(overrides)
    return bind_modal_inference_workload(source, **values)


def test_authenticated_workload_binding_uses_exact_retained_workload_without_weights(
    monkeypatch,
):
    source, transport, values = _case(monkeypatch)
    admitted = _bind(source, transport)
    document = parse_canonical_object(admitted.workload_bytes, name="workload")
    workload_member = next(
        item
        for item in source.native_members
        if item.role is ArtifactRole.WORKLOAD_RECORD
    )
    assert type(admitted) is ModalInferenceWorkloadBinding
    assert admitted.run_id == source.run.run_id
    assert admitted.project_ref == source.run.project_ref
    assert (
        admitted.workload_size == len(admitted.workload_bytes) == workload_member.size
    )
    assert (
        admitted.workload_sha256 == hashlib.sha256(admitted.workload_bytes).hexdigest()
    )
    assert admitted.workload_sha256 == workload_member.sha256
    assert admitted.model_revision == admitted.tokenizer_revision
    assert (
        document["configuration"]["document"]["model"]
        == document["identities"]["model"]
    )
    assert values[6] == []


@pytest.mark.parametrize("field", ("size", "sha256"))
def test_workload_binding_rejects_native_workload_hash_or_size_mutation(
    monkeypatch, field
):
    source, transport, _ = _case(monkeypatch)
    members = list(source.native_members)
    index = next(
        i for i, item in enumerate(members) if item.role is ArtifactRole.WORKLOAD_RECORD
    )
    changed = (
        {"size": members[index].size + 1} if field == "size" else {"sha256": "0" * 64}
    )
    members[index] = replace(members[index], **changed)
    object.__setattr__(source, "native_members", tuple(members))
    with pytest.raises(
        ModalInferenceWorkloadError, match="^modal_inference_workload_invalid$"
    ) as caught:
        _bind(source, transport)
    assert caught.value.__cause__ is None


def test_workload_binding_is_factory_issued():
    with pytest.raises(TypeError):
        ModalInferenceWorkloadBinding()


@pytest.mark.parametrize("target", ("retained", "admitted"))
def test_workload_binding_rejects_launch_mutation_after_authentication(
    monkeypatch, target
):
    source, transport, _ = _case(monkeypatch)
    retained = transport._launch_source.value
    original = workload_module.prepare_modal_submit_dispatch
    calls = []

    def authenticated_then_changed(envelope, **kwargs):
        dispatch = original(envelope, **kwargs)
        calls.append("authenticated")
        changed = retained if target == "retained" else envelope
        object.__setattr__(changed, "claim", changed.claim + b" ")
        return dispatch

    monkeypatch.setattr(
        workload_module, "prepare_modal_submit_dispatch", authenticated_then_changed
    )
    with pytest.raises(
        ModalInferenceWorkloadError, match="^modal_inference_workload_invalid$"
    ):
        _bind(source, transport)
    assert calls == ["authenticated"]


@pytest.mark.parametrize(
    ("attribute", "replacement"),
    (
        ("manifest_digest", "0" * 64),
        ("effect_id", "effect-alien"),
        ("artifact_volume_id", "volume-alien"),
    ),
)
def test_workload_binding_rejects_changed_source_projection(
    monkeypatch, attribute, replacement
):
    source, transport, _ = _case(monkeypatch)
    object.__setattr__(source, attribute, replacement)
    with pytest.raises(
        ModalInferenceWorkloadError, match="^modal_inference_workload_invalid$"
    ):
        _bind(source, transport)


def test_workload_binding_rejects_wrong_native_key_without_body_reads(monkeypatch):
    source, transport, values = _case(monkeypatch)
    evidence = parse_canonical_object(source.native_evidence, name="native evidence")
    evidence["identity"]["key_ref"] = "alien-key"
    object.__setattr__(source, "native_evidence", canonical_bytes(evidence))
    with pytest.raises(
        ModalInferenceWorkloadError, match="^modal_inference_workload_invalid$"
    ):
        _bind(source, transport)
    assert values[6] == []


def test_workload_binding_rejects_unequal_model_and_tokenizer_revisions(monkeypatch):
    source, transport, _ = _case(monkeypatch, equal_revisions=False)
    with pytest.raises(
        ModalInferenceWorkloadError, match="^modal_inference_workload_invalid$"
    ):
        _bind(source, transport)


def test_workload_binding_closes_and_redacts_foreign_errors(monkeypatch):
    source, transport, _ = _case(monkeypatch)

    class BrokenLaunchSource:
        def resolve(self, digest):
            raise RuntimeError("HF_TOKEN=private")

    with pytest.raises(ModalInferenceWorkloadError) as caught:
        _bind(source, transport, launch_source=BrokenLaunchSource())
    assert str(caught.value) == "modal_inference_workload_invalid"
    assert caught.value.__cause__ is None


def test_workload_binding_propagates_control_interrupt(monkeypatch):
    source, transport, _ = _case(monkeypatch)

    class InterruptedLaunchSource:
        def resolve(self, digest):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _bind(source, transport, launch_source=InterruptedLaunchSource())


def test_workload_binding_rejects_source_mutation_during_launch_resolution(monkeypatch):
    source, transport, _ = _case(monkeypatch)

    class MutatingLaunchSource:
        def resolve(self, digest):
            object.__setattr__(source, "manifest_digest", "0" * 64)
            return transport._launch_source.resolve(digest)

    with pytest.raises(
        ModalInferenceWorkloadError, match="^modal_inference_workload_invalid$"
    ):
        _bind(source, transport, launch_source=MutatingLaunchSource())


def test_workload_binding_does_not_evaluate_nonmethod_resolve_property(monkeypatch):
    source, transport, _ = _case(monkeypatch)
    accessed = []

    class PropertyLaunchSource:
        @property
        def resolve(self):
            accessed.append(True)
            raise RuntimeError("private")

    with pytest.raises(
        ModalInferenceWorkloadError, match="^modal_inference_workload_invalid$"
    ):
        _bind(source, transport, launch_source=PropertyLaunchSource())
    assert accessed == []


def test_workload_binding_rewraps_foreign_same_error_class(monkeypatch):
    source, transport, _ = _case(monkeypatch)

    class BrokenLaunchSource:
        def resolve(self, digest):
            raise ModalInferenceWorkloadError("credential-shaped-private-text")

    with pytest.raises(ModalInferenceWorkloadError) as caught:
        _bind(source, transport, launch_source=BrokenLaunchSource())
    assert str(caught.value) == "modal_inference_workload_invalid"
    assert caught.value.__cause__ is None
