"""Provider-free tests for exact Modal chat command retention."""

from __future__ import annotations

import hashlib
import json

import pytest
from tuner.execution.foundation_v2.canonical import (
    canonical_bytes,
    parse_canonical_object,
)
from tuner.execution.foundation_v2.identities import EffectKind, derive_effect
from tuner.execution.foundation_v2.references import StagePredecessorV2
from tuner.execution.providers.modal.inference_commands import (
    ModalInferenceCommandBinding,
)
from tuner.execution.providers.modal.inference_retention import (
    ModalInferenceRetentionError,
    load_modal_chat_command,
    retain_modal_chat_command,
)
from tests.execution.providers.test_modal_inference_preparation import _prepared
from tests.execution.foundation_v2.helpers import cancel_command
from tuner.execution.foundation_v2.commands import (
    parse_exact_command,
    build_stage_command,
)
from tuner.execution.providers.modal.inference_preparation import _derive_preparation


def _retag(binding):
    document = parse_canonical_object(binding.preparation_snapshot, name="snapshot")
    old = document["configuration_tag"]
    document["configuration_tag"] = ("00" if old[:2] != "00" else "01") + old[2:]
    return ModalInferenceCommandBinding(
        binding.command_bytes, canonical_bytes(document)
    )


def test_same_command_different_content_is_a_conflict_even_if_both_authenticated(
    monkeypatch,
):
    candidate = _bindings(monkeypatch, include_submit=False)[0]
    other = _retag(candidate)
    assert other.command_digest == candidate.command_digest
    assert other.binding_digest != candidate.binding_digest
    catalog = Catalog()
    catalog.values[candidate.command_digest] = other
    authority = Authority([candidate.canonical_bytes, other.canonical_bytes])
    with pytest.raises(ModalInferenceRetentionError):
        retain_modal_chat_command(candidate, catalog=catalog, authority=authority)
    assert all(name != "publish" for name, _ in catalog.calls)
    assert catalog.values[candidate.command_digest] is other


def test_retained_snapshot_is_structural_not_fresh_evidence_or_authentication(
    monkeypatch,
):
    candidate = _bindings(monkeypatch, include_submit=False)[0]
    other = _retag(candidate)
    catalog = Catalog()
    catalog.values[candidate.command_digest] = other
    # A structurally valid tag replacement is not an authentic binding. Recovery
    # invokes the content authority, not expiring source/config/quote preflight.
    with pytest.raises(ModalInferenceRetentionError):
        load_modal_chat_command(
            candidate.command_digest,
            catalog=catalog,
            authority=Authority([candidate.canonical_bytes]),
        )
    loaded = load_modal_chat_command(
        other.command_digest,
        catalog=catalog,
        authority=Authority([other.canonical_bytes]),
    )
    assert loaded.canonical_bytes == other.canonical_bytes


@pytest.mark.parametrize("target", ("caller", "owned"))
def test_resolve_callback_cannot_change_either_retained_input(monkeypatch, target):
    candidate = _bindings(monkeypatch, include_submit=False)[0]
    replacement = _retag(candidate)
    authority = Authority([candidate.canonical_bytes, replacement.canonical_bytes])
    catalog = Catalog()

    def resolve(key):
        victim = candidate if target == "caller" else authority.calls[0]
        object.__setattr__(
            victim, "_preparation_snapshot", replacement.preparation_snapshot
        )
        return None

    catalog.resolve = resolve
    with pytest.raises(ModalInferenceRetentionError):
        retain_modal_chat_command(candidate, catalog=catalog, authority=authority)
    assert not catalog.values


@pytest.mark.parametrize("bad", (None, 1, True, b"a" * 64, "bad", "A" * 64))
def test_load_rejects_bad_keys_before_catalog_access(bad):
    catalog = Catalog()
    with pytest.raises(ModalInferenceRetentionError):
        load_modal_chat_command(bad, catalog=catalog, authority=Authority([]))
    assert catalog.calls == []


@pytest.mark.parametrize(
    "mutation",
    (
        "member-extra",
        "member-entry",
        "member-path",
        "workload-hash",
        "model-ref",
        "source-volume",
        "source-revision",
        "source-provider",
        "tag-case",
    ),
)
def test_rehashed_semantically_invalid_snapshot_is_still_rejected(
    monkeypatch, mutation
):
    candidate = _bindings(monkeypatch, include_submit=False)[0]
    snapshot = parse_canonical_object(candidate.preparation_snapshot, name="snapshot")
    chat = snapshot["chat_input"]
    source = chat["source"]
    workload = chat["workload"]
    if mutation == "member-extra":
        source["members"][0]["unknown"] = "field"
    elif mutation == "member-entry":
        source["members"][0]["provider_entry_id"] = "0" * 64
    elif mutation == "member-path":
        source["members"][0]["path"] = "operations/other/output/final_model"
    elif mutation == "workload-hash":
        workload["workload_bytes"] = workload["workload_sha256"] = "0" * 64
    elif mutation == "model-ref":
        workload["model_ref"] = {"unexpected": "field"}
    elif mutation == "source-volume":
        chat["configuration"]["volumes"]["source_artifact_volume_id"] = "other"
        # Keep the derived resource quote self-consistent too: the source-volume
        # equality remains independently required after rehashing all content.
        from tuner.execution.foundation_v2.canonical import domain_digest

        configuration = chat["configuration"]
        chat["quote"]["resource_digest"] = domain_digest(
            "synaptic-modal-inference-resource/v1",
            canonical_bytes(
                {
                    "resources": configuration["resources"],
                    "volumes": configuration["volumes"],
                    "application": configuration["application"],
                }
            ),
        )
    elif mutation == "source-revision":
        source["source_revision"] = True
    elif mutation == "source-provider":
        source["provider"][0] = "other"
    else:
        snapshot["configuration_tag"] = "AA"
    snapshot["configuration"] = chat["configuration"]
    snapshot["quote"] = chat["quote"]
    from tuner.execution.foundation_v2.canonical import domain_digest
    from tuner.execution.providers.modal.inference_preparation import (
        CONFIG_EVIDENCE_PURPOSE,
    )

    snapshot["configuration_digest"] = domain_digest(
        CONFIG_EVIDENCE_PURPOSE, canonical_bytes(snapshot["configuration"])
    )
    snapshot["chat_input_sha256"] = hashlib.sha256(canonical_bytes(chat)).hexdigest()
    original = parse_exact_command(candidate.command_bytes)
    prep = _derive_preparation(chat, original.executor)
    snapshot["preparation"] = prep.to_dict()
    from tuner.execution.foundation_v2.commands import CanonicalProviderPayloadV1

    command = build_stage_command(
        prep,
        "rehash-stage",
        CanonicalProviderPayloadV1.build(
            "modal", "stage-payload/v2", prep.workload_digest
        ),
        original.executor,
    )
    with pytest.raises((TypeError, ValueError)):
        ModalInferenceCommandBinding(command.canonical_bytes, canonical_bytes(snapshot))


def test_binding_envelope_is_canonical_and_components_keep_their_bounds(monkeypatch):
    stage, submit, _ = _bindings(monkeypatch)
    for binding in (stage, submit):
        raw = binding.canonical_bytes
        assert (
            json.dumps(
                json.loads(raw),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode()
            == raw
        )
        assert len(binding.command_bytes) <= 16384
        assert len(binding.preparation_snapshot) <= 16384
        assert len(raw) <= 32768 + 99
        with pytest.raises((TypeError, ValueError)):
            ModalInferenceCommandBinding(b" " * 16385, binding.preparation_snapshot)
        with pytest.raises((TypeError, ValueError)):
            ModalInferenceCommandBinding(
                binding.command_bytes, bytearray(binding.preparation_snapshot)
            )


def _bindings(monkeypatch, *, include_submit=True):
    preparation, source, *_ = _prepared(monkeypatch)
    prep = preparation.preparation
    stage = preparation.stage("chat-stage-nonce")
    effect = derive_effect(prep, EffectKind.STAGE)
    predecessor = StagePredecessorV2(
        prep.provider.provider_id,
        prep.provider.profile_ref,
        prep.scope.account_ref,
        prep.scope.namespace_ref,
        prep.project_ref,
        prep.run_id,
        prep.plan_fingerprint,
        prep.preparation_digest,
        prep.workload_digest,
        effect.effect_id,
        "a" * 64,
        "b" * 64,
    )
    submit = preparation.submit("chat-submit-nonce", predecessor)
    stage_binding = ModalInferenceCommandBinding(
        stage.canonical_bytes, preparation.canonical_bytes
    )
    submit_binding = (
        ModalInferenceCommandBinding(
            submit.canonical_bytes, preparation.canonical_bytes
        )
        if include_submit
        else None
    )
    return stage_binding, submit_binding, source


class Catalog:
    def __init__(self):
        self.values = {}
        self.calls = []

    def resolve(self, key):
        self.calls.append(("resolve", key))
        return self.values.get(key)

    def publish_if_absent(self, key, value):
        self.calls.append(("publish", key))
        self.values.setdefault(key, value)


class Authority:
    def __init__(self, accepted):
        self.accepted = set(accepted)
        self.calls = []

    def authenticate(self, binding):
        self.calls.append(binding)
        return binding.canonical_bytes in self.accepted


@pytest.mark.parametrize("index", (0, 1))
def test_binding_roundtrip_retention_and_restart_are_exact(monkeypatch, index):
    bindings = _bindings(monkeypatch)[:2]
    candidate = bindings[index]
    catalog = Catalog()
    authority = Authority([candidate.canonical_bytes])
    retained = retain_modal_chat_command(
        candidate, catalog=catalog, authority=authority
    )
    loaded = load_modal_chat_command(
        candidate.command_digest, catalog=catalog, authority=authority
    )
    again = retain_modal_chat_command(candidate, catalog=catalog, authority=authority)
    assert type(retained) is type(loaded) is ModalInferenceCommandBinding
    assert retained is not candidate and loaded is not retained
    assert (
        retained.canonical_bytes
        == loaded.canonical_bytes
        == again.canonical_bytes
        == candidate.canonical_bytes
    )
    assert [name for name, _ in catalog.calls].count("publish") == 1


def test_training_submit_is_not_a_chat_command_binding(monkeypatch):
    _, _, source = _bindings(monkeypatch, include_submit=False)
    with pytest.raises((TypeError, ValueError)):
        ModalInferenceCommandBinding(
            source.command_binding.command_bytes,
            source.command_binding.preparation_snapshot,
        )


def test_cancel_is_not_a_chat_command_binding(monkeypatch):
    stage, _, _ = _bindings(monkeypatch, include_submit=False)
    with pytest.raises(ValueError):
        ModalInferenceCommandBinding(
            cancel_command().canonical_bytes, stage.preparation_snapshot
        )


@pytest.mark.parametrize(
    "field",
    (
        "provider_id",
        "profile_ref",
        "account_ref",
        "namespace_ref",
        "project_ref",
        "run_id",
        "plan_fingerprint",
        "preparation_digest",
        "workload_digest",
        "stage_effect_id",
    ),
)
def test_submit_rejects_every_wrong_stage_predecessor_identity(monkeypatch, field):
    _, submit, _ = _bindings(monkeypatch)
    document = parse_canonical_object(submit.command_bytes, name="submit")
    predecessor = document["stage_predecessor"]
    predecessor[field] = (
        "0" * 64 if field.endswith(("digest", "fingerprint", "effect_id")) else "other"
    )
    with pytest.raises(ValueError):
        ModalInferenceCommandBinding(
            canonical_bytes(document), submit.preparation_snapshot
        )


@pytest.mark.parametrize("mutation", ("chat-extra", "source-extra", "executor"))
def test_snapshot_rejects_extra_nested_chat_fields_and_wrong_executor(
    monkeypatch, mutation
):
    stage, _, _ = _bindings(monkeypatch, include_submit=False)
    document = parse_canonical_object(stage.preparation_snapshot, name="snapshot")
    if mutation == "chat-extra":
        document["chat_input"]["unexpected"] = True
    elif mutation == "source-extra":
        document["chat_input"]["source"]["unexpected"] = True
    else:
        document["executor"]["executor_id"] = "other-executor"
    with pytest.raises(ValueError):
        ModalInferenceCommandBinding(stage.command_bytes, canonical_bytes(document))


@pytest.mark.parametrize(
    "field", ("resource_digest", "workload_digest", "run_id", "profile_ref")
)
def test_binding_rejects_rehashed_preparation_snapshot_mismatch(monkeypatch, field):
    candidate = _bindings(monkeypatch, include_submit=False)[0]
    document = parse_canonical_object(candidate.preparation_snapshot, name="snapshot")
    if field == "profile_ref":
        document["preparation"]["provider"][field] = "other"
    elif field == "run_id":
        document["preparation"][field] = "other-session"
    else:
        document["preparation"][field] = "0" * 64
    with pytest.raises((TypeError, ValueError)):
        ModalInferenceCommandBinding(candidate.command_bytes, canonical_bytes(document))


def test_binding_is_immutable_final_and_factory_content_is_fresh(monkeypatch):
    candidate = _bindings(monkeypatch, include_submit=False)[0]
    copied = ModalInferenceCommandBinding(
        candidate.command_bytes, candidate.preparation_snapshot
    )
    assert (
        copied is not candidate and copied.canonical_bytes == candidate.canonical_bytes
    )
    with pytest.raises(AttributeError):
        candidate._command_bytes = b"changed"
    with pytest.raises(AttributeError):
        candidate._preparation_snapshot = b"changed"
    with pytest.raises(TypeError):
        type("Subclass", (ModalInferenceCommandBinding,), {})


@pytest.mark.parametrize("mode", ("deny", "raise", "mutate"))
def test_authentication_failure_or_mutation_precedes_publication(monkeypatch, mode):
    candidate = _bindings(monkeypatch, include_submit=False)[0]
    catalog = Catalog()

    class BadAuthority:
        def authenticate(self, binding):
            if mode == "raise":
                raise RuntimeError("credential-shaped-private")
            if mode == "mutate":
                object.__setattr__(binding, "_command_bytes", b"changed")
            return mode == "mutate"

    with pytest.raises(
        ModalInferenceRetentionError, match="^modal_inference_retention_invalid$"
    ) as caught:
        retain_modal_chat_command(candidate, catalog=catalog, authority=BadAuthority())
    assert caught.value.__cause__ is None and catalog.calls == []


@pytest.mark.parametrize("mode", ("conflict", "disappear"))
def test_catalog_conflict_missing_reread_or_wrong_key_fails_closed(monkeypatch, mode):
    stage, submit, _ = _bindings(monkeypatch)
    catalog = Catalog()
    authority = Authority([stage.canonical_bytes, submit.canonical_bytes])
    if mode == "conflict":
        catalog.values[stage.command_digest] = submit
    else:
        catalog.publish_if_absent = lambda key, value: catalog.calls.append(
            ("publish", key)
        )
    with pytest.raises(
        ModalInferenceRetentionError, match="^modal_inference_retention_invalid$"
    ):
        retain_modal_chat_command(stage, catalog=catalog, authority=authority)


def test_duck_binding_properties_are_not_evaluated():
    reads = []

    class Duck:
        @property
        def command_bytes(self):
            reads.append("read")
            raise AssertionError

    with pytest.raises(ModalInferenceRetentionError):
        retain_modal_chat_command(Duck(), catalog=Catalog(), authority=Authority([]))
    assert reads == []


@pytest.mark.parametrize("control", (KeyboardInterrupt, SystemExit))
@pytest.mark.parametrize("callback", ("authenticate", "resolve", "publish"))
def test_control_flow_from_collaborators_is_preserved(monkeypatch, control, callback):
    candidate = _bindings(monkeypatch, include_submit=False)[0]
    catalog = Catalog()
    authority = Authority([candidate.canonical_bytes])
    if callback == "authenticate":
        authority.authenticate = lambda value: (_ for _ in ()).throw(control())
    elif callback == "resolve":
        catalog.resolve = lambda key: (_ for _ in ()).throw(control())
    else:
        catalog.resolve = lambda key: None
        catalog.publish_if_absent = lambda key, value: (_ for _ in ()).throw(control())
    with pytest.raises(control):
        retain_modal_chat_command(candidate, catalog=catalog, authority=authority)


def test_publish_ambiguity_is_not_retried_and_later_load_recovers(monkeypatch):
    candidate = _bindings(monkeypatch, include_submit=False)[0]
    catalog = Catalog()
    authority = Authority([candidate.canonical_bytes])
    calls = []

    def publish(key, value):
        calls.append(key)
        catalog.values.setdefault(key, value)
        raise TimeoutError("private")

    catalog.publish_if_absent = publish
    with pytest.raises(
        ModalInferenceRetentionError, match="^modal_inference_retention_invalid$"
    ):
        retain_modal_chat_command(candidate, catalog=catalog, authority=authority)
    assert calls == [candidate.command_digest]
    assert (
        load_modal_chat_command(
            candidate.command_digest, catalog=catalog, authority=authority
        ).canonical_bytes
        == candidate.canonical_bytes
    )


def test_catalog_mutation_of_candidate_is_detected(monkeypatch):
    candidate = _bindings(monkeypatch, include_submit=False)[0]
    catalog = Catalog()
    authority = Authority([candidate.canonical_bytes])

    def publish(key, value):
        catalog.values[key] = value
        object.__setattr__(value, "_preparation_snapshot", b"changed")

    catalog.publish_if_absent = publish
    with pytest.raises(ModalInferenceRetentionError):
        retain_modal_chat_command(candidate, catalog=catalog, authority=authority)


def test_valid_binding_size_is_measured_without_foundation_command_bound_assumption(
    monkeypatch,
):
    stage, submit, _ = _bindings(monkeypatch)
    assert len(stage.canonical_bytes) > len(stage.command_bytes)
    assert len(submit.canonical_bytes) > len(submit.command_bytes)
    assert max(map(lambda value: len(value.canonical_bytes), (stage, submit))) > 16_384
