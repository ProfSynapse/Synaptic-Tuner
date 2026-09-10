"""Provider-free acceptance tests for pure Modal chat preparation."""

from __future__ import annotations
from dataclasses import replace
import hashlib
import pytest
from tuner.execution.foundation_v2.canonical import (
    canonical_bytes,
    domain_digest,
    parse_canonical_object,
)
from tuner.execution.foundation_v2.identities import EffectKind, derive_effect
from tuner.execution.foundation_v2.authority import GrantAuthorityV2
from tuner.execution.foundation_v2.references import StagePredecessorV2
from tuner.execution.providers.modal.coordinator_preflight import (
    AuthenticatedModalQuote,
    TrustedEvidenceIdentity,
    QUOTE_PURPOSE,
)
from tuner.execution.providers.modal.inference_preparation import (
    CONFIG_EVIDENCE_PURPOSE,
    AuthenticatedModalInferencePreparationConfig,
    ModalInferencePreparation,
    ModalInferencePreparationConfig,
    ModalInferencePreparationError,
    prepare_modal_chat,
)
from tests.execution.providers.test_modal_coordinator_preflight import (
    body as quote_body,
    evidence_tag,
)
from tests.execution.providers.test_modal_inference_workload import (
    _case as workload_case,
    _bind,
)


class Auth:
    def sign(self, *args):
        return b"tag"

    def verify(self, purpose, payload, tag, key_ref):
        return tag == evidence_tag(purpose, payload, key_ref)


class Clock:
    def now_iso(self):
        return "2026-09-09T12:02:00Z"


def _document(source, **changes):
    client = source.command_binding.client_binding
    value = {
        "schema_version": "synaptic-modal-inference-preparation-config/v1",
        "provider": {"provider_id": "modal", "profile_ref": "chat-a10"},
        "client": {
            "account_ref": client.account_ref,
            "workspace_ref": client.workspace_ref,
            "environment_ref": client.environment_ref,
            "client_ref": client.client_ref,
            "sdk_version": client.sdk_version,
        },
        "application": {
            "app_name": "synaptic-chat-v1",
            "app_ref": "chat-app",
            "sandbox_entrypoint": "chat-entry",
            "worker_ref": "chat-worker",
        },
        "image": {
            "registry_reference": "registry/chat@sha256:" + "1" * 64,
            "image_digest": "1" * 64,
        },
        "runtime": {
            "dependency_lock_digest": "2" * 64,
            "runtime_lock_digest": "3" * 64,
            "source_lock_digest": "4" * 64,
            "worker_closure_digest": "5" * 64,
            "python_version": "3.11.14",
            "python_executable": "/opt/conda/bin/python3",
            "python_executable_digest": "6" * 64,
        },
        "volumes": {
            "source_artifact_volume_ref": "source-artifacts",
            "source_artifact_volume_id": source.artifact_volume_id,
            "chat_control_volume_ref": "chat-control",
            "chat_control_volume_id": "chat-control-id",
            "model_cache_volume_ref": "model-cache",
            "model_cache_volume_id": "model-cache-id",
            "key_ref": "chat-evidence-key",
        },
        "resources": {
            "accelerator": "A10G",
            "accelerator_count": 1,
            "cpu_millicores": 4000,
            "memory_mb": 16384,
            "service_port": 8000,
            "provider_timeout_seconds": 900,
            "provider_idle_timeout_seconds": 300,
            "max_retries": 0,
        },
        "policy": {
            "startup_timeout_seconds": 600,
            "request_timeout_seconds": 60,
            "idle_timeout_seconds": 300,
            "absolute_lifetime_seconds": 900,
            "max_turns": 32,
            "max_history_bytes": 65536,
            "max_request_bytes": 8192,
            "max_response_bytes": 65536,
        },
        "secrets": [{"name": "model-download", "required_keys": ["HF_TOKEN"]}],
        "evidence": {
            "issuer_ref": "host-config",
            "evidence_ref": "config-1",
            "audience_ref": "chat-session",
            "challenge_nonce": "config-nonce",
            "key_ref": "config-key",
            "verified_at": "2026-09-09T12:01:00Z",
            "expires_at": "2026-09-09T12:05:00Z",
        },
    }
    value.update(changes)
    return value


def _prepared(monkeypatch, **config_changes):
    source, transport, _ = workload_case(monkeypatch)
    workload = _bind(source, transport)
    document = _document(source, **config_changes)
    config = ModalInferencePreparationConfig.build(document)
    authenticated = AuthenticatedModalInferencePreparationConfig(
        config.canonical_bytes,
        evidence_tag(CONFIG_EVIDENCE_PURPOSE, config.canonical_bytes, "config-key"),
    )
    resource = domain_digest(
        "synaptic-modal-inference-resource/v1",
        canonical_bytes(
            {
                "resources": document["resources"],
                "volumes": document["volumes"],
                "application": document["application"],
            }
        ),
    )
    raw = quote_body(
        provider_id="modal",
        profile_ref=document["provider"]["profile_ref"],
        account_ref=source.account_ref,
        namespace_ref=source.namespace_ref,
        resource_digest=resource,
    )
    quote = AuthenticatedModalQuote(raw, evidence_tag(QUOTE_PURPOSE, raw, "quote-key"))
    value = prepare_modal_chat(
        source,
        workload,
        configuration=authenticated,
        configuration_trust=TrustedEvidenceIdentity(
            "host-config", "config-key", "chat-session"
        ),
        quote=quote,
        quote_trust=TrustedEvidenceIdentity("host-quoter", "quote-key", "project-run"),
        evidence_authenticator=Auth(),
        clock=Clock(),
        session_id="chat-session-1",
        executor_version="v1",
    )
    return value, source, workload, authenticated, quote


def test_chat_preparation_is_deterministic_complete_and_provider_free(monkeypatch):
    first, source, workload, configuration, quote = _prepared(monkeypatch)
    second = prepare_modal_chat(
        source,
        workload,
        configuration=configuration,
        configuration_trust=TrustedEvidenceIdentity(
            "host-config", "config-key", "chat-session"
        ),
        quote=quote,
        quote_trust=TrustedEvidenceIdentity("host-quoter", "quote-key", "project-run"),
        evidence_authenticator=Auth(),
        clock=Clock(),
        session_id="chat-session-1",
        executor_version="v1",
    )
    assert first.canonical_bytes == second.canonical_bytes
    assert first.preparation.run_id == "chat-session-1"
    assert first.preparation.provider.profile_ref == "chat-a10"
    assert first.preparation.source_digest == source.artifact_source_digest
    assert first.executor.executor_id == "modal-chat-executor"
    assert parse_canonical_object(first.canonical_bytes, name="preparation")[
        "configuration"
    ]["secrets"] == [{"name": "model-download", "required_keys": ["HF_TOKEN"]}]


def test_stage_and_submit_use_exact_chat_predecessor(monkeypatch):
    value, *_ = _prepared(monkeypatch)
    stage = value.stage("chat-stage-nonce")
    prep = value.preparation
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
    submit = value.submit("chat-submit-nonce", predecessor)
    assert submit.stage_predecessor == predecessor
    assert stage.operation.effect.effect_id == predecessor.stage_effect_id


@pytest.mark.parametrize("field", ("profile", "resource", "stale", "tag"))
def test_chat_preparation_rejects_wrong_authenticated_configuration_or_quote(
    monkeypatch, field
):
    value, source, workload, configuration, quote = _prepared(monkeypatch)
    if field == "tag":
        configuration = AuthenticatedModalInferencePreparationConfig(
            configuration.body_bytes, b"wrong"
        )
    elif field == "stale":
        raw = quote_body(
            profile_ref="chat-a10",
            account_ref=source.account_ref,
            namespace_ref=source.namespace_ref,
            resource_digest=quote.body.resource_digest,
            expires_at="2026-09-09T12:01:30Z",
        )
        quote = AuthenticatedModalQuote(
            raw, evidence_tag(QUOTE_PURPOSE, raw, "quote-key")
        )
    else:
        doc = parse_canonical_object(configuration.body_bytes, name="config")
        if field == "profile":
            doc["provider"]["profile_ref"] = "other-chat"
        else:
            doc["resources"]["memory_mb"] += 1
        raw = canonical_bytes(doc)
        configuration = AuthenticatedModalInferencePreparationConfig(
            raw, evidence_tag(CONFIG_EVIDENCE_PURPOSE, raw, "config-key")
        )
    with pytest.raises(
        ModalInferencePreparationError, match="^modal_inference_preparation_invalid$"
    ) as caught:
        prepare_modal_chat(
            source,
            workload,
            configuration=configuration,
            configuration_trust=TrustedEvidenceIdentity(
                "host-config", "config-key", "chat-session"
            ),
            quote=quote,
            quote_trust=TrustedEvidenceIdentity(
                "host-quoter", "quote-key", "project-run"
            ),
            evidence_authenticator=Auth(),
            clock=Clock(),
            session_id="chat-session-1",
            executor_version="v1",
        )
    assert caught.value.__cause__ is None


def test_chat_preparation_is_factory_issued_and_config_is_canonical():
    with pytest.raises(TypeError):
        ModalInferencePreparation(preparation=None, executor=None, snapshot=b"")
    with pytest.raises(ValueError):
        ModalInferencePreparationConfig.parse(b'{"provider":1}')


def test_training_submit_grant_cannot_authorize_chat_submit(monkeypatch):
    value, source, *_ = _prepared(monkeypatch)
    prep = value.preparation
    stage_effect = derive_effect(prep, EffectKind.STAGE)
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
        stage_effect.effect_id,
        "a" * 64,
        "b" * 64,
    )
    chat = value.submit("chat-submit-nonce", predecessor)
    authority = GrantAuthorityV2("grant-authority", b"g" * 32)
    training = authority.issue(
        source.command_binding.command_bytes,
        grant_ref="training-grant",
        policy_digest="c" * 64,
        requirement_digest="d" * 64,
        not_before_epoch=1,
        expires_at_epoch=100,
    )
    assert authority.verify(training, chat.canonical_bytes, now_epoch=2) is False


@pytest.mark.parametrize(
    "field", ("profile_ref", "run_id", "preparation_digest", "stage_effect_id")
)
def test_submit_rejects_any_foreign_stage_predecessor(monkeypatch, field):
    value, *_ = _prepared(monkeypatch)
    prep = value.preparation
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
    replacement = "0" * 64 if field == "preparation_digest" else "other"
    with pytest.raises(ValueError, match="predecessor"):
        value.submit("submit-nonce", replace(predecessor, **{field: replacement}))


def test_source_mutation_during_evidence_authentication_is_rejected(monkeypatch):
    _, source, workload, configuration, quote = _prepared(monkeypatch)
    original = source.manifest_digest

    class MutatingAuth(Auth):
        def verify(self, purpose, payload, tag, key_ref):
            result = super().verify(purpose, payload, tag, key_ref)
            if purpose == CONFIG_EVIDENCE_PURPOSE:
                object.__setattr__(source, "manifest_digest", "0" * 64)
            return result

    try:
        with pytest.raises(
            ModalInferencePreparationError,
            match="^modal_inference_preparation_invalid$",
        ):
            prepare_modal_chat(
                source,
                workload,
                configuration=configuration,
                configuration_trust=TrustedEvidenceIdentity(
                    "host-config", "config-key", "chat-session"
                ),
                quote=quote,
                quote_trust=TrustedEvidenceIdentity(
                    "host-quoter", "quote-key", "project-run"
                ),
                evidence_authenticator=MutatingAuth(),
                clock=Clock(),
                session_id="chat-session-1",
                executor_version="v1",
            )
    finally:
        object.__setattr__(source, "manifest_digest", original)


def test_clock_instance_shadowing_is_rejected(monkeypatch):
    _, source, workload, configuration, quote = _prepared(monkeypatch)
    clock = Clock()
    clock.now_iso = lambda: "2026-09-09T12:02:00Z"
    with pytest.raises(
        ModalInferencePreparationError, match="^modal_inference_preparation_invalid$"
    ):
        prepare_modal_chat(
            source,
            workload,
            configuration=configuration,
            configuration_trust=TrustedEvidenceIdentity(
                "host-config", "config-key", "chat-session"
            ),
            quote=quote,
            quote_trust=TrustedEvidenceIdentity(
                "host-quoter", "quote-key", "project-run"
            ),
            evidence_authenticator=Auth(),
            clock=clock,
            session_id="chat-session-1",
            executor_version="v1",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("model_ref", "other/model"),
        ("model_revision", "0" * 40),
        ("tokenizer_revision", "0" * 40),
        ("load_in_4bit", True),
    ),
)
def test_workload_model_projection_must_match_canonical_workload(
    monkeypatch, field, value
):
    _, source, workload, configuration, quote = _prepared(monkeypatch)
    original = getattr(workload, field)
    object.__setattr__(workload, field, value)
    try:
        with pytest.raises(
            ModalInferencePreparationError,
            match="^modal_inference_preparation_invalid$",
        ):
            prepare_modal_chat(
                source,
                workload,
                configuration=configuration,
                configuration_trust=TrustedEvidenceIdentity(
                    "host-config", "config-key", "chat-session"
                ),
                quote=quote,
                quote_trust=TrustedEvidenceIdentity(
                    "host-quoter", "quote-key", "project-run"
                ),
                evidence_authenticator=Auth(),
                clock=Clock(),
                session_id="chat-session-1",
                executor_version="v1",
            )
    finally:
        object.__setattr__(workload, field, original)


@pytest.mark.parametrize(
    "target", ("configuration", "quote", "configuration_trust", "quote_trust")
)
def test_evidence_callback_mutation_is_rejected(monkeypatch, target):
    _, source, workload, configuration, quote = _prepared(monkeypatch)
    configuration_trust = TrustedEvidenceIdentity(
        "host-config", "config-key", "chat-session"
    )
    quote_trust = TrustedEvidenceIdentity("host-quoter", "quote-key", "project-run")
    changed = []

    class MutatingAuth(Auth):
        def verify(self, purpose, payload, tag, key_ref):
            result = super().verify(purpose, payload, tag, key_ref)
            if not changed:
                changed.append(target)
                victim = {
                    "configuration": configuration,
                    "quote": quote,
                    "configuration_trust": configuration_trust,
                    "quote_trust": quote_trust,
                }[target]
                name = "tag" if target in ("configuration", "quote") else "issuer_ref"
                object.__setattr__(
                    victim, name, b"changed" if name == "tag" else "changed"
                )
            return result

    with pytest.raises(
        ModalInferencePreparationError, match="^modal_inference_preparation_invalid$"
    ):
        prepare_modal_chat(
            source,
            workload,
            configuration=configuration,
            configuration_trust=configuration_trust,
            quote=quote,
            quote_trust=quote_trust,
            evidence_authenticator=MutatingAuth(),
            clock=Clock(),
            session_id="chat-session-1",
            executor_version="v1",
        )
    assert changed == [target]


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("resources", "max_retries", True),
        ("resources", "service_port", 65536),
        ("resources", "provider_timeout_seconds", 86401),
        ("policy", "startup_timeout_seconds", 86401),
        ("policy", "request_timeout_seconds", 0),
        ("policy", "idle_timeout_seconds", 901),
        ("volumes", "chat_control_volume_id", "model-cache-id"),
        ("volumes", "chat_control_volume_ref", "model-cache"),
    ),
)
def test_configuration_rejects_wrong_types_bounds_and_colliding_volumes(
    monkeypatch, section, field, value
):
    source, _, _ = workload_case(monkeypatch)
    document = _document(source)
    document[section][field] = value
    with pytest.raises((TypeError, ValueError)):
        ModalInferencePreparationConfig.build(document)


def test_commands_are_fresh_immutable_views(monkeypatch):
    value, *_ = _prepared(monkeypatch)
    first = value.stage("nonce-a")
    second = value.stage("nonce-a")
    assert first is not second and first.canonical_bytes == second.canonical_bytes
    raw = first.canonical_bytes
    with pytest.raises(AttributeError):
        first._raw = b"changed"
    assert value.stage("nonce-a").canonical_bytes == raw


@pytest.mark.parametrize("field", ("resource_digest", "workload_digest"))
def test_retained_preparation_snapshot_mutation_cannot_issue_commands(
    monkeypatch, field
):
    value, *_ = _prepared(monkeypatch)
    document = parse_canonical_object(
        value.canonical_bytes, name="preparation snapshot"
    )
    document["preparation"][field] = "0" * 64
    object.__setattr__(value, "_snapshot", canonical_bytes(document))
    with pytest.raises((ModalInferencePreparationError, ValueError)):
        value.stage("nonce-after-mutation")


def test_retained_values_reject_normal_assignment(monkeypatch):
    value, _, _, configuration, _ = _prepared(monkeypatch)
    with pytest.raises(AttributeError):
        value._snapshot = b"changed"
    config = ModalInferencePreparationConfig.parse(configuration.body_bytes)
    with pytest.raises(AttributeError):
        config._raw = b"changed"


def test_configuration_duck_properties_are_not_evaluated(monkeypatch):
    _, source, workload, _, quote = _prepared(monkeypatch)
    reads = []

    class Duck:
        @property
        def body_bytes(self):
            reads.append("body")
            raise AssertionError("accessed")

        @property
        def tag(self):
            reads.append("tag")
            raise AssertionError("accessed")

    with pytest.raises(
        ModalInferencePreparationError, match="^modal_inference_preparation_invalid$"
    ):
        prepare_modal_chat(
            source,
            workload,
            configuration=Duck(),
            configuration_trust=TrustedEvidenceIdentity(
                "host-config", "config-key", "chat-session"
            ),
            quote=quote,
            quote_trust=TrustedEvidenceIdentity(
                "host-quoter", "quote-key", "project-run"
            ),
            evidence_authenticator=Auth(),
            clock=Clock(),
            session_id="chat-session-1",
            executor_version="v1",
        )
    assert reads == []
