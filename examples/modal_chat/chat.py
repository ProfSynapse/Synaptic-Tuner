"""Consumer composition for one authenticated, remote Modal chat session."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from tuner.execution.foundation_v2.broker import EffectBrokerV2
from tuner.execution.foundation_v2.canonical import (
    canonical_bytes,
    digest_text,
    parse_canonical_object,
    safe_ref,
)
from tuner.execution.foundation_v2.commands import parse_exact_command
from tuner.execution.foundation_v2.identities import EffectKind
from tuner.execution.foundation_v2.repository import InMemoryEffectRepositoryV2
from tuner.execution.providers.modal.inference_binding import ModalInferenceSourceBinder
from tuner.execution.providers.modal.inference_commands import (
    ModalInferenceCommandBinding,
)
from tuner.execution.providers.modal.inference_effects import (
    ModalChatEffectExecutor,
    ModalChatExecutorResolver,
)
from tuner.execution.providers.modal.inference_run_chat import ModalRunChatRuntime
from tuner.execution.providers.modal.inference_transport import (
    ModalChatLaunchSettings,
    ModalChatLeaseHandoff,
    ModalInferenceSdkTransport,
)
from tuner.execution.providers.modal.inference_preparation import (
    _validate_preparation_snapshot,
)

from .authority import UnavailableRecoveryVerifier
from .host import ModalChatHost


class ModalChatRuntimeCompositionError(RuntimeError):
    """Closed failure from consumer-owned chat composition."""


class _BindingCatalog:
    def __init__(self):
        self._values = {}
        self._lock = RLock()

    def publish_if_absent(self, key, value):
        key = safe_ref(key, "command_digest")
        if (
            type(value) is not ModalInferenceCommandBinding
            or value.command_digest != key
        ):
            raise ModalChatRuntimeCompositionError("modal_chat_runtime_invalid")
        owned = ModalInferenceCommandBinding(
            value.command_bytes, value.preparation_snapshot
        )
        with self._lock:
            prior = self._values.get(key)
            if prior is None:
                self._values[key] = owned
                return True
            if prior.canonical_bytes == owned.canonical_bytes:
                return False
        raise ModalChatRuntimeCompositionError("modal_chat_runtime_conflict")

    def resolve(self, key):
        key = safe_ref(key, "command_digest")
        with self._lock:
            value = self._values.get(key)
        return (
            None
            if value is None
            else ModalInferenceCommandBinding(
                value.command_bytes, value.preparation_snapshot
            )
        )

    def submit_digests(self):
        with self._lock:
            return tuple(
                key
                for key, value in self._values.items()
                if parse_exact_command(value.command_bytes).operation.effect.kind
                is EffectKind.SUBMIT
            )


@dataclass(frozen=True, slots=True)
class ModalChatRuntimeComposition:
    """Keep ownership reachable even if no ready lease was produced."""

    runtime: ModalRunChatRuntime
    transport: ModalInferenceSdkTransport
    catalog: _BindingCatalog

    def pending_ownership(self):
        return tuple(
            ownership
            for digest in self.catalog.submit_digests()
            if (
                ownership := self.transport.pending_cleanup(
                    submit_command_digest=digest
                )
            )
            is not None
        )


class _BindingAuthority:
    def __init__(self, catalog, configuration_bytes, session_id, executor_version):
        self._catalog = catalog
        self._configuration = bytes(configuration_bytes)
        self._session = safe_ref(session_id, "session_id")
        self._executor = safe_ref(executor_version, "executor_version")

    def authenticate(self, value):
        try:
            if type(value) is not ModalInferenceCommandBinding:
                return False
            snapshot = _validate_preparation_snapshot(value.preparation_snapshot)
            command = parse_exact_command(value.command_bytes)
            retained = self._catalog.resolve(value.command_digest)
            policy = (
                canonical_bytes(snapshot["configuration"]) == self._configuration
                and command.preparation.run_id == self._session
                and command.executor.executor_id == "modal-chat-executor"
                and command.executor.implementation_version == self._executor
            )
            return bool(
                policy
                and (
                    retained is None
                    or (
                        type(retained) is ModalInferenceCommandBinding
                        and retained.canonical_bytes == value.canonical_bytes
                    )
                )
            )
        except Exception:
            return False


class _Foundation:
    def __init__(self, repository, assessments):
        self._repository, self._assessments = repository, assessments

    def get(self, effect_id):
        return self._repository.get(effect_id)

    def assess(self, record):
        return self._assessments.assess(record)


class _ChatGrants:
    """One bounded grant for each exact phase of one chat session."""

    def __init__(
        self,
        authority,
        clock,
        *,
        session_id,
        policy_digest,
        requirement_digest,
        maximum_seconds,
    ):
        self._authority, self._clock = authority, clock
        self._session = safe_ref(session_id, "session_id")
        self._policy = digest_text(policy_digest, "policy_digest")
        self._requirement = digest_text(requirement_digest, "requirement_digest")
        if type(maximum_seconds) is not int or not 1 <= maximum_seconds <= 900:
            raise ValueError("chat grant bound is invalid")
        self._maximum = maximum_seconds
        self._issued = set()
        self._lock = RLock()

    def grant(self, command_bytes, *, phase):
        command = parse_exact_command(command_bytes)
        expected = {"stage": EffectKind.STAGE, "submit": EffectKind.SUBMIT}
        if (
            phase not in expected
            or command.operation.effect.kind is not expected[phase]
            or command.preparation.run_id != self._session
        ):
            raise ModalChatRuntimeCompositionError("modal_chat_grant_invalid")
        with self._lock:
            if phase in self._issued:
                raise ModalChatRuntimeCompositionError(
                    "modal_chat_grant_already_issued"
                )
            now = self._clock.now_epoch()
            if type(now) is not int or now < 0:
                raise ModalChatRuntimeCompositionError("modal_chat_grant_invalid")
            self._issued.add(phase)
        return self._authority.issue(
            command_bytes,
            grant_ref=f"{self._session}-{phase}-grant",
            policy_digest=self._policy,
            requirement_digest=self._requirement,
            not_before_epoch=now,
            expires_at_epoch=now + self._maximum,
        )


def compose_modal_run_chat(
    *,
    host: ModalChatHost,
    deployment,
    launch_source,
    recipes,
    configuration,
    configuration_trust,
    quote,
    quote_trust,
    evidence_authenticator,
    clock,
    session_id: str,
    stage_nonce: str,
    submit_nonce: str,
    executor_version: str,
    policy_digest: str,
    requirement_digest: str,
    launch_settings: ModalChatLaunchSettings,
    launch_signer,
    issued_at: str,
    expires_at: str,
    evidence_ref: str,
    profile_ref: str,
    namespace_ref: str,
    prequalified_image_id: str,
    maximum_grant_seconds: int = 300,
) -> ModalChatRuntimeComposition:
    """Internal graph constructor; invoke only through storage-claimed chat_once.

    Configuration issuance must use the reviewed-capture helper. The image
    comparison below preserves that upstream selection, not a second runtime
    qualification. Construction neither claims an attempt nor authorizes retry.
    """
    if type(host) is not ModalChatHost:
        raise TypeError("exact Modal chat host required")
    if clock is not host.clock:
        raise ModalChatRuntimeCompositionError("modal_chat_clock_mismatch")
    config_document = parse_canonical_object(
        configuration.body_bytes, name="authenticated chat configuration"
    )
    if config_document["image"]["provider_image_id"] != safe_ref(
        prequalified_image_id, "prequalified_image_id"
    ) or config_document["provider"]["profile_ref"] != safe_ref(
        profile_ref, "profile_ref"
    ):
        raise ModalChatRuntimeCompositionError("modal_chat_image_not_qualified")
    runs = host.api.runs
    ports = host.foundation_ports
    foundation = host.composition.foundation
    binder = ModalInferenceSourceBinder(
        runs=runs,
        workflows=host.stores.workflow_store,
        foundation=foundation,
        foundation_authenticator=ports.foundation_authenticator,
        assessment_authenticator=ports.assessment_authority,
        reader=host.reader,
    )
    catalog = _BindingCatalog()
    authority = _BindingAuthority(
        catalog, configuration.body_bytes, session_id, executor_version
    )
    handoff = ModalChatLeaseHandoff()
    repository = InMemoryEffectRepositoryV2(
        ports.receipt_authority,
        ports.invalid_evidence_authority,
        UnavailableRecoveryVerifier(),
        UnavailableRecoveryVerifier(),
        ports.grant_authority,
    )
    chat_foundation = _Foundation(repository, ports.assessment_authority)
    transport = ModalInferenceSdkTransport(
        facade=deployment.facade(),
        binding_authority=authority,
        handoff=handoff,
        foundation=chat_foundation,
        catalog=catalog,
        foundation_authenticator=ports.foundation_authenticator,
        assessment_authenticator=ports.assessment_authority,
        signer=launch_signer,
        launch_settings=launch_settings,
        clock=clock,
        issued_at=issued_at,
        expires_at=expires_at,
        evidence_ref=evidence_ref,
    )
    selection = deployment.selection
    executor = ModalChatEffectExecutor(
        profile_ref=profile_ref,
        account_ref=selection.account_ref,
        namespace_ref=safe_ref(namespace_ref, "namespace_ref"),
        implementation_version=executor_version,
        catalog=catalog,
        authority=authority,
        transport=transport,
    )
    broker = EffectBrokerV2(
        repository,
        ModalChatExecutorResolver(executor),
        ports.grant_authority,
        ports.receipt_authority,
        ports.invalid_evidence_authority,
    )
    grants = _ChatGrants(
        ports.grant_authority,
        clock,
        session_id=session_id,
        policy_digest=policy_digest,
        requirement_digest=requirement_digest,
        maximum_seconds=maximum_grant_seconds,
    )
    runtime = ModalRunChatRuntime(
        runs=runs,
        source_binder=binder,
        launch_source=launch_source,
        foundation_authenticator=ports.foundation_authenticator,
        assessment_authenticator=ports.assessment_authority,
        workload_binding_authority=ports.binding_authority,
        stage_verifier=ports.stage_authority,
        launch_verifier=ports.launch_authority,
        recipes=recipes,
        configuration=configuration,
        configuration_trust=configuration_trust,
        quote=quote,
        quote_trust=quote_trust,
        evidence_authenticator=evidence_authenticator,
        clock=clock,
        session_id=session_id,
        executor_version=executor_version,
        stage_nonce=stage_nonce,
        submit_nonce=submit_nonce,
        catalog=catalog,
        chat_binding_authority=authority,
        grants=grants,
        broker=broker,
        foundation=chat_foundation,
        handoff=handoff,
    )
    return ModalChatRuntimeComposition(runtime, transport, catalog)


__all__ = [
    "ModalChatRuntimeComposition",
    "ModalChatRuntimeCompositionError",
    "compose_modal_run_chat",
]
