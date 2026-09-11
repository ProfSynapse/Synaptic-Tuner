"""Consumer composition of verified runs, Foundation effects and bounded chat.

The consumer owns every store, content authority and grant. This adapter never
mints permission or retries submission, and never downloads models locally.
One instance represents one session attempt, including an interrupted attempt.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
import threading
from typing import Iterator, Literal, Protocol

from Evaluator.chat_session import ChatSession, ChatSessionPolicy
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.runs_facade import RunsAPI
from tuner.execution.coordinator_v1.model import (
    AuthenticatedFoundationRecordAssessmentV1,
    BoundProviderRunRefV1,
    BoundProviderStageRefV1,
    EffectIntentV1,
    FoundationDispositionV1,
)
from tuner.execution.coordinator_v1.state_machine import _derive_foundation
from tuner.execution.foundation_v2.authority import AuthenticatedGrantV2
from tuner.execution.foundation_v2.broker import EffectBrokerV2
from tuner.execution.foundation_v2.canonical import safe_ref
from tuner.execution.foundation_v2.references import StagePredecessorV2
from tuner.execution.foundation_v2.repository import EffectRecordV2
from tuner.inference.run_chat import PreparedRunChat

from .contracts import BoundsPolicyV1
from .coordinator_preflight import AuthenticatedModalQuote, TrustedEvidenceIdentity
from .inference_binding import ModalInferenceSourceBinder, _method, _run
from .inference_commands import ModalInferenceCommandBinding
from .inference_launch import modal_chat_stage_ref
from .inference_preparation import (
    AuthenticatedModalInferencePreparationConfig,
    prepare_modal_chat,
    _validate_preparation_snapshot,
)
from .inference_retention import retain_modal_chat_command
from .inference_transport import ModalChatLeaseHandoff, ModalChatSandboxLease
from .inference_workload import bind_modal_inference_workload


class ModalRunChatError(RuntimeError):
    """Closed non-secret failure; retained Foundation state remains authoritative."""


class ModalChatGrantPort(Protocol):
    """Consumer authorization, not an engine-owned signing key or approval UI."""

    def grant(
        self, command_bytes: bytes, *, phase: Literal["stage", "submit"]
    ) -> AuthenticatedGrantV2: ...


class ModalRunChatRuntime:
    """One-use remote implementation of the existing RunChatRuntime protocol.

    Construction performs no provider I/O. All supplied collaborators are
    trusted consumer composition, not arbitrary plugins authenticated by their
    Python types. Retain the transport and its pending-cleanup ownership after
    a failed open; a missing ready lease never proves that creation did not run.
    """

    def __init__(
        self,
        *,
        runs: RunsAPI,
        source_binder: ModalInferenceSourceBinder,
        launch_source,
        foundation_authenticator,
        assessment_authenticator,
        workload_binding_authority,
        stage_verifier,
        launch_verifier,
        recipes,
        configuration: AuthenticatedModalInferencePreparationConfig,
        configuration_trust: TrustedEvidenceIdentity,
        quote: AuthenticatedModalQuote,
        quote_trust: TrustedEvidenceIdentity,
        evidence_authenticator,
        clock,
        session_id: str,
        executor_version: str,
        stage_nonce: str,
        submit_nonce: str,
        catalog,
        chat_binding_authority,
        grants: ModalChatGrantPort,
        broker: EffectBrokerV2,
        foundation,
        handoff: ModalChatLeaseHandoff,
        bounds: BoundsPolicyV1 = BoundsPolicyV1(),
    ) -> None:
        if type(runs) is not RunsAPI:
            raise TypeError("exact RunsAPI required")
        if type(source_binder) is not ModalInferenceSourceBinder:
            raise TypeError("exact Modal source binder required")
        if type(broker) is not EffectBrokerV2:
            raise TypeError("exact Foundation broker required")
        if type(handoff) is not ModalChatLeaseHandoff:
            raise TypeError("exact Modal chat handoff required")
        if (
            type(configuration) is not AuthenticatedModalInferencePreparationConfig
            or type(configuration_trust) is not TrustedEvidenceIdentity
            or type(quote) is not AuthenticatedModalQuote
            or type(quote_trust) is not TrustedEvidenceIdentity
            or type(bounds) is not BoundsPolicyV1
        ):
            raise TypeError("exact preparation configuration required")
        for value, names in (
            (grants, ("grant",)),
            (foundation, ("get", "assess")),
            (catalog, ("resolve", "publish_if_absent")),
            (chat_binding_authority, ("authenticate",)),
            (clock, ("now_iso", "now_epoch")),
        ):
            for name in names:
                _method(value, name)
        for name, value in (
            ("session_id", session_id),
            ("executor_version", executor_version),
            ("stage_nonce", stage_nonce),
            ("submit_nonce", submit_nonce),
        ):
            if type(value) is not str:
                raise TypeError("exact chat identity required")
            safe_ref(value, name)
        self._runs = runs
        self._source_binder = source_binder
        self._workload_inputs = dict(
            launch_source=launch_source,
            foundation_authenticator=foundation_authenticator,
            assessment_authenticator=assessment_authenticator,
            binding_authority=workload_binding_authority,
            stage_verifier=stage_verifier,
            launch_verifier=launch_verifier,
            recipes=recipes,
            bounds=copy.deepcopy(bounds),
        )
        self._preparation_inputs = dict(
            configuration=AuthenticatedModalInferencePreparationConfig(
                configuration.body_bytes, configuration.tag
            ),
            configuration_trust=copy.deepcopy(configuration_trust),
            quote=AuthenticatedModalQuote(quote.body_bytes, quote.tag),
            quote_trust=copy.deepcopy(quote_trust),
            evidence_authenticator=evidence_authenticator,
            clock=clock,
            session_id=session_id,
            executor_version=executor_version,
        )
        self._session_id = session_id
        self._stage_nonce = stage_nonce
        self._submit_nonce = submit_nonce
        self._catalog = catalog
        self._authority = chat_binding_authority
        self._grants = grants
        self._broker = broker
        self._foundation = foundation
        self._foundation_authenticator = foundation_authenticator
        self._assessment_authenticator = assessment_authenticator
        self._clock = clock
        self._handoff = handoff
        self._lock = threading.Lock()
        self._attempted = False
        self._owned_lease: ModalChatSandboxLease | None = None

    @property
    def owned_lease(self) -> ModalChatSandboxLease | None:
        """Retain exact cleanup ownership, including after an unsuccessful open."""
        return self._owned_lease

    def _found(self, command):
        """Authenticate actual retained Foundation evidence, not broker projection."""
        raw = command.canonical_bytes
        record = self._foundation.get(command.operation.effect.effect_id)
        if type(record) is not EffectRecordV2:
            raise ValueError("retained Foundation record unavailable")
        record_digest = record.record_digest
        owned = copy.deepcopy(record)
        assessment = self._foundation.assess(owned)
        if type(assessment) is not AuthenticatedFoundationRecordAssessmentV1:
            raise ValueError("retained Foundation assessment unavailable")
        _, outcome, bound = _derive_foundation(
            EffectIntentV1.from_command_bytes(raw),
            owned,
            assessment,
            self._foundation_authenticator,
            self._assessment_authenticator,
            None,
        )
        current = self._foundation.get(command.operation.effect.effect_id)
        if (
            outcome.disposition is not FoundationDispositionV1.FOUND
            or command.canonical_bytes != raw
            or record.record_digest != record_digest
            or owned.record_digest != record_digest
            or type(current) is not EffectRecordV2
            or current.record_digest != record_digest
            or current != owned
        ):
            raise ValueError("Foundation outcome unavailable or changed")
        return owned, bound

    def _retain(self, command, snapshot):
        return retain_modal_chat_command(
            ModalInferenceCommandBinding(command.canonical_bytes, snapshot),
            catalog=self._catalog,
            authority=self._authority,
        )

    def _execute(self, binding, phase, guard):
        raw = binding.command_bytes
        grant = self._grants.grant(raw, phase=phase)
        guard()
        if type(grant) is not AuthenticatedGrantV2 or binding.command_bytes != raw:
            raise ValueError("consumer grant or command changed")
        now = self._clock.now_epoch()
        guard()
        if type(now) is not int or now < 0:
            raise ValueError("invalid Foundation clock")
        # Foundation verifies and durably consumes the exact consumer grant.
        # Its uncertain outcome is never retried by this adapter.
        self._broker.execute(raw, grant, now_epoch=now)

    @contextmanager
    def open(self, runs: RunsAPI, run: TrainingRunRef) -> Iterator[PreparedRunChat]:
        if runs is not self._runs:
            raise ModalRunChatError("modal_run_chat_invalid")
        requested = _run(run)
        with self._lock:
            if self._attempted:
                raise ModalRunChatError("modal_run_chat_already_attempted")
            self._attempted = True
        lease = None
        session = None
        submit_binding = None
        try:
            source = self._source_binder.bind(runs, requested)
            workload = bind_modal_inference_workload(source, **self._workload_inputs)
            preparation = prepare_modal_chat(
                source, workload, **self._preparation_inputs
            )
            snapshot = preparation.canonical_bytes

            def guard():
                if _run(run) != requested or source.run != requested:
                    raise ValueError("requested run changed")
                # No repeated bind/reverify: those intentionally change the
                # workflow revision and would invalidate every valid attempt.
                self._source_binder.assert_current(source)
                current = prepare_modal_chat(
                    source, workload, **self._preparation_inputs
                )
                if current.canonical_bytes != snapshot:
                    raise ValueError("chat preparation changed")

            stage = preparation.stage(self._stage_nonce)
            stage_binding = self._retain(stage, snapshot)
            guard()
            self._execute(stage_binding, "stage", guard)
            stage_record, staged = self._found(stage)
            if type(
                staged
            ) is not BoundProviderStageRefV1 or staged.reference.stage_ref != modal_chat_stage_ref(
                stage_binding
            ):
                raise ValueError("authenticated stage reference differs")
            prep = preparation.preparation
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
                stage.operation.effect.effect_id,
                staged.authenticated_receipt_digest,
                stage_record.record_digest,
            )
            submit = preparation.submit(self._submit_nonce, predecessor)
            submit_binding = self._retain(submit, snapshot)
            guard()
            self._execute(submit_binding, "submit", guard)
            _, submitted = self._found(submit)
            if type(submitted) is not BoundProviderRunRefV1:
                raise ValueError("authenticated submitted run unavailable")
            lease = self._handoff.take(submit_command_digest=submit.digest)
            if type(lease) is not ModalChatSandboxLease:
                raise ValueError("exact owned Modal chat lease unavailable")
            self._owned_lease = lease
            model = lease.channel.model
            if (
                lease.submit_command_digest != submit.digest
                or lease.session_id != self._session_id
                or lease.preparation_snapshot != snapshot
                or lease.sandbox.object_id != submitted.reference.provider_job_ref
                or (model.model_ref, model.model_revision, model.tokenizer_revision)
                != (
                    workload.model_ref,
                    workload.model_revision,
                    workload.tokenizer_revision,
                )
            ):
                raise ValueError("ready lease differs from authenticated submission")
            guard()
            policy = _validate_preparation_snapshot(snapshot)["configuration"]["policy"]
            session = ChatSession(
                lease.channel,
                lease,
                ChatSessionPolicy(
                    **{
                        name: policy[name]
                        for name in ChatSessionPolicy.__dataclass_fields__
                    }
                ),
                deadline=lease.deadline,
            )
            prepared = PreparedRunChat(session, requested, source.artifacts, model)
        except BaseException as error:
            if lease is None and submit_binding is not None:
                # A successful transport can have published before Foundation
                # receipt persistence failed. Keep the slot retained, but close
                # its exact resource; never infer absence from a missing slot.
                lease = self._handoff.peek(
                    submit_command_digest=submit_binding.command_digest
                )
            if type(lease) is ModalChatSandboxLease:
                self._owned_lease = lease
            if session is not None:
                session.close()
                session.wait_closed(10.5)
            elif lease is not None:
                lease.close()
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise ModalRunChatError("modal_run_chat_invalid") from None
        with session:
            yield prepared


__all__: list[str] = []
