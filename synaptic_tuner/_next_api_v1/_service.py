"""Private lifecycle service implementing the durable S1/S2/S3 protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ._log_policy import MessageCode
from ._provider import (
    AuthenticationUnavailable, CancelRequest, DefinitiveNoEffect,
    EffectKind, EffectObservation, EffectOutcomeUnknown, EffectReceipt,
    ExecutionProvider, LookupResult, ProtocolViolation, ProviderAuth,
    ProviderRunState, ProviderUnavailable, SubmitRequest,
)
from ._store import GrantBinding, InvalidTransition, JobStore, SubmissionClaim
from .execution import (
    AccessContext, CancelResult, ExecutionGrant, LogCursor, LogPage, RunRef,
    RunState, RunStatus,
)
from .training import TrainingPlan


class ProviderNotConfigured(RuntimeError):
    pass


class JobStoreUnconfigured(RuntimeError):
    code = "job_store_unconfigured"

    def __init__(self) -> None:
        super().__init__(self.code)


@dataclass(frozen=True, slots=True)
class StartResult:
    run: RunRef
    state: RunState
    replayed: bool


_PROVIDER_STATE = {
    ProviderRunState.SUBMITTED: RunState.SUBMITTED,
    ProviderRunState.QUEUED: RunState.QUEUED,
    ProviderRunState.RUNNING: RunState.RUNNING,
    ProviderRunState.SUCCEEDED: RunState.SUCCEEDED,
    ProviderRunState.FAILED: RunState.FAILED,
    ProviderRunState.CANCELLED: RunState.CANCELLED,
    ProviderRunState.UNKNOWN: RunState.RECONCILE_REQUIRED,
}


class JobService:
    """Coordinates durable effects; providers are explicitly injected only."""

    def __init__(self, store: JobStore | None, providers: Mapping[str, ExecutionProvider]) -> None:
        self._store_value = store
        self._provider_values = providers if store is None else dict(providers)
        if store is not None:
            for name, provider in self._provider_values.items():
                if name != provider.scope.provider:
                    raise ValueError("provider mapping key must match provider scope")

    @property
    def _store(self) -> JobStore:
        if self._store_value is None:
            raise JobStoreUnconfigured()
        return self._store_value

    def _provider(self, name: str) -> ExecutionProvider:
        self._store
        try:
            return self._provider_values[name]
        except KeyError as exc:
            raise ProviderNotConfigured("execution provider is not configured") from exc

    @staticmethod
    def _validate_call_boundary(
        provider: ExecutionProvider, auth: ProviderAuth, scope: object
    ) -> None:
        if (not isinstance(auth, ProviderAuth) or auth.scope != scope
                or provider.scope != scope):
            raise AuthenticationUnavailable("provider/auth/effect scope mismatch")

    def start(
        self, access: AccessContext, plan: TrainingPlan, grant: ExecutionGrant,
        binding: GrantBinding, auth: ProviderAuth,
    ) -> StartResult:
        self._store
        provider = self._provider(binding.scope.provider)
        if provider.scope != binding.scope:
            raise ProviderNotConfigured("execution scope is not configured")
        if binding.plan_fingerprint != plan.fingerprint:
            raise ValueError("grant binding does not match training plan")
        claim = self._store.claim_submission(
            access, grant, binding, canonical_plan=_canonical_plan(plan)
        )
        if not claim.new_claim:
            return StartResult(claim.run, claim.state, True)
        if not isinstance(auth, ProviderAuth) or auth.scope != binding.scope:
            self._store.finish_effect(
                access, claim.run, EffectKind.SUBMIT, effect_status="absent",
                state=RunState.NOT_SUBMITTED,
                message=MessageCode.AUTHENTICATION_UNAVAILABLE,
            )
            return StartResult(claim.run, RunState.NOT_SUBMITTED, False)
        started = self._store.mark_effect_started(access, claim.run, EffectKind.SUBMIT)
        request = SubmitRequest(
            started.identity, plan.fingerprint, plan.source_digest,
            plan.workload_digest, plan.artifact_slot_ref,
        )
        try:
            self._validate_call_boundary(provider, auth, started.identity.scope)
            receipt = provider.submit(auth, request)
            self._validate_receipt(receipt, started.identity)
        except DefinitiveNoEffect:
            state = self._store.finish_effect(
                access, claim.run, EffectKind.SUBMIT, effect_status="absent",
                state=RunState.NOT_SUBMITTED,
                message=MessageCode.EFFECT_DEFINITIVELY_ABSENT,
            )
        except AuthenticationUnavailable:
            state = self._ambiguous(access, claim.run, EffectKind.SUBMIT,
                                    MessageCode.AUTHENTICATION_UNAVAILABLE)
        except ProviderUnavailable:
            state = self._ambiguous(access, claim.run, EffectKind.SUBMIT,
                                    MessageCode.PROVIDER_UNAVAILABLE)
        except (EffectOutcomeUnknown, ProtocolViolation):
            state = self._ambiguous(access, claim.run, EffectKind.SUBMIT,
                                    MessageCode.EFFECT_OUTCOME_UNKNOWN)
        except Exception:
            state = self._ambiguous(access, claim.run, EffectKind.SUBMIT,
                                    MessageCode.EFFECT_OUTCOME_UNKNOWN)
        else:
            state = self._store.finish_effect(
                access, claim.run, EffectKind.SUBMIT, effect_status="confirmed",
                state=RunState.SUBMITTED, message=MessageCode.EFFECT_CONFIRMED,
                provider_job=receipt.job, receipt_digest=receipt.receipt_digest,
            )
        return StartResult(claim.run, state.state, False)

    def reconcile(
        self, access: AccessContext, run: RunRef, auth: ProviderAuth,
        *, kind: EffectKind = EffectKind.SUBMIT,
    ) -> RunStatus:
        claim = self._store.claim_reconcile(access, run, kind)
        effect = claim.effect
        provider = self._provider(effect.identity.scope.provider)
        if not isinstance(auth, ProviderAuth) or auth.scope != effect.identity.scope:
            return self._ambiguous(access, run, kind, MessageCode.AUTHENTICATION_UNAVAILABLE,
                                   reconciliation_token=claim.claim_token)
        try:
            self._validate_call_boundary(provider, auth, effect.identity.scope)
            observation = (
                provider.lookup_submission(auth, effect.identity)
                if kind is EffectKind.SUBMIT
                else provider.lookup_cancellation(auth, effect.identity)
            )
            self._validate_observation(observation, effect.identity)
        except Exception:
            return self._ambiguous(access, run, kind, MessageCode.EFFECT_OUTCOME_UNKNOWN,
                                   reconciliation_token=claim.claim_token)
        if observation.result is LookupResult.FOUND:
            if kind is EffectKind.CANCEL and observation.job != effect.provider_job:
                return self._ambiguous(
                    access, run, kind, MessageCode.PROVIDER_PROTOCOL_VIOLATION,
                    reconciliation_token=claim.claim_token,
                )
            state = RunState.SUBMITTED if kind is EffectKind.SUBMIT else RunState.CANCELLING
            return self._store.finish_effect(
                access, run, kind, effect_status="confirmed", state=state,
                message=MessageCode.EFFECT_CONFIRMED, provider_job=observation.job,
                receipt_digest=observation.receipt_digest,
                observation_result=observation.result.value,
                reconciliation_token=claim.claim_token,
            )
        if observation.result is LookupResult.DEFINITIVELY_ABSENT:
            state = RunState.NOT_SUBMITTED if kind is EffectKind.SUBMIT else RunState.CANCEL_FAILED
            return self._store.finish_effect(
                access, run, kind, effect_status="absent", state=state,
                message=MessageCode.EFFECT_DEFINITIVELY_ABSENT,
                observation_result=observation.result.value,
                reconciliation_token=claim.claim_token,
            )
        message = (MessageCode.EFFECT_COLLISION if observation.result is LookupResult.COLLISION
                   else MessageCode.EFFECT_OUTCOME_UNKNOWN)
        return self._store.finish_effect(
            access, run, kind, effect_status="ambiguous",
            state=(RunState.SUBMISSION_AMBIGUOUS if kind is EffectKind.SUBMIT else RunState.CANCEL_AMBIGUOUS),
            message=message, observation_result=observation.result.value,
            reconciliation_token=claim.claim_token,
        )

    def refresh(self, access: AccessContext, run: RunRef, auth: ProviderAuth) -> RunStatus:
        effect = self._store.effect(access, run, EffectKind.SUBMIT)
        if effect.provider_job is None:
            return self._store.status(access, run)
        provider = self._provider(effect.identity.scope.provider)
        if not isinstance(auth, ProviderAuth) or auth.scope != effect.identity.scope:
            return self._store.status(access, run)
        try:
            self._validate_call_boundary(provider, auth, effect.identity.scope)
            observation = provider.observe(auth, effect.provider_job)
            if observation.job != effect.provider_job:
                raise ProtocolViolation("provider returned a mismatched job")
            state = _PROVIDER_STATE[observation.state]
        except Exception:
            return self._store.status(access, run)
        return self._store.observe_state(access, run, state, observation.observation_digest)

    def cancel(
        self, access: AccessContext, run: RunRef, auth: ProviderAuth,
    ) -> CancelResult:
        current = self._store.status(access, run)
        if current.state in {
            RunState.SUBMISSION_AMBIGUOUS, RunState.RECONCILE_REQUIRED,
            RunState.RECONCILING,
        }:
            current = self.reconcile(
                access, run, auth, kind=self._store.reconciliation_kind(access, run)
            )
        try:
            effect = self._store.create_cancel_effect(access, run)
        except InvalidTransition:
            state = self._store.status(access, run)
            return CancelResult(run, RunState.CANCEL_FAILED, False, state.message_code)
        if effect is None:
            return CancelResult(run, RunState.CANCEL_FAILED, False, "run_already_terminal")
        if not isinstance(auth, ProviderAuth) or auth.scope != effect.identity.scope:
            status = self._store.finish_effect(
                access, run, EffectKind.CANCEL, effect_status="absent",
                state=RunState.CANCEL_FAILED, message=MessageCode.AUTHENTICATION_UNAVAILABLE,
            )
            return CancelResult(run, status.state, False, status.message_code)
        if effect.status != "claimed":
            state = self._store.status(access, run)
            accepted = state.state in {RunState.CANCEL_REQUESTED, RunState.CANCELLING}
            return CancelResult(run, state.state, accepted, state.message_code)
        started = self._store.mark_effect_started(access, run, EffectKind.CANCEL)
        if started.provider_job is None:
            # The store preserves the target on the cancellation effect.
            started = self._store.effect(access, run, EffectKind.CANCEL)
        if started.provider_job is None:
            return self._cancel_ambiguous(access, run, MessageCode.PROVIDER_PROTOCOL_VIOLATION)
        provider = self._provider(started.identity.scope.provider)
        try:
            self._validate_call_boundary(provider, auth, started.identity.scope)
            receipt = provider.cancel(auth, CancelRequest(started.identity, started.provider_job))
            self._validate_receipt(receipt, started.identity)
            if receipt.job != started.provider_job:
                raise ProtocolViolation("cancellation receipt job mismatch")
        except DefinitiveNoEffect:
            status = self._store.finish_effect(
                access, run, EffectKind.CANCEL, effect_status="absent",
                state=RunState.CANCEL_FAILED,
                message=MessageCode.EFFECT_DEFINITIVELY_ABSENT,
            )
            return CancelResult(run, status.state, False, status.message_code)
        except Exception:
            return self._cancel_ambiguous(access, run, MessageCode.EFFECT_OUTCOME_UNKNOWN)
        status = self._store.finish_effect(
            access, run, EffectKind.CANCEL, effect_status="confirmed",
            state=RunState.CANCELLING, message=MessageCode.EFFECT_CONFIRMED,
            provider_job=receipt.job, receipt_digest=receipt.receipt_digest,
        )
        return CancelResult(run, status.state, True, status.message_code)

    def show(self, access: AccessContext, run: RunRef) -> RunStatus:
        return self._store.status(access, run)

    def list(self, access: AccessContext, *, cursor: LogCursor | None = None,
             limit: int = 50) -> tuple[RunRef, ...]:
        if cursor is not None:
            raise ValueError("run-list cursors are not implemented in the private slice")
        return self._store.list_runs(access, limit)

    def logs(self, access: AccessContext, run: RunRef, *,
             cursor: LogCursor | None = None, limit: int = 100) -> LogPage:
        return self._store.logs(access, run, cursor, limit)

    @staticmethod
    def _validate_receipt(receipt: EffectReceipt, identity: object) -> None:
        if not isinstance(receipt, EffectReceipt) or receipt.identity != identity:
            raise ProtocolViolation("provider receipt identity mismatch")

    @staticmethod
    def _validate_observation(observation: EffectObservation, identity: object) -> None:
        if not isinstance(observation, EffectObservation) or observation.identity != identity:
            raise ProtocolViolation("provider observation identity mismatch")

    def _ambiguous(self, access: AccessContext, run: RunRef, kind: EffectKind,
                   message: MessageCode, *,
                   reconciliation_token: str | None = None) -> RunStatus:
        return self._store.finish_effect(
            access, run, kind, effect_status="ambiguous",
            state=(RunState.SUBMISSION_AMBIGUOUS if kind is EffectKind.SUBMIT else RunState.CANCEL_AMBIGUOUS),
            message=message, reconciliation_token=reconciliation_token,
        )

    def _cancel_ambiguous(self, access: AccessContext, run: RunRef,
                          message: MessageCode) -> CancelResult:
        status = self._ambiguous(access, run, EffectKind.CANCEL, message)
        return CancelResult(run, status.state, False, status.message_code)


def _canonical_plan(plan: TrainingPlan) -> str:
    import json
    return json.dumps(plan.to_dict(), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


__all__ = ["JobService", "JobStoreUnconfigured", "ProviderNotConfigured", "StartResult"]
