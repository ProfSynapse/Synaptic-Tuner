"""Pure lifecycle transition rules for provider-neutral execution."""

from __future__ import annotations

from dataclasses import replace

from .contracts import (
    EffectRecord,
    EffectState,
    EventCode,
    InvalidTransition,
    LifecycleEvent,
    LifecyclePhase,
    LifecycleRecord,
    MessageCode,
    VerificationStatus,
)


_TERMINAL_PHASES = {
    LifecyclePhase.SUCCEEDED,
    LifecyclePhase.FAILED,
    LifecyclePhase.CANCELLED,
}


def initial_record(
    *, project_ref: str, run_id: str, occurred_at: str
) -> LifecycleRecord:
    event = LifecycleEvent(
        EventCode.RUN_PLANNED, occurred_at, MessageCode.PLANNED
    )
    return LifecycleRecord(
        run_id=run_id,
        project_ref=project_ref,
        revision=1,
        phase=LifecyclePhase.PLANNED,
        verification=VerificationStatus.NOT_READY,
        updated_at=event.occurred_at,
        message_code=event.message_code,
        events=(event,),
    )


def apply_event(record: LifecycleRecord, event: LifecycleEvent) -> LifecycleRecord:
    """Apply one already-durable event to a lifecycle snapshot."""

    reopening_invalid_verification = (
        event.code is EventCode.VERIFICATION_REOPENED
        and record.phase is LifecyclePhase.FAILED
        and record.verification is VerificationStatus.INVALID
    )
    if record.phase in _TERMINAL_PHASES and not reopening_invalid_verification:
        raise InvalidTransition("terminal runs cannot transition")

    phase = record.phase
    verification = record.verification
    grant = record.grant_binding
    effects = record.effects

    if event.code is EventCode.VERIFICATION_REOPENED:
        if not reopening_invalid_verification:
            raise InvalidTransition("only invalid verification may be reopened")
        phase = LifecyclePhase.VERIFYING
        verification = VerificationStatus.VERIFYING
    elif event.code is EventCode.AUTHORITY_ACCEPTED:
        if event.grant_binding is None:
            raise InvalidTransition("authority event requires a grant binding")
        if event.grant_binding.effect_kind.value == "cancel":
            if phase not in {
                LifecyclePhase.QUEUED,
                LifecyclePhase.RUNNING,
                LifecyclePhase.VERIFYING,
            }:
                raise InvalidTransition("cancellation authority cannot be bound in the current phase")
        else:
            if phase not in {LifecyclePhase.PLANNED, LifecyclePhase.RECONCILE_REQUIRED}:
                raise InvalidTransition("submission authority cannot be bound in the current phase")
            if phase is LifecyclePhase.RECONCILE_REQUIRED:
                if not effects or any(
                    effect.state is not EffectState.DEFINITELY_ABSENT for effect in effects
                ):
                    raise InvalidTransition("new authority requires closed absent prior effects")
                if grant is not None and event.grant_binding.fingerprint == grant.fingerprint:
                    raise InvalidTransition("new effect authority must be freshly bound")
            phase = LifecyclePhase.READY
        grant = event.grant_binding
    elif event.code is EventCode.AUTHORIZATION_REJECTED:
        phase = LifecyclePhase.FAILED
    elif event.code is EventCode.PREPARATION_STARTED:
        _require_phase(phase, LifecyclePhase.READY)
        phase = LifecyclePhase.PREPARING
    elif event.code is EventCode.PREPARATION_COMPLETED:
        _require_phase(phase, LifecyclePhase.PREPARING)
        phase = LifecyclePhase.READY
    elif event.code is EventCode.EFFECT_CLAIMED:
        if event.effect is None or event.effect.state is not EffectState.CLAIMED:
            raise InvalidTransition("effect claim event requires a claimed effect")
        if grant is None or event.effect.grant_fingerprint != grant.fingerprint:
            raise InvalidTransition("effect claim must match current authority")
        if event.effect.identity.kind.value == "submit":
            _require_phase(phase, LifecyclePhase.READY)
            phase = LifecyclePhase.SUBMITTING
        else:
            if phase not in {
                LifecyclePhase.QUEUED,
                LifecyclePhase.RUNNING,
                LifecyclePhase.VERIFYING,
            }:
                raise InvalidTransition("cancellation cannot start in the current phase")
            phase = LifecyclePhase.CANCELLING
        effects = effects + (event.effect,)
    elif event.code in {
        EventCode.EFFECT_FOUND,
        EventCode.EFFECT_DEFINITELY_ABSENT,
        EventCode.EFFECT_INDETERMINATE,
    }:
        if event.effect is None:
            raise InvalidTransition("effect outcome event requires an effect")
        effects = _replace_effect(effects, event.effect)
        if event.code is EventCode.EFFECT_FOUND:
            phase = (
                LifecyclePhase.QUEUED
                if event.effect.identity.kind.value == "submit"
                else LifecyclePhase.CANCELLING
            )
        else:
            phase = LifecyclePhase.RECONCILE_REQUIRED
    elif event.code is EventCode.EFFECT_ATTEMPTED:
        if event.effect is None or event.effect.state is not EffectState.ATTEMPTED:
            raise InvalidTransition("attempt event requires an attempted effect")
        if event.effect.identity.kind.value == "submit":
            if phase is not LifecyclePhase.READY:
                raise InvalidTransition("submit attempt is out of order")
        elif phase not in {LifecyclePhase.QUEUED, LifecyclePhase.RUNNING, LifecyclePhase.VERIFYING}:
            raise InvalidTransition("cancel attempt is out of order")
        effects = effects + (event.effect,)
        phase = LifecyclePhase.SUBMITTING if event.effect.identity.kind.value == "submit" else LifecyclePhase.CANCELLING
    elif event.code is EventCode.PROVIDER_QUEUED:
        _require_confirmed_submit(effects)
        if phase not in {LifecyclePhase.SUBMITTING, LifecyclePhase.QUEUED}:
            raise InvalidTransition("provider queue observation is out of order")
        phase = LifecyclePhase.QUEUED
    elif event.code is EventCode.PROVIDER_RUNNING:
        _require_confirmed_submit(effects)
        if phase not in {LifecyclePhase.QUEUED, LifecyclePhase.RUNNING}:
            raise InvalidTransition("provider running observation is out of order")
        phase = LifecyclePhase.RUNNING
    elif event.code is EventCode.PROVIDER_SUCCEEDED:
        _require_confirmed_submit(effects)
        if phase not in {LifecyclePhase.QUEUED, LifecyclePhase.RUNNING}:
            raise InvalidTransition("provider success is out of order")
        phase = LifecyclePhase.VERIFYING
        verification = VerificationStatus.PENDING
    elif event.code is EventCode.PROVIDER_FAILED:
        _require_confirmed_submit(effects)
        if phase not in {
            LifecyclePhase.SUBMITTING,
            LifecyclePhase.QUEUED,
            LifecyclePhase.RUNNING,
        }:
            raise InvalidTransition("provider failure is out of order")
        phase = LifecyclePhase.FAILED
    elif event.code is EventCode.PROVIDER_CANCELLED:
        if phase not in {
            LifecyclePhase.QUEUED,
            LifecyclePhase.RUNNING,
            LifecyclePhase.VERIFYING,
            LifecyclePhase.CANCELLING,
        }:
            raise InvalidTransition("provider cancellation is out of order")
        phase = LifecyclePhase.CANCELLED
    elif event.code is EventCode.PROVIDER_UNKNOWN:
        phase = LifecyclePhase.RECONCILE_REQUIRED
    elif event.code is EventCode.VERIFICATION_STARTED:
        if phase is LifecyclePhase.RECONCILE_REQUIRED and verification is VerificationStatus.INCONCLUSIVE:
            phase = LifecyclePhase.VERIFYING
        else:
            _require_phase(phase, LifecyclePhase.VERIFYING)
        if verification not in {
            VerificationStatus.PENDING,
            VerificationStatus.VERIFYING,
            VerificationStatus.INCONCLUSIVE,
        }:
            raise InvalidTransition("verification cannot start from the current status")
        verification = VerificationStatus.VERIFYING
    elif event.code is EventCode.VERIFICATION_VERIFIED:
        _require_phase(phase, LifecyclePhase.VERIFYING)
        if verification not in {VerificationStatus.PENDING, VerificationStatus.VERIFYING}:
            raise InvalidTransition("verification cannot complete from the current status")
        phase = LifecyclePhase.SUCCEEDED
        verification = VerificationStatus.VERIFIED
    elif event.code is EventCode.VERIFICATION_INVALID:
        _require_phase(phase, LifecyclePhase.VERIFYING)
        phase = LifecyclePhase.FAILED
        verification = VerificationStatus.INVALID
    elif event.code is EventCode.VERIFICATION_INCONCLUSIVE:
        _require_phase(phase, LifecyclePhase.VERIFYING)
        phase = LifecyclePhase.RECONCILE_REQUIRED
        verification = VerificationStatus.INCONCLUSIVE
    else:
        raise InvalidTransition("unsupported lifecycle event")

    return replace(
        record,
        revision=record.revision + 1,
        phase=phase,
        verification=verification,
        updated_at=event.occurred_at,
        message_code=event.message_code,
        events=record.events + (event,),
        effects=effects,
        grant_binding=grant,
    )


def _replace_effect(
    effects: tuple[EffectRecord, ...], replacement: EffectRecord
) -> tuple[EffectRecord, ...]:
    updated: list[EffectRecord] = []
    matched = False
    for effect in effects:
        if effect.identity == replacement.identity:
            if effect.grant_fingerprint != replacement.grant_fingerprint:
                raise InvalidTransition("effect grant binding cannot change")
            updated.append(replacement)
            matched = True
        else:
            updated.append(effect)
    if not matched:
        raise InvalidTransition("effect outcome has no durable claim")
    return tuple(updated)


def _require_phase(actual: LifecyclePhase, expected: LifecyclePhase) -> None:
    if actual is not expected:
        raise InvalidTransition(
            f"transition requires {expected.value} phase"
        )


def _require_confirmed_submit(effects: tuple[EffectRecord, ...]) -> None:
    if not any(
        effect.identity.kind.value == "submit" and effect.state is EffectState.FOUND
        for effect in effects
    ):
        raise InvalidTransition(
            "provider lifecycle requires a durably confirmed submit effect"
        )


__all__ = ["apply_event", "initial_record"]
