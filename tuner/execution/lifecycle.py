"""Pure lifecycle transition rules for provider-neutral execution."""

from __future__ import annotations

from dataclasses import replace

from .contracts import (
    EffectIdentity,
    EffectRecord,
    EffectKind,
    EffectState,
    EventCode,
    InvalidTransition,
    ExecutionScope,
    GrantBinding,
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
    """Authenticate one canonical predecessor history, then apply one event."""
    failed = False
    try:
        result = _authenticate_and_apply(record, event)
    except Exception:
        failed = True
    if failed:
        raise InvalidTransition("invalid lifecycle transition") from None
    return result


def _authenticate_and_apply(
    record: LifecycleRecord, event: LifecycleEvent
) -> LifecycleRecord:
    if type(record) is not LifecycleRecord or type(event) is not LifecycleEvent:
        raise InvalidTransition("exact lifecycle values are required")
    presented_bytes = record.canonical_bytes
    canonical = LifecycleRecord.from_canonical_bytes(presented_bytes)
    if canonical != record or canonical.canonical_bytes != presented_bytes:
        raise InvalidTransition("lifecycle predecessor is not canonical")
    origin = canonical.events[0]
    if origin.code is not EventCode.RUN_PLANNED:
        raise InvalidTransition("lifecycle history has no canonical origin")
    replayed = initial_record(
        project_ref=canonical.project_ref,
        run_id=canonical.run_id,
        occurred_at=origin.occurred_at,
    )
    if replayed.events[0] != origin:
        raise InvalidTransition("lifecycle origin disagrees")
    for retained in canonical.events[1:]:
        replayed = _reduce_event(replayed, retained)
    if replayed.canonical_bytes != canonical.canonical_bytes:
        raise InvalidTransition("lifecycle snapshot disagrees with its history")
    detached_event = LifecycleEvent.from_dict(event.to_dict())
    if detached_event != event:
        raise InvalidTransition("lifecycle event is not canonical")
    return _reduce_event(replayed, detached_event)


def _reduce_event(record: LifecycleRecord, event: LifecycleEvent) -> LifecycleRecord:
    """Reduce one detached event; callers authenticate the predecessor once."""
    if type(record) is not LifecycleRecord or type(event) is not LifecycleEvent:
        raise InvalidTransition("exact lifecycle values are required")

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
        _require_authority(record, event.grant_binding)
        if event.grant_binding.effect_kind is EffectKind.CANCEL:
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
        _require_phase(phase, LifecyclePhase.PLANNED)
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
        _require_effect_authority(record, grant, event.effect)
        if any(item.identity == event.effect.identity for item in effects):
            raise InvalidTransition("effect identity is already retained")
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
        _require_effect_authority(record, grant, event.effect)
        effects = _replace_effect(effects, event.effect, event.code)
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
        _require_effect_authority(record, grant, event.effect)
        matching = tuple(
            (index, item)
            for index, item in enumerate(effects)
            if item.identity == event.effect.identity
        )
        if event.effect.identity.kind is EffectKind.SUBMIT:
            expected_phase = (
                LifecyclePhase.SUBMITTING if matching else LifecyclePhase.READY
            )
            if phase is not expected_phase:
                raise InvalidTransition("submit attempt is out of order")
        elif matching:
            if phase is not LifecyclePhase.CANCELLING:
                raise InvalidTransition("cancel attempt is out of order")
        elif phase not in {
            LifecyclePhase.QUEUED,
            LifecyclePhase.RUNNING,
            LifecyclePhase.VERIFYING,
        }:
            raise InvalidTransition("cancel attempt is out of order")
        if matching:
            if len(matching) != 1 or matching[0][1].state is not EffectState.CLAIMED:
                raise InvalidTransition("effect attempt has invalid prior state")
            _require_immutable_effect(matching[0][1], event.effect)
            updated = list(effects)
            updated[matching[0][0]] = event.effect
            effects = tuple(updated)
        else:
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
        _require_confirmed_submit(effects)
        if phase not in {
            LifecyclePhase.QUEUED,
            LifecyclePhase.RUNNING,
            LifecyclePhase.VERIFYING,
            LifecyclePhase.CANCELLING,
        }:
            raise InvalidTransition("unknown provider state is out of order")
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
    effects: tuple[EffectRecord, ...], replacement: EffectRecord, code: EventCode
) -> tuple[EffectRecord, ...]:
    if type(replacement) is not EffectRecord:
        raise InvalidTransition("effect outcome is not canonical")
    updated: list[EffectRecord] = []
    matched = False
    for effect in effects:
        if effect.identity == replacement.identity:
            if matched or effect.state not in {
                EffectState.ATTEMPTED,
                EffectState.INDETERMINATE,
            }:
                raise InvalidTransition("effect outcome has invalid prior state")
            _require_immutable_effect(effect, replacement)
            expected_state = {
                EventCode.EFFECT_FOUND: EffectState.FOUND,
                EventCode.EFFECT_DEFINITELY_ABSENT: EffectState.DEFINITELY_ABSENT,
                EventCode.EFFECT_INDETERMINATE: EffectState.INDETERMINATE,
            }[code]
            if replacement.state is not expected_state:
                raise InvalidTransition("effect outcome state disagrees with event")
            updated.append(replacement)
            matched = True
        else:
            updated.append(effect)
    if not matched:
        raise InvalidTransition("effect outcome has no durable claim")
    return tuple(updated)


def _require_authority(record: LifecycleRecord, binding: GrantBinding) -> None:
    from .operation import OperationBindingV1

    if (
        type(binding) is not GrantBinding
        or type(binding.operation) is not OperationBindingV1
        or type(binding.operation.effect) is not EffectIdentity
        or type(binding.operation.effect.scope) is not ExecutionScope
        or GrantBinding.from_dict(binding.to_dict()) != binding
    ):
        raise InvalidTransition("authority is not canonical")
    operation = binding.operation
    if operation.project_ref != record.project_ref or operation.run_id != record.run_id:
        raise InvalidTransition("authority targets a different run")
    if operation.grant_ref != binding.grant_ref:
        raise InvalidTransition("authority grant reference disagrees")
    if operation.effect.kind is EffectKind.CANCEL:
        target = tuple(
            effect
            for effect in record.effects
            if effect.identity.kind is EffectKind.SUBMIT
            and effect.state is EffectState.FOUND
            and effect.provider_job_ref == operation.target_provider_job_ref
        )
        if len(target) != 1 or target[0].identity.scope != operation.effect.scope:
            raise InvalidTransition("cancel authority target is not confirmed")


def _require_effect_authority(
    record: LifecycleRecord,
    binding: GrantBinding | None,
    effect: EffectRecord,
) -> None:
    if type(effect) is not EffectRecord or type(effect.identity) is not EffectIdentity:
        raise InvalidTransition("effect is not canonical")
    if binding is None:
        raise InvalidTransition("effect requires active authority")
    _require_authority(record, binding)
    from .broker import MutationCommandV1

    command = MutationCommandV1.from_bytes(effect.canonical_command)
    if (
        command.digest != effect.command_digest
        or command.operation != binding.operation
        or command.effect != effect.identity
        or effect.grant_ref != binding.grant_ref
        or effect.grant_fingerprint != binding.fingerprint
        or command.project_ref != record.project_ref
        or command.run_id != record.run_id
    ):
        raise InvalidTransition("effect authority or command disagrees")


def _require_immutable_effect(before: EffectRecord, after: EffectRecord) -> None:
    if (
        type(before) is not EffectRecord
        or type(after) is not EffectRecord
        or type(before.identity) is not EffectIdentity
        or type(after.identity) is not EffectIdentity
        or before.identity != after.identity
        or before.grant_fingerprint != after.grant_fingerprint
        or before.grant_ref != after.grant_ref
        or before.command_digest != after.command_digest
        or before.canonical_command != after.canonical_command
        or after.attempt_count != 1
    ):
        raise InvalidTransition("effect authority or command changed")


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
