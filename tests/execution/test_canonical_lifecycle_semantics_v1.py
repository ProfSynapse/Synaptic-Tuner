from __future__ import annotations

from dataclasses import replace
import inspect
import json
from types import MappingProxyType

import pytest

from tests.execution.test_mutation_broker import D, NOW, command, grant, ready
from tuner.execution import contracts
from tuner.execution.contracts import (
    EffectKind,
    EffectRecord,
    EffectState,
    EventCode,
    GrantBinding,
    InvalidTransition,
    LifecycleEvent,
    LifecyclePhase,
    LifecycleRecord,
    MessageCode,
)
from tuner.execution import lifecycle
from tuner.execution.broker import MutationCommandV1
from tuner.execution.lifecycle import apply_event


def _effect(value, binding, state: EffectState) -> EffectRecord:
    found = state is EffectState.FOUND
    return EffectRecord(
        value.effect,
        binding.fingerprint,
        state,
        "fc-1" if found else None,
        D if found else None,
        value.grant_ref,
        value.digest,
        value.canonical_bytes,
        0 if state is EffectState.CLAIMED else 1,
    )


def _event(code: EventCode, effect: EffectRecord) -> LifecycleEvent:
    return LifecycleEvent(
        code,
        NOW,
        contracts._EVENT_MESSAGE_CODES[code],
        effect=effect,
    )


def _found_submit():
    value = command()
    _, record = ready(value)
    attempted = _effect(value, record.grant_binding, EffectState.ATTEMPTED)
    record = apply_event(record, _event(EventCode.EFFECT_ATTEMPTED, attempted))
    found = replace(
        attempted,
        state=EffectState.FOUND,
        provider_job_ref="fc-1",
        receipt_digest=D,
    )
    return value, apply_event(record, _event(EventCode.EFFECT_FOUND, found)), found


def test_event_message_inventory_is_exact_and_exhaustive() -> None:
    assert set(contracts._EVENT_MESSAGE_CODES) == set(EventCode)
    assert contracts._EVENT_MESSAGE_CODES[EventCode.RUN_PLANNED] is MessageCode.PLANNED
    assert (
        contracts._EVENT_MESSAGE_CODES[EventCode.PROVIDER_SUCCEEDED]
        is MessageCode.SEMANTIC_VERIFICATION_PENDING
    )
    assert (
        contracts._EVENT_MESSAGE_CODES[EventCode.VERIFICATION_REOPENED]
        is MessageCode.SEMANTIC_VERIFICATION_REOPENED
    )


def test_event_constructor_closes_wrong_message_and_payload_adjacency() -> None:
    value = command()
    binding = grant(value)
    attempted = _effect(value, binding, EffectState.ATTEMPTED)
    with pytest.raises(ValueError, match="message"):
        LifecycleEvent(
            EventCode.EFFECT_ATTEMPTED,
            NOW,
            MessageCode.EFFECT_CONFIRMED,
            effect=attempted,
        )
    with pytest.raises(ValueError, match="payload"):
        LifecycleEvent(
            EventCode.AUTHORITY_ACCEPTED,
            NOW,
            MessageCode.AUTHORITY_BOUND,
            effect=attempted,
            grant_binding=binding,
        )
    with pytest.raises(ValueError, match="payload"):
        LifecycleEvent(
            EventCode.PROVIDER_RUNNING,
            NOW,
            MessageCode.PROVIDER_STATE_OBSERVED,
            effect=attempted,
        )


def test_event_payloads_require_exact_contract_types() -> None:
    value = command()
    binding = grant(value)

    class DerivedEffect(EffectRecord):
        pass

    class DerivedGrant(GrantBinding):
        pass

    derived_effect = DerivedEffect(
        value.effect,
        binding.fingerprint,
        EffectState.ATTEMPTED,
        grant_ref=value.grant_ref,
        command_digest=value.digest,
        canonical_command=value.canonical_bytes,
        attempt_count=1,
    )
    derived_grant = DerivedGrant(
        binding.operation,
        issued_at=binding.issued_at,
        expires_at=binding.expires_at,
    )
    with pytest.raises(TypeError, match="effect"):
        LifecycleEvent(
            EventCode.EFFECT_ATTEMPTED,
            NOW,
            MessageCode.EFFECT_MUTATION_ATTEMPTED,
            effect=derived_effect,
        )
    with pytest.raises(TypeError, match="grant"):
        LifecycleEvent(
            EventCode.AUTHORITY_ACCEPTED,
            NOW,
            MessageCode.AUTHORITY_BOUND,
            grant_binding=derived_grant,
        )


def test_effect_structure_requires_full_command_authority_at_every_state() -> None:
    value = command()
    binding = grant(value)
    for state in EffectState:
        effect = _effect(value, binding, state)
        assert effect.command_digest == value.digest
        assert effect.canonical_command == value.canonical_bytes
        assert effect.attempt_count == (0 if state is EffectState.CLAIMED else 1)
    with pytest.raises((TypeError, ValueError)):
        EffectRecord(value.effect, binding.fingerprint, EffectState.CLAIMED)
    with pytest.raises(ValueError, match="attempt count"):
        replace(_effect(value, binding, EffectState.ATTEMPTED), attempt_count=0)


def test_claim_to_attempt_to_outcome_replaces_one_exact_effect() -> None:
    value = command()
    _, record = ready(value)
    binding = record.grant_binding
    claimed = _effect(value, binding, EffectState.CLAIMED)
    record = apply_event(record, _event(EventCode.EFFECT_CLAIMED, claimed))
    assert record.phase is LifecyclePhase.SUBMITTING
    assert record.effects == (claimed,)

    attempted = replace(claimed, state=EffectState.ATTEMPTED, attempt_count=1)
    record = apply_event(record, _event(EventCode.EFFECT_ATTEMPTED, attempted))
    assert record.effects == (attempted,)

    found = replace(
        attempted,
        state=EffectState.FOUND,
        provider_job_ref="fc-1",
        receipt_digest=D,
    )
    record = apply_event(record, _event(EventCode.EFFECT_FOUND, found))
    assert record.phase is LifecyclePhase.QUEUED
    assert record.effects == (found,)

    with pytest.raises(InvalidTransition):
        apply_event(
            replace(record, phase=LifecyclePhase.SUBMITTING, effects=(attempted,)),
            _event(
                EventCode.EFFECT_FOUND,
                replace(found, command_digest="b" * 64),
            ),
        )


def test_direct_attempt_is_canonical_and_tampering_is_closed_context_free() -> None:
    value = command()
    _, record = ready(value)
    binding = record.grant_binding
    attempted = _effect(value, binding, EffectState.ATTEMPTED)
    record = apply_event(record, _event(EventCode.EFFECT_ATTEMPTED, attempted))
    assert record.effects == (attempted,)

    _, fresh = ready(value)
    tampered = replace(attempted, command_digest="b" * 64)
    with pytest.raises(InvalidTransition) as captured:
        apply_event(fresh, _event(EventCode.EFFECT_ATTEMPTED, tampered))
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


def test_submit_to_cancel_authority_lineage_is_exact() -> None:
    submit = command()
    _, record = ready(submit)
    submit_binding = record.grant_binding
    submitted = _effect(submit, submit_binding, EffectState.ATTEMPTED)
    record = apply_event(record, _event(EventCode.EFFECT_ATTEMPTED, submitted))
    found = replace(
        submitted,
        state=EffectState.FOUND,
        provider_job_ref="fc-1",
        receipt_digest=D,
    )
    record = apply_event(record, _event(EventCode.EFFECT_FOUND, found))

    cancel = command(EffectKind.CANCEL, "fc-1", key="cancel", eid="cancel-effect")
    cancel_binding = grant(cancel)
    record = apply_event(
        record,
        LifecycleEvent(
            EventCode.AUTHORITY_ACCEPTED,
            NOW,
            MessageCode.AUTHORITY_BOUND,
            grant_binding=cancel_binding,
        ),
    )
    cancel_attempt = _effect(cancel, cancel_binding, EffectState.ATTEMPTED)
    record = apply_event(
        record, _event(EventCode.EFFECT_ATTEMPTED, cancel_attempt)
    )
    assert record.phase is LifecyclePhase.CANCELLING
    assert record.effects == (found, cancel_attempt)

    wrong_target = command(
        EffectKind.CANCEL, "foreign-job", key="cancel-2", eid="cancel-effect-2"
    )
    with pytest.raises(InvalidTransition):
        apply_event(
            replace(record, phase=LifecyclePhase.QUEUED, grant_binding=submit_binding),
            LifecycleEvent(
                EventCode.AUTHORITY_ACCEPTED,
                NOW,
                MessageCode.AUTHORITY_BOUND,
                grant_binding=grant(wrong_target),
            ),
        )


def test_public_apply_authenticates_complete_predecessor_history() -> None:
    value, queued, found = _found_submit()
    observation = LifecycleEvent(
        EventCode.PROVIDER_RUNNING,
        NOW,
        MessageCode.PROVIDER_STATE_OBSERVED,
    )
    attempted = replace(
        found,
        state=EffectState.ATTEMPTED,
        provider_job_ref=None,
        receipt_digest=None,
    )
    fabricated = (
        replace(queued, phase=LifecyclePhase.RUNNING),
        replace(queued, effects=()),
        replace(
            ready(value)[1],
            phase=LifecyclePhase.QUEUED,
            effects=(found,),
        ),
        replace(
            queued,
            revision=queued.revision + 1,
            events=queued.events
            + (_event(EventCode.EFFECT_ATTEMPTED, attempted),),
        ),
        replace(
            ready(value)[1],
            revision=1,
            events=(ready(value)[1].events[0],),
        ),
    )
    for predecessor in fabricated:
        with pytest.raises(InvalidTransition) as captured:
            apply_event(predecessor, observation)
        assert captured.value.__cause__ is None
        assert captured.value.__context__ is None


@pytest.mark.parametrize(
    ("code", "state"),
    (
        (EventCode.EFFECT_FOUND, EffectState.FOUND),
        (EventCode.EFFECT_DEFINITELY_ABSENT, EffectState.DEFINITELY_ABSENT),
        (EventCode.EFFECT_INDETERMINATE, EffectState.INDETERMINATE),
    ),
)
def test_every_outcome_rejects_a_substituted_active_authority(
    code: EventCode, state: EffectState
) -> None:
    value = command()
    _, record = ready(value)
    attempted = _effect(value, record.grant_binding, EffectState.ATTEMPTED)
    record = apply_event(record, _event(EventCode.EFFECT_ATTEMPTED, attempted))
    replacement = replace(
        attempted,
        state=state,
        provider_job_ref="fc-1" if state is EffectState.FOUND else None,
        receipt_digest=D if state is EffectState.FOUND else None,
    )
    other = command(nonce="other", key="other", eid="other-effect")
    substituted = replace(record, grant_binding=grant(other))
    with pytest.raises(InvalidTransition):
        apply_event(substituted, _event(code, replacement))


def test_authorization_rejection_is_only_a_pre_authority_transition() -> None:
    value = command()
    _, authorized = ready(value)
    rejection = LifecycleEvent(
        EventCode.AUTHORIZATION_REJECTED,
        NOW,
        MessageCode.AUTHORIZATION_MISMATCH,
    )
    planned = lifecycle.initial_record(project_ref="p", run_id="r", occurred_at=NOW)
    assert apply_event(planned, rejection).phase is LifecyclePhase.FAILED
    with pytest.raises(InvalidTransition):
        apply_event(authorized, rejection)


def test_provider_unknown_accepts_only_confirmed_observation_phases() -> None:
    value, queued, _ = _found_submit()
    unknown = LifecycleEvent(
        EventCode.PROVIDER_UNKNOWN,
        NOW,
        MessageCode.PROVIDER_STATE_OBSERVED,
    )
    assert apply_event(queued, unknown).phase is LifecyclePhase.RECONCILE_REQUIRED

    running = apply_event(
        queued,
        LifecycleEvent(
            EventCode.PROVIDER_RUNNING,
            NOW,
            MessageCode.PROVIDER_STATE_OBSERVED,
        ),
    )
    assert apply_event(running, unknown).phase is LifecyclePhase.RECONCILE_REQUIRED
    verifying = apply_event(
        running,
        LifecycleEvent(
            EventCode.PROVIDER_SUCCEEDED,
            NOW,
            MessageCode.SEMANTIC_VERIFICATION_PENDING,
        ),
    )
    assert apply_event(verifying, unknown).phase is LifecyclePhase.RECONCILE_REQUIRED

    cancel = command(EffectKind.CANCEL, "fc-1", key="cancel", eid="cancel-effect")
    cancelling = apply_event(
        queued,
        LifecycleEvent(
            EventCode.AUTHORITY_ACCEPTED,
            NOW,
            MessageCode.AUTHORITY_BOUND,
            grant_binding=grant(cancel),
        ),
    )
    cancel_attempt = _effect(
        cancel, cancelling.grant_binding, EffectState.ATTEMPTED
    )
    cancelling = apply_event(
        cancelling, _event(EventCode.EFFECT_ATTEMPTED, cancel_attempt)
    )
    assert apply_event(cancelling, unknown).phase is LifecyclePhase.RECONCILE_REQUIRED

    planned = lifecycle.initial_record(project_ref="p", run_id="r", occurred_at=NOW)
    with pytest.raises(InvalidTransition):
        apply_event(planned, unknown)


def test_generic_lifecycle_has_no_modal_or_provider_dependency() -> None:
    source = inspect.getsource(lifecycle)
    assert "Modal" not in source
    assert ".providers" not in source


class _HostileDict(dict):
    def __iter__(self):
        raise RuntimeError("raw-secret")


class _HostileList(list):
    def __iter__(self):
        raise RuntimeError("raw-secret")


class _TextSubclass(str):
    pass


class _HostileKey(str):
    armed = False

    def __hash__(self):
        if self.armed:
            raise RuntimeError("raw-secret")
        return str.__hash__(self)


class _BytesSubclass(bytes):
    pass


def _assert_closed(error: BaseException) -> None:
    assert error.__cause__ is None
    assert error.__context__ is None
    assert "raw-secret" not in str(error)


def test_lifecycle_parsers_reject_proxy_and_subclass_containers_closed() -> None:
    _, record, _ = _found_submit()
    record_doc = record.to_dict()
    event_doc = record.events[-1].to_dict()
    effect_doc = record.effects[-1].to_dict()
    grant_doc = record.grant_binding.to_dict()
    hostile_values = (
        (LifecycleRecord.from_dict, MappingProxyType(record_doc)),
        (LifecycleRecord.from_dict, _HostileDict(record_doc)),
        (LifecycleEvent.from_dict, MappingProxyType(event_doc)),
        (LifecycleEvent.from_dict, _HostileDict(event_doc)),
        (EffectRecord.from_dict, MappingProxyType(effect_doc)),
        (EffectRecord.from_dict, _HostileDict(effect_doc)),
        (GrantBinding.from_dict, MappingProxyType(grant_doc)),
        (GrantBinding.from_dict, _HostileDict(grant_doc)),
    )
    for parser, value in hostile_values:
        with pytest.raises((TypeError, ValueError)) as captured:
            parser(value)
        _assert_closed(captured.value)

    list_doc = record.to_dict()
    list_doc["events"] = _HostileList(list_doc["events"])
    with pytest.raises(ValueError) as captured:
        LifecycleRecord.from_dict(list_doc)
    _assert_closed(captured.value)

    text_doc = record.to_dict()
    text_doc["schema_version"] = _TextSubclass(
        "synaptic-lifecycle-record/v1"
    )
    with pytest.raises(ValueError) as captured:
        LifecycleRecord.from_dict(text_doc)
    _assert_closed(captured.value)

    hostile_key = _HostileKey("schema_version")
    keyed_doc = {hostile_key: "synaptic-lifecycle-record/v1"}
    _HostileKey.armed = True
    try:
        with pytest.raises(ValueError) as captured:
            LifecycleRecord.from_dict(keyed_doc)
        _assert_closed(captured.value)
    finally:
        _HostileKey.armed = False


def test_canonical_byte_parsers_close_duplicates_nonfinite_and_subclasses() -> None:
    value = command()
    _, record = ready(value)
    cases = (
        (LifecycleRecord.from_canonical_bytes, _BytesSubclass(record.canonical_bytes)),
        (MutationCommandV1.from_bytes, _BytesSubclass(value.canonical_bytes)),
        (LifecycleRecord.from_canonical_bytes, b'{"a":1,"a":2}'),
        (MutationCommandV1.from_bytes, b'{"a":NaN}'),
        (LifecycleRecord.from_canonical_bytes, b'{"a":Infinity}'),
    )
    for parser, raw in cases:
        with pytest.raises((TypeError, ValueError)) as captured:
            parser(raw)
        _assert_closed(captured.value)

    assert LifecycleRecord.from_canonical_bytes(record.canonical_bytes) == record
    assert MutationCommandV1.from_bytes(value.canonical_bytes) == value


def test_lifecycle_parser_preserves_exact_integer_not_boolean_semantics() -> None:
    value, record, _ = _found_submit()
    document = record.to_dict()
    document["revision"] = True
    with pytest.raises(ValueError):
        LifecycleRecord.from_dict(document)
    effect = record.effects[0].to_dict()
    effect["attempt_count"] = True
    with pytest.raises(ValueError):
        EffectRecord.from_dict(effect)
    command_document = json.loads(value.canonical_bytes)
    command_document["operation_binding"]["stage_target"]["generation"] = True
    raw = json.dumps(command_document, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError):
        MutationCommandV1.from_bytes(raw)
