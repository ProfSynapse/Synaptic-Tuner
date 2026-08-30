from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from synaptic_tuner.api.v1.execution import ExecutionGrant
from tests.execution.providers.test_modal_sdk154_adapter import FakeCall, FakeFunction
from tests.execution.providers.test_modal_training_operations import Auth, NOW, operations
from tuner.execution.contracts import (
    ExecutionScope,
    EffectState,
    EffectIdentity,
    EffectKind,
    EffectRecord,
    EventCode,
    GrantBinding,
    LifecycleEvent,
    LifecycleRecord,
    LifecyclePhase,
    MessageCode,
    RunAlreadyExists,
)
from tuner.execution.providers.modal.training import (
    ModalDurablePreparationV1,
    ModalPreparedRunV1,
    ModalTrainingRepository,
    _prepared_record_prefix,
)
from tuner.execution.providers.modal.staging import StageMaterialV1, prepare_modal_stage
from tuner.execution.providers.modal.control import StageExpectationV1
from tuner.execution.lifecycle import apply_event
from tuner.execution.broker import MutationCommandV1
from tuner.execution.providers.modal.contracts import operation_path


@pytest.fixture(autouse=True)
def _reset_provider_fakes() -> None:
    FakeFunction.calls = []
    FakeFunction.spawn_calls = []
    FakeFunction.fail = False
    FakeCall.result = TimeoutError()


def _captured_pair(tmp_path, monkeypatch: pytest.MonkeyPatch) -> ModalPreparedRunV1:
    value, repository, plan = operations(tmp_path)
    captured: list[ModalPreparedRunV1] = []

    def interrupt(presented: ModalPreparedRunV1) -> None:
        captured.append(presented)
        raise RuntimeError("simulated pre-admission interruption")

    monkeypatch.setattr(repository, "create_modal_prepared_run", interrupt)
    with pytest.raises(RuntimeError, match="host Modal start failed"):
        value.start(plan, value.preflight(plan), ExecutionGrant("grant-run-1"))
    assert len(captured) == 1
    return captured[0]


def test_prepared_run_is_exact_revision_four_ready_pair(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = _captured_pair(tmp_path, monkeypatch)
    assert type(pair) is ModalPreparedRunV1
    assert pair.record.revision == 4
    assert pair.record.phase is LifecyclePhase.READY
    assert pair.record.grant_binding.operation == pair.preparation.operation
    assert len({event.occurred_at for event in pair.record.events}) == 1

    with pytest.raises(ValueError, match="identities disagree"):
        ModalPreparedRunV1(
            replace(pair.record, phase=LifecyclePhase.PREPARING),
            pair.preparation,
        )


def test_interruption_before_atomic_callback_leaves_neither_value(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, repository, plan = operations(tmp_path)

    def interrupt(_presented: ModalPreparedRunV1) -> None:
        raise RuntimeError("before atomic admission")

    monkeypatch.setattr(repository, "create_modal_prepared_run", interrupt)
    with pytest.raises(RuntimeError, match="host Modal start failed"):
        value.start(plan, value.preflight(plan), ExecutionGrant("grant-run-1"))
    assert repository.load("project-1", "run-1") is None
    assert repository.load_modal_preparation("project-1", "run-1") is None
    assert FakeFunction.spawn_calls == []


def test_interruption_after_atomic_callback_leaves_both_values(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, repository, plan = operations(tmp_path)
    original = repository.create_modal_prepared_run

    def interrupt(presented: ModalPreparedRunV1) -> None:
        original(presented)
        raise RuntimeError("after atomic admission")

    monkeypatch.setattr(repository, "create_modal_prepared_run", interrupt)
    with pytest.raises(RuntimeError, match="host Modal start failed"):
        value.start(plan, value.preflight(plan), ExecutionGrant("grant-run-1"))
    record = repository.load("project-1", "run-1")
    preparation = repository.load_modal_preparation("project-1", "run-1")
    assert record is not None and preparation is not None
    assert ModalPreparedRunV1(record, preparation)
    assert FakeFunction.spawn_calls == []


def test_callback_return_claim_is_ignored_and_exact_readback_controls(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, repository, plan = operations(tmp_path)
    original = repository.create_modal_prepared_run

    def claiming_callback(presented: ModalPreparedRunV1):
        original(presented)
        return object()

    monkeypatch.setattr(repository, "create_modal_prepared_run", claiming_callback)
    value.start(plan, value.preflight(plan), ExecutionGrant("grant-run-1"))
    assert len(FakeFunction.spawn_calls) == 1


def test_callback_mutation_is_rejected_before_provider_mutation(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, repository, plan = operations(tmp_path)

    def poisoned(presented: ModalPreparedRunV1) -> None:
        object.__setattr__(
            presented.preparation, "public_plan_fingerprint", "f" * 64
        )
        key = (presented.record.project_ref, presented.record.run_id)
        repository.records[key] = presented.record
        repository.preparations[key] = presented.preparation

    monkeypatch.setattr(repository, "create_modal_prepared_run", poisoned)
    with pytest.raises(RuntimeError, match="readback failed"):
        value.start(plan, value.preflight(plan), ExecutionGrant("grant-run-1"))
    assert FakeFunction.spawn_calls == []


def test_exact_run_already_exists_subclass_is_not_convergence(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, repository, plan = operations(tmp_path)

    class DerivedAlreadyExists(RunAlreadyExists):
        pass

    def collide(_presented: ModalPreparedRunV1) -> None:
        raise DerivedAlreadyExists("not an exact convergence signal")

    monkeypatch.setattr(repository, "create_modal_prepared_run", collide)
    with pytest.raises(RuntimeError, match="host Modal start failed") as captured:
        value.start(plan, value.preflight(plan), ExecutionGrant("grant-run-1"))
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert FakeFunction.spawn_calls == []


def test_exact_run_already_exists_with_conflicting_pair_fails_closed(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, repository, plan = operations(tmp_path)

    def conflict(presented: ModalPreparedRunV1) -> None:
        changed = ModalDurablePreparationV1(
            "f" * 64,
            presented.preparation.context,
            presented.preparation.operation,
            presented.preparation.stage,
        )
        key = (presented.record.project_ref, presented.record.run_id)
        repository.records[key] = presented.record
        repository.preparations[key] = changed
        raise RunAlreadyExists("conflicting prepared run")

    monkeypatch.setattr(repository, "create_modal_prepared_run", conflict)
    with pytest.raises(RuntimeError, match="host Modal start failed"):
        value.start(plan, value.preflight(plan), ExecutionGrant("grant-run-1"))
    assert FakeFunction.spawn_calls == []


def test_concurrent_exact_replay_converges_to_one_provider_spawn(tmp_path) -> None:
    value, repository, plan = operations(tmp_path)
    preflight = value.preflight(plan)
    original = repository.create_modal_prepared_run
    lock = threading.Lock()

    def atomic_create(presented: ModalPreparedRunV1) -> None:
        with lock:
            original(presented)

    repository.create_modal_prepared_run = atomic_create

    def start_once(_index: int):
        return value.start(plan, preflight, ExecutionGrant("grant-run-1"))

    with ThreadPoolExecutor(max_workers=2) as pool:
        submissions = tuple(pool.map(start_once, range(2)))
    assert submissions[0] == submissions[1]
    assert len(FakeFunction.spawn_calls) == 1


def test_old_split_commit_protocol_is_absent() -> None:
    assert not hasattr(ModalTrainingRepository, "commit_modal_preparation")
    assert hasattr(ModalTrainingRepository, "create_modal_prepared_run")


def test_prepared_pair_rejects_incomplete_record_adjacency(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = _captured_pair(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="identities disagree"):
        ModalPreparedRunV1(
            replace(pair.record, message_code=MessageCode.PREPARING),
            pair.preparation,
        )


def test_prepared_pair_requires_authority_event_without_effect(tmp_path) -> None:
    value, repository, plan = operations(tmp_path)
    value.start(plan, value.preflight(plan), ExecutionGrant("grant-run-1"))
    advanced = repository.load("project-1", "run-1")
    prefix = _prepared_record_prefix(advanced)
    with pytest.raises(ValueError, match="payload"):
        replace(prefix.events[1], effect=advanced.effects[0])


def _pair_with_operation(
    pair: ModalPreparedRunV1, operation, *, defer_scope_validation: bool = False,
) -> tuple[object, ModalDurablePreparationV1]:
    if defer_scope_validation:
        original = pair.preparation.stage.expectation
        expectation = object.__new__(StageExpectationV1)
        for name, value in (
            ("operation", operation),
            ("binding", original.binding),
            ("claim_digest", original.claim_digest),
            ("bundle_digest", original.bundle_digest),
            ("bundle_size", original.bundle_size),
        ):
            object.__setattr__(expectation, name, value)
        stage = StageMaterialV1(
            expectation,
            pair.preparation.stage.bundle,
            pair.preparation.stage.claim,
            pair.preparation.stage.claim_tag,
        )
    else:
        stage = prepare_modal_stage(
            operation,
            pair.preparation.context.binding,
            pair.preparation.stage.bundle,
            Auth(),
        )
    preparation = replace(pair.preparation, operation=operation, stage=stage)
    binding = replace(pair.record.grant_binding, operation=operation)
    authority = replace(pair.record.events[1], grant_binding=binding)
    record = replace(
        pair.record,
        events=(
            pair.record.events[0],
            authority,
            pair.record.events[2],
            pair.record.events[3],
        ),
        grant_binding=binding,
    )
    return record, preparation


@pytest.mark.parametrize(
    "field,value",
    (
        ("deployment_attestation_digest", "a" * 64),
        ("resource_digest", "b" * 64),
        ("quote_digest", "c" * 64),
    ),
)
def test_prepared_pair_rejects_operation_context_digest_disagreement(
    tmp_path, monkeypatch: pytest.MonkeyPatch, field: str, value: str,
) -> None:
    pair = _captured_pair(tmp_path, monkeypatch)
    operation = replace(pair.preparation.operation, **{field: value})
    record, preparation = _pair_with_operation(pair, operation)
    with pytest.raises(ValueError, match="identities disagree"):
        ModalPreparedRunV1(record, preparation)


def test_prepared_pair_owns_cross_operation_effect_adjacency_rejection(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = _captured_pair(tmp_path, monkeypatch)
    effect = replace(
        pair.preparation.operation.effect,
        effect_key="other-run",
    )
    operation = replace(pair.preparation.operation, effect=effect)
    record, preparation = _pair_with_operation(pair, operation)
    with pytest.raises(ValueError, match="identities disagree"):
        ModalPreparedRunV1(record, preparation)


@pytest.mark.parametrize(
    "scope",
    (
        ExecutionScope("docker", "acct", "env"),
        ExecutionScope("modal", "other-account", "env"),
        ExecutionScope("modal", "acct", "other-environment"),
    ),
)
def test_prepared_pair_owns_provider_account_namespace_rejection(
    tmp_path, monkeypatch: pytest.MonkeyPatch, scope,
) -> None:
    pair = _captured_pair(tmp_path, monkeypatch)
    effect = replace(pair.preparation.operation.effect, scope=scope)
    operation = replace(pair.preparation.operation, effect=effect)
    record, preparation = _pair_with_operation(
        pair, operation, defer_scope_validation=True
    )
    with pytest.raises(ValueError, match="not canonical"):
        ModalPreparedRunV1(record, preparation)


def test_later_clock_exact_replay_uses_persisted_winner_and_short_circuits(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, repository, plan = operations(tmp_path)
    preflight = value.preflight(plan)
    first = value.start(plan, preflight, ExecutionGrant("grant-run-1"))
    calls = len(FakeFunction.spawn_calls)
    object.__setattr__(value._ports, "clock", lambda: "2026-08-25T12:04:00Z")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("advanced durable lifecycle must short-circuit")

    monkeypatch.setattr(value._stage_control, "validate", forbidden)
    monkeypatch.setattr(repository, "compare_and_consume_attempt", forbidden)
    second = value.start(plan, preflight, ExecutionGrant("grant-run-1"))
    assert second == first
    assert repository.load("project-1", "run-1").phase is LifecyclePhase.QUEUED
    assert len(FakeFunction.spawn_calls) == calls


@pytest.mark.parametrize(
    "phase,message",
    (
        (LifecyclePhase.CANCELLING, MessageCode.PROVIDER_STATE_OBSERVED),
        (LifecyclePhase.CANCELLED, MessageCode.PROVIDER_STATE_OBSERVED),
        (LifecyclePhase.SUCCEEDED, MessageCode.SEMANTIC_VERIFICATION_PASSED),
    ),
)
def test_phase_only_rebound_authority_is_rejected_without_effects(
    tmp_path, monkeypatch: pytest.MonkeyPatch, phase, message,
) -> None:
    value, repository, plan = operations(tmp_path)
    preflight = value.preflight(plan)
    value.start(plan, preflight, ExecutionGrant("grant-run-1"))
    key = ("project-1", "run-1")
    repository.records[key] = replace(
        repository.records[key],
        phase=phase,
        message_code=message,
        grant_binding=None,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("advanced lineage must not re-enter effects")

    monkeypatch.setattr(value._stage_control, "validate", forbidden)
    monkeypatch.setattr(repository, "compare_and_consume_attempt", forbidden)
    calls = len(FakeFunction.spawn_calls)
    with pytest.raises(RuntimeError, match="prepared-run readback failed"):
        value.start(plan, preflight, ExecutionGrant("grant-run-1"))
    assert len(FakeFunction.spawn_calls) == calls


def test_state_only_reconcile_and_unrecorded_effect_are_rejected(tmp_path) -> None:
    value, repository, plan = operations(tmp_path)
    preflight = value.preflight(plan)
    value.start(plan, preflight, ExecutionGrant("grant-run-1"))
    key = ("project-1", "run-1")
    record = repository.records[key]
    reconciled = replace(
        record.effects[0],
        state=EffectState.INDETERMINATE,
        provider_job_ref=None,
        receipt_digest=None,
    )
    foreign_identity = replace(
        record.effects[0].identity,
        effect_id="unrecorded-submit-effect",
    )
    foreign = replace(record.effects[0], identity=foreign_identity)
    repository.records[key] = replace(
        record,
        phase=LifecyclePhase.RECONCILE_REQUIRED,
        message_code=MessageCode.EFFECT_OUTCOME_UNKNOWN,
        effects=(reconciled, foreign),
    )
    calls = len(FakeFunction.spawn_calls)
    with pytest.raises(RuntimeError, match="prepared-run readback failed"):
        value.start(plan, preflight, ExecutionGrant("grant-run-1"))
    assert len(FakeFunction.spawn_calls) == calls


@pytest.mark.parametrize("field", ("command_digest", "canonical_command"))
def test_advanced_submit_effect_requires_exact_persisted_command(
    tmp_path, field: str,
) -> None:
    value, repository, plan = operations(tmp_path)
    preflight = value.preflight(plan)
    value.start(plan, preflight, ExecutionGrant("grant-run-1"))
    key = ("project-1", "run-1")
    record = repository.records[key]
    changed = (
        "d" * 64 if field == "command_digest" else b'{"wrong":"command"}'
    )
    repository.records[key] = replace(
        record,
        effects=(replace(record.effects[0], **{field: changed}),),
    )
    calls = len(FakeFunction.spawn_calls)
    with pytest.raises(RuntimeError, match="prepared-run readback failed"):
        value.start(plan, preflight, ExecutionGrant("grant-run-1"))
    assert len(FakeFunction.spawn_calls) == calls


def _append_real_cancel_authority(record, preparation):
    submit = preparation.operation
    cancel_effect = EffectIdentity(
        "cancel-effect-run-1",
        submit.effect.effect_key,
        EffectKind.CANCEL,
        submit.effect.scope,
    )
    cancel_target = replace(
        submit.stage_target,
        output_prefix=operation_path(cancel_effect.effect_id, "output"),
    )
    cancel_operation = replace(
        submit,
        effect=cancel_effect,
        grant_ref="cancel-grant-run-1",
        stage_target=cancel_target,
        target_provider_job_ref="fc-1",
    )
    cancel_binding = GrantBinding.from_operation(
        cancel_operation,
        issued_at="2026-08-25T12:03:00Z",
        expires_at="2026-08-25T12:10:00Z",
    )
    authorized = apply_event(
        record,
        LifecycleEvent(
            EventCode.AUTHORITY_ACCEPTED,
            "2026-08-25T12:04:00Z",
            MessageCode.AUTHORITY_BOUND,
            grant_binding=cancel_binding,
        ),
    )
    cancel_command = MutationCommandV1(
        cancel_operation,
        "a" * 64,
        "b" * 64,
    )
    cancel_attempt = EffectRecord(
        cancel_effect,
        cancel_binding.fingerprint,
        EffectState.ATTEMPTED,
        grant_ref=cancel_binding.grant_ref,
        command_digest=cancel_command.digest,
        canonical_command=cancel_command.canonical_bytes,
        attempt_count=1,
    )
    return apply_event(
        authorized,
        LifecycleEvent(
            EventCode.EFFECT_ATTEMPTED,
            "2026-08-25T12:05:00Z",
            MessageCode.EFFECT_MUTATION_ATTEMPTED,
            effect=cancel_attempt,
        ),
    )


def test_real_cancel_authority_preserves_submit_and_never_respawns(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, repository, plan = operations(tmp_path)
    preflight = value.preflight(plan)
    first = value.start(plan, preflight, ExecutionGrant("grant-run-1"))
    key = ("project-1", "run-1")
    preparation = repository.preparations[key]
    repository.records[key] = _append_real_cancel_authority(
        repository.records[key], preparation
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("cancel authority replay must not re-enter effects")

    monkeypatch.setattr(value._stage_control, "validate", forbidden)
    monkeypatch.setattr(repository, "compare_and_consume_attempt", forbidden)
    calls = len(FakeFunction.spawn_calls)
    second = value.start(plan, preflight, ExecutionGrant("grant-run-1"))
    assert second == first
    assert repository.records[key].phase is LifecyclePhase.CANCELLING
    assert repository.records[key].grant_binding.operation.effect.kind is EffectKind.CANCEL
    assert len(FakeFunction.spawn_calls) == calls


def test_cancel_authority_with_corrupt_submit_command_rejects_without_respawn(
    tmp_path,
) -> None:
    value, repository, plan = operations(tmp_path)
    preflight = value.preflight(plan)
    value.start(plan, preflight, ExecutionGrant("grant-run-1"))
    key = ("project-1", "run-1")
    preparation = repository.preparations[key]
    record = _append_real_cancel_authority(repository.records[key], preparation)
    repository.records[key] = replace(
        record,
        effects=(replace(record.effects[0], command_digest="e" * 64),),
    )
    calls = len(FakeFunction.spawn_calls)
    with pytest.raises(RuntimeError, match="prepared-run readback failed"):
        value.start(plan, preflight, ExecutionGrant("grant-run-1"))
    assert len(FakeFunction.spawn_calls) == calls


def test_clock_mutation_of_plan_fails_before_repository_or_provider(
    tmp_path,
) -> None:
    value, repository, plan = operations(tmp_path)
    preflight = value.preflight(plan)

    def mutating_clock() -> str:
        object.__setattr__(plan, "workload", plan.execution_context)
        return NOW

    object.__setattr__(value._ports, "clock", mutating_clock)
    with pytest.raises(RuntimeError, match="host Modal start failed"):
        value.start(plan, preflight, ExecutionGrant("grant-run-1"))
    assert repository.load("project-1", "run-1") is None
    assert FakeFunction.spawn_calls == []


def test_grant_callback_mutation_of_preflight_fails_before_durable_create(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, repository, plan = operations(tmp_path)
    preflight = value.preflight(plan)
    original = value._ports.grants.bind

    def mutating_bind(*args, **kwargs):
        result = original(*args, **kwargs)
        object.__setattr__(preflight, "plan_fingerprint", "e" * 64)
        return result

    monkeypatch.setattr(value._ports.grants, "bind", mutating_bind)
    with pytest.raises(RuntimeError, match="host Modal start failed"):
        value.start(plan, preflight, ExecutionGrant("grant-run-1"))
    assert repository.load("project-1", "run-1") is None
    assert FakeFunction.spawn_calls == []


def test_readback_callback_mutation_of_plan_fails_before_provider(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, repository, plan = operations(tmp_path)
    original = repository.load

    def mutating_load(*args, **kwargs):
        result = original(*args, **kwargs)
        object.__setattr__(plan, "resolved_config", plan.execution_context)
        return result

    monkeypatch.setattr(repository, "load", mutating_load)
    with pytest.raises(RuntimeError, match="prepared-run readback failed"):
        value.start(plan, value.preflight(plan), ExecutionGrant("grant-run-1"))
    assert FakeFunction.spawn_calls == []


def test_collaborator_replacement_during_clock_fails_closed(
    tmp_path,
) -> None:
    value, repository, plan = operations(tmp_path)
    preflight = value.preflight(plan)

    def replacing_clock() -> str:
        object.__setattr__(value, "_repository", object())
        return NOW

    object.__setattr__(value._ports, "clock", replacing_clock)
    with pytest.raises(RuntimeError, match="host Modal start failed"):
        value.start(plan, preflight, ExecutionGrant("grant-run-1"))
    assert repository.load("project-1", "run-1") is None
    assert FakeFunction.spawn_calls == []


def test_host_ports_field_replacement_during_clock_fails_before_durability(
    tmp_path,
) -> None:
    value, repository, plan = operations(tmp_path)
    preflight = value.preflight(plan)

    def replacing_clock() -> str:
        object.__setattr__(value._ports, "lifecycle", object())
        return NOW

    object.__setattr__(value._ports, "clock", replacing_clock)
    with pytest.raises(RuntimeError, match="host Modal start failed"):
        value.start(plan, preflight, ExecutionGrant("grant-run-1"))
    assert repository.load("project-1", "run-1") is None
    assert FakeFunction.spawn_calls == []


def test_outcome_callback_replacement_after_admission_stops_before_provider(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, repository, plan = operations(tmp_path)
    original = repository.compare_and_consume_attempt

    def replacing_admission(*args, **kwargs):
        result = original(*args, **kwargs)
        repository.record_attempt_outcome = lambda *_args, **_kwargs: None
        return result

    monkeypatch.setattr(repository, "compare_and_consume_attempt", replacing_admission)
    with pytest.raises(RuntimeError, match="host Modal start failed"):
        value.start(plan, value.preflight(plan), ExecutionGrant("grant-run-1"))
    record = repository.load("project-1", "run-1")
    assert record.effects[0].state is EffectState.ATTEMPTED
    assert FakeFunction.spawn_calls == []


def test_stage_expectation_callback_replacement_stops_before_provider(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, repository, plan = operations(tmp_path)
    original = repository.load_modal_preparation_by_effect

    def replacing_lookup(effect_id):
        result = original(effect_id)
        repository.load_modal_preparation_by_effect = lambda _effect_id: result
        return result

    monkeypatch.setattr(
        repository, "load_modal_preparation_by_effect", replacing_lookup
    )
    with pytest.raises(RuntimeError, match="host Modal start failed"):
        value.start(plan, value.preflight(plan), ExecutionGrant("grant-run-1"))
    assert FakeFunction.spawn_calls == []


def test_outcome_failure_preserves_attempted_durable_state_and_is_redacted(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, repository, plan = operations(tmp_path)

    def secret_failure(*_args, **_kwargs):
        raise RuntimeError("PRIVATE-OUTCOME-DETAIL")

    monkeypatch.setattr(repository, "record_attempt_outcome", secret_failure)
    with pytest.raises(RuntimeError, match="host Modal start failed") as captured:
        value.start(plan, value.preflight(plan), ExecutionGrant("grant-run-1"))
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert "PRIVATE" not in str(captured.value)
    record = repository.load("project-1", "run-1")
    assert record.effects[0].state is EffectState.ATTEMPTED
    assert len(FakeFunction.spawn_calls) == 1


def test_public_submission_uses_validated_durable_run_plan_and_effect_time(
    tmp_path,
) -> None:
    value, repository, plan = operations(tmp_path)
    submission = value.start(
        plan, value.preflight(plan), ExecutionGrant("grant-run-1")
    )
    preparation = repository.load_modal_preparation("project-1", "run-1")
    record = repository.load("project-1", "run-1")
    effect_id = preparation.operation.effect.effect_id
    event = next(
        item
        for item in record.events
        if item.effect is not None and item.effect.identity.effect_id == effect_id
    )
    assert submission.run.project_ref == record.project_ref == "project-1"
    assert submission.run.run_id == record.run_id == "run-1"
    assert submission.plan_fingerprint == preparation.public_plan_fingerprint
    assert submission.submitted_at == event.occurred_at


@pytest.mark.parametrize("mode", ("error", "foreign", "subclass", "wrong_effect"))
def test_final_lifecycle_readback_is_exact_bound_and_closed(
    tmp_path, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    value, repository, plan = operations(tmp_path)
    original = repository.load
    load_count = 0

    class DerivedRecord(LifecycleRecord):
        pass

    def hostile_load(project_ref, run_id):
        nonlocal load_count
        load_count += 1
        current = original(project_ref, run_id)
        if load_count == 1:
            return current
        if mode == "error":
            raise RuntimeError("PRIVATE-FINAL-LOAD")
        if mode == "foreign":
            return replace(current, project_ref="foreign-project")
        if mode == "subclass":
            return DerivedRecord(
                current.run_id,
                current.project_ref,
                current.revision,
                current.phase,
                current.verification,
                current.updated_at,
                current.message_code,
                current.events,
                current.effects,
                current.grant_binding,
            )
        changed = replace(
            current.effects[0].identity,
            effect_id="foreign-effect",
        )
        return replace(
            current,
            effects=(replace(current.effects[0], identity=changed),),
        )

    monkeypatch.setattr(repository, "load", hostile_load)
    with pytest.raises(RuntimeError) as captured:
        value.start(plan, value.preflight(plan), ExecutionGrant("grant-run-1"))
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert "PRIVATE" not in str(captured.value)
    assert len(FakeFunction.spawn_calls) == 1
    assert repository.records[("project-1", "run-1")].effects[0].state is EffectState.FOUND


@pytest.mark.parametrize("method", ("create_modal_prepared_run", "load", "load_modal_preparation"))
def test_repository_errors_are_closed_without_context(
    tmp_path, monkeypatch: pytest.MonkeyPatch, method: str,
) -> None:
    value, repository, plan = operations(tmp_path)

    def secret_failure(*_args, **_kwargs):
        raise RuntimeError("PRIVATE-CALLBACK-DETAIL")

    monkeypatch.setattr(repository, method, secret_failure)
    with pytest.raises(RuntimeError) as captured:
        value.start(plan, value.preflight(plan), ExecutionGrant("grant-run-1"))
    assert "PRIVATE" not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert FakeFunction.spawn_calls == []
