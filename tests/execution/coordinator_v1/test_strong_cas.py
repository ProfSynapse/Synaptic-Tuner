from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import pytest

from synaptic_tuner.api.v1.results import TrainingRunRef

from tuner.execution.coordinator_v1.coordinator import (
    ApplyStageEffectTransitionV1,
    ApplySubmitEffectTransitionV1,
    BeginPreparationTransitionV1,
    ExecutionGrantSlotV1,
    ReconciliationGrantSlotV1,
    RecordStageIntentTransitionV1,
    RecordSubmitIntentTransitionV1,
)
from tuner.execution.coordinator_v1.state_machine import (
    apply_stage_effect_record,
    apply_submit_effect_record,
    begin_preparation,
    record_stage_intent,
    record_submit_intent,
)
from tuner.execution.coordinator_v1.model import WorkflowPhaseV1
from tuner.execution.coordinator_v1.stores import (
    CoordinatorStoreCode,
    CoordinatorStoreError,
    InMemoryExecutionGrantStoreV1,
    InMemoryPreparationStoreV1,
    InMemoryReconciliationGrantStoreV1,
    InMemoryWorkflowStoreV1,
)
from tuner.execution.foundation_v2.authority import (
    GrantAuthorityV2,
    ReconciliationGrantContentV1,
)
from tuner.execution.foundation_v2.canonical import canonical_bytes, domain_digest
from tuner.execution.foundation_v2.executors import AdapterDescriptorV1
from tuner.execution.foundation_v2.preparation import CanonicalPreparationV2
from tuner.execution.foundation_v2.repository import (
    EffectState,
    ReconciliationGrantBindingV2,
    ReconciliationOwnershipV2,
)

from .test_state_machine import (
    AssessmentAuth,
    Auth,
    D,
    STAGE_REF,
    PROVIDER_RUN,
    assessment,
    intent,
    planned,
    prep,
    record,
)


def workflow_store():
    return InMemoryWorkflowStoreV1(Auth(), AssessmentAuth())


def execution_grant(raw_intent=None, *, grant_ref="execution-grant"):
    raw_intent = raw_intent or intent("stage")
    authority = GrantAuthorityV2("grant-authority", b"g" * 32)
    grant = authority.issue(
        raw_intent.canonical_command_bytes,
        grant_ref=grant_ref,
        policy_digest=D[10],
        requirement_digest=D[11],
        not_before_epoch=1,
        expires_at_epoch=100,
    )
    slot = ExecutionGrantSlotV1(
        grant.content.effect_id,
        grant.content.command_digest,
        domain_digest(
            "synaptic-foundation-command-bytes/v1", raw_intent.canonical_command_bytes
        ),
    )
    return slot, grant, authority, raw_intent.canonical_command_bytes


def reconciliation_grant(
    *, owner="owner-a", generation=1, epoch=1, grant_ref="recon-a",
    authority=None, raw_intent=None, prior_claim=None, predecessor_grant=None,
):
    raw_intent = raw_intent or intent("stage")
    command = raw_intent.canonical_command_bytes
    from tuner.execution.foundation_v2.commands import parse_exact_command

    parsed = parse_exact_command(command)
    preparation = parsed.preparation
    effect = parsed.operation.effect
    adapter = AdapterDescriptorV1(
        preparation.provider.provider_id, "adapter-a", "1.0.0"
    )
    authority = authority or GrantAuthorityV2("grant-authority", b"r" * 32)
    content = ReconciliationGrantContentV1(
        grant_ref,
        parsed.digest,
        effect.effect_id,
        preparation.preparation_digest,
        adapter.digest,
        preparation.provider.provider_id,
        preparation.provider.profile_ref,
        preparation.scope.account_ref,
        preparation.scope.namespace_ref,
        owner,
        generation,
        epoch,
        D[9],
        D[10],
        1,
        100,
        authority.epoch,
        authority.revocation_generation,
    )
    grant = authority.issue_reconciliation(content)
    slot = ReconciliationGrantSlotV1(
        content.effect_id,
        content.command_digest,
        domain_digest("synaptic-foundation-command-bytes/v1", command),
        generation,
        epoch,
        prior_claim,
        predecessor_grant,
    )
    return slot, grant, authority, command


def test_workflow_cas_accepts_reconstructed_expected_and_exact_named_successor():
    store = workflow_store()
    source = planned()
    assert store.create(source)
    reconstructed = replace(source)
    successor = begin_preparation(source)
    assert store.compare_and_swap(
        reconstructed,
        successor,
        transition=BeginPreparationTransitionV1(),
    )
    assert store.get(source.run) == successor
    assert store.get_by_plan(source.run.project_ref, source.plan_fingerprint) == successor
    assert store.list(source.run.project_ref) == (successor,)


def test_workflow_cas_stale_is_false_and_same_revision_fork_is_closed():
    store = workflow_store()
    source = planned()
    first = begin_preparation(source)
    assert store.create(source)
    assert store.compare_and_swap(
        source, first, transition=BeginPreparationTransitionV1()
    )
    assert not store.compare_and_swap(
        source, first, transition=BeginPreparationTransitionV1()
    )
    fork = replace(first, preflight_digest=D[14])
    with pytest.raises(CoordinatorStoreError, match="conflict"):
        store.compare_and_swap(
            fork,
            record_stage_intent(first, prep(), intent("stage")),
            transition=RecordStageIntentTransitionV1(prep(), intent("stage")),
        )


def test_workflow_cas_rejects_arbitrary_replacement_and_forged_descriptor():
    store = workflow_store()
    source = planned()
    assert store.create(source)
    legitimate = begin_preparation(source)
    arbitrary = replace(legitimate, preflight_digest=D[14])
    with pytest.raises(CoordinatorStoreError, match="transition_invalid"):
        store.compare_and_swap(
            source, arbitrary, transition=BeginPreparationTransitionV1()
        )
    with pytest.raises(CoordinatorStoreError, match="transition_invalid"):
        store.compare_and_swap(
            source,
            legitimate,
            transition=RecordStageIntentTransitionV1(prep(), intent("stage")),
        )
    assert store.get(source.run) == source


def test_workflow_cas_rejects_replacement_run_revision_and_retained_corruption():
    source = planned()
    successor = begin_preparation(source)
    transition = BeginPreparationTransitionV1()
    store = workflow_store()
    store.create(source)
    with pytest.raises(CoordinatorStoreError, match="binding_mismatch"):
        store.compare_and_swap(
            source,
            replace(successor, run=TrainingRunRef("other-run", source.run.project_ref)),
            transition=transition,
        )
    with pytest.raises(CoordinatorStoreError, match="transition_invalid"):
        store.compare_and_swap(
            source, replace(successor, revision=2), transition=transition
        )
    assert store.get(source.run) == source

    retained = store._records[(source.run.project_ref, source.run.run_id)]
    object.__setattr__(retained, "revision", 1)
    with pytest.raises(CoordinatorStoreError, match="integrity_error"):
        store.get(source.run)


def test_workflow_project_plan_index_is_scoped_and_corruption_is_closed():
    source = planned()
    store = workflow_store()
    store.create(source)
    assert store.get_by_plan("other-project", source.plan_fingerprint) is None
    index = (source.run.project_ref, source.plan_fingerprint)
    store._plans[index] = (source.run.project_ref, "missing-run")
    with pytest.raises(CoordinatorStoreError) as caught:
        store.get_by_plan(*index)
    assert caught.value.code is CoordinatorStoreCode.INTEGRITY_ERROR
    with pytest.raises(CoordinatorStoreError, match="integrity_error"):
        store.get(source.run)


@pytest.mark.parametrize("operation", ["get", "create", "cas", "list", "get_by_plan"])
def test_workflow_foreign_run_under_primary_key_is_integrity_error(operation):
    source = planned()
    successor = begin_preparation(source)
    store = workflow_store()
    store.create(source)
    key = (source.run.project_ref, source.run.run_id)
    foreign = replace(
        source, run=TrainingRunRef("foreign-run", "foreign-project")
    )
    store._records[key] = foreign
    store._digests[key] = foreign.record_digest
    store._plans = {
        (foreign.run.project_ref, foreign.plan_fingerprint): key
    }
    actions = {
        "get": lambda: store.get(source.run),
        "create": lambda: store.create(source),
        "cas": lambda: store.compare_and_swap(
            source, successor, transition=BeginPreparationTransitionV1()
        ),
        "list": lambda: store.list(source.run.project_ref),
        "get_by_plan": lambda: store.get_by_plan(
            foreign.run.project_ref, foreign.plan_fingerprint
        ),
    }
    with pytest.raises(CoordinatorStoreError) as caught:
        actions[operation]()
    assert caught.value.code is CoordinatorStoreCode.INTEGRITY_ERROR


def test_workflow_cas_replays_effect_with_store_owned_exact_true_authentication():
    source = planned()
    preparing = begin_preparation(source)
    raw = intent("stage")
    awaiting = record_stage_intent(preparing, prep(), raw)
    evidence = record(raw, EffectState.FOUND, STAGE_REF)
    authenticated_assessment = assessment(evidence)
    successor = apply_stage_effect_record(
        awaiting, evidence, authenticated_assessment, Auth(), AssessmentAuth()
    )
    transition = ApplyStageEffectTransitionV1(evidence, authenticated_assessment)

    store = workflow_store()
    store.create(source)
    store.compare_and_swap(
        source, preparing, transition=BeginPreparationTransitionV1()
    )
    store.compare_and_swap(
        preparing,
        awaiting,
        transition=RecordStageIntentTransitionV1(prep(), raw),
    )
    assert store.compare_and_swap(awaiting, successor, transition=transition)

    denied = InMemoryWorkflowStoreV1(Auth(), AssessmentAuth(allowed="yes"))
    denied.create(source)
    denied.compare_and_swap(
        source, preparing, transition=BeginPreparationTransitionV1()
    )
    denied.compare_and_swap(
        preparing,
        awaiting,
        transition=RecordStageIntentTransitionV1(prep(), raw),
    )
    with pytest.raises(CoordinatorStoreError, match="transition_invalid"):
        denied.compare_and_swap(awaiting, successor, transition=transition)
    assert denied.get(source.run) == awaiting


def test_workflow_throwing_reducer_authenticator_is_closed_without_raw_leakage():
    class ThrowingFoundation(Auth):
        def authenticate_grant(self, value, command_bytes):
            raise RuntimeError("credential-secret")

    source = planned()
    preparing = begin_preparation(source)
    raw = intent("stage")
    awaiting = record_stage_intent(preparing, prep(), raw)
    evidence = record(raw, EffectState.FOUND, STAGE_REF)
    authenticated_assessment = assessment(evidence)
    successor = apply_stage_effect_record(
        awaiting, evidence, authenticated_assessment, Auth(), AssessmentAuth()
    )
    store = InMemoryWorkflowStoreV1(ThrowingFoundation(), AssessmentAuth())
    store.create(source)
    store.compare_and_swap(
        source, preparing, transition=BeginPreparationTransitionV1()
    )
    store.compare_and_swap(
        preparing,
        awaiting,
        transition=RecordStageIntentTransitionV1(prep(), raw),
    )
    with pytest.raises(CoordinatorStoreError) as caught:
        store.compare_and_swap(
            awaiting,
            successor,
            transition=ApplyStageEffectTransitionV1(
                evidence, authenticated_assessment
            ),
        )
    assert caught.value.code is CoordinatorStoreCode.TRANSITION_INVALID
    assert "credential-secret" not in repr(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert store.get(source.run) == awaiting


@pytest.mark.parametrize(
    "effect_state,target",
    [
        (EffectState.FOUND, WorkflowPhaseV1.QUEUED),
        (EffectState.INDETERMINATE, WorkflowPhaseV1.SUBMIT_RECONCILE_REQUIRED),
        (EffectState.DEFINITELY_ABSENT, WorkflowPhaseV1.FAILED),
        (EffectState.CONTRADICTED, WorkflowPhaseV1.CONTRADICTED),
    ],
)
def test_workflow_cas_replays_all_submit_effect_branches(effect_state, target):
    source = planned()
    preparing = begin_preparation(source)
    stage_intent = intent("stage")
    stage_pending = record_stage_intent(preparing, prep(), stage_intent)
    stage_record = record(stage_intent, EffectState.FOUND, STAGE_REF)
    stage_assessment = assessment(stage_record)
    staged_record = apply_stage_effect_record(
        stage_pending, stage_record, stage_assessment, Auth(), AssessmentAuth()
    )
    submit_intent = intent("submit")
    submit_pending = record_submit_intent(staged_record, submit_intent)
    submit_record = record(
        submit_intent,
        effect_state,
        PROVIDER_RUN if effect_state in {EffectState.FOUND, EffectState.CONTRADICTED} else None,
    )
    submit_assessment = assessment(submit_record)
    successor = apply_submit_effect_record(
        submit_pending, submit_record, submit_assessment, Auth(), AssessmentAuth()
    )

    store = workflow_store()
    store.create(source)
    assert store.compare_and_swap(
        source, preparing, transition=BeginPreparationTransitionV1()
    )
    assert store.compare_and_swap(
        preparing,
        stage_pending,
        transition=RecordStageIntentTransitionV1(prep(), stage_intent),
    )
    assert store.compare_and_swap(
        stage_pending,
        staged_record,
        transition=ApplyStageEffectTransitionV1(stage_record, stage_assessment),
    )
    assert store.compare_and_swap(
        staged_record,
        submit_pending,
        transition=RecordSubmitIntentTransitionV1(submit_intent),
    )
    assert store.compare_and_swap(
        submit_pending,
        successor,
        transition=ApplySubmitEffectTransitionV1(submit_record, submit_assessment),
    )
    assert store.get(source.run).phase is target


def test_workflow_cas_two_threads_have_one_winner():
    store = workflow_store()
    source = planned()
    successor = begin_preparation(source)
    store.create(source)
    gate = Barrier(2)

    def compete():
        gate.wait()
        return store.compare_and_swap(
            source, successor, transition=BeginPreparationTransitionV1()
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(lambda _: compete(), range(2)))
    assert sorted(results) == [False, True]
    assert store.get(source.run) == successor


def test_preparation_put_if_absent_reload_and_concurrent_winner():
    store = InMemoryPreparationStoreV1()
    value = prep()
    gate = Barrier(2)

    def compete():
        gate.wait()
        return store.put_if_absent(value)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(lambda _: compete(), range(2)))
    assert sorted(results) == [False, True]
    assert store.get(value.preparation_digest) == value
    with pytest.raises(CoordinatorStoreError):
        store.put_if_absent(object())
    alternate_doc = value.to_dict()
    alternate_doc["quote_digest"] = D[14]
    alternate = CanonicalPreparationV2.parse(canonical_bytes(alternate_doc))
    store._values[value.preparation_digest] = alternate
    with pytest.raises(CoordinatorStoreError, match="conflict"):
        store.put_if_absent(value)


def test_execution_grant_slot_is_exact_and_concurrent_one_winner():
    slot, first, authority, command = execution_grant()
    raw = intent("stage")
    competitor = authority.issue(
        raw.canonical_command_bytes,
        grant_ref="execution-grant-b",
        policy_digest=D[10],
        requirement_digest=D[11],
        not_before_epoch=1,
        expires_at_epoch=100,
    )
    store = InMemoryExecutionGrantStoreV1(authority)
    gate = Barrier(2)

    def compete(grant):
        gate.wait()
        try:
            return store.put_if_absent(slot, grant, command)
        except CoordinatorStoreError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(compete, (first, competitor)))
    assert sorted(results) == [False, True]
    assert store.get(slot, command) in (first, competitor)
    wrong = replace(slot, command_bytes_digest=D[14])
    with pytest.raises(CoordinatorStoreError, match="binding_mismatch"):
        store.put_if_absent(wrong, first, command)


def test_reconciliation_grant_competing_owners_share_exact_genesis_slot():
    raw = intent("stage")
    foundation = record(raw, EffectState.INDETERMINATE)
    slot, owner_a, authority, command = reconciliation_grant(
        owner="owner-a", grant_ref="recon-a", raw_intent=raw
    )
    _, owner_b, _, _ = reconciliation_grant(
        owner="owner-b", grant_ref="recon-b", authority=authority, raw_intent=raw
    )
    store = InMemoryReconciliationGrantStoreV1(authority)
    gate = Barrier(2)

    def compete(grant):
        gate.wait()
        try:
            return store.put_if_absent(slot, grant, command, foundation)
        except CoordinatorStoreError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(compete, (owner_a, owner_b)))
    assert sorted(results) == [False, True]
    winner = store._values[slot]
    loser = owner_b if winner == owner_a else owner_a
    assert loser != winner
    assert winner in (owner_a, owner_b)
    # A caller holding the losing candidate reloads by durable slot, not candidate identity.
    assert store.get(slot, command_bytes=command, record=foundation) == winner
    with pytest.raises(CoordinatorStoreError, match="binding_mismatch"):
        store.put_if_absent(
            replace(slot, ownership_epoch=2), owner_a, command, foundation
        )


@pytest.mark.parametrize("result", [False, None, object()])
def test_reconciliation_authentication_requires_exact_true(result):
    class Authenticator:
        def authenticate_reconciliation(self, grant, command_bytes):
            return result

    raw = intent("stage")
    foundation = record(raw, EffectState.INDETERMINATE)
    slot, grant, _, command = reconciliation_grant(raw_intent=raw)
    store = InMemoryReconciliationGrantStoreV1(Authenticator())
    with pytest.raises(CoordinatorStoreError) as caught:
        store.put_if_absent(slot, grant, command, foundation)
    assert caught.value.code is CoordinatorStoreCode.AUTHORITY_INVALID
    assert store._values == {}


def test_timeless_reconciliation_authentication_and_retained_tamper():
    raw = intent("stage")
    foundation = record(raw, EffectState.INDETERMINATE)
    slot, grant, authority, command = reconciliation_grant(raw_intent=raw)
    assert authority.authenticate_reconciliation(grant, command) is True
    assert authority.verify_reconciliation(grant, now_epoch=101) is False
    store = InMemoryReconciliationGrantStoreV1(authority)
    assert store.put_if_absent(slot, grant, command, foundation)
    retained = store._values[slot]
    object.__setattr__(retained, "tag", D[14])
    with pytest.raises(CoordinatorStoreError) as caught:
        store.get(slot, command_bytes=command, record=foundation)
    assert caught.value.code is CoordinatorStoreCode.AUTHORITY_INVALID


def test_reconciliation_throwing_authenticator_and_record_tamper_are_closed():
    class Throwing:
        def authenticate_reconciliation(self, grant, command_bytes):
            raise RuntimeError("credential-secret")

    raw = intent("stage")
    foundation = record(raw, EffectState.INDETERMINATE)
    slot, grant, authority, command = reconciliation_grant(raw_intent=raw)
    with pytest.raises(CoordinatorStoreError, match="authority_invalid") as caught:
        InMemoryReconciliationGrantStoreV1(Throwing()).put_if_absent(
            slot, grant, command, foundation
        )
    assert "credential-secret" not in repr(caught.value)
    object.__setattr__(foundation, "state", EffectState.FOUND)
    with pytest.raises(CoordinatorStoreError) as caught:
        InMemoryReconciliationGrantStoreV1(authority).put_if_absent(
            slot, grant, command, foundation
        )
    assert caught.value.code is CoordinatorStoreCode.INTEGRITY_ERROR


@pytest.mark.parametrize("result", [False, None, object()])
def test_execution_grant_authentication_requires_exact_true_and_zero_write(result):
    class Authenticator:
        def authenticate(self, grant, command_bytes):
            return result

    slot, grant, _, command = execution_grant()
    store = InMemoryExecutionGrantStoreV1(Authenticator())
    with pytest.raises(CoordinatorStoreError) as caught:
        store.put_if_absent(slot, grant, command)
    assert caught.value.code is CoordinatorStoreCode.AUTHORITY_INVALID
    assert store._values == {}


def test_grant_authenticator_exception_is_closed_and_retained_tamper_fails_read():
    class Throwing:
        def authenticate(self, grant, command_bytes):
            raise RuntimeError("secret")

    slot, grant, authority, command = execution_grant()
    with pytest.raises(CoordinatorStoreError, match="authority_invalid") as caught:
        InMemoryExecutionGrantStoreV1(Throwing()).put_if_absent(
            slot, grant, command
        )
    assert "secret" not in repr(caught.value)
    store = InMemoryExecutionGrantStoreV1(authority)
    store.put_if_absent(slot, grant, command)
    retained = store._values[slot]
    object.__setattr__(retained, "tag", D[14])
    with pytest.raises(CoordinatorStoreError, match="authority_invalid"):
        store.get(slot, command)


def _claim(record_value, grant, *, active, completed, prior=None, lineage=()):
    content = grant.content
    target = domain_digest(
        "synaptic-reconciliation-target/v2",
        canonical_bytes(
            {
                "command_digest": content.command_digest,
                "effect_id": content.effect_id,
                "preparation_digest": content.preparation_digest,
                "adapter_digest": content.adapter_digest,
                "provider_id": content.provider_id,
                "profile_ref": content.profile_ref,
                "account_ref": content.account_ref,
                "namespace_ref": content.namespace_ref,
                "owner_ref": content.owner_ref,
                "policy_digest": content.policy_digest,
                "requirement_digest": content.requirement_digest,
            }
        ),
    )
    if not lineage:
        lineage = (
            ReconciliationGrantBindingV2(
                content.grant_ref,
                grant.authenticated_grant_digest,
                len(record_value.receipt_admissions),
                None,
            ),
        )
    return ReconciliationOwnershipV2(
        content.owner_ref,
        content.generation,
        content.ownership_epoch,
        10,
        target,
        content.grant_ref,
        grant.authenticated_grant_digest,
        lineage,
        active,
        completed,
    )


def test_reconciliation_resume_second_resume_retry_and_active_reconstruction():
    raw = intent("stage")
    base = record(raw, EffectState.INDETERMINATE)
    genesis_slot, genesis, authority, command = reconciliation_grant(raw_intent=raw)
    active = _claim(base, genesis, active=True, completed=False)
    active_record = replace(base, reconciliation=active, reconciliation_claims=(active,))
    store = InMemoryReconciliationGrantStoreV1(authority)
    assert store.put_if_absent(genesis_slot, genesis, command, active_record)
    assert store.get(
        genesis_slot, command_bytes=command, record=active_record
    ) == genesis

    interrupted = replace(active, active=False)
    interrupted_record = replace(
        base, reconciliation=interrupted, reconciliation_claims=(interrupted,)
    )
    resume_slot, resumed_grant, _, _ = reconciliation_grant(
        authority=authority,
        raw_intent=raw,
        grant_ref="resume-a",
        prior_claim=interrupted.claim_digest,
        predecessor_grant=interrupted.grant_digest,
    )
    assert store.put_if_absent(resume_slot, resumed_grant, command, interrupted_record)
    resumed_binding = ReconciliationGrantBindingV2(
        resumed_grant.content.grant_ref,
        resumed_grant.authenticated_grant_digest,
        len(base.receipt_admissions),
        interrupted.grant_lineage[-1].binding_digest,
    )
    resumed = _claim(
        base,
        resumed_grant,
        active=False,
        completed=False,
        lineage=interrupted.grant_lineage + (resumed_binding,),
    )
    resumed_record = replace(base, reconciliation=resumed, reconciliation_claims=(resumed,))
    active_resumed = replace(resumed, active=True)
    active_resumed_record = replace(
        base, reconciliation=active_resumed, reconciliation_claims=(active_resumed,)
    )
    assert store.get(
        resume_slot, command_bytes=command, record=active_resumed_record
    ) == resumed_grant
    second_slot, second_grant, _, _ = reconciliation_grant(
        authority=authority,
        raw_intent=raw,
        grant_ref="resume-b",
        prior_claim=resumed.claim_digest,
        predecessor_grant=resumed.grant_digest,
    )
    assert store.put_if_absent(second_slot, second_grant, command, resumed_record)

    completed = replace(resumed, completed=True)
    completed_record = replace(base, reconciliation=completed, reconciliation_claims=(completed,))
    retry_slot, retry_grant, _, _ = reconciliation_grant(
        authority=authority,
        raw_intent=raw,
        generation=2,
        epoch=2,
        grant_ref="retry-a",
        prior_claim=completed.claim_digest,
        predecessor_grant=completed.grant_digest,
    )
    assert store.put_if_absent(retry_slot, retry_grant, command, completed_record)

    with pytest.raises(CoordinatorStoreError, match="binding_mismatch"):
        store.put_if_absent(
            replace(resume_slot, predecessor_grant_digest=D[14]),
            resumed_grant,
            command,
            interrupted_record,
        )


def test_retained_reconciliation_grant_must_bind_containing_claim_identity():
    raw = intent("stage")
    base = record(raw, EffectState.INDETERMINATE)
    slot, foreign_grant, authority, command = reconciliation_grant(
        raw_intent=raw,
        owner="foreign-owner",
        generation=7,
        epoch=9,
        grant_ref="foreign-grant",
    )
    binding = ReconciliationGrantBindingV2(
        foreign_grant.content.grant_ref,
        foreign_grant.authenticated_grant_digest,
        len(base.receipt_admissions),
        None,
    )
    claim = ReconciliationOwnershipV2(
        "claim-owner",
        1,
        1,
        10,
        D[0],
        binding.grant_ref,
        binding.grant_digest,
        (binding,),
        True,
        False,
    )
    forged_record = replace(
        base, reconciliation=claim, reconciliation_claims=(claim,)
    )
    store = InMemoryReconciliationGrantStoreV1(authority)
    with pytest.raises(CoordinatorStoreError) as caught:
        store.put_if_absent(slot, foreign_grant, command, forged_record)
    assert caught.value.code is CoordinatorStoreCode.BINDING_MISMATCH
    assert store._values == {}

    store._values[slot] = foreign_grant
    store._canonical[slot] = foreign_grant.canonical_bytes
    with pytest.raises(CoordinatorStoreError) as caught:
        store.get(slot, command_bytes=command, record=forged_record)
    assert caught.value.code is CoordinatorStoreCode.BINDING_MISMATCH
