from dataclasses import replace

import pytest

from synaptic_tuner.api.v1.results import VerifiedArtifact
from tuner.execution.coordinator_v1.coordinator import (
    ApplyArtifactVerificationTransitionV1,
    ApplyCancelEffectTransitionV1,
    ApplyProviderObservationTransitionV1,
    ApplyReverificationTransitionV1,
    ApplyStageEffectTransitionV1,
    ApplySubmitEffectTransitionV1,
    BeginPreparationTransitionV1,
    RecordStageIntentTransitionV1,
    RecordSubmitIntentTransitionV1,
    RecordCancelIntentTransitionV1,
)
from tuner.execution.coordinator_v1.model import (
    ArtifactManifestV1,
    ProviderReadPurposeV1,
    ProviderRunPhaseV1,
    VerificationVerdictV1,
    WorkflowPhaseV1,
)
from tuner.execution.coordinator_v1.state_machine import (
    WorkflowTransitionError,
    apply_artifact_verification,
    apply_cancel_effect_record,
    apply_provider_observation,
    apply_reverification,
    apply_stage_effect_record,
    apply_submit_effect_record,
    begin_preparation,
    provider_run_read_request,
    record_cancel_intent,
    record_stage_intent,
    record_submit_intent,
)
from tuner.execution.coordinator_v1.stores import (
    CoordinatorStoreCode,
    CoordinatorStoreError,
    InMemoryWorkflowStoreV1,
)
from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.execution.foundation_v2.repository import EffectState
from tuner.execution.foundation_v2.references import CancellationRefV1, ProviderRunRefV1

from .test_state_machine import (
    AssessmentAuth,
    Auth,
    D,
    PROVIDER_RUN,
    RUN,
    ObservationAuth,
    STAGE_REF,
    Verifier,
    assessment,
    intent,
    observation,
    queued_evidence,
    planned,
    prep,
    record,
    verification_receipt,
)


def _store(*, observation_auth=None, verifier=None):
    store = InMemoryWorkflowStoreV1(
        Auth(),
        AssessmentAuth(),
        observation_auth or ObservationAuth(),
        verifier or Verifier(),
    )
    assert store.create(planned())
    return store


def _queued_store(*, observation_auth=None, verifier=None):
    store = _store(observation_auth=observation_auth, verifier=verifier)
    source = planned()
    preparing = begin_preparation(source)
    assert store.compare_and_swap(
        source, preparing, transition=BeginPreparationTransitionV1()
    )
    stage_intent = intent("stage")
    stage_pending = record_stage_intent(preparing, prep(), stage_intent)
    assert store.compare_and_swap(
        preparing,
        stage_pending,
        transition=RecordStageIntentTransitionV1(prep(), stage_intent),
    )
    stage_evidence = record(stage_intent, EffectState.FOUND, STAGE_REF)
    stage_assessment = assessment(stage_evidence)
    staged = apply_stage_effect_record(
        stage_pending, stage_evidence, stage_assessment, Auth(), AssessmentAuth()
    )
    assert store.compare_and_swap(
        stage_pending,
        staged,
        transition=ApplyStageEffectTransitionV1(
            stage_evidence, stage_assessment
        ),
    )
    submit_intent = intent("submit")
    submit_pending = record_submit_intent(staged, submit_intent)
    assert store.compare_and_swap(
        staged,
        submit_pending,
        transition=RecordSubmitIntentTransitionV1(submit_intent),
    )
    foundation = record(submit_intent, EffectState.FOUND, PROVIDER_RUN)
    submit_assessment = assessment(foundation)
    queued = apply_submit_effect_record(
        submit_pending, foundation, submit_assessment, Auth(), AssessmentAuth()
    )
    assert store.compare_and_swap(
        submit_pending,
        queued,
        transition=ApplySubmitEffectTransitionV1(
            foundation, submit_assessment
        ),
    )
    return store, queued, foundation


def _succeeded_store(*, verifier=None):
    store, queued, foundation = _queued_store(verifier=verifier)
    request, envelope = observation(
        queued, foundation, ProviderRunPhaseV1.SUCCEEDED, {"phase": "succeeded"}
    )
    succeeded = apply_provider_observation(
        queued, request, envelope, ObservationAuth()
    )
    assert store.compare_and_swap(
        queued,
        succeeded,
        transition=ApplyProviderObservationTransitionV1(request, envelope),
    )
    return store, succeeded, foundation


def _manifest():
    artifact = VerifiedArtifact("adapter", D[4], 10)
    return ArtifactManifestV1.build(
        run=RUN,
        provider_run=PROVIDER_RUN,
        artifacts=(artifact,),
        artifact_source_digest=D[5],
        canonical_evidence=canonical_bytes({"inventory": 1}),
    )


def test_strong_cas_replays_cancel_intent_and_effect():
    store, queued, _ = _queued_store()
    cancel_intent = intent("cancel")
    pending = record_cancel_intent(queued, cancel_intent)
    assert store.compare_and_swap(
        queued,
        pending,
        transition=RecordCancelIntentTransitionV1(cancel_intent),
    )

    reference = CancellationRefV1(
        ProviderRunRefV1(PROVIDER_RUN.provider_job_ref), D[14]
    )
    evidence = record(cancel_intent, EffectState.FOUND, reference)
    authenticated_assessment = assessment(evidence)
    requested = apply_cancel_effect_record(
        pending, evidence, authenticated_assessment, Auth(), AssessmentAuth()
    )
    assert store.compare_and_swap(
        pending,
        requested,
        transition=ApplyCancelEffectTransitionV1(
            evidence, authenticated_assessment
        ),
    )
    assert store.get(queued.run).phase is WorkflowPhaseV1.CANCEL_REQUESTED


@pytest.mark.parametrize(
    "effect_state,target",
    [
        (EffectState.INDETERMINATE, WorkflowPhaseV1.CANCEL_RECONCILE_REQUIRED),
        (EffectState.DEFINITELY_ABSENT, WorkflowPhaseV1.QUEUED),
        (EffectState.CONTRADICTED, WorkflowPhaseV1.CONTRADICTED),
    ],
)
def test_strong_cas_replays_cancel_effect_branches(effect_state, target):
    store, queued, _ = _queued_store()
    cancel_intent = intent("cancel")
    pending = record_cancel_intent(queued, cancel_intent)
    reference = None
    if effect_state is EffectState.CONTRADICTED:
        reference = CancellationRefV1(
            ProviderRunRefV1(PROVIDER_RUN.provider_job_ref), D[14]
        )
    evidence = record(cancel_intent, effect_state, reference)
    authenticated_assessment = assessment(evidence)
    successor = apply_cancel_effect_record(
        pending, evidence, authenticated_assessment, Auth(), AssessmentAuth()
    )
    assert store.compare_and_swap(
        queued,
        pending,
        transition=RecordCancelIntentTransitionV1(cancel_intent),
    )
    assert store.compare_and_swap(
        pending,
        successor,
        transition=ApplyCancelEffectTransitionV1(
            evidence, authenticated_assessment
        ),
    )
    assert store.get(queued.run).phase is target


def test_strong_cas_replays_provider_observation_with_store_authenticator():
    store, queued, foundation = _queued_store()
    request, envelope = observation(
        queued, foundation, ProviderRunPhaseV1.RUNNING, {"phase": "running"}
    )
    running = apply_provider_observation(
        queued, request, envelope, ObservationAuth()
    )
    assert store.compare_and_swap(
        queued,
        running,
        transition=ApplyProviderObservationTransitionV1(request, envelope),
    )

    denied, denied_queued, denied_foundation = _queued_store(
        observation_auth=ObservationAuth(allowed="truthy")
    )
    denied_request, denied_envelope = observation(
        denied_queued,
        denied_foundation,
        ProviderRunPhaseV1.RUNNING,
        {"phase": "running-denied"},
    )
    denied_running = apply_provider_observation(
        denied_queued, denied_request, denied_envelope, ObservationAuth()
    )
    with pytest.raises(CoordinatorStoreError) as caught:
        denied.compare_and_swap(
            denied_queued,
            denied_running,
            transition=ApplyProviderObservationTransitionV1(
                denied_request, denied_envelope
            ),
        )
    assert caught.value.code is CoordinatorStoreCode.TRANSITION_INVALID
    assert denied.get(denied_queued.run) == denied_queued


def test_strong_cas_replays_verification_and_reverification():
    store, succeeded, _ = _succeeded_store()
    manifest = _manifest()
    receipt = verification_receipt(
        succeeded, manifest, VerificationVerdictV1.VERIFIED
    )
    verified = apply_artifact_verification(
        succeeded, manifest, receipt, Verifier()
    )
    assert store.compare_and_swap(
        succeeded,
        verified,
        transition=ApplyArtifactVerificationTransitionV1(manifest, receipt),
    )
    next_receipt = verification_receipt(
        verified, manifest, VerificationVerdictV1.VERIFIED, "b"
    )
    reverified = apply_reverification(
        verified, manifest, next_receipt, Verifier()
    )
    assert store.compare_and_swap(
        verified,
        reverified,
        transition=ApplyReverificationTransitionV1(manifest, next_receipt),
    )


def test_strong_cas_replays_rejected_verification_and_requires_exact_true():
    store, succeeded, _ = _succeeded_store()
    manifest = _manifest()
    receipt = verification_receipt(
        succeeded, manifest, VerificationVerdictV1.REJECTED, "c"
    )
    rejected = apply_artifact_verification(
        succeeded, manifest, receipt, Verifier()
    )
    assert store.compare_and_swap(
        succeeded,
        rejected,
        transition=ApplyArtifactVerificationTransitionV1(manifest, receipt),
    )
    assert rejected.phase is WorkflowPhaseV1.VERIFICATION_FAILED

    denied, denied_succeeded, _ = _succeeded_store(
        verifier=Verifier(allowed="truthy")
    )
    denied_receipt = verification_receipt(
        denied_succeeded, manifest, VerificationVerdictV1.REJECTED, "d"
    )
    denied_rejected = apply_artifact_verification(
        denied_succeeded, manifest, denied_receipt, Verifier()
    )
    with pytest.raises(CoordinatorStoreError) as caught:
        denied.compare_and_swap(
            denied_succeeded,
            denied_rejected,
            transition=ApplyArtifactVerificationTransitionV1(
                manifest, denied_receipt
            ),
        )
    assert caught.value.code is CoordinatorStoreCode.TRANSITION_INVALID
    assert denied.get(denied_succeeded.run) == denied_succeeded

def test_provider_read_purpose_is_canonical_and_observation_only_accepts_observe():
    queued, foundation = queued_evidence()
    authenticated_assessment = assessment(foundation)
    logs_request = provider_run_read_request(
        queued,
        foundation,
        authenticated_assessment,
        Auth(),
        AssessmentAuth(),
        purpose=ProviderReadPurposeV1.LOGS,
    )
    assert b'"purpose":"logs"' in logs_request.canonical_bytes
    with pytest.raises(ValueError, match="identity"):
        replace(logs_request, purpose=ProviderReadPurposeV1.OBSERVE)
    _, envelope = observation(
        queued, foundation, ProviderRunPhaseV1.RUNNING, {"phase": "running"}
    )
    with pytest.raises(WorkflowTransitionError, match="purpose"):
        apply_provider_observation(
            queued, logs_request, envelope, ObservationAuth()
        )


def test_provider_read_purpose_has_phase_specific_eligibility():
    queued, foundation = queued_evidence()
    authenticated_assessment = assessment(foundation)
    with pytest.raises(WorkflowTransitionError, match="eligible"):
        provider_run_read_request(
            queued,
            foundation,
            authenticated_assessment,
            Auth(),
            AssessmentAuth(),
            purpose=ProviderReadPurposeV1.ARTIFACTS,
        )
    _, succeeded, _ = _succeeded_store()
    artifact_request = provider_run_read_request(
        succeeded,
        foundation,
        authenticated_assessment,
        Auth(),
        AssessmentAuth(),
        purpose=ProviderReadPurposeV1.ARTIFACTS,
    )
    assert artifact_request.purpose is ProviderReadPurposeV1.ARTIFACTS
    with pytest.raises(WorkflowTransitionError, match="eligible"):
        provider_run_read_request(
            succeeded,
            foundation,
            authenticated_assessment,
            Auth(),
            AssessmentAuth(),
            purpose=ProviderReadPurposeV1.OBSERVE,
        )


def test_operational_descendant_reachability_is_history_and_terminal_aware():
    store, queued, foundation = _queued_store()
    request, running_observation = observation(
        queued, foundation, ProviderRunPhaseV1.RUNNING, {"phase": "running"}
    )
    running = apply_provider_observation(
        queued, request, running_observation, ObservationAuth()
    )
    assert store.compare_and_swap(
        queued,
        running,
        transition=ApplyProviderObservationTransitionV1(
            request, running_observation
        ),
    )
    assert store.is_descendant(queued, running) is True
    assert store.is_descendant(running, queued) is False

    cancel_intent = intent("cancel")
    pending = record_cancel_intent(running, cancel_intent)
    assert store.compare_and_swap(
        running,
        pending,
        transition=RecordCancelIntentTransitionV1(cancel_intent),
    )
    absent = record(cancel_intent, EffectState.DEFINITELY_ABSENT)
    restored = apply_cancel_effect_record(
        pending, absent, assessment(absent), Auth(), AssessmentAuth()
    )
    assert restored.phase is WorkflowPhaseV1.RUNNING
    assert store.compare_and_swap(
        pending,
        restored,
        transition=ApplyCancelEffectTransitionV1(absent, assessment(absent)),
    )
    assert store.is_descendant(running, restored) is True
    assert store.is_descendant(restored, running) is False

    verification_store, succeeded, _ = _succeeded_store()
    manifest = _manifest()
    receipt = verification_receipt(
        succeeded, manifest, VerificationVerdictV1.VERIFIED
    )
    verified = apply_artifact_verification(
        succeeded, manifest, receipt, Verifier()
    )
    assert verification_store.compare_and_swap(
        succeeded,
        verified,
        transition=ApplyArtifactVerificationTransitionV1(manifest, receipt),
    )
    assert verification_store.is_descendant(succeeded, verified) is True
    assert verification_store.is_descendant(verified, succeeded) is False


def test_compressed_running_to_verified_is_not_store_ancestry():
    running_store, queued, foundation = _queued_store()
    request, running_observation = observation(
        queued, foundation, ProviderRunPhaseV1.RUNNING, {"phase": "running"}
    )
    running = apply_provider_observation(
        queued, request, running_observation, ObservationAuth()
    )
    assert running_store.compare_and_swap(
        queued,
        running,
        transition=ApplyProviderObservationTransitionV1(
            request, running_observation
        ),
    )
    verification_store, succeeded, _ = _succeeded_store()
    manifest = _manifest()
    receipt = verification_receipt(
        succeeded, manifest, VerificationVerdictV1.VERIFIED
    )
    verified = apply_artifact_verification(
        succeeded, manifest, receipt, Verifier()
    )
    assert verification_store.compare_and_swap(
        succeeded,
        verified,
        transition=ApplyArtifactVerificationTransitionV1(manifest, receipt),
    )
    assert verified.revision == running.revision + 1
    assert running_store.is_descendant(running, verified) is False


def test_operational_store_closes_throwing_authenticator_without_raw_leakage():
    class ThrowingObservation:
        def authenticate(self, value):
            raise RuntimeError("provider-credential-secret")

    store, queued, foundation = _queued_store(
        observation_auth=ThrowingObservation()
    )
    request, envelope = observation(
        queued, foundation, ProviderRunPhaseV1.RUNNING, {"phase": "running"}
    )
    running = apply_provider_observation(
        queued, request, envelope, ObservationAuth()
    )
    with pytest.raises(CoordinatorStoreError) as caught:
        store.compare_and_swap(
            queued,
            running,
            transition=ApplyProviderObservationTransitionV1(request, envelope),
        )
    assert caught.value.code is CoordinatorStoreCode.TRANSITION_INVALID
    assert "provider-credential-secret" not in repr(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert store.get(queued.run) == queued
