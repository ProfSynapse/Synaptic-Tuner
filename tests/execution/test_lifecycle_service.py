from tuner.execution._effect_executor import _ProviderEffectExecutor
from tuner.execution.broker import MutationBroker,MutationCommandV1
from tuner.execution.contracts import *
from tuner.execution.operation import ModalStageTargetV1,OperationBindingV1
from tuner.execution.service import LifecycleService
from tuner.runtime.offline_sft_worker import load_packaged_offline_sft_worker_manifest
from tests.execution.fakes import InMemoryLifecycleRepository
D="a"*64;NOW="2026-08-25T12:00:00Z";LATER="2026-08-25T13:00:00Z"
class Driver:
    def __init__(self,effect):self.effect=effect
    def execute_once(self,raw):return EffectObservation(self.effect,EffectDisposition.FOUND,"fc-1",D)
def operation(effect):
    return OperationBindingV1(project_ref="p",run_id="r",effect=effect,grant_ref="g",plan_fingerprint=D,execution_source_digest=D,workload_digest=D,deployment_attestation_digest=D,artifact_contract_digest=D,log_policy_digest=D,invocation_intent_digest=D,worker_closure_manifest_digest=load_packaged_offline_sft_worker_manifest().sha256,resource_digest=D,quote_digest=D,secret_requirements_digest=D,invocation_arguments_digest=D,invocation_nonce="nonce",stage_target=ModalStageTargetV1("slot","cv","av","operations/e/output",1,"key"))
def test_provider_completion_enters_verifying():
    effect=EffectIdentity("e","op",EffectKind.SUBMIT,ExecutionScope("modal","acct","env"));bound_operation=operation(effect);command=MutationCommandV1(bound_operation,D,D)
    binding=GrantBinding.from_operation(bound_operation,issued_at=NOW,expires_at=LATER)
    repo=InMemoryLifecycleRepository();service=LifecycleService(repo,clock=lambda:NOW);record=service.plan(project_ref="p",run_id="r");record=service.authorize(project_ref="p",run_id="r",expected_revision=record.revision,binding=binding)
    MutationBroker(repo,_ProviderEffectExecutor(Driver(effect))).execute(command,expected_revision=record.revision);record=repo.load("p","r")
    record=service.record_provider_phase(project_ref="p",run_id="r",expected_revision=record.revision,provider_phase=ProviderRunPhase.RUNNING)
    record=service.record_provider_phase(project_ref="p",run_id="r",expected_revision=record.revision,provider_phase=ProviderRunPhase.SUCCEEDED)
    assert record.phase is LifecyclePhase.VERIFYING and record.verification is VerificationStatus.PENDING

def test_invalid_verification_can_be_explicitly_reopened_and_verified():
    effect=EffectIdentity("e","op",EffectKind.SUBMIT,ExecutionScope("modal","acct","env"));bound_operation=operation(effect);command=MutationCommandV1(bound_operation,D,D)
    binding=GrantBinding.from_operation(bound_operation,issued_at=NOW,expires_at=LATER)
    repo=InMemoryLifecycleRepository();service=LifecycleService(repo,clock=lambda:NOW);record=service.plan(project_ref="p",run_id="r");record=service.authorize(project_ref="p",run_id="r",expected_revision=record.revision,binding=binding)
    MutationBroker(repo,_ProviderEffectExecutor(Driver(effect))).execute(command,expected_revision=record.revision);record=repo.load("p","r")
    record=service.record_provider_phase(project_ref="p",run_id="r",expected_revision=record.revision,provider_phase=ProviderRunPhase.SUCCEEDED)
    record=service.record_verification(project_ref="p",run_id="r",expected_revision=record.revision,verification=VerificationStatus.VERIFYING)
    record=service.record_verification(project_ref="p",run_id="r",expected_revision=record.revision,verification=VerificationStatus.INVALID)
    record=service.reopen_verification(project_ref="p",run_id="r",expected_revision=record.revision)
    assert record.phase is LifecyclePhase.VERIFYING and record.verification is VerificationStatus.VERIFYING
    record=service.record_verification(project_ref="p",run_id="r",expected_revision=record.revision,verification=VerificationStatus.VERIFIED)
    assert record.phase is LifecyclePhase.SUCCEEDED and record.verification is VerificationStatus.VERIFIED
