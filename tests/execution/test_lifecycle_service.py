from tuner.execution._effect_executor import _ProviderEffectExecutor
from tuner.execution.broker import MutationBroker,MutationCommandV1
from tuner.execution.contracts import *
from tuner.execution.operation import ModalStageTargetV1,OperationBindingV1
from tuner.execution.service import LifecycleService
from tests.execution.fakes import InMemoryLifecycleRepository
D="a"*64;NOW="2026-08-25T12:00:00Z";LATER="2026-08-25T13:00:00Z"
class Driver:
    def __init__(self,effect):self.effect=effect
    def execute_once(self,raw):return EffectObservation(self.effect,EffectDisposition.FOUND,"fc-1",D)
def test_provider_completion_enters_verifying():
    effect=EffectIdentity("e","op",EffectKind.SUBMIT,ExecutionScope("modal","acct","env"));operation=OperationBindingV1("p","r",effect,"g",D,D,D,D,D,D,D,D,D,D,D,"nonce",ModalStageTargetV1("slot","cv","av","operations/e/output",1,"key"));command=MutationCommandV1(operation,D,D)
    binding=GrantBinding.from_operation(operation,issued_at=NOW,expires_at=LATER)
    repo=InMemoryLifecycleRepository();service=LifecycleService(repo,clock=lambda:NOW);record=service.plan(project_ref="p",run_id="r");record=service.authorize(project_ref="p",run_id="r",expected_revision=record.revision,binding=binding)
    MutationBroker(repo,_ProviderEffectExecutor(Driver(effect))).execute(command,expected_revision=record.revision);record=repo.load("p","r")
    record=service.record_provider_phase(project_ref="p",run_id="r",expected_revision=record.revision,provider_phase=ProviderRunPhase.RUNNING)
    record=service.record_provider_phase(project_ref="p",run_id="r",expected_revision=record.revision,provider_phase=ProviderRunPhase.SUCCEEDED)
    assert record.phase is LifecyclePhase.VERIFYING and record.verification is VerificationStatus.PENDING

def test_invalid_verification_can_be_explicitly_reopened_and_verified():
    effect=EffectIdentity("e","op",EffectKind.SUBMIT,ExecutionScope("modal","acct","env"));operation=OperationBindingV1("p","r",effect,"g",D,D,D,D,D,D,D,D,D,D,D,"nonce",ModalStageTargetV1("slot","cv","av","operations/e/output",1,"key"));command=MutationCommandV1(operation,D,D)
    binding=GrantBinding.from_operation(operation,issued_at=NOW,expires_at=LATER)
    repo=InMemoryLifecycleRepository();service=LifecycleService(repo,clock=lambda:NOW);record=service.plan(project_ref="p",run_id="r");record=service.authorize(project_ref="p",run_id="r",expected_revision=record.revision,binding=binding)
    MutationBroker(repo,_ProviderEffectExecutor(Driver(effect))).execute(command,expected_revision=record.revision);record=repo.load("p","r")
    record=service.record_provider_phase(project_ref="p",run_id="r",expected_revision=record.revision,provider_phase=ProviderRunPhase.SUCCEEDED)
    record=service.record_verification(project_ref="p",run_id="r",expected_revision=record.revision,verification=VerificationStatus.VERIFYING)
    record=service.record_verification(project_ref="p",run_id="r",expected_revision=record.revision,verification=VerificationStatus.INVALID)
    record=service.reopen_verification(project_ref="p",run_id="r",expected_revision=record.revision)
    assert record.phase is LifecyclePhase.VERIFYING and record.verification is VerificationStatus.VERIFYING
    record=service.record_verification(project_ref="p",run_id="r",expected_revision=record.revision,verification=VerificationStatus.VERIFIED)
    assert record.phase is LifecyclePhase.SUCCEEDED and record.verification is VerificationStatus.VERIFIED
