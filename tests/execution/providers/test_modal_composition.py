from __future__ import annotations

from synaptic_tuner.api.v1.host import HostPorts
from tests.execution.providers.test_modal_verification import Auth,Facade
from tests.execution.providers.test_modal_source_resolution import Replay,_deployment
from tuner.execution.providers.modal.composition import ModalVerificationPolicyV1,compose_modal_source_finalizer
from tuner.project.git_verification import GitCliLocalSourceInspector,GitLsRemotePushedCommitVerifier
from tuner.execution.providers.modal.verification import ModalSdkDeploymentVerifier


class Remote:
    def read_ref(self,*,canonical_url,exact_ref):return b""


def test_single_composition_root_selects_production_verification_algorithms():
    ports=HostPorts(lifecycle=object(),grants=object(),secrets=object(),evidence_replay=Replay(),authenticator=Auth(),clock=lambda:"2026-08-25T12:00:00Z",git_remote=Remote(),modal_reads=Facade(_deployment()),training_resolver=object())
    policy=ModalVerificationPolicyV1("project/run-1","git-verifier","modal-verifier","git-key","modal-key",lambda purpose:purpose.replace("/","-"),lambda purpose:"evidence-"+purpose.split("/")[0])
    value=compose_modal_source_finalizer(ports,policy)
    assert type(value._local_sources) is GitCliLocalSourceInspector
    assert type(value._pushed_sources) is GitLsRemotePushedCommitVerifier
    assert type(value._deployments) is ModalSdkDeploymentVerifier
