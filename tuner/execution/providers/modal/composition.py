"""Single Modal composition root over host-owned ports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from tuner.project.git_verification import GitCliLocalSourceInspector,GitLsRemotePushedCommitVerifier

from ...contracts import safe_ref
from ...evidence import SOURCE_EVIDENCE_PURPOSE, EvidenceAuthenticator, EvidenceReplayRepository
from .resolution import ModalDualCloneSourceFinalizer
from .verification import ModalSdkDeploymentVerifier


class _GitRemoteReader(Protocol):
    def read_ref(self, *, canonical_url: str, exact_ref: str) -> bytes: ...


class _DeploymentReader(Protocol):
    def bound_scope(self): ...
    def capability_proof(self, binding): ...
    def inspect_deployment(self, *, app_name: str, function_name: str): ...


@dataclass(frozen=True, slots=True)
class ModalSourceVerificationPorts:
    """Explicit verification collaborators, outside provider-neutral HostPorts."""

    git_remote: _GitRemoteReader
    deployment_reads: _DeploymentReader
    authenticator: EvidenceAuthenticator
    evidence_replay: EvidenceReplayRepository
    clock: Callable[[], str]


@dataclass(frozen=True,slots=True)
class ModalVerificationPolicyV1:
    audience_ref:str
    source_issuer_ref:str
    deployment_issuer_ref:str
    source_key_ref:str
    deployment_key_ref:str
    challenge_factory:Callable[[str],str]
    evidence_ref_factory:Callable[[str],str]
    def __post_init__(self):
        for name in ("audience_ref","source_issuer_ref","deployment_issuer_ref","source_key_ref","deployment_key_ref"):object.__setattr__(self,name,safe_ref(getattr(self,name),name))
        if not callable(self.challenge_factory) or not callable(self.evidence_ref_factory):raise TypeError("evidence factories must be callable")


def compose_modal_source_finalizer(ports:ModalSourceVerificationPorts,policy:ModalVerificationPolicyV1)->ModalDualCloneSourceFinalizer:
    if type(ports) is not ModalSourceVerificationPorts or type(policy) is not ModalVerificationPolicyV1:raise TypeError("exact Modal verification ports and policy are required")
    pushed=GitLsRemotePushedCommitVerifier(ports.git_remote,ports.authenticator,clock=ports.clock,audience_ref=policy.audience_ref,issuer_ref=policy.source_issuer_ref,key_ref=policy.source_key_ref,challenge_factory=lambda:policy.challenge_factory(SOURCE_EVIDENCE_PURPOSE),evidence_ref_factory=lambda:policy.evidence_ref_factory(SOURCE_EVIDENCE_PURPOSE))
    deployment=ModalSdkDeploymentVerifier(ports.deployment_reads,ports.authenticator,clock=ports.clock,audience_ref=policy.audience_ref,issuer_ref=policy.deployment_issuer_ref,key_ref=policy.deployment_key_ref,challenge_factory=lambda:policy.challenge_factory("modal-deployment-evidence/v1"),evidence_ref_factory=lambda:policy.evidence_ref_factory("modal-deployment-evidence/v1"))
    return ModalDualCloneSourceFinalizer(GitCliLocalSourceInspector(),pushed,deployment,authenticator=ports.authenticator,replay=ports.evidence_replay,clock=ports.clock,source_issuer_ref=policy.source_issuer_ref,deployment_issuer_ref=policy.deployment_issuer_ref,source_key_ref=policy.source_key_ref,deployment_key_ref=policy.deployment_key_ref)


__all__=[
    "ModalVerificationPolicyV1",
    "ModalSourceVerificationPorts",
    "compose_modal_source_finalizer",
]
