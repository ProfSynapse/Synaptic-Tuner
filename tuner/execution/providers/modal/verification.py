"""Read-only Modal 1.5.4 deployment verification algorithm."""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import timedelta
from typing import Callable, Protocol, runtime_checkable

from tuner.execution.evidence import DEPLOYMENT_EVIDENCE_POLICY, EvidenceAuthenticator, parse_utc

from .binding import CapabilityProofV1, ModalClientBinding
from .resolution import (
    ModalDeploymentSelectionV1, ModalDeploymentVerificationPort,
    VerifiedModalDeploymentIdentityV1,
)


@runtime_checkable
class ModalDeploymentReadFacade(Protocol):
    def bound_scope(self) -> tuple[str, str, str, str]: ...
    def capability_proof(self, binding: ModalClientBinding) -> CapabilityProofV1: ...
    def inspect_deployment(self, *, app_name: str, function_name: str) -> ModalDeploymentSelectionV1: ...


class ModalSdkDeploymentVerifier(ModalDeploymentVerificationPort):
    """Verify provider-observed immutable deployment facts, then host-seal them."""

    def __init__(self,facade:ModalDeploymentReadFacade,authenticator:EvidenceAuthenticator,*,clock:Callable[[],str],audience_ref:str,issuer_ref:str,key_ref:str,challenge_factory:Callable[[],str],evidence_ref_factory:Callable[[],str]):
        if not isinstance(facade,ModalDeploymentReadFacade) or not isinstance(authenticator,EvidenceAuthenticator):raise TypeError("Modal read facade and authenticator are required")
        self.facade=facade;self.authenticator=authenticator;self.clock=clock;self.audience_ref=audience_ref;self.issuer_ref=issuer_ref;self.key_ref=key_ref;self.challenge_factory=challenge_factory;self.evidence_ref_factory=evidence_ref_factory

    def verify(self,selection:ModalDeploymentSelectionV1)->VerifiedModalDeploymentIdentityV1:
        if not isinstance(selection,ModalDeploymentSelectionV1):raise TypeError("deployment selection is required")
        binding=ModalClientBinding(selection.account_ref,selection.workspace_ref,selection.environment_ref,selection.client_ref,selection.sdk_version)
        try:
            scope=self.facade.bound_scope();capabilities=self.facade.capability_proof(binding)
            observed=self.facade.inspect_deployment(app_name=selection.app_name,function_name=selection.function_name)
        except Exception as exc:raise ValueError("deployment_identity_unverifiable") from exc
        if scope!=(binding.account_ref,binding.workspace_ref,binding.environment_ref,binding.client_ref):raise ValueError("deployment_identity_unverifiable")
        if not isinstance(capabilities,CapabilityProofV1) or not capabilities.complete:raise ValueError("deployment_identity_unverifiable")
        if observed!=selection:raise ValueError("deployment_identity_unverifiable")
        verified=self.clock();expires=(parse_utc(verified)+timedelta(seconds=DEPLOYMENT_EVIDENCE_POLICY.maximum_lifetime_seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")
        fields={"selection":observed,"issuer_ref":self.issuer_ref,"evidence_ref":self.evidence_ref_factory(),"audience_ref":self.audience_ref,"challenge_nonce":self.challenge_factory(),"verified_at":verified,"expires_at":expires,"key_ref":self.key_ref}
        unsigned={"schema_version":"synaptic-verified-modal-deployment/v1","selection":observed.to_dict(),**{name:fields[name] for name in ("issuer_ref","evidence_ref","audience_ref","challenge_nonce","verified_at","expires_at","key_ref")}}
        payload=json.dumps(unsigned,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")
        attestation=hashlib.sha256(payload).hexdigest();tag=self.authenticator.sign("modal-deployment-evidence/v1",payload,self.key_ref)
        return VerifiedModalDeploymentIdentityV1(**fields,attestation_digest=attestation,tag_base64=base64.b64encode(tag).decode("ascii"))


__all__=["ModalDeploymentReadFacade","ModalSdkDeploymentVerifier"]
