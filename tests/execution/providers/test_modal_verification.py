from __future__ import annotations

from dataclasses import replace

import pytest

from tests.execution.providers.test_modal_source_resolution import _deployment
from tuner.execution.providers.modal.binding import CapabilityProofV1
from tuner.execution.providers.modal.verification import ModalSdkDeploymentVerifier


class Auth:
    def sign(self,purpose,payload,key_ref):return b"authenticated-tag"
    def verify(self,purpose,payload,tag,key_ref):return tag==b"authenticated-tag"


class Facade:
    def __init__(self,selection):self.selection=selection;self.proof=CapabilityProofV1(True,True,True,True,True,True,True)
    def bound_scope(self):return ("acct","workspace","env","client")
    def capability_proof(self,binding):return self.proof
    def inspect_deployment(self,*,app_name,function_name):return self.selection


def verifier(facade):return ModalSdkDeploymentVerifier(facade,Auth(),clock=lambda:"2026-08-25T12:00:00Z",audience_ref="project/run-1",issuer_ref="modal-verifier",key_ref="modal-key",challenge_factory=lambda:"deployment-challenge",evidence_ref_factory=lambda:"deployment-evidence")


def test_modal_verifier_seals_exact_provider_observation():
    selection=_deployment();evidence=verifier(Facade(selection)).verify(selection)
    assert evidence.selection==selection and evidence.tag==b"authenticated-tag" and evidence.audience_ref=="project/run-1"


def test_modal_verifier_rejects_scope_capability_and_deployment_drift():
    selection=_deployment();facade=Facade(selection);facade.bound_scope=lambda:("other","workspace","env","client")
    with pytest.raises(ValueError,match="unverifiable"):verifier(facade).verify(selection)
    facade=Facade(selection);facade.proof=replace(facade.proof,image_identity=False)
    with pytest.raises(ValueError,match="unverifiable"):verifier(facade).verify(selection)
    with pytest.raises(ValueError,match="unverifiable"):verifier(Facade(replace(selection,function_version="v2"))).verify(selection)
