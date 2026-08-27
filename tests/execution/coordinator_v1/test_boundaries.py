import inspect
import sys
from dataclasses import replace
import pytest
from tuner.execution.coordinator_v1 import model, ports, state_machine
from tuner.execution.coordinator_v1.model import (AuthenticatedArtifactVerificationReceiptV1,
 BoundCancellationRefV1, FoundationEffectBindingV1, FoundationEffectOutcomeV1,
 WorkflowRecordV1)

def test_evidence_records_are_ordinary_exact_frozen_dataclasses():
 from .test_state_machine import planned
 assert planned().__dataclass_params__.frozen is True
 assert "_DerivationOwned" not in vars(model)

def test_only_named_mutation_api_is_public():
 exported=set(state_machine.__all__)
 assert "transition_workflow" not in exported
 assert "refine_effect_intent" not in exported
 assert not hasattr(WorkflowRecordV1,"_successor")
 assert not any(name.endswith("ISSUER") or name in {"_advance","_event","_apply_effect","_check_verification","_install_named_reducers","effect_reducer","successor","mint"} for name in vars(state_machine))
 assert not any("ISSUER" in name or "TOKEN" in name for name in vars(model))
 assert {"begin_preparation","record_stage_intent","apply_stage_effect_record",
         "record_submit_intent","apply_submit_effect_record","record_cancel_intent",
         "apply_cancel_effect_record","apply_provider_observation",
         "apply_artifact_verification","apply_reverification"} <= exported

def test_reader_and_verifier_ports_use_bound_and_authenticated_types():
 observe=inspect.signature(ports.ProviderRunReaderPortV1.observe)
 assert "ProviderRunReadRequestV1" in str(observe.parameters["request"].annotation)
 verify=inspect.signature(ports.ArtifactVerifierPortV1.verify)
 assert "AuthenticatedArtifactVerificationReceiptV1" in str(verify.return_annotation)
 assert hasattr(ports.FoundationEvidenceAuthenticatorPortV1,"authenticate_grant")

def test_coordinator_import_is_provider_sdk_and_storage_neutral():
 sources="\n".join(inspect.getsource(x) for x in (model,ports,state_machine)).lower()
 assert "sqlite" not in sources
 assert "huggingface_hub" not in sources and "modal" not in sources and "runpod" not in sources
 assert not any(name.startswith(("huggingface_hub","modal","runpod")) for name in sys.modules)

def test_package_root_remains_unexported():
 import tuner.execution.coordinator_v1 as package
 assert not hasattr(package,"WorkflowRecordV1")

def test_bound_references_expose_no_public_construction_factory():
 for cls in (model.BoundProviderStageRefV1,model.BoundProviderRunRefV1,model.BoundCancellationRefV1):
  assert not ({"bind","build","issue","parse","from_dict"}&set(vars(cls)))
 assert not ({"build","issue","from_dict"}&set(vars(model.ArtifactVerificationContentV1)))
 assert not ({"build","issue","from_dict"}&set(vars(model.AuthenticatedArtifactVerificationReceiptV1)))

def test_direct_evidence_replacement_has_no_authority_at_consumer():
 from .test_state_machine import D, queued
 with pytest.raises(ValueError): replace(queued().provider_run_ref,binding_digest=D[0])
