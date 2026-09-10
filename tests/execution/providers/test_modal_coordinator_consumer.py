"""Provider-free proof of the inactive Modal coordinator composition."""
from __future__ import annotations

import base64
import hashlib
from dataclasses import replace

from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.training_facade import TrainingRequest
from tuner.execution.coordinator_v1.coordinator import TrainingCoordinatorV1
from tuner.execution.coordinator_v1.foundation import ComposedEffectFoundationV1, FoundationRecordAssessmentAuthorityV1
from tuner.execution.coordinator_v1.stores import InMemoryExecutionGrantStoreV1, InMemoryPreparationStoreV1, InMemoryReconciliationGrantStoreV1, InMemoryWorkflowStoreV1
from tuner.execution.foundation_v2.broker import EffectBrokerV2
from tuner.execution.foundation_v2.canonical import canonical_bytes, parse_canonical_object
from tuner.execution.foundation_v2.reconciliation import ReconciliationServiceV1
from tuner.execution.providers.modal.coordinator_adapter import ModalPreparationAdapter
from tuner.execution.providers.modal.coordinator_effects import ModalFoundationEffectExecutor, ModalFoundationReconciliationAdapter
from tuner.execution.providers.modal.coordinator_factories import modal_coordinator_registration
from tuner.execution.providers.modal.coordinator_preflight import AuthenticatedModalQuote, ModalOperationalPreflightAdapter, QUOTE_PURPOSE, TrustedEvidenceIdentity
from tuner.execution.providers.modal.coordinator_reader import ModalCoordinatorRunReader
from tuner.execution.providers.modal.coordinator_retention import ModalFoundationRetentionDelegate, ModalRetainedPreparation
from tuner.execution.providers.modal.coordinator_transport import ModalFoundationHostTransport
from tuner.execution.providers.modal.facade import ExplicitModal154ReadFacade
from tuner.execution.foundation_v2.registry import LazyProviderRegistryV2
from tuner.project.execution_source import ExecutionSourceV1
from tuner.training.coordinator_service import CoordinatorTrainingService

from tests.execution.coordinator_v1 import test_start_reconcile_service as generic
from tests.execution.foundation_v2.helpers import AdapterResolver, environment
from tests.execution.providers import test_modal_coordinator_bundle as bundle_tests
from tests.execution.providers.test_modal_coordinator_adapter import inputs
from tests.execution.providers.test_modal_coordinator_bundle import _fixture
from tests.execution.providers.test_modal_coordinator_launch import Authenticator
from tests.execution.providers.test_modal_coordinator_preflight import body, evidence_tag
from tests.execution.providers.test_modal_coordinator_transport import SDK, Volume


class Catalog:
    def __init__(self): self.values = {}
    def resolve(self, key): return self.values.get(key)
    def publish_if_absent(self, key, value): self.values.setdefault(key, value)


class BindingAuthority:
    def __init__(self, catalog): self.catalog = catalog
    def authenticate(self, value):
        return any(type(item) is type(value) and item.canonical_bytes == value.canonical_bytes
                   for item in self.catalog.values.values())


class Inputs:
    def __init__(self, value): self.value = value
    def resolve(self, digest):
        return self.value


class NoCalls:
    def __getattr__(self, name):
        def unavailable(*args, **kwargs): raise AssertionError(name)
        return unavailable


class DeferredTransport:
    def __init__(self): self.target = None; self.calls = []
    def bind(self, target):
        assert self.target is None
        self.target = target
    def execute_once(self, binding, command):
        assert self.target is not None
        self.calls.append(type(command).__name__)
        return self.target.execute_once(binding, command)
    def lookup_once(self, binding, command):
        assert self.target is not None
        return self.target.lookup_once(binding, command)


class PlanningStore:
    def __init__(self): self.plans = {}; self.contexts = {}
    def put_plan_if_absent(self, value): return self.plans.setdefault(value.plan_fingerprint, value) is value
    def get_plan(self, key): return self.plans.get(key)
    def put_context_if_absent(self, value): return self.contexts.setdefault(value.provider_context_digest, value) is value
    def get_context(self, key): return self.contexts.get(key)


class Clock:
    def now(self): return "2026-08-25T12:02:00Z"
    def now_iso(self): return self.now()
    def now_epoch(self): return 150


def _material(monkeypatch):
    original_source, original_verified = bundle_tests._execution_source, bundle_tests.verified
    def source():
        value = original_source(); evidence = value.source_evidence
        evidence = replace(evidence, tag_base64=base64.b64encode(evidence_tag(
            "source-lock-evidence/v1", evidence.authenticated_payload, evidence.key_ref,
        )).decode(), attestation_digest=hashlib.sha256(evidence.authenticated_payload).hexdigest())
        return replace(value, source_evidence=evidence)
    def deployment(selection):
        value = original_verified(selection).to_dict(); value["expires_at"] = "2026-08-25T12:05:00Z"
        unsigned = dict(value); unsigned.pop("tag_base64"); unsigned.pop("attestation_digest")
        payload = canonical_bytes(unsigned)
        value["tag_base64"] = base64.b64encode(evidence_tag("modal-deployment-evidence/v1", payload, value["key_ref"])).decode()
        value["attestation_digest"] = hashlib.sha256(payload).hexdigest()
        return type(original_verified(selection)).from_dict(value)
    with monkeypatch.context() as scoped:
        scoped.setattr(bundle_tests, "_execution_source", source)
        scoped.setattr(bundle_tests, "verified", deployment)
        return _fixture()


def _preflight(monkeypatch, template, material):
    values = inputs()
    first = ModalPreparationAdapter(profile=values["profile"], binding=values["binding"],
        resolved=material.planning_request, runtime_environment=values["runtime_environment"],
        quote_digest="0"*64, timeout_seconds=values["timeout_seconds"], clock=Clock())
    _, execution, _ = first._snapshot()
    quote_bytes = body(profile_ref=values["profile"].profile, account_ref=execution.scope.account_ref,
        namespace_ref=execution.scope.namespace_ref, resource_digest=execution.resource_digest,
        issued_at="2026-08-25T12:01:00Z", expires_at="2026-08-25T12:05:00Z")
    quote = AuthenticatedModalQuote(quote_bytes, evidence_tag(QUOTE_PURPOSE, quote_bytes, "quote-key"))
    preparation = ModalPreparationAdapter(profile=values["profile"], binding=values["binding"],
        resolved=material.planning_request, runtime_environment=values["runtime_environment"],
        quote_digest=quote.body.quote_digest, timeout_seconds=values["timeout_seconds"], clock=Clock())
    selection = template.deployment.selection
    Volume.registry = {values["profile"].control_volume_ref: Volume("cv"), values["profile"].artifact_volume_ref: Volume("av")}
    Volume.calls = []
    class Secret:
        @classmethod
        def from_name(cls, name, **kwargs):
            value=cls(); value.name=name; value.object_id="st-one"; value.is_hydrated=False; return value
        def hydrate(self, client): self.is_hydrated=True; return self
    sdk = type("SDK", (), {"__version__":"1.5.4", "Volume":Volume, "Function":SDK.Function,
        "FunctionCall":SDK.FunctionCall, "Secret":Secret, "exception":SDK.exception})
    facade = ExplicitModal154ReadFacade(values["binding"], sdk=sdk, client=object(),
        scope_observer=lambda client:(selection.account_ref,selection.workspace_ref,selection.environment_ref,selection.client_ref),
        deployment_observer=lambda **kwargs:selection,
        volume_names={"cv":values["profile"].control_volume_ref,"av":values["profile"].artifact_volume_ref})
    source = ExecutionSourceV1.from_dict(parse_canonical_object(material.execution_source_bytes, name="source"))
    class Auth:
        def sign(self,purpose,payload,key_ref): return evidence_tag(purpose,payload,key_ref)
        def verify(self,purpose,payload,tag,key_ref): return tag == evidence_tag(purpose,payload,key_ref)
    operational = ModalOperationalPreflightAdapter(preparation,
        execution_source_bytes=material.execution_source_bytes, deployment_bytes=template.deployment_bytes,
        quote=quote, control_volume_id="cv", artifact_volume_id="av", facade=facade,
        authenticator=Auth(), clock=Clock(),
        source_trust=TrustedEvidenceIdentity(source.source_evidence.issuer_ref,source.source_evidence.key_ref,source.source_evidence.audience_ref),
        deployment_trust=TrustedEvidenceIdentity(template.deployment.issuer_ref,template.deployment.key_ref,template.deployment.audience_ref),
        quote_trust=TrustedEvidenceIdentity(quote.body.issuer_ref,quote.body.key_ref,quote.body.audience_ref))
    return preparation, operational, facade


def test_inactive_consumer_composes_fresh_stage_and_spawn_once(monkeypatch):
    template, material, recipes, policy, closure = _material(monkeypatch)
    preparation, planning, facade = _preflight(monkeypatch, template, material)
    context, binding, plan = preparation._snapshot(); run = TrainingRunRef(material.run_id, material.planning_request.project_ref)
    for name,value in {"PROVIDER":context.provider,"CONTEXT":context,"PLAN":plan,"DESC":preparation.describe(context.provider),"RUN":run,"SCOPE":binding.scope}.items():
        monkeypatch.setattr(generic,name,value)
    bindings, stages, launches = Catalog(), Catalog(), Catalog(); authority = BindingAuthority(bindings)
    deferred = DeferredTransport()
    executor = ModalFoundationEffectExecutor(profile_ref=binding.provider.profile_ref, account_ref=binding.scope.account_ref,
        namespace_ref=binding.scope.namespace_ref, catalog=bindings, authority=authority, transport=deferred)
    lookup = ModalFoundationReconciliationAdapter(profile_ref=binding.provider.profile_ref, account_ref=binding.scope.account_ref,
        namespace_ref=binding.scope.namespace_ref, catalog=bindings, authority=authority, transport=deferred)
    repository, grants, receipts, invalid, verifier, executor_resolver = environment(executor)
    broker = EffectBrokerV2(repository, executor_resolver, grants, receipts, invalid)
    reconciliation = ReconciliationServiceV1(repository, grants, AdapterResolver(lookup), receipts, invalid)
    assessments = FoundationRecordAssessmentAuthorityV1("assessment-authority","assessment-key",b"a"*32,
        assessor_ref="foundation-assessor",assessor_version="1.0.0",clock=Clock(),receipt_authority=receipts,
        invalid_evidence_authority=invalid,grant_authority=grants)
    foundation = ComposedEffectFoundationV1(repository,broker,reconciliation,grant_authority=grants,
        receipt_authority=receipts,invalid_evidence_authority=invalid,assessment_authority=assessments,
        trusted_quiescence_evidence=generic.TrustedEvidence(repository,verifier))
    authenticator = generic.FoundationAuthenticator(grants,receipts,invalid); signer=Authenticator()
    retained = ModalRetainedPreparation(preparation.snapshot(),template.deployment_bytes,material.canonical_bytes,
        recipes,policy,closure,"cv","av","stage-key")
    wrapped = ModalFoundationRetentionDelegate(foundation,foundation_authenticator=authenticator,
        assessment_authority=assessments,binding_authority=authority,stage_authority=signer,launch_authority=signer,
        binding_catalog=bindings,stage_catalog=stages,launch_catalog=launches,retained_inputs=Inputs(retained))
    transport = ModalFoundationHostTransport(facade=facade,deployment=template.deployment,stage_source=stages,
        launch_source=launches,binding_authority=authority,foundation_authenticator=authenticator,
        assessment_authenticator=assessments,stage_verifier=signer,launch_verifier=signer,recipes=recipes)
    assert deferred.calls == []; deferred.bind(transport)
    authorization = generic.Authorization(grants, executor, lookup.descriptor)
    executor.effects={}; executor.commands={}
    workflows=InMemoryWorkflowStoreV1(authenticator,assessments,assessments,assessments)
    store=PlanningStore()
    coordinator = TrainingCoordinatorV1(planning,store,workflows,InMemoryPreparationStoreV1(),
        InMemoryExecutionGrantStoreV1(grants),InMemoryReconciliationGrantStoreV1(grants),preparation,preparation,
        authorization,wrapped,authenticator,Clock(),generic.Identity())
    class Loader:
        def load(self,text): return TrainingRequest(material.planning_request.request_id,material.planning_request.project_ref,text)
    class Resolver:
        def resolve(self,request): return material.planning_request
    service=CoordinatorTrainingService(loader=Loader(),resolver=Resolver(),planning=planning,planning_store=store,
        coordinator=coordinator,clock=Clock())
    request=service.load('{"method":"sft"}'); resolved=service.resolve(request); planned=service.plan(resolved,context.provider)
    ready=service.preflight(planned); started=service.start(planned,ready)
    assert started.accepted is True
    assert deferred.calls == ["StageCommandV2","SubmitCommandV2"]
    from tests.execution.providers.test_modal_coordinator_transport import Function
    assert len(Function.spawns) == 1 and sum(len(volume.files) for volume in Volume.registry.values()) == 3
    restarted=CoordinatorTrainingService(loader=Loader(),resolver=Resolver(),planning=planning,planning_store=store,
        coordinator=coordinator,clock=Clock())
    assert restarted.start(planned,ready) == started
    assert deferred.calls == ["StageCommandV2","SubmitCommandV2"] and len(Function.spawns) == 1
    reader = ModalCoordinatorRunReader(
        catalog=NoCalls(), binding_authority=NoCalls(),
        foundation_authenticator=NoCalls(), assessment_authenticator=NoCalls(),
        evidence_authority=NoCalls(), transport=NoCalls(),
        observed_at="2026-08-25T12:02:00Z",
    )
    registration = modal_coordinator_registration(preparation,executor,lookup,reader)
    registry=LazyProviderRegistryV2(); registry.register(registration)
    assert registry.list() == (registration.provider,)
    assert not any(registration.provider.capabilities.to_dict().values())
