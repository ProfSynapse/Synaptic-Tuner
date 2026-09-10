"""Concrete construction and execution proof for Modal coordinator composition."""
from dataclasses import replace

import pytest
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.training_facade import TrainingRequest
from synaptic_tuner.api.v1.host import APIHost, HostPorts
from synaptic_tuner.api.v1.runs_facade import RunArtifactRequest, RunListRequest, RunLogsRequest, RunOperationError
from tuner.execution.coordinator_v1.foundation import FoundationRecordAssessmentAuthorityV1
from tuner.execution.coordinator_v1.stores import InMemoryExecutionGrantStoreV1, InMemoryPreparationStoreV1, InMemoryReconciliationGrantStoreV1, InMemoryWorkflowStoreV1
from tuner.execution.foundation_v2.registry import LazyProviderRegistryV2, ProviderReaderFactoryRequestV1, ResolvedProviderReaderV1
from tuner.execution.providers.modal.coordinator_composition import (
    ModalCoordinatorStorePorts, ModalFoundationCompositionPorts,
    compose_modal_coordinator,
)
from tuner.execution.providers.modal.coordinator_retention import ModalRetainedPreparation
from tuner.execution.providers.modal.facade import ExplicitModal154ReadFacade
from tuner.execution.providers.modal.coordinator_preflight import AuthenticatedModalQuote
from tuner.project.errors import SourceLockError

from tests.execution.coordinator_v1 import test_start_reconcile_service as generic
from tests.execution.foundation_v2.helpers import environment
from tests.execution.providers.test_modal_coordinator_consumer import (
    BindingAuthority, Catalog, Clock, Inputs, PlanningStore,
    _material, _preflight,
)
from tests.execution.providers.test_modal_coordinator_launch import Authenticator
from tests.execution.providers.test_modal_coordinator_preflight import body, evidence_tag
from tests.execution.providers.test_modal_coordinator_transport import Function, Volume
from tests.execution.providers.test_modal_sdk154_adapter import verified


class Authorization:
    def __init__(self, grants): self.grants, self.effect_issues = grants, 0
    def commit_preflight(self, plan, preflight): return "7" * 64
    def issue_effect_grant(self, command_bytes, *, preflight_digest, now_epoch):
        from tuner.execution.foundation_v2.commands import parse_exact_command
        command=parse_exact_command(command_bytes); self.effect_issues += 1
        return self.grants.issue(command_bytes, grant_ref=f"grant-{command.operation.effect.kind.value}",
            policy_digest=preflight_digest, requirement_digest="8"*64,
            not_before_epoch=100, expires_at_epoch=200)
    def issue_reconciliation_grant(self, *args, **kwargs): raise AssertionError("not reconciled")


class Loader:
    def __init__(self, material): self.material=material
    def load(self, text): return TrainingRequest(self.material.planning_request.request_id,
        self.material.planning_request.project_ref, text)


class Resolver:
    def __init__(self, material): self.material=material
    def resolve(self, request): return self.material.planning_request


class EvidenceAuthority:
    def observation(self, content): raise AssertionError("reader not invoked")
    def log_page(self, content): raise AssertionError("reader not invoked")


class ReadAuthenticator:
    def authenticate(self, value): raise AssertionError("inactive provider read")


class ArtifactVerifier:
    def verify(self, *args): raise AssertionError("inactive provider read")
    def replay(self, *args): raise AssertionError("inactive provider read")
    def authenticate(self, *args): raise AssertionError("inactive provider read")


class CursorAuthority:
    def issue(self, value): raise AssertionError("single-page fixture")
    def verify(self, value): raise AssertionError("cursor-free fixture")


def composition_case(monkeypatch):
    template, material, recipes, policy, closure = _material(monkeypatch)
    preparation, operational, facade = _preflight(monkeypatch, template, material)
    clock=operational._clock; preparation._clock=clock
    context,binding,plan=preparation._snapshot()
    for name,value in {"PROVIDER":context.provider,"CONTEXT":context,"PLAN":plan,
        "DESC":preparation.describe(context.provider),"RUN":TrainingRunRef(material.run_id,material.planning_request.project_ref),
        "SCOPE":binding.scope}.items(): monkeypatch.setattr(generic,name,value)
    placeholder=generic.DynamicExecutor(binding.executor_descriptor,{kind:None for kind in ("stage","submit","cancel")})
    repository,grants,receipts,invalid,verifier,_=environment(placeholder)
    assessments=FoundationRecordAssessmentAuthorityV1("assessment-authority","assessment-key",b"a"*32,
        assessor_ref="foundation-assessor",assessor_version="1.0.0",clock=clock,
        receipt_authority=receipts,invalid_evidence_authority=invalid,grant_authority=grants)
    authenticator=generic.FoundationAuthenticator(grants,receipts,invalid)
    bindings,stages,launches=Catalog(),Catalog(),Catalog(); signer=Authenticator()
    retained=ModalRetainedPreparation(preparation.snapshot(),template.deployment_bytes,material.canonical_bytes,
        recipes,policy,closure,"cv","av","stage-key")
    ports=ModalFoundationCompositionPorts(repository,grants,receipts,invalid,assessments,
        authenticator,generic.TrustedEvidence(repository,verifier),BindingAuthority(bindings),
        signer,signer,bindings,stages,launches,Inputs(retained))
    stores=ModalCoordinatorStorePorts(PlanningStore(),
        InMemoryWorkflowStoreV1(authenticator,assessments,assessments,assessments),
        InMemoryPreparationStoreV1(),InMemoryExecutionGrantStoreV1(grants),
        InMemoryReconciliationGrantStoreV1(grants))
    return dict(preparation=preparation,operational_preflight=operational,facade=facade,
        deployment=template.deployment,recipes=recipes,evidence_authority=EvidenceAuthority(),
        evidence_verifier=signer,observed_at="2026-08-25T12:02:00Z",loader=Loader(material),
        observation_authenticator=ReadAuthenticator(),log_authenticator=ReadAuthenticator(),
        artifact_verifier=ArtifactVerifier(),cursor_authority=CursorAuthority(),
        resolver=Resolver(material),authorization=Authorization(grants),clock=clock,
        run_identity=generic.Identity(),foundation_ports=ports,stores=stores)


def test_factory_is_provider_io_free_then_stages_and_spawns_once(monkeypatch):
    monkeypatch.setattr(Function, "spawns", [])
    monkeypatch.setattr(Function, "resolutions", [])
    template, material, recipes, policy, closure = _material(monkeypatch)
    preparation, operational, facade = _preflight(monkeypatch, template, material)
    # The local fixture historically allocated equivalent clocks separately;
    # composition requires the one retained instance used by both adapters.
    clock = operational._clock
    preparation._clock = clock
    context, binding, plan = preparation._snapshot()
    run=TrainingRunRef(material.run_id, material.planning_request.project_ref)
    for name,value in {"PROVIDER":context.provider,"CONTEXT":context,"PLAN":plan,
        "DESC":preparation.describe(context.provider),"RUN":run,"SCOPE":binding.scope}.items():
        monkeypatch.setattr(generic,name,value)
    # Build the consumer-owned Foundation authorities without an executor; the
    # factory supplies its reviewed Modal executor to the broker.
    placeholder=generic.DynamicExecutor(binding.executor_descriptor, {kind:None for kind in ("stage","submit","cancel")})
    repository, grants, receipts, invalid, verifier, _ = environment(placeholder)
    assessments=FoundationRecordAssessmentAuthorityV1("assessment-authority","assessment-key",b"a"*32,
        assessor_ref="foundation-assessor",assessor_version="1.0.0",clock=Clock(),
        receipt_authority=receipts,invalid_evidence_authority=invalid,grant_authority=grants)
    authenticator=generic.FoundationAuthenticator(grants,receipts,invalid)
    bindings,stages,launches=Catalog(),Catalog(),Catalog(); binding_authority=BindingAuthority(bindings)
    signer=Authenticator()
    retained=ModalRetainedPreparation(preparation.snapshot(),template.deployment_bytes,
        material.canonical_bytes,recipes,policy,closure,"cv","av","stage-key")
    planning_store=PlanningStore()
    before=(len(Volume.calls),len(Function.resolutions),len(Function.spawns))
    composed=compose_modal_coordinator(
        preparation=preparation,operational_preflight=operational,facade=facade,
        deployment=template.deployment,recipes=recipes,
        evidence_authority=EvidenceAuthority(), evidence_verifier=signer,
        observation_authenticator=ReadAuthenticator(),
        log_authenticator=ReadAuthenticator(), artifact_verifier=ArtifactVerifier(),
        cursor_authority=CursorAuthority(),
        observed_at="2026-08-25T12:02:00Z",
        loader=Loader(material),resolver=Resolver(material),authorization=Authorization(grants),
        clock=clock,run_identity=generic.Identity(),
        foundation_ports=ModalFoundationCompositionPorts(repository,grants,receipts,invalid,
            assessments,authenticator,generic.TrustedEvidence(repository,verifier),binding_authority,
            signer,signer,bindings,stages,launches,Inputs(retained)),
        stores=ModalCoordinatorStorePorts(planning_store,
            InMemoryWorkflowStoreV1(authenticator,assessments,assessments,assessments),
            InMemoryPreparationStoreV1(),InMemoryExecutionGrantStoreV1(grants),
            InMemoryReconciliationGrantStoreV1(grants)),
    )
    assert (len(Volume.calls),len(Function.resolutions),len(Function.spawns)) == before
    request=composed.training.load('{"method":"sft"}')
    resolved=composed.training.resolve(request)
    planned=composed.training.plan(resolved,context.provider)
    ready=composed.training.preflight(planned)
    first=composed.training.start(planned,ready)
    assert first.accepted is True and len(Function.spawns)==1
    assert sum(len(volume.files) for volume in Volume.registry.values()) == 3
    assert composed.training.start(planned,ready) == first and len(Function.spawns)==1
    host=APIHost(composed.training,HostPorts(composed.runs,clock))
    assert host.runs.show(first.run).run == first.run
    page=host.runs.list(RunListRequest(first.run.project_ref))
    assert tuple(item.run for item in page.outcomes) == (first.run,)
    with pytest.raises(RunOperationError): host.runs.outcome(first.run)
    with pytest.raises(RunOperationError): host.runs.logs(RunLogsRequest(first.run))
    with pytest.raises(RunOperationError): host.runs.artifacts(RunArtifactRequest(first.run,"final_model",1))
    registry=LazyProviderRegistryV2(); registry.register(composed.registration)
    assert registry.list()==(composed.registration.provider,)
    assert not any(composed.registration.provider.capabilities.to_dict().values())
    reader_request=ProviderReaderFactoryRequestV1(
        context.provider, composed.registration.provider.descriptor_digest,
        binding.profile_digest, binding.scope.account_ref, binding.scope.namespace_ref,
    )
    resolved_reader=registry.resolve_reader(reader_request)
    assert type(resolved_reader) is ResolvedProviderReaderV1
    assert resolved_reader.reader._catalog is bindings
    assert resolved_reader.reader._transport._facade is facade


def test_factory_rejects_preflight_from_another_preparation_before_provider_io(monkeypatch):
    monkeypatch.setattr(Function, "spawns", [])
    monkeypatch.setattr(Function, "resolutions", [])
    template, material, recipes, _, _ = _material(monkeypatch)
    preparation, operational, facade = _preflight(monkeypatch, template, material)
    clock = operational._clock; preparation._clock = clock
    other, _, _ = _preflight(monkeypatch, template, material)
    before=(len(Volume.calls),len(Function.resolutions),len(Function.spawns))
    import pytest
    with pytest.raises(ValueError, match="exact preparation"):
        compose_modal_coordinator(preparation=other,operational_preflight=operational,
            facade=facade,deployment=template.deployment,recipes=recipes,
            evidence_authority=object(),evidence_verifier=object(),
            observation_authenticator=object(),log_authenticator=object(),
            artifact_verifier=object(),cursor_authority=object(),
            observed_at="2026-08-25T12:02:00Z",
            loader=object(),resolver=object(),authorization=object(),clock=clock,
            run_identity=object(),foundation_ports=object(),stores=object())
    assert (len(Volume.calls),len(Function.resolutions),len(Function.spawns)) == before


def test_static_port_validation_does_not_invoke_hostile_properties():
    from tuner.execution.providers.modal.coordinator_composition import _methods
    import pytest
    touched=[]
    class Hostile:
        @property
        def get(self): touched.append(True); raise AssertionError
    with pytest.raises(TypeError, match="incomplete"):
        _methods(Hostile(), "get")
    assert touched == []


@pytest.mark.parametrize("mode", [
    "facade", "clock", "deployment", "source", "quote",
    "repository", "grants", "receipts", "invalid", "planning_store",
    "workflow_store", "preparation_store", "execution_store",
    "reconciliation_store", "evidence_authority", "evidence_verifier",
    "observation_authenticator", "log_authenticator",
    "artifact_missing_verify", "artifact_missing_replay", "artifact_missing_authenticate",
    "cursor_missing_issue", "cursor_missing_verify",
])
def test_composition_substitutions_fail_before_provider_io(monkeypatch, mode):
    arguments=composition_case(monkeypatch)
    if mode == "facade":
        original=arguments["facade"]
        arguments["facade"]=ExplicitModal154ReadFacade(
            original.binding,sdk=original.sdk,client=object(),
            scope_observer=lambda client:(),deployment_observer=lambda **kwargs:None,
            volume_names=dict(original._volume_names),
        )
    elif mode == "clock": arguments["clock"] = Clock()
    elif mode == "deployment":
        arguments["deployment"] = verified(replace(
            arguments["deployment"].selection, environment_ref="alien-environment",
        ))
    elif mode == "source": arguments["operational_preflight"]._source_bytes = b"{}"
    elif mode == "quote":
        raw=body(profile_ref="different-profile")
        arguments["operational_preflight"]._quote=AuthenticatedModalQuote(
            raw,evidence_tag("modal-quote-evidence/v1",raw,"quote-key"),
        )
    elif mode in {"repository","grants","receipts","invalid"}:
        field={"repository":"effect_repository","grants":"grant_authority",
            "receipts":"receipt_authority","invalid":"invalid_evidence_authority"}[mode]
        arguments["foundation_ports"]=replace(arguments["foundation_ports"],**{field:object()})
    elif mode in {"planning_store","workflow_store","preparation_store","execution_store","reconciliation_store"}:
        field={"planning_store":"planning_store","workflow_store":"workflow_store",
            "preparation_store":"preparation_store","execution_store":"execution_grant_store",
            "reconciliation_store":"reconciliation_grant_store"}[mode]
        arguments["stores"]=replace(arguments["stores"],**{field:object()})
    elif mode == "evidence_authority": arguments["evidence_authority"] = object()
    elif mode == "evidence_verifier": arguments["evidence_verifier"] = object()
    elif mode == "observation_authenticator": arguments["observation_authenticator"] = object()
    elif mode == "log_authenticator": arguments["log_authenticator"] = object()
    elif mode.startswith("artifact_missing_"):
        missing=mode.removeprefix("artifact_missing_")
        members={name:(lambda self,*args: True) for name in {"verify","replay","authenticate"}-{missing}}
        arguments["artifact_verifier"]=type("PartialArtifactVerifier",(),members)()
    elif mode.startswith("cursor_missing_"):
        missing=mode.removeprefix("cursor_missing_")
        members={name:(lambda self,*args: True) for name in {"issue","verify"}-{missing}}
        arguments["cursor_authority"]=type("PartialCursorAuthority",(),members)()
    else: raise AssertionError(f"unhandled substitution mode: {mode}")
    before=(len(Volume.calls),len(Function.resolutions),len(Function.spawns))
    with pytest.raises((TypeError, ValueError, SourceLockError)):
        compose_modal_coordinator(**arguments)
    assert (len(Volume.calls),len(Function.resolutions),len(Function.spawns)) == before
