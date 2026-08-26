from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema.validators import validator_for

from synaptic_tuner.api.v1.execution import ExecutionGrant, RunState
from synaptic_tuner.api.v1.artifacts import ArtifactPublicationReceipt,PublishedArtifact
from synaptic_tuner.api.v1.host import APIHost,HostPorts
from synaptic_tuner.api.v1.training import (
    ArtifactPolicy,
    CanonicalDocument,
    ResourceSpec,
    RuntimeSpec,
    TrainingPlan,
)
from tests.contract.test_public_training_api_v1 import _execution_source
from tests.execution.fakes import InMemoryLifecycleRepository
from tests.execution.providers.test_modal_sdk154_adapter import FakeCall,FakeFunction,make_facade
from tests.execution.providers.test_modal_source_resolution import DeploymentPort
from tests.training.test_training_service import _resolved_config
from tuner.execution.contracts import (
    EventCode,
    GrantBinding,
    LifecycleEvent,
    LifecyclePhase,
    MessageCode,
)
from tuner.execution.lifecycle import apply_event
from tuner.execution.providers.modal.config import (
    ModalProviderProfileV1,
    ModalRuntimeLockV1,
    ModalSecretProfileV1,
)
from tuner.execution.providers.modal.deployment_identity import modal_function_name
from tuner.execution.providers.modal.contracts import canonical_json, sha
from tuner.execution.providers.modal.producer import MountedCompletionProducerV1
from tuner.execution.providers.modal.remote import ProcessResultV1,admit_remote_invocation
from tuner.execution.providers.modal.resolution import ModalDeploymentSelectionV1
from tuner.execution.providers.modal.training import (
    MODAL_PLAN_CONTEXT_SCHEMA,
    ModalDurablePreparationV1,
    ModalPlanContextV1,
    ModalTrainingOperations,
    _provider_runtime_requirements_digest,
    _resource_digest,
    _secret_requirements_digest,
    compose_modal_training_operations,
)
from tuner.project.context import ProjectContext
from tuner.training import ResolvedTrainingComponents,TrainingService, default_recipe_registry
from tuner.training.methods.sft import compile_sft_workload
from tuner.execution.broker import MutationCommandV1
from tuner.execution.providers.modal.contracts import ArtifactRole,StageReceiptV1
from tuner.runtime.verification import VerificationStatus as RuntimeVerificationStatus


NOW = "2026-08-25T12:03:00Z"


class Auth:
    def sign(self, purpose, payload, key_ref):
        return hashlib.sha256(
            purpose.encode() + b"\0" + key_ref.encode() + b"\0" + payload
        ).digest()

    def verify(self, purpose, payload, tag, key_ref):
        return tag == self.sign(purpose, payload, key_ref)


class Repository(InMemoryLifecycleRepository):
    def __init__(self):
        super().__init__(clock=lambda: NOW)
        self.preparations = {}

    def commit_modal_preparation(
        self, project_ref, run_id, *, expected_revision, occurred_at, preparation
    ):
        assert type(preparation) is ModalDurablePreparationV1
        key = (project_ref, run_id)
        old = self.records[key]
        assert old.revision == expected_revision
        event = LifecycleEvent(
            EventCode.PREPARATION_COMPLETED,
            occurred_at,
            MessageCode.READY,
        )
        new = apply_event(old, event)
        self.preparations[key] = preparation
        self.records[key] = new
        return new

    def load_modal_preparation(self, project_ref, run_id):
        return self.preparations.get((project_ref, run_id))

    def load_modal_preparation_by_effect(self, effect_id):
        return next(
            (
                value
                for value in self.preparations.values()
                if value.operation.effect.effect_id == effect_id
            ),
            None,
        )


class Grants:
    def bind(self, grant, *, operation, requirements):
        assert requirements[0].operation == "training.start"
        assert grant.grant_ref == operation.grant_ref
        return GrantBinding.from_operation(
            operation,
            issued_at="2026-08-25T12:02:00Z",
            expires_at="2026-08-25T12:10:00Z",
        )


class Resolver:
    def resolve(self, request, *, context):  # pragma: no cover - not used here
        raise AssertionError


class PlanningResolver:
    def __init__(self, plan):
        self.plan = plan

    def resolve(self, request, *, context):
        return ResolvedTrainingComponents(
            execution_source=self.plan.execution_source,
            execution_context=self.plan.execution_context,
            resolved_config=self.plan.resolved_config,
            runtime=self.plan.runtime,
            resources=self.plan.resources,
            artifact_policy=self.plan.artifact_policy,
        )


def selection() -> ModalDeploymentSelectionV1:
    deployment_ref = "modal-deployment-" + "1" * 32
    runtime_lock = ModalRuntimeLockV1.packaged()
    selected = ModalDeploymentSelectionV1(
        account_ref="acct",
        workspace_ref="workspace",
        environment_ref="env",
        client_ref="client",
        app_name="synaptic-training-v1",
        deployment_ref=deployment_ref,
        function_name=modal_function_name(deployment_ref),
        image_digest=runtime_lock.image_digest,
        dependency_lock_digest=runtime_lock.locked_digest("dependency_lock"),
        wrapper_digest=runtime_lock.locked_digest("deployment_wrapper"),
        runtime_digest=runtime_lock.locked_digest("sft_runtime"),
        python_version=runtime_lock.python_version,
        python_executable=runtime_lock.python_executable,
        python_executable_digest=runtime_lock.python_executable_digest,
        secret_requirements_digest=_secret_requirements_digest(profile()),
        provider_runtime_requirements_digest="4" * 64,
        runtime_environment={"PATH": "/opt/conda/bin"},
    )
    return replace(
        selected,
        provider_runtime_requirements_digest=_provider_runtime_requirements_digest(
            runtime_lock, profile(), selected
        ),
    )


def profile() -> ModalProviderProfileV1:
    deployment_ref = "modal-deployment-" + "1" * 32
    return ModalProviderProfileV1(
        "modal-a10-v1",
        "synaptic-training-v1",
        modal_function_name(deployment_ref),
        deployment_ref,
        "engine://tuner/execution/providers/modal/modal-runtime-v1.lock.json",
        "control-name",
        "artifact-name",
        (
            ModalSecretProfileV1(
                "runtime-secrets", ("HF_TOKEN", "SYNAPTIC_EVIDENCE_MAC_KEY")
            ),
        ),
    )


def plan_and_facade(selected=None):
    selected = selected or selection()
    verified = DeploymentPort().verify(selected)
    deployment_bytes = canonical_json(verified.to_dict())
    source = replace(
        _execution_source(),
        deployment_member_sha256=sha(deployment_bytes),
        python_version=selected.python_version,
        python_executable=selected.python_executable,
        python_executable_digest=selected.python_executable_digest,
        secret_requirements_digest=selected.secret_requirements_digest,
        provider_runtime_requirements_digest=selected.provider_runtime_requirements_digest,
    )
    workload = compile_sft_workload(
        resolved_config=_resolved_config(), execution_source=source
    )
    resources = ResourceSpec("A10", 1, 3600)
    provisional = TrainingPlan(
        source,
        CanonicalDocument.from_mapping({"placeholder": True}),
        _resolved_config(),
        CanonicalDocument(workload.canonical_bytes.decode()),
        RuntimeSpec(
            ModalRuntimeLockV1.packaged().registry_reference,
            selected.dependency_lock_digest,
            selected.python_version,
        ),
        resources,
        ArtifactPolicy(),
    )
    context = CanonicalDocument.from_mapping(
        {
            "schema_version": MODAL_PLAN_CONTEXT_SCHEMA,
            "project_ref": "project-1",
            "profile": "modal-a10-v1",
            "deployment": verified.to_dict(),
            "binding": {
                "account_ref": "acct",
                "workspace_ref": "workspace",
                "environment_ref": "env",
                "client_ref": "client",
                "sdk_version": "1.5.4",
            },
            "volumes": {"control_volume_id": "cv", "artifact_volume_id": "av"},
            "authority": {
                "key_ref": "evidence-key",
                "quote_digest": "7" * 64,
                "quote_expires_at": "2026-08-25T12:07:00Z",
                "maximum_cost_minor_units": 100,
                "currency": "USD",
            },
            "operation": {
                "effect_id": "effect-run-1",
                "effect_key": source.run_id,
                "artifact_slot_ref": "slot-run-1",
                "invocation_nonce": "nonce-run-1",
                "generation": 1,
            },
            "resource_digest": _resource_digest(provisional),
        }
    )
    plan = replace(provisional, execution_context=context)
    facade, _ = make_facade(selection=selected)
    return plan, facade


def operations(tmp_path, *, now=NOW, plan_facade=None, publisher=None):
    plan, facade = plan_facade or plan_and_facade()
    project = tmp_path / "host"
    engine = project / "vendor" / "engine"
    engine.mkdir(parents=True)
    project_context = ProjectContext.host(engine_root=engine, project_root=project)
    repository = Repository()
    repository.clock = lambda: now
    ports = HostPorts(
        lifecycle=repository,
        runs=object(),
        grants=Grants(),
        secrets=object(),
        evidence_replay=object(),
        authenticator=Auth(),
        clock=lambda: now,
        git_remote=object(),
        modal_reads=facade,
        training_resolver=Resolver(),
        artifact_publisher=publisher,
    )
    planning = TrainingService(
        context=project_context,
        resolver=Resolver(),
        recipes=default_recipe_registry(),
    )
    return (
        ModalTrainingOperations(
            planning=planning,
            context=project_context,
            ports=ports,
            profile=profile(),
        ),
        repository,
        plan,
    )


def test_start_uses_host_durability_then_one_spawn(tmp_path):
    value, repository, plan = operations(tmp_path)
    preflight = value.preflight(plan)
    assert preflight.ready

    submission = value.start(plan, preflight, ExecutionGrant("grant-run-1"))

    preparation = repository.load_modal_preparation("project-1", "run-1")
    assert type(preparation) is ModalDurablePreparationV1
    record = repository.load("project-1", "run-1")
    assert record.phase is LifecyclePhase.QUEUED
    assert record.effects[0].attempt_count == 1
    assert record.effects[0].provider_job_ref == "fc-1"
    assert value._expectations.load_terminal_expectation("effect-run-1").job_ref == "fc-1"
    assert submission.plan_fingerprint == plan.fingerprint
    assert value.outcome(submission).status.state is RunState.RUNNING


def test_modal_preparation_has_a_strict_canonical_persistence_round_trip(tmp_path):
    value, repository, plan = operations(tmp_path)
    value.start(plan, value.preflight(plan), ExecutionGrant("grant-run-1"))
    preparation = repository.load_modal_preparation("project-1", "run-1")
    assert ModalDurablePreparationV1.from_canonical_bytes(
        preparation.canonical_bytes
    ) == preparation
    with pytest.raises(ValueError, match="canonical"):
        ModalDurablePreparationV1.from_canonical_bytes(
            preparation.canonical_bytes + b" "
        )


def test_repeated_start_reuses_durable_operation_and_never_spawns_twice(tmp_path):
    value, repository, plan = operations(tmp_path)
    preflight = value.preflight(plan)
    first = value.start(plan, preflight, ExecutionGrant("grant-run-1"))
    second = value.start(plan, preflight, ExecutionGrant("grant-run-1"))
    assert second == first
    assert len(FakeFunction.spawn_calls) == 1
    assert repository.load("project-1", "run-1").effects[0].attempt_count == 1


def test_engine_defines_only_a_repository_protocol(tmp_path):
    value, repository, _ = operations(tmp_path)
    assert value._repository is repository
    assert not hasattr(value._repository, "database_path")


def test_returned_call_without_authenticated_terminal_requires_reconciliation(tmp_path):
    value, _, plan = operations(tmp_path)
    submission = value.start(
        plan, value.preflight(plan), ExecutionGrant("grant-run-1")
    )
    FakeCall.result = {
        "schema_version": "synaptic-modal-worker-result/v1",
        "effect_id": "effect-run-1",
        "returncode": 0,
        "status_code": "completed",
    }
    assert value.outcome(submission).status.state is RunState.RECONCILE_REQUIRED


def test_resolved_modal_plan_context_matches_checked_in_schema():
    plan, _ = plan_and_facade()
    root = Path(__file__).parents[3]
    schema = json.loads(
        (root / "schemas" / "synaptic-modal-plan-context-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    validator = validator_for(schema)
    validator.check_schema(schema)
    validator(schema).validate(plan.execution_context.to_dict())


def test_public_modal_context_owns_the_resource_digest_algorithm():
    resources = ResourceSpec("A10", 1, 3600)
    plan, _ = plan_and_facade()
    assert ModalPlanContextV1.digest_resources(resources) == _resource_digest(plan)


def test_expired_quote_or_deployment_cannot_produce_a_ready_preflight(tmp_path):
    value, _, plan = operations(tmp_path, now="2026-08-25T12:08:00Z")
    preflight = value.preflight(plan)
    assert not preflight.ready
    assert preflight.errors[0].code.value == "preflight_failed"


def test_public_api_composes_from_host_ports_without_a_provider_specific_verb(tmp_path):
    expected, facade = plan_and_facade()
    project = tmp_path / "host"
    engine = project / "vendor" / "engine"
    engine.mkdir(parents=True)
    context = ProjectContext.host(engine_root=engine, project_root=project)
    repository = Repository()
    ports = HostPorts(
        lifecycle=repository,
        runs=object(),
        grants=Grants(),
        secrets=object(),
        evidence_replay=object(),
        authenticator=Auth(),
        clock=lambda: NOW,
        git_remote=object(),
        modal_reads=facade,
        training_resolver=PlanningResolver(expected),
    )
    operations = compose_modal_training_operations(
        context=context,
        host_ports=ports,
        provider_config=profile(),
    )
    host = APIHost(operations, ports)
    request = host.training.load(CanonicalDocument.from_mapping({"method": "sft"}))
    assert host.training.plan(host.training.resolve(request)) == expected


def _publish_unrelated_completed_run(value, repository, tmp_path):
    preparation=repository.load_modal_preparation("project-1","run-1")
    expected=preparation.stage.expectation
    receipt=StageReceiptV1(expected.effect.effect_id,expected.operation_binding_digest,expected.control_volume_id,expected.artifact_volume_id,expected.claim_digest,expected.bundle_digest)
    command=MutationCommandV1.from_stage(preparation.operation,receipt)
    invocation=admit_remote_invocation(command.canonical_bytes,claim=preparation.stage.claim,claim_tag=preparation.stage.claim_tag,bundle_transport=preparation.stage.bundle,verifier=Auth())
    control=tmp_path/"remote-control";artifact_volume=tmp_path/"remote-artifact"
    runtime=artifact_volume/invocation.source.run_id
    roots=dict(invocation.source.roots)
    for name in ("artifacts","state","tracking","cache","tmp"):
        roots[name]=str(runtime/name);(runtime/name).mkdir(parents=True)
    environment=dict(invocation.source.environment)
    for name, variable in {
        "artifacts": "SYNAPTIC_ARTIFACT_ROOT",
        "state": "SYNAPTIC_STATE_ROOT",
        "tracking": "SYNAPTIC_TRACKING_ROOT",
        "cache": "SYNAPTIC_CACHE_ROOT",
        "tmp": "SYNAPTIC_TMP_ROOT",
    }.items():
        environment[variable]=roots[name]
    environment["HF_HOME"]=roots["cache"]+"/huggingface"
    environment["TRANSFORMERS_CACHE"]=roots["cache"]+"/transformers"
    invocation=replace(
        invocation,
        source=replace(
            invocation.source,
            roots=roots,
            writable_capability_root=str(artifact_volume),
            environment=environment,
        ),
    )
    records=[]
    for role in ArtifactRole:
        name=role.value+".bin";content=("unrelated-"+role.value).encode();(runtime/"artifacts"/name).write_bytes(content);records.append({"role":role.value,"path":name,"sha256":hashlib.sha256(content).hexdigest(),"size":len(content)})
    workload_fingerprint=hashlib.sha256(b"synaptic-training-workload/v1\0"+invocation.workload).hexdigest()
    (runtime/"state"/"runtime-v1-inventory.json").write_bytes(canonical_json({"schema_version":"synaptic-artifact-inventory/v1","workload_fingerprint":workload_fingerprint,"artifacts":records}))
    MountedCompletionProducerV1(Auth(),control_root=str(control),artifact_root=str(artifact_volume)).finalize(invocation,ProcessResultV1(0),job_ref="fc-1")
    from tests.execution.providers.test_modal_sdk154_adapter import FakeVolume
    for root,volume_name in ((control,"control-name"),(artifact_volume,"artifact-name")):
        for path in root.rglob("*"):
            if path.is_file():FakeVolume.registry[volume_name].files[path.relative_to(root).as_posix()]=path.read_bytes()


def test_structurally_valid_but_unrelated_artifacts_cannot_succeed(tmp_path):
    value,repository,plan=operations(tmp_path)
    submission=value.start(plan,value.preflight(plan),ExecutionGrant("grant-run-1"))
    _publish_unrelated_completed_run(value,repository,tmp_path)
    outcome=value.outcome(submission)
    assert outcome.status.state is RunState.FAILED
    assert not outcome.artifacts and not outcome.success


def test_explicit_reverify_can_correct_a_terminal_false_negative(tmp_path):
    value,repository,plan=operations(tmp_path)
    submission=value.start(plan,value.preflight(plan),ExecutionGrant("grant-run-1"))
    _publish_unrelated_completed_run(value,repository,tmp_path)
    value._verify_semantics=lambda preparation,manifest: RuntimeVerificationStatus.INVALID
    assert value.outcome(submission).status.state is RunState.FAILED
    value._verify_semantics=lambda preparation,manifest: RuntimeVerificationStatus.VERIFIED
    corrected=value.reverify(submission)
    assert corrected.status.state is RunState.SUCCEEDED
    assert corrected.success and len(corrected.artifacts)==5
    record=repository.load("project-1","run-1")
    assert record.events[-2].code is EventCode.VERIFICATION_REOPENED
    assert record.events[-1].code is EventCode.VERIFICATION_VERIFIED


def test_publish_delegates_verified_bytes_to_the_host_owned_destination(tmp_path):
    class Publisher:
        def __init__(self):self.calls=[]
        def publish(self,source,destination_ref):
            values={item.kind:b"".join(source.iter_bytes(item.kind,maximum=item.size)) for item in source.artifacts}
            self.calls.append((source.run,source.plan_fingerprint,destination_ref,values))
            return ArtifactPublicationReceipt(
                source.run,source.plan_fingerprint,destination_ref,
                tuple(PublishedArtifact(item.kind,f"memory://{destination_ref}/{item.kind}",item.sha256,item.size) for item in source.artifacts),
            )
    publisher=Publisher();value,repository,plan=operations(tmp_path,publisher=publisher)
    submission=value.start(plan,value.preflight(plan),ExecutionGrant("grant-run-1"))
    _publish_unrelated_completed_run(value,repository,tmp_path)
    value._verify_semantics=lambda preparation,manifest: RuntimeVerificationStatus.VERIFIED
    assert value.outcome(submission).success
    receipt=value.publish(submission,"local-test")
    assert receipt.destination_ref=="local-test" and len(receipt.artifacts)==5
    assert len(publisher.calls)==1 and set(publisher.calls[0][3])=={role.value for role in ArtifactRole}


def test_transient_completion_read_becomes_inconclusive_not_invalid(tmp_path):
    value,repository,plan=operations(tmp_path)
    submission=value.start(plan,value.preflight(plan),ExecutionGrant("grant-run-1"))
    _publish_unrelated_completed_run(value,repository,tmp_path)
    from tests.execution.providers.test_modal_sdk154_adapter import FakeVolume
    manifest="operations/effect-run-1/evidence/completion-manifest.v1.json"
    saved=FakeVolume.registry["control-name"].files.pop(manifest)
    outcome=value.outcome(submission)
    assert outcome.status.state is RunState.RECONCILE_REQUIRED
    assert repository.load("project-1","run-1").verification.value=="inconclusive"
    FakeVolume.registry["control-name"].files[manifest]=saved
    assert value.outcome(submission).status.state is RunState.FAILED


@pytest.mark.parametrize(
    "field,value",
    (
        ("image_digest", "a" * 64),
        ("dependency_lock_digest", "a" * 64),
        ("wrapper_digest", "a" * 64),
        ("runtime_digest", "a" * 64),
        ("python_version", "3.11.13"),
        ("python_executable", "/usr/bin/python3"),
        ("python_executable_digest", "a" * 64),
    ),
)
def test_packaged_runtime_lock_rejects_each_critical_substitution(field,value):
    with pytest.raises(ValueError,match="packaged runtime lock"):
        ModalRuntimeLockV1.packaged().validate_selection(
            replace(selection(),**{field:value})
        )


def test_composed_preflight_rejects_self_consistent_unlocked_deployment_before_spawn(tmp_path):
    substituted=replace(selection(),wrapper_digest="a"*64)
    plan,facade=plan_and_facade(substituted)
    value,_,plan=operations(tmp_path,plan_facade=(plan,facade))
    with pytest.raises(ValueError,match="packaged runtime lock"):
        value.preflight(plan)
    assert FakeFunction.spawn_calls==[]
