"""Provider-read-only RunsAPI for new, privately retained known Modal calls.

Local authenticated observations and verification may advance durable records.
This never restores executable sources, effect grants, log proof or continuation.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from synaptic_tuner.api.v1.planning import TrainingPlan, TrainingPlanBasisV1
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.runs_facade import RunsAPI, RunOperationCode, RunOperationError
from synaptic_tuner.api.v1.secrets import SecretRef
from synaptic_tuner.api.v1.training_facade import TrainingRequest, TrainingPreflight
from synaptic_tuner.api.v1.reference.authority import compose_reference_authority, UnavailableRecoveryVerifierV1
from synaptic_tuner.api.v1.reference.repositories import (
    DurableEffectRepositoryV1, DurablePlanningStoreV1, DurableWorkflowStoreV1,
)
from synaptic_tuner.api.v1.reference.runs import build_run_operations
from synaptic_tuner.api.v1.reference.training import _decode_commitment, _AUTHORIZATION
from tuner.execution.foundation_v2.canonical import canonical_bytes, domain_digest
from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
from tuner.execution.coordinator_v1.model import ProviderReadPurposeV1, WorkflowPhaseV1
from tuner.execution.coordinator_v1.foundation import ComposedEffectFoundationV1
from synaptic_tuner.api.v1.reference.authority import UnavailableQuiescenceEvidenceV1
from tuner.execution.providers.modal.binding import ModalClientBinding
from tuner.execution.providers.modal.facade import ExplicitModal154ReadFacade
from tuner.execution.providers.modal.packaged_binding import ModalPackagedRuntimeFactsV1
from tuner.execution.providers.modal.packaged_deployment import ModalPackagedDeploymentObserver
from tuner.execution.providers.modal.packaged_reader import ModalPackagedReader
from tuner.execution.providers.modal.packaged_progress import PROGRESS_PURPOSE
from tuner.execution.providers.modal.runtime_release_deployment import ModalRuntimeReleaseDeploymentFactsV1
from tuner.runtime.releases import parse_packaged_runtime_release, PackagedExecutionBindingV1
from tuner.training.contracts import CanonicalDocument, ResolvedTrainingComponents, RuntimeSpec, ResourceSpec, ArtifactPolicy
from tuner.training.modal_host_authority import HMACAuthenticator, UTCClock, ObservationAuthenticator, LogAuthenticator
from tuner.training.modal_host_composition import _modal_packaged_descriptor, _modal_read_authorities
from tuner.training.modal_host_effects import ModalPackagedHostEffectsV1
from tuner.training.modal_host_runtime import _CurrentPackagedDeploymentReader
from tuner.training.modal_host_scope import open_modal_host_scope, observe_modal_host_scope
from tuner.training.modal_host_storage import ModalHostStorageV1
from tuner.training.modal_local_preparation import validate_local_launcher
from tuner.training.modal_run_retention import load_retained_modal_run, _decode_bytes, WORKFLOW_FILE
from tuner.training.packaged_compilation import (
    compile_packaged_sft_workload, packaged_configuration_digest, packaged_artifact_policy_digest,
)
from tuner.training import default_recipe_registry
from tuner.training.contracts import ResolvedTrainingRequest, TrainingRequest as MaterialRequest
from tuner.training.packaged_boundary import derive_packaged_coordinator_material

_TERMINAL = frozenset({
    WorkflowPhaseV1.SUCCEEDED_UNVERIFIED, WorkflowPhaseV1.VERIFIED, WorkflowPhaseV1.VERIFICATION_FAILED,
})
_ACTIVE = frozenset({WorkflowPhaseV1.QUEUED, WorkflowPhaseV1.RUNNING,
                     WorkflowPhaseV1.CANCEL_REQUESTED})
_READABLE = _TERMINAL | _ACTIVE


class ModalRetainedReadUnavailable(RuntimeError):
    def __init__(self):
        super().__init__("modal_retained_run_unavailable")


class _Keys:
    __slots__ = ("_keys",)
    def __init__(self, keys):
        self._keys = keys
    def resolve(self, ref):
        if type(ref) is not SecretRef or ref.provider != "env" or ref.name not in self._keys:
            raise ModalRetainedReadUnavailable()
        return self._keys[ref.name]
    def __repr__(self):
        return "<retained host signing keys>"


class _BindingReads(ModalPackagedHostEffectsV1):
    def authorize(self, requirements):
        raise ModalRetainedReadUnavailable()
    def bind(self, grant, *, operation, requirements):
        raise ModalRetainedReadUnavailable()


class _PlanningReads:
    def __init__(self, provider, descriptor):
        self._provider, self._descriptor = provider, descriptor
    def describe(self, provider):
        if provider != self._provider:
            raise ModalRetainedReadUnavailable()
        return self._descriptor


class _TerminalOperations:
    __slots__ = ("_operations", "_workflows", "_run")
    def __init__(self, operations, workflows, run):
        self._operations, self._workflows, self._run = operations, workflows, run
    def _check(self, run):
        if run != self._run:
            raise RunOperationError(RunOperationCode.RUN_MISSING)
        record = self._workflows.get(run)
        if record is None or record.phase not in _READABLE:
            raise RunOperationError(RunOperationCode.READ_INELIGIBLE)
    def show(self, run):
        self._check(run)
        return self._operations.show(run)
    def outcome(self, run):
        self._check(run)
        return self._operations.outcome(run)
    def verify(self, run):
        self._check(run)
        if self._workflows.get(run).phase is WorkflowPhaseV1.VERIFIED:
            raise RunOperationError(RunOperationCode.READ_INELIGIBLE)
        return self._operations.verify(run)
    def artifacts(self, request):
        self._check(request.run)
        return self._operations.artifacts(request)
    def list(self, request):
        self._check(self._run)
        if request.project_ref != self._run.project_ref:
            raise RunOperationError(RunOperationCode.RUN_MISSING)
        result = self._operations.list(request)
        if any(item.run != self._run for item in result.outcomes):
            raise RunOperationError(RunOperationCode.INTEGRITY_ERROR)
        return result
    def logs(self, request):
        raise RunOperationError(RunOperationCode.CAPABILITY_UNAVAILABLE)
    def cancel(self, run, reason):
        raise RunOperationError(RunOperationCode.CAPABILITY_UNAVAILABLE)
    def reconcile(self, run):
        raise RunOperationError(RunOperationCode.CAPABILITY_UNAVAILABLE)
    def reverify(self, run):
        raise RunOperationError(RunOperationCode.CAPABILITY_UNAVAILABLE)


@dataclass(frozen=True, slots=True)
class RetainedModalRunsV1:
    run: TrainingRunRef
    runs: RunsAPI


def _exact(document, fields):
    if type(document) is not dict or set(document) != set(fields):
        raise ValueError
    return document


def _typed_inputs(retained):
    inputs = retained.inputs
    release = parse_packaged_runtime_release(inputs["runtime_release"])
    deployment = ModalRuntimeReleaseDeploymentFactsV1.from_dict(inputs["deployment_facts"])
    facts = ModalPackagedRuntimeFactsV1.from_dict(inputs["runtime_facts"])
    scope = ModalClientBinding(**inputs["client_binding"])
    facts.validate_release(release)
    if (facts.client_binding != scope or deployment.client_binding != scope
            or ModalPackagedRuntimeFactsV1.from_release_deployment(deployment) != facts
            or deployment.runtime_release_digest != release.manifest_digest):
        raise ValueError
    raw = inputs["components"]
    execution = PackagedExecutionBindingV1.from_dict(raw["execution_source"])
    provider_binding = facts.build_provider_binding(release)
    execution.validate_bindings(release, provider_binding)
    runtime = RuntimeSpec(**_exact(raw["runtime"], (
        "image", "dependency_lock_digest", "python_version", "material_kind", "material_digest",
    )))
    resources = ResourceSpec(**_exact(raw["resources"], ("accelerator", "accelerator_count", "timeout_seconds")))
    policy_doc = _exact(raw["artifact_policy"], ("required_kinds", "retain_checkpoints"))
    if type(policy_doc["required_kinds"]) is not list:
        raise ValueError
    policy = ArtifactPolicy(tuple(policy_doc["required_kinds"]), policy_doc["retain_checkpoints"])
    components = ResolvedTrainingComponents(
        execution, CanonicalDocument(_decode_bytes(raw["execution_context"]).decode("utf-8")),
        CanonicalDocument(_decode_bytes(raw["resolved_config"]).decode("utf-8")), runtime, resources, policy,
    )
    workload = compile_packaged_sft_workload(resolved_config=components.resolved_config)
    if (workload.canonical_bytes != _decode_bytes(inputs["workload"])
            or execution.run_ref != retained.run.run_id
            or execution.workload_digest != workload.fingerprint
            or execution.configuration_digest != packaged_configuration_digest(components.resolved_config)
            or execution.artifact_policy_digest != packaged_artifact_policy_digest(policy)
            or components.execution_context.to_dict() != {
                "schema_version": "synaptic-packaged-execution-context/v1", "runtime_release": release.to_dict(),
            }):
        raise ValueError
    if (runtime.material_kind != "modal_build" or runtime.material_digest != release.material_digest
            or runtime.dependency_lock_digest != release.installed_distributions_digest
            or runtime.python_version != release.python_version):
        raise ValueError
    request = TrainingRequest(**_exact(inputs["prepared_request"], ("request_id", "project_ref", "canonical_json")))
    plan = TrainingPlan.from_dict(inputs["training_plan"])
    preflight = TrainingPreflight.from_dict(inputs["training_preflight"])
    if request.project_ref != retained.run.project_ref or not preflight.binds(plan):
        raise ValueError
    resolved = ResolvedTrainingRequest(
        MaterialRequest(CanonicalDocument(request.canonical_json)), execution,
        components.execution_context, components.resolved_config,
        CanonicalDocument(workload.canonical_bytes.decode("utf-8")),
        runtime, resources, policy,
    )
    material = derive_packaged_coordinator_material(
        resolved, default_recipe_registry(), request_id=request.request_id,
        project_ref=request.project_ref, run_id=retained.run.run_id,
    )
    if TrainingPlanBasisV1.from_resolved(material.planning_request) != plan.basis:
        raise ValueError
    volumes = inputs["volume_names"]
    if (type(volumes) is not list or any(type(row) is not list or len(row) != 2 for row in volumes)
            or len(volumes) != 3 or len(dict(volumes)) != 3
            or dict(volumes) != {item.volume_id: item.spec.name for item in deployment.volumes}):
        raise ValueError
    return inputs, release, deployment, facts, scope, components, workload, provider_binding, plan


@contextmanager
def open_retained_modal_runs(*, root, run, modal_profile, modal_environment, sdk, clock=None):
    """Open one exact retained call; provider reads only, local CAS writes allowed."""
    opened = False
    try:
        if type(run) is not TrainingRunRef or not isinstance(root, Path):
            raise ValueError
        validate_local_launcher()
        retained = load_retained_modal_run(root, expected_run=run)
        inputs, release, deployment_facts, facts, scope, components, workload, provider_binding, plan = _typed_inputs(retained)
        if (modal_profile != inputs["modal_profile"] or modal_environment != inputs["modal_environment"]
                or getattr(sdk, "__version__", None) != "1.5.4"):
            raise ValueError
        journal = Path(inputs["attempt_journal"])
        # Reopening never creates a missing state/catalog database.
        if not journal.is_file() or not (root / WORKFLOW_FILE).is_file():
            raise ValueError
        signing = _Keys(retained.signing_keys)
        selected_clock = UTCClock() if clock is None else clock
        authority = compose_reference_authority(
            clock=selected_clock, secrets=signing,
            authority_secret=SecretRef("env", "SYNAPTIC_TRAIN_AUTHORITY_KEY"),
        )
        with ModalHostStorageV1(journal, run.project_ref, create=False) as storage, SQLiteDurableStoreV1(root / WORKFLOW_FILE, create=False) as records:
            signer_ref = "modal-runtime-release-qualification-hmac-v1"
            signer = HMACAuthenticator({signer_ref: retained.qualification_key}, allowed_purposes=frozenset({
                "modal-packaged-dispatch/v1", "modal-packaged-dispatch/v2",
                "modal-packaged-completion/v1", "modal-packaged-evaluation/v1", PROGRESS_PURPOSE,
            }))
            effects = _BindingReads(
                storage=storage, runtime_release=release, provider_binding=provider_binding,
                provider_facts=facts, execution_binding=components.execution_source, retained_source=None,
                workload_bytes=workload.canonical_bytes, artifact_policy=components.artifact_policy,
                signer=signer, key_ref=signer_ref, maximum_cost_minor_units=inputs["maximum_cost_minor_units"],
            )
            evidence, reader, verifier = _modal_read_authorities(
                key=retained.signing_keys["SYNAPTIC_TRAIN_READER_KEY"].encode("utf-8"), clock=authority.clock,
                bindings=effects.bindings, binding_authority=effects,
                foundation_authenticator=authority.foundation_authenticator,
                assessment_authenticator=authority.assessment_authority, packaged_reader=None, facade=None,
            )
            recovery = UnavailableRecoveryVerifierV1()
            repository = DurableEffectRepositoryV1(
                records, receipt_authority=authority.receipt_authority,
                invalid_evidence_authority=authority.invalid_evidence_authority,
                recovery_verifier=recovery, finality_verifier=recovery, grant_authority=authority.grant_authority,
            )
            foundation = ComposedEffectFoundationV1(
                repository, None, None, grant_authority=authority.grant_authority,
                receipt_authority=authority.receipt_authority,
                invalid_evidence_authority=authority.invalid_evidence_authority,
                assessment_authority=authority.assessment_authority,
                trusted_quiescence_evidence=UnavailableQuiescenceEvidenceV1(),
            )
            reader.bind_foundation(foundation)
            verifier.bind_foundation(foundation)
            workflows = DurableWorkflowStoreV1(
                records, records, foundation_authenticator=authority.foundation_authenticator,
                assessment_authenticator=authority.assessment_authority,
                observation_authenticator=ObservationAuthenticator(evidence), artifact_verifier=verifier,
            )
            workflow = workflows.get(run)
            if workflow is None or workflow.phase not in _READABLE or workflow.submit is None:
                raise ValueError
            reader.bind_workflows(workflows)
            commitment_record = records.read(
                partition=_AUTHORIZATION, key=f"policy/{workflow.preflight_digest}",
            )
            if commitment_record is None:
                raise ValueError
            commitment = _decode_commitment(commitment_record.canonical, workflow.preflight_digest)
            preflight = TrainingPreflight.from_dict(inputs["training_preflight"])
            if commitment.preflight != preflight or commitment.plan_fingerprint != plan.plan_fingerprint:
                raise ValueError
            paid = [item for item in preflight.authorization if item.paid_effect]
            if (len(paid) != 1 or paid[0].maximum_cost_minor_units != inputs["maximum_cost_minor_units"]
                    or paid[0].currency != "USD"):
                raise ValueError
            plans = DurablePlanningStoreV1(records)
            context = plans.get_context(workflow.provider_context_digest)
            if (plans.get_plan(workflow.plan_fingerprint) != plan or workflow.plan_fingerprint != plan.plan_fingerprint
                    or context is None or context.provider != workflow.provider
                    or context.descriptor_digest != workflow.provider_descriptor_digest
                    or context.profile_digest != inputs["profile_digest"]):
                raise ValueError
            descriptors = [_modal_packaged_descriptor(logs=value) for value in (False, True)]
            matches = [d for d in descriptors if d.descriptor_digest == workflow.provider_descriptor_digest]
            if len(matches) != 1:
                raise ValueError
            operations = build_run_operations(
                planning=_PlanningReads(workflow.provider, matches[0]), planning_store=plans,
                workflow_store=workflows, coordinator=None, foundation=foundation,
                foundation_authenticator=authority.foundation_authenticator,
                assessment_authenticator=authority.assessment_authority, reader=reader,
                observation_authenticator=ObservationAuthenticator(evidence), log_authenticator=LogAuthenticator(evidence),
                artifact_verifier=verifier, cursor_authority=authority.cursor_authority, clock=authority.clock,
            )
            # This uses the exact assessment retained in the original submit.
            purpose = (ProviderReadPurposeV1.OBSERVE if workflow.phase in _ACTIVE
                       else ProviderReadPurposeV1.ARTIFACTS)
            request = operations._request(workflow, purpose)
            binding = reader._binding(request, purpose)
            call = effects.calls.resolve(binding.command_digest)
            resource_digest = domain_digest(
                "synaptic-modal-packaged-resource-request/v1", canonical_bytes({
                    "accelerator": components.resources.accelerator,
                    "accelerator_count": components.resources.accelerator_count,
                    "timeout_seconds": components.resources.timeout_seconds,
                }),
            )
            if (type(call) is not str or call != request.provider_run.reference.provider_job_ref
                    or binding.command.preparation.quote_digest != inputs["quote_digest"]
                    or binding.command.preparation.resource_digest != resource_digest):
                raise ValueError
            # No SDK scope/lookup occurs until the stored authenticated terminal chain above passes.
            client, observed_scope = open_modal_host_scope(
                sdk=sdk, profile=modal_profile, environment_name=modal_environment, client_ref=scope.client_ref,
            )
            if observed_scope != scope:
                raise ValueError
            names = {item.spec.role: item.spec.name for item in deployment_facts.volumes}
            current = _CurrentPackagedDeploymentReader(
                sdk=sdk, client=client, binding=scope, facts=facts, names=names,
                secret_ids=tuple((item.spec.name, item.secret_id, item.spec.required_keys)
                                 for item in deployment_facts.secrets),
            )
            observer = ModalPackagedDeploymentObserver(sdk=sdk, client=client, client_binding=scope, reader=current)
            facade = ExplicitModal154ReadFacade(
                scope, sdk=sdk, client=client,
                scope_observer=lambda supplied: observe_modal_host_scope(sdk=sdk, client=supplied, binding=scope),
                deployment_observer=current.observe, volume_names=dict(inputs["volume_names"]),
            )
            reader._facade = facade
            reader._reader = ModalPackagedReader(
                facade=facade, deployment_observer=observer, verifier=signer, key_ref=signer_ref,
            )
            view = RetainedModalRunsV1(run, RunsAPI(_TerminalOperations(operations, workflows, run)))
            opened = True
            yield view
        return
    except Exception:
        if opened:
            raise
        pass
    raise ModalRetainedReadUnavailable()
