"""Explicit Docker host assembly over the existing public reference ports.

No daemon is contacted during assembly. Source publication is lazy inside the
claimed stage effect. Supplied stores and original named keys own persistence;
this module does not create a database or qualify a production capability.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import time

from synaptic_tuner.api.v1.planning import (
    ProviderPlanContextV1, ProviderPlanRef, TrainingPlan, TrainingPlanBasisV1,
)
from synaptic_tuner.api.v1.ports import StoragePartition
from synaptic_tuner.api.v1.reference.authority import compose_reference_authority, derive_authority_key
from synaptic_tuner.api.v1.reference.composition import compose_reference_host
from synaptic_tuner.api.v1.reference.provider_family import (
    ProviderFamilyV1, ReferenceHostPortsV1, require_methods,
)
from synaptic_tuner.api.v1.reference.repositories import _decode, _encode, _dump, _read
from synaptic_tuner.api.v1.reference.training import ReferenceRequestPortsV1
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.secrets import SecretRef
from synaptic_tuner.api.v1.training_facade import AuthorizationRequirement, TrainingPreflight, TrainingRequest
from tuner.execution.coordinator_v1.model import (
    ArtifactVerificationContentV1, AuthenticatedFoundationRecordAssessmentV1,
    ProviderReadPurposeV1, VerificationVerdictV1,
)
from tuner.execution.coordinator_v1.state_machine import _derive_foundation, provider_run_read_request
from tuner.execution.foundation_v2.canonical import canonical_bytes, domain_digest
from tuner.execution.foundation_v2.commands import CancelCommandV2, parse_exact_command
from tuner.execution.providers.docker_provider_v1.cancellation_transport import DockerCancellationTransportV1
from tuner.execution.providers.docker_provider_v1.cli_transport import DockerCLISessionV1
from tuner.execution.providers.docker_provider_v1.effects import (
    DockerEffectExecutorV1, DockerExecutorResolverV1,
    DockerReconciliationAdapterV1, DockerReconciliationResolverV1,
)
from tuner.execution.providers.docker_provider_v1.local_artifacts import DockerLocalArtifactsV1
from tuner.execution.providers.docker_provider_v1.local_source import DockerLocalSourceSealsV1
from tuner.execution.providers.docker_provider_v1.model import (
    AuthenticatedDockerCommandBindingV1, DockerCommandBindingV1, DockerEffectIdentityV1,
    DockerLookupDispositionV1, DockerSourceSealLookupResultV1, labels_for,
    validated_profile_snapshot,
)
from tuner.execution.providers.docker_provider_v1.preparation import (
    DockerBindingResolverV1, DockerPreparationMaterializerV1, DockerTrainingPreparationBridgeV1,
)
from tuner.execution.providers.docker_provider_v1.reader import DockerProviderRunReaderV1
from tuner.training.docker_host_authority import DockerHostEvidenceAuthorityV1
from tuner.training.modal_host_authority import ObservationAuthenticator, LogAuthenticator


class _BindingCatalog:
    def __init__(self, records, authority, profile):
        self._records, self._authority, self._profile = records, authority, profile
        self._prefix = "docker-binding/" + domain_digest(
            "synaptic-docker-host-scope/v1", canonical_bytes({
                "provider": profile.provider.to_dict(),
                "scope": {"account": profile.scope.account_ref, "namespace": profile.scope.namespace_ref},
            })) + "/"

    def _raw(self, value):
        if (type(value) is not AuthenticatedDockerCommandBindingV1
                or self._authority.authenticate(value) is not True
                or value.content.plan.profile != self._profile):
            raise ValueError("docker_host_binding_invalid")
        raw = _dump(_encode(value, AuthenticatedDockerCommandBindingV1))
        if len(raw) > 1048576:
            raise ValueError("docker_host_binding_invalid")
        return raw

    def resolve(self, command_digest):
        from tuner.execution.foundation_v2.canonical import digest_text
        key = self._prefix + digest_text(command_digest, "command_digest")
        stored = _read(self._records, StoragePartition.EFFECTS.value, key)
        if stored is None:
            return None
        from tuner.training.contracts import bounded_json_object
        value = _decode(bounded_json_object(stored.canonical.decode('utf-8'), maximum_bytes=1048576),
                        AuthenticatedDockerCommandBindingV1)
        if self._raw(value) != stored.canonical or value.content.command_digest != command_digest:
            raise ValueError("docker_host_binding_invalid")
        return value

    def publish_once(self, value):
        raw = self._raw(value)
        key = self._prefix + value.content.command_digest
        accepted = self._records.put_if_absent(partition=StoragePartition.EFFECTS.value,
                                              key=key, canonical=raw)
        if type(accepted) is not bool:
            raise ValueError("docker_host_binding_invalid")
        retained = self.resolve(value.content.command_digest)
        if retained != value:
            raise ValueError("docker_host_binding_conflict")
        return retained


class _PlanPorts:
    def __init__(self, profile, rich_plan, material, recipes, clock, available):
        bridge = DockerTrainingPreparationBridgeV1(profile)
        self.material, self.resolved = bridge.coordinator_material(rich_plan, material, recipes)
        self.run = TrainingRunRef(self.material.run_id, self.resolved.project_ref)
        basis = TrainingPlanBasisV1.from_resolved(self.resolved)
        self.context_value = ProviderPlanContextV1(
            "synaptic-provider-plan-context/v1", profile.provider, basis.basis_digest,
            profile.descriptor.descriptor_digest, profile.profile_digest)
        self.plan = TrainingPlan("synaptic-training-plan/v2", basis,
                                 ProviderPlanRef(self.context_value.provider_context_digest))
        self.preparation, self.prepared = bridge.coordinator_prepared(
            plan=rich_plan, material=self.material, coordinator_plan=self.plan,
            run=self.run, recipes=recipes)
        self._profile, self._clock, self._available = profile, clock, available
        self._resolver, self._materializer = DockerBindingResolverV1(profile), DockerPreparationMaterializerV1(profile)
        self._request = TrainingRequest(self.resolved.request_id, self.resolved.project_ref,
                                       self.material.request_bytes.decode("utf-8"))

    def load(self, canonical_json):
        if type(canonical_json) is not str or canonical_json != self._request.canonical_json:
            raise ValueError("docker_host_request_invalid")
        return self._request

    def resolve(self, *args):
        if len(args) == 1:
            if args[0] != self._request:
                raise ValueError("docker_host_request_invalid")
            return self.resolved
        if len(args) == 2 and args[1] == self.context_value:
            return self._resolver.resolve(*args)
        raise ValueError("docker_host_plan_invalid")

    def for_plan(self, plan):
        if type(plan) is not TrainingPlan or plan != self.plan:
            raise ValueError("docker_host_plan_invalid")
        return self.run

    def describe(self, provider):
        if provider != self._profile.provider:
            raise ValueError("docker_host_provider_invalid")
        return self._profile.descriptor

    def context(self, resolved, provider):
        if resolved != self.resolved or provider != self._profile.provider:
            raise ValueError("docker_host_plan_invalid")
        return self.context_value

    def preflight(self, plan):
        self.for_plan(plan)
        now = self._clock.now()
        expiry = (datetime.fromisoformat(now.replace("Z", "+00:00")) + timedelta(minutes=15))
        return TrainingPreflight(plan.plan_fingerprint, self._available, now,
            expiry.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            (AuthorizationRequirement("training.start", False),),
            () if self._available else ('docker_source_unavailable',))

    def prepare(self, plan, run, binding):
        if self.for_plan(plan) != run:
            raise ValueError("docker_host_plan_invalid")
        return self._materializer.prepare(plan, run, binding)

    def payload(self, preparation, kind):
        return self._materializer.payload(preparation, kind)


class _Source:
    def __init__(self, planning, rich_plan, recipes, source_inputs, evidence):
        self._planning, self._rich, self._recipes = planning, rich_plan, recipes
        self._inputs = None if source_inputs is None else dict(source_inputs)
        self._evidence, self._port, self._attempted = evidence, None, False

    def seal_read_only(self, request):
        if self._inputs is None or self._attempted or request.identity.plan != self._planning.prepared:
            raise ValueError("docker_host_source_unavailable")
        # Invoked only by DockerEffectExecutor after Foundation owns stage.
        self._attempted = True
        self._port = DockerLocalSourceSealsV1(
            **self._inputs, training_plan=self._rich, prepared_plan=self._planning.prepared,
            sign_seal=self._evidence.source_seal, coordinator_plan=self._planning.plan,
            coordinator_material=self._planning.material, recipes=self._recipes)
        return self._port.seal_read_only(request)

    def lookup(self, request):
        if self._port is None:
            return DockerSourceSealLookupResultV1(DockerLookupDispositionV1.INDETERMINATE)
        return self._port.lookup(request)

    def revalidate_for_submit(self, **kwargs):
        if self._port is None:
            raise ValueError("docker_host_source_unavailable")
        return self._port.revalidate_for_submit(**kwargs)

    def close(self):
        if self._port is not None:
            self._port.close()
        elif self._inputs is not None:
            self._inputs['prepared_lease'].close()


class _WorkflowAccess:
    def __init__(self, authority, planning):
        self._authority, self._run = authority, planning.run
        self._plan, self._context = planning.plan, planning.context_value
        self._composition = None

    def bind(self, composition):
        if self._composition is not None:
            raise ValueError("docker_host_already_bound")
        self._composition = composition

    def workflow(self):
        if self._composition is None:
            raise ValueError("docker_host_not_bound")
        workflow = self._composition.stores.workflow_store.get(self._run)
        if (workflow is None or workflow.submit is None or workflow.provider_run_ref is None
                or workflow.plan_fingerprint != self._plan.plan_fingerprint
                or workflow.provider_context_digest != self._context.provider_context_digest
                or workflow.provider_descriptor_digest != self._context.descriptor_digest):
            raise ValueError("docker_host_submit_unavailable")
        record = self._composition.foundation.get(workflow.submit.effect_id)
        assessment = AuthenticatedFoundationRecordAssessmentV1.parse(
            workflow.submit.foundation_bindings[-1].canonical_assessment_bytes)
        binding, outcome, target = _derive_foundation(workflow.submit, record, assessment,
            self._authority.foundation_authenticator, self._authority.assessment_authority,
            workflow.provider_run_ref)
        if (binding != workflow.submit.foundation_bindings[-1]
                or outcome != workflow.submit.foundation_outcomes[-1]
                or outcome.disposition.value != "found" or target != workflow.provider_run_ref):
            raise ValueError("docker_host_submit_unavailable")
        return workflow, record, assessment

    def request(self, purpose):
        workflow, record, assessment = self.workflow()
        return provider_run_read_request(workflow, record, assessment,
            self._authority.foundation_authenticator, self._authority.assessment_authority,
            purpose=purpose)

    def authenticate(self, request):
        try:
            return request == self.request(request.purpose)
        except Exception:
            return False


class _Grants:
    def __init__(self, delegate, planning, catalog, evidence, access):
        self._delegate, self._planning, self._catalog = delegate, planning, catalog
        self._evidence, self._access = evidence, access

    def authorize(self, requirements):
        return self._delegate.authorize(requirements)

    def bind(self, grant, *, operation, requirements):
        command = parse_exact_command(operation.canonical_bytes)
        if command.preparation != self._planning.preparation:
            raise ValueError("docker_host_command_invalid")
        kind = command.operation.effect.kind.value
        identity = DockerEffectIdentityV1(command.digest, command.operation.effect.effect_id,
                                         kind, self._planning.prepared)
        if type(command) is CancelCommandV2:
            workflow, _, _ = self._access.workflow()
            if workflow.cancel is None or workflow.cancel.canonical_command_bytes != command.canonical_bytes:
                raise ValueError("docker_host_cancel_invalid")
            original = self._catalog.resolve(workflow.submit.command_digest)
            cancellation = command.to_dict()['cancellation']
            if original is None or cancellation['provider_job_ref'] != workflow.provider_run_ref.reference.provider_job_ref:
                raise ValueError("docker_host_cancel_invalid")
            labels = labels_for(original.content.identity)
            binding = DockerCommandBindingV1(identity, command.canonical_bytes,
                workflow.submit.canonical_command_bytes,
                workflow.provider_run_ref.reference.provider_job_ref,
                cancellation['reason_digest'], labels,
                domain_digest("synaptic-docker-cancel-authorization/v1", canonical_bytes({
                    "grant_ref": grant.grant_ref, "command_digest": command.digest,
                    "requirements": [_encode(value, type(value)) for value in requirements]})))
        else:
            binding = DockerCommandBindingV1(identity, command.canonical_bytes)
        result = self._delegate.bind(grant, operation=operation, requirements=requirements)
        self._catalog.publish_once(self._evidence.issue(binding))
        return result


class _LocalReads:
    def __init__(self, planning, control, access, catalog, evidence, artifact_root,
                 role_paths, read_seconds):
        self._planning, self._control, self._access = planning, control, access
        self._catalog, self._evidence = catalog, evidence
        self._root, self._paths, self._seconds = artifact_root, dict(role_paths), read_seconds
        self._inventory_request = None

    def lookup(self, request):
        return self._control.lookup(request)

    def logs(self, request):
        raise ValueError("docker_host_logs_unavailable")

    def terminal_phase(self, request):
        from tuner.execution.providers.docker_provider_v1.model import (
            DockerLookupRequestV1, DockerLookupPurposeV1, DockerLookupResultV1,
        )
        if self._access.authenticate(request) is not True:
            raise ValueError('docker_host_completion_unavailable')
        workflow, _, _ = self._access.workflow()
        bound = self._catalog.resolve(workflow.submit.command_digest)
        labels = labels_for(bound.content.identity)
        result = self._control.lookup(DockerLookupRequestV1(labels,
            DockerLookupPurposeV1.OBSERVE, request.source_revision))
        if (type(result) is not DockerLookupResultV1
                or result.disposition is not DockerLookupDispositionV1.FOUND
                or result.labels != labels
                or result.container_ref != request.provider_run.reference.provider_job_ref):
            raise ValueError('docker_host_completion_unavailable')
        return result.phase

    @contextmanager
    def _reader(self, labels):
        workflow, _, _ = self._access.workflow()
        bound = self._catalog.resolve(workflow.submit.command_digest)
        if bound is None or labels != labels_for(bound.content.identity):
            raise ValueError("docker_host_artifact_binding_invalid")
        port = DockerLocalArtifactsV1(plan=bound.content.plan, labels=labels,
            container_ref=workflow.provider_run_ref.reference.provider_job_ref,
            control=self._control, root=self._root, role_paths=self._paths,
            deadline=time.monotonic() + self._seconds,
            sign_inventory=self._evidence.inventory, sign_eof=self._evidence.eof)
        try:
            yield port
        finally:
            port.close()

    def artifact_inventory(self, request):
        with self._reader(request.labels) as reader:
            inventory = reader.artifact_inventory(request)
            self._inventory_request = (inventory.content.content_digest, request)
            return inventory

    def iter_artifact_events(self, request):
        # Reconstruct the matching inventory on the same retained descriptors
        # before streaming; never restore an unauthenticated process-local cache.
        retained = self._inventory_request
        if retained is None or retained[0] != request.inventory_digest:
            raise ValueError("docker_host_artifact_inventory_changed")
        with self._reader(request.labels) as reader:
            inventory = reader.artifact_inventory(retained[1])
            if inventory.content.content_digest != request.inventory_digest:
                raise ValueError("docker_host_artifact_inventory_changed")
            yield from reader.iter_artifact_events(request)


class _Reader:
    def __init__(self, profile, catalog, evidence, access, local, clock):
        self._profile, self._catalog, self._evidence = profile, catalog, evidence
        self._access, self._local, self._clock = access, local, clock

    def _reader(self):
        return DockerProviderRunReaderV1(self._profile, self._catalog, self._evidence,
            self._access, self._evidence, self._local, observed_at=self._clock.now())

    def observe(self, request):
        return self._reader().observe(request)

    def terminal_phase(self, request):
        return self._local.terminal_phase(request)

    def logs(self, request, query):
        raise ValueError("docker_host_logs_unavailable")

    def artifacts(self, request):
        return self._reader().artifacts(request)

    def iter_artifact_bytes(self, request, manifest, role, *, maximum_bytes):
        return self._reader().iter_artifact_bytes(request, manifest, role, maximum_bytes=maximum_bytes)


class _Verifier:
    def __init__(self, reader, access, evidence, clock, rich_plan):
        self._reader, self._access, self._evidence, self._clock = reader, access, evidence, clock
        from tuner.runtime.dispatch import (build_source_worker_invocation, materialize_worker_bundle,
            WorkerControlLocationV1, CanonicalWorkloadFileLocationV1)
        from pathlib import PurePosixPath
        from tuner.training.recipes import CompiledWorkload
        from tuner.runtime.artifacts import ArtifactContract, ArtifactRequirement
        from tuner.runtime.verification import VerificationService, WorkloadBindingVerifier
        control = WorkerControlLocationV1(PurePosixPath('/source/control'))
        bundle = materialize_worker_bundle(build_source_worker_invocation(rich_plan, control,
            CanonicalWorkloadFileLocationV1(control.control_root)))
        doc = rich_plan.workload.to_dict()
        self._workload = CompiledWorkload(doc['method'], doc['schema_version'], doc['entrypoint'],
                                        rich_plan.workload.canonical_json.encode('utf-8'))
        contract = doc['artifacts']
        self._contract = ArtifactContract(contract['schema_version'],
            tuple(ArtifactRequirement(**item) for item in contract['requirements']),
            contract.get('allow_unlisted_roles', False))
        self._service = VerificationService(WorkloadBindingVerifier(
            closure_digest=bundle.closure_digest,
            closure_manifest_path=bundle.closure_manifest_runtime_path.as_posix()))

    def verify(self, workflow, manifest):
        current, _, _ = self._access.workflow()
        if workflow != current:
            raise ValueError("docker_host_verification_changed")
        request = self._access.request(ProviderReadPurposeV1.ARTIFACTS)
        if self._reader.artifacts(request) != manifest:
            raise ValueError("docker_host_verification_changed")
        from tuner.runtime.artifacts import ArtifactEntry, ArtifactInventory
        from tuner.runtime.dispatch import ProcessResult
        from tuner.runtime.verification import ArtifactReadLimitExceeded
        inventory = ArtifactInventory(tuple(ArtifactEntry(item.role, item.role,
            item.sha256, item.size_bytes) for item in manifest.artifacts))
        owner = self
        class AuthenticatedStreams:
            def read_bytes(self, artifact, *, maximum):
                if artifact.size > maximum:
                    raise ArtifactReadLimitExceeded('docker_host_artifact_bound')
                content = bytearray()
                for chunk in owner._reader.iter_artifact_bytes(request, manifest, artifact.role,
                        maximum_bytes=max(1, artifact.size)):
                    if len(chunk) > maximum - len(content):
                        raise ArtifactReadLimitExceeded('docker_host_artifact_bound')
                    content.extend(chunk)
                return bytes(content)
        from tuner.execution.providers.docker_provider_v1.model import DockerRunPhaseV1
        phase = self._reader.terminal_phase(request)
        completed = phase in {DockerRunPhaseV1.SUCCEEDED, DockerRunPhaseV1.FAILED,
                              DockerRunPhaseV1.CANCELLED}
        # Exact CLI identity/status/exit-code lookup supplies this phase; a stop
        # acknowledgement alone never supplies successful process completion.
        report = self._service.verify(provider_completed=completed,
            process=ProcessResult(0 if phase is DockerRunPhaseV1.SUCCEEDED else 1),
            workload=self._workload, contract=self._contract, inventory=inventory,
            reader=AuthenticatedStreams())
        final_phase = self._reader.terminal_phase(request)
        succeeded = report.success and final_phase is DockerRunPhaseV1.SUCCEEDED
        verdict = VerificationVerdictV1.VERIFIED if succeeded else VerificationVerdictV1.REJECTED
        diagnostic = ('docker_provider_exit_invalid'
            if phase is not DockerRunPhaseV1.SUCCEEDED or final_phase is not DockerRunPhaseV1.SUCCEEDED
            else 'docker_native_artifacts_invalid')
        content = ArtifactVerificationContentV1(
            "synaptic-artifact-verification-content/v1", workflow.record_digest,
            workflow.revision, workflow.run, workflow.provider_run_ref.binding_digest,
            manifest.manifest_digest, manifest.artifact_source_digest, manifest.artifacts,
            manifest.artifacts if succeeded else (), verdict,
            None if succeeded else diagnostic,
            "docker-local-artifact-verifier", "1.0.0",
            canonical_bytes({"request_digest": request.request_digest, "native_status": report.status.value,
                "initial_phase": phase.value, "final_phase": final_phase.value,
                "semantic_checks": [{"code": check.code, "passed": check.passed}
                    for check in report.semantic_checks]}), self._clock.now())
        return self._evidence.verification(content)

    def replay(self, workflow, manifest, prior_receipt):
        if self.authenticate(prior_receipt) is not True or workflow.artifact_manifest != manifest:
            raise ValueError("docker_host_verification_invalid")
        return self.verify(workflow, manifest)

    def authenticate(self, value):
        from tuner.execution.coordinator_v1.model import AuthenticatedArtifactVerificationReceiptV1
        return type(value) is AuthenticatedArtifactVerificationReceiptV1 and self._evidence.authenticate(value)


@dataclass(frozen=True, slots=True)
class DockerReferenceHostV1:
    composition: object
    control: DockerCLISessionV1
    source: object

    @property
    def api(self):
        return self.composition.api()

    def close(self):
        self.source.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def compose_docker_reference_host(*, profile, rich_plan, material, recipes,
        ports: ReferenceHostPortsV1, authority_secret, evidence_secret,
        cli_configuration, source_inputs, artifact_root, role_paths,
        read_seconds=60, stop_grace_seconds=1, allow_current_cancel_authorization=False):
    """Assemble explicit host configuration; source_inputs=None permits cold reads.

    Existing descriptor flags are retained, never promoted. Logs are unavailable.
    CLI deadlines bound provider commands; synchronous host stores/signers retain
    their existing semantics. No arbitrary cold-stage continuation is promised.
    """
    if type(ports) is not ReferenceHostPortsV1:
        raise TypeError("exact reference host ports required")
    if type(authority_secret) is not SecretRef or type(evidence_secret) is not SecretRef:
        raise TypeError('exact named host secrets required')
    if type(allow_current_cancel_authorization) is not bool:
        raise TypeError('exact cancellation option required')
    profile = validated_profile_snapshot(profile)
    if profile.descriptor.capabilities.logs:
        raise ValueError("docker_host_logs_unavailable")
    if type(read_seconds) is not int or not 1 <= read_seconds <= 300:
        raise ValueError("docker_host_read_bound_invalid")
    from pathlib import Path, PurePosixPath
    cli = dict(cli_configuration)
    artifact_host, artifact_destination = cli['artifact_roots'][profile.roots.artifact_ref]
    if Path(artifact_host) != Path(artifact_root):
        raise ValueError('docker_host_artifact_root_invalid')
    if source_inputs is not None:
        required = {'bundle', 'closure_members', 'prepared_lease', 'source_root',
            'source_container_root', 'artifact_root', 'artifact_container_root', 'deadline'}
        source_inputs = dict(source_inputs)
        if set(source_inputs) != required:
            raise ValueError('docker_host_source_inputs_invalid')
        source_host, source_destination = cli['source_roots'][profile.roots.source_ref]
        if (Path(source_host) != Path(source_inputs['source_root'])
                or PurePosixPath(source_destination) != PurePosixPath(source_inputs['source_container_root'])
                or Path(artifact_root) != Path(source_inputs['artifact_root'])
                or PurePosixPath(artifact_destination) != PurePosixPath(source_inputs['artifact_container_root'])):
            raise ValueError('docker_host_source_mount_invalid')
        source_inputs['closure_members'] = dict(source_inputs['closure_members'])
    require_methods(ports.records, "create", "read", "compare_and_swap", "put_if_absent", "list_page")
    require_methods(ports.streams, "append", "read_page")
    require_methods(ports.grants, "authorize", "bind")
    authority = compose_reference_authority(clock=ports.clock, secrets=ports.secrets,
                                            authority_secret=authority_secret)
    raw_key = ports.secrets.resolve(evidence_secret)
    if type(raw_key) is not str or len(raw_key) < 32:
        raise ValueError("docker_host_evidence_secret_unavailable")
    evidence = DockerHostEvidenceAuthorityV1(
        key=derive_authority_key(raw_key.encode("utf-8"), "docker-evidence"),
        authority_ref="docker-host-evidence", key_ref="docker-host-evidence-key")
    del raw_key
    planning = _PlanPorts(profile, rich_plan, material, recipes, authority.clock, source_inputs is not None)
    catalog = _BindingCatalog(ports.records, evidence, profile)
    source = _Source(planning, rich_plan, recipes, source_inputs, evidence)
    control = DockerCLISessionV1(**cli, profile=profile,
                                 source_submit_check=source.revalidate_for_submit)
    access = _WorkflowAccess(authority, planning)
    local = _LocalReads(planning, control, access, catalog, evidence,
                        artifact_root, role_paths, read_seconds)
    reader = _Reader(profile, catalog, evidence, access, local, authority.clock)
    verifier = _Verifier(reader, access, evidence, authority.clock, rich_plan)
    cancellation = DockerCancellationTransportV1(session=control, records=ports.records,
        issue_evidence=evidence.cancellation, evidence_authority=evidence, grace_seconds=stop_grace_seconds)
    executor = DockerEffectExecutorV1(profile, catalog, evidence, control, source,
        control, cancellation, evidence, source_submit_check=source.revalidate_for_submit)
    adapter = DockerReconciliationAdapterV1(profile, catalog, evidence, control,
                                             source, cancellation, evidence)
    family = ProviderFamilyV1(profile.descriptor, planning, planning, reader,
        DockerExecutorResolverV1(executor), DockerReconciliationResolverV1(adapter),
        evidence, verifier, ObservationAuthenticator(evidence), LogAuthenticator(evidence))
    grants = _Grants(ports.grants, planning, catalog, evidence, access)
    composed = compose_reference_host(family=family,
        ports=ReferenceHostPortsV1(ports.records, ports.streams, ports.clock, grants, ports.secrets),
        requests=ReferenceRequestPortsV1(planning, planning, planning), authority=authority,
        allow_current_cancel_authorization=allow_current_cancel_authorization)
    access.bind(composed)
    return DockerReferenceHostV1(composed, control, source)
