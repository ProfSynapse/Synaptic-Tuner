from __future__ import annotations

import pytest

from synaptic_tuner.api.v1.planning import ProviderPlanRef, TrainingPlan, TrainingPlanBasisV1
from synaptic_tuner.api.v1.providers import ProviderCapabilities, ProviderDescriptor, ProviderRef
from synaptic_tuner.api.v1.results import TrainingRunRef
from tuner.execution.foundation_v2.executors import AdapterDescriptorV1, ExecutorDescriptorV1
from tuner.execution.foundation_v2.references import ExecutionScopeV1
from tuner.execution.providers.docker_provider_v1.model import (
    AuthenticatedDockerAbsenceV1, AuthenticatedDockerCancellationEvidenceV1,
    AuthenticatedDockerCancellationAbsenceV1,
    AuthenticatedDockerSourceSealV1, DockerAbsenceContentV1,
    AuthenticatedDockerCommandBindingV1,
    DockerArtifactContractV1, DockerCreateDispositionV1, DockerCreateResultV1,
    DockerCancellationAbsenceContentV1, DockerCancellationContentV1,
    DockerCancellationLookupResultV1,
    DockerImageV1, DockerLookupDispositionV1, DockerLookupResultV1, DockerProfileV1,
    DockerRootsV1, DockerRunPhaseV1, DockerRuntimeV1, DockerSourceSealContentV1,
    DockerSourceSealLookupResultV1, DockerLookupPurposeV1,
    DockerWorkloadV1,
)
from tuner.execution.foundation_v2.canonical import canonical_bytes, domain_digest

D = tuple(character * 64 for character in "123456789abcdef")


@pytest.fixture(params=("opaque/local-cpu", "opaque/registry-cpu"))
def profile(request):
    provider = ProviderRef("docker", request.param)
    descriptor = ProviderDescriptor(
        "synaptic-provider-descriptor/v1", "docker", "Docker", "1.0.0",
        ProviderCapabilities(True, True, True, True, True, False),
    )
    runtime = DockerRuntimeV1(2, 1_073_741_824, 3600)
    workload = DockerWorkloadV1(("python", "/source/run_fixture.py", "/source", "/artifacts"), (), D[0])
    artifacts = DockerArtifactContractV1(("result",), 1_048_576, 1_048_576)
    return DockerProfileV1(
        provider, descriptor, D[1], ExecutionScopeV1("account", "namespace"),
        ExecutorDescriptorV1("docker", "docker-executor-v1", "1.0.0"),
        AdapterDescriptorV1("docker", "docker-reconcile-v1", "1.0.0"),
        DockerImageV1("fixture-image", "sha256:" + "a" * 64), runtime, workload,
        DockerRootsV1("source-root", "artifact-root"), artifacts, D[2], D[3], D[4],
    )


@pytest.fixture
def plan(profile):
    basis = TrainingPlanBasisV1(
        "synaptic-training-plan-basis/v1", "request", "project", D[5], D[6],
        profile.workload.workload_digest, profile.runtime.digest, profile.artifacts.digest,
    )
    return TrainingPlan("synaptic-training-plan/v2", basis, ProviderPlanRef(D[7]))


@pytest.fixture
def run():
    return TrainingRunRef("run", "project")


class BindingAuthority:
    authority_ref = "docker-binding-authority-v1"
    key_ref = "docker-binding-key-v1"
    def tag_for(self, binding):
        return domain_digest("synaptic-test-docker-binding-tag/v1", canonical_bytes({
            "binding_digest": binding.binding_digest,
            "authority_ref": self.authority_ref,
            "key_ref": self.key_ref,
        }))
    def issue(self, binding):
        return AuthenticatedDockerCommandBindingV1(
            binding, binding.binding_digest, self.authority_ref,
            self.key_ref, self.tag_for(binding),
        )
    def authenticate(self, value):
        return (type(value) is AuthenticatedDockerCommandBindingV1
                and value.authority_ref == self.authority_ref
                and value.key_ref == self.key_ref
                and value.binding_digest == value.content.binding_digest
                and value.tag == self.tag_for(value.content))


class Catalog:
    def __init__(self, binding_authority=None):
        self.values = {}
        self.binding_authority = binding_authority or BindingAuthority()
    def resolve(self, digest):
        value = self.values[digest]
        if type(value) is AuthenticatedDockerCommandBindingV1:
            return value
        return self.binding_authority.issue(value)


class ImageInventory:
    def __init__(self, present=True):
        self.present, self.calls = present, 0
    def require_present(self, image):
        self.calls += 1
        return self.present


class Source:
    def __init__(self):
        self.calls = 0
        self.mutations = 0
        self.lookup_calls = 0
        self.retained = {}
        self.lost_return = False
        self.lookup_disposition = DockerLookupDispositionV1.FOUND
    def seal_read_only(self, request):
        self.calls += 1
        if request.digest not in self.retained:
            self.mutations += 1
        content = DockerSourceSealContentV1(
            request.digest, request.identity.digest, request.source_ref,
            request.source_digest, True, "stage-sealed", D[8],
        )
        seal = AuthenticatedDockerSourceSealV1(content, "docker-test", "key-v1", D[9])
        self.retained[request.digest] = seal
        if self.lost_return:
            raise RuntimeError("lost source seal return")
        return seal
    def lookup(self, request):
        self.lookup_calls += 1
        if self.lookup_disposition is DockerLookupDispositionV1.FOUND:
            seal = self.retained.get(request.source_request.digest)
            return (DockerSourceSealLookupResultV1(DockerLookupDispositionV1.FOUND, seal=seal)
                    if seal is not None else DockerSourceSealLookupResultV1(DockerLookupDispositionV1.INDETERMINATE))
        if self.lookup_disposition is DockerLookupDispositionV1.DEFINITELY_ABSENT:
            content = DockerAbsenceContentV1(
                request.digest, request.source_request.identity.digest,
                DockerLookupPurposeV1.RECONCILE_STAGE, request.generation, D[10],
            )
            absence = AuthenticatedDockerAbsenceV1(content, "docker-test", "key-v1", D[11])
            return DockerSourceSealLookupResultV1(
                DockerLookupDispositionV1.DEFINITELY_ABSENT, absence=absence
            )
        return DockerSourceSealLookupResultV1(self.lookup_disposition)


class Control:
    def __init__(self):
        self.trace = []
        self.create_disposition = DockerCreateDispositionV1.CREATED
        self.start_result = True
        self.lookup_result = None
        self.lookup_disposition = DockerLookupDispositionV1.FOUND
        self.created = {}
        self.create_mutations = 0
        self.started = set()
        self.start_mutations = 0
    def create_once(self, **values):
        self.trace.append(("create", values["labels"].digest))
        key = values["labels"].digest
        if key in self.created:
            return self.created[key]
        if self.create_disposition is DockerCreateDispositionV1.CREATED:
            self.create_mutations += 1
            result = DockerCreateResultV1(self.create_disposition, values["labels"], "container-1")
            self.created[key] = result
            return result
        return DockerCreateResultV1(self.create_disposition)
    def start_once(self, container_ref, labels):
        self.trace.append(("start", container_ref))
        key = (container_ref, labels.digest)
        if key not in self.started and self.start_result:
            self.start_mutations += 1
            self.started.add(key)
        return self.start_result
    def lookup(self, request):
        self.trace.append(("lookup", request.labels.digest))
        if self.lookup_result is not None:
            return self.lookup_result
        if self.lookup_disposition is DockerLookupDispositionV1.DEFINITELY_ABSENT:
            content = DockerAbsenceContentV1(
                request.digest, request.labels.digest, request.purpose,
                request.generation, D[12],
            )
            absence = AuthenticatedDockerAbsenceV1(content, "docker-test", "key-v1", D[13])
            return DockerLookupResultV1(DockerLookupDispositionV1.DEFINITELY_ABSENT, absence=absence)
        if self.lookup_disposition is not DockerLookupDispositionV1.FOUND:
            return DockerLookupResultV1(self.lookup_disposition)
        retained = self.created.get(request.labels.digest)
        if retained is None:
            return DockerLookupResultV1(DockerLookupDispositionV1.INDETERMINATE)
        return DockerLookupResultV1(
            DockerLookupDispositionV1.FOUND, retained.labels,
            retained.container_ref, DockerRunPhaseV1.RUNNING,
        )


class Cancellations:
    def __init__(self):
        self.trace = []
        self.requests = []
        self.retained = {}
        self.lost_return = False
        self.lookup_disposition = DockerLookupDispositionV1.FOUND
        self.mutations = 0
    def stop_once(self, request):
        self.requests.append(request)
        self.trace.append(("stop", request.container_ref))
        if request.digest not in self.retained:
            self.mutations += 1
        content = DockerCancellationContentV1(
            request.digest, request.cancellation_identity.digest,
            request.submit_labels.digest, request.container_ref,
            request.reason_digest, request.authorization_digest, D[10],
        )
        evidence = AuthenticatedDockerCancellationEvidenceV1(
            content, "docker-test", "key-v1", D[11]
        )
        self.retained[request.digest] = evidence
        if self.lost_return:
            raise RuntimeError("lost cancellation return")
        return evidence
    def lookup(self, request):
        self.trace.append(("cancel_lookup", request.cancellation_request.digest))
        if self.lookup_disposition is DockerLookupDispositionV1.FOUND:
            evidence = self.retained.get(request.cancellation_request.digest)
            return (DockerCancellationLookupResultV1(DockerLookupDispositionV1.FOUND, evidence=evidence)
                    if evidence is not None else DockerCancellationLookupResultV1(DockerLookupDispositionV1.INDETERMINATE))
        if self.lookup_disposition is DockerLookupDispositionV1.DEFINITELY_ABSENT:
            value = request.cancellation_request
            content = DockerCancellationAbsenceContentV1(
                request.digest, value.digest, value.cancellation_identity.digest,
                value.authorization_digest, value.submit_labels.digest,
                value.container_ref, value.reason_digest, request.generation,
                DockerRunPhaseV1.RUNNING, D[12],
            )
            absence = AuthenticatedDockerCancellationAbsenceV1(
                content, "docker-test", "key-v1", D[13]
            )
            return DockerCancellationLookupResultV1(
                DockerLookupDispositionV1.DEFINITELY_ABSENT, absence=absence
            )
        return DockerCancellationLookupResultV1(self.lookup_disposition)


class Authority:
    def authenticate_source_seal(self, value): return type(value) is AuthenticatedDockerSourceSealV1
    def authenticate_cancellation(self, value): return type(value) is AuthenticatedDockerCancellationEvidenceV1
    def authenticate_absence(self, value): return type(value) is AuthenticatedDockerAbsenceV1
    def authenticate_cancellation_absence(self, value): return type(value) is AuthenticatedDockerCancellationAbsenceV1


@pytest.fixture
def seams():
    return Catalog(), ImageInventory(), Source(), Control(), Cancellations()
