"""Full admitted SFT material projection into the existing public plan contracts."""

from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from synaptic_tuner.api.v1.planning import (
    ProviderPlanContextV1, ProviderPlanRef, ResolvedTrainingRequest,
    TrainingPlan, TrainingPlanBasisV1,
)
from synaptic_tuner.api.v1.providers import ProviderCapabilities, ProviderDescriptor
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.training_facade import TrainingPreflight, AuthorizationRequirement
from synaptic_tuner.api.v1._contract import PreparedTrainingInputIdentity
from tuner.execution.coordinator_v1.model import ProviderExecutionBindingV1
from tuner.execution.foundation_v2.commands import CanonicalProviderPayloadV1
from tuner.execution.foundation_v2.executors import ExecutorDescriptorV1, AdapterDescriptorV1
from tuner.execution.foundation_v2.identities import EffectKind
from tuner.execution.foundation_v2.preparation import CanonicalPreparationV2
from tuner.runtime.dispatch import (
    WorkerBundleMaterializationV1, WorkerControlLocationV1,
    CanonicalWorkloadFileLocationV1, build_source_worker_invocation,
    materialize_worker_bundle,
)
from tuner.runtime.offline_sft_worker import parse_offline_sft_worker_manifest
from tuner.training.contracts import (
    TrainingPlan as RichTrainingPlan, RetainedTrainingInputStreamLease,
    execution_material_digest,
)
from .model import HFJobsSelection, HFJobsError, document_bytes, MAX_INPUT_BYTES, EXACT_ROLES


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class HFJobsPreparation:
    """One supplied rich plan/bundle/one-use input, with no provider operations.

    The public source digest remains execution identity. Full source-member,
    bootstrap, dispatch, selected runtime and input evidence is committed in
    resolved_config_digest, avoiding a digest-only executable scaffold.
    """

    def __init__(self, *, selection, training_plan, bundle, closure_members,
                 parent_members, input_lease, request_id, project_ref, clock,
                 maximum_cost_minor_units, currency, preflight_seconds=300):
        if (type(selection) is not HFJobsSelection
                or type(training_plan) is not RichTrainingPlan
                or type(bundle) is not WorkerBundleMaterializationV1
                or type(input_lease) is not RetainedTrainingInputStreamLease
                or type(preflight_seconds) is not int or not 1 <= preflight_seconds <= 3600):
            raise HFJobsError("input_invalid")
        if (set(training_plan.artifact_policy.required_kinds) != EXACT_ROLES
                or training_plan.artifact_policy.retain_checkpoints is not False):
            raise HFJobsError("capability_unavailable")
        control = WorkerControlLocationV1(PurePosixPath("/workspace/private/control"))
        expected = materialize_worker_bundle(build_source_worker_invocation(
            training_plan, control, CanonicalWorkloadFileLocationV1(control.control_root)))
        if any(getattr(bundle, name) != getattr(expected, name)
               for name in expected.__dataclass_fields__):
            raise HFJobsError("binding_mismatch")
        workload = json.loads(bundle.canonical_workload_bytes)
        configuration = workload["configuration"]["document"]
        if workload["method"] != "sft":
            raise HFJobsError("capability_unavailable")
        identity = PreparedTrainingInputIdentity.from_dict(configuration["dataset"])
        if (input_lease.identity != identity or not 0 < identity.size_bytes <= MAX_INPUT_BYTES
                or identity.format != "syntunia-sft-row/v1"):
            raise HFJobsError("binding_mismatch")
        model = configuration["model"]
        if model["tokenizer_revision"] != model["revision"]:
            raise HFJobsError("capability_unavailable")
        selection.runtime_profile.resolve(model=model["ref"],
                                          model_revision=model["revision"], method="sft")
        if (training_plan.runtime.image != selection.runtime_profile.image
                or training_plan.resources != selection.resources):
            raise HFJobsError("binding_mismatch")
        source = training_plan.execution_source
        if (dict(source.roots) != {name: "/workspace/private/" + ("" if name in {"engine", "project"} else "run/") + name for name in
                ("engine", "project", "artifacts", "state", "tracking", "cache", "tmp")}
                or source.python_version != selection.runtime_profile.runtime_facts["python_version"]
                or training_plan.runtime.python_version != source.python_version):
            raise HFJobsError("binding_mismatch")
        manifest = parse_offline_sft_worker_manifest(
            bundle.closure_manifest_bytes, source_ref=bundle.closure_manifest_source,
            manifest_path=Path(bundle.closure_manifest_runtime_path),
            expected_digest=bundle.closure_digest)
        copied = {}
        if set(closure_members) != {member.path.as_posix() for member in manifest.closure.members}:
            raise HFJobsError("binding_mismatch")
        for member in manifest.closure.members:
            raw = closure_members[member.path.as_posix()]
            if type(raw) is not bytes or len(raw) != member.size_bytes or sha(raw) != member.sha256:
                raise HFJobsError("binding_mismatch")
            copied["engine/" + member.path.as_posix()] = raw
        copied["control/offline-sft-worker-v1.json"] = bundle.closure_manifest_bytes
        copied["control/workload.json"] = bundle.canonical_workload_bytes
        # Parent files must equal the packaged reviewed inventory, not merely
        # self-consistent caller files. Their engine boundary stays separate.
        from importlib.resources import files
        parent_manifest = json.loads(files(__package__).joinpath("hf-parent-closure.json").read_bytes())
        records = parent_manifest["members"]
        if set(parent_members) != {row["path"] for row in records}:
            raise HFJobsError("binding_mismatch")
        for row in records:
            raw = parent_members[row["path"]]
            if type(raw) is not bytes or len(raw) != row["size_bytes"] or sha(raw) != row["sha256"]:
                raise HFJobsError("binding_mismatch")
            copied["parent/" + row["path"]] = raw
        copied["control/runtime-inventory.json"] = selection.inventory_bytes
        self.selection, self.bundle, self.rich_plan = selection, bundle, training_plan
        self.input_identity, self.input_lease = identity, input_lease
        self.source_files = MappingProxyType(copied)
        self.clock, self.preflight_seconds = clock, preflight_seconds
        self.requirement = AuthorizationRequirement("training.start", True,
                                                   maximum_cost_minor_units, currency)
        self.descriptor = ProviderDescriptor("synaptic-provider-descriptor/v1", "hf_jobs",
            "Ordinary HF Jobs SFT", "0.1.0", ProviderCapabilities(True, False, False, False, True, False))
        self.executor = ExecutorDescriptorV1("hf_jobs", "hf-jobs-sft", "0.1.0")
        self.adapter = AdapterDescriptorV1("hf_jobs", "hf-jobs-unavailable-reconcile", "0.1.0")
        projection = {"schema_version": "synaptic-hf-jobs-full-material/v1",
            "selection": selection.projection(), "rich_plan_fingerprint": training_plan.fingerprint,
            "host_sdk_lock_sha256": sha(files(__package__).joinpath("host-dependencies.lock").read_bytes()),
            "bundle_projection_sha256": bundle.projection_sha256,
            "members": [{"path": name, "size_bytes": len(raw), "sha256": sha(raw)}
                        for name, raw in sorted(copied.items())],
            "input": identity.to_dict(), "authorization": {
                "maximum_cost_minor_units": maximum_cost_minor_units, "currency": currency}}
        self.material_bytes = document_bytes(projection)
        self.material_digest = sha(self.material_bytes)
        self.resolved = ResolvedTrainingRequest("synaptic-resolved-training-request/v1",
            request_id, project_ref, execution_material_digest(training_plan.execution_source),
            self.material_digest, bundle.workload_fingerprint,
            sha(document_bytes({"runtime": selection.digest,
                "rich_runtime": training_plan.runtime.to_dict() if hasattr(training_plan.runtime, "to_dict")
                else {"image": training_plan.runtime.image, "python_version": training_plan.runtime.python_version,
                      "dependency_lock_digest": training_plan.runtime.dependency_lock_digest}})),
            sha(document_bytes({"artifact_policy": list(training_plan.artifact_policy.required_kinds)})))
        self.context_value = ProviderPlanContextV1("synaptic-provider-plan-context/v1",
            selection.provider, TrainingPlanBasisV1.from_resolved(self.resolved).basis_digest,
            self.descriptor.descriptor_digest, selection.digest)
        self.plan = TrainingPlan("synaptic-training-plan/v2", TrainingPlanBasisV1.from_resolved(self.resolved),
                                 ProviderPlanRef(self.context_value.provider_context_digest))
        self.binding = ProviderExecutionBindingV1(selection.provider, self.descriptor.descriptor_digest,
            selection.digest, selection.scope, self.executor, self.adapter.digest,
            sha(document_bytes({"flavor": selection.flavor, "resources": selection.projection()["resources"]})),
            selection.quote_digest, sha(document_bytes({"job_secret_key": "HF_TOKEN",
                                                        "reference": selection.job_token.to_dict()})))

    def describe(self, provider):
        if provider != self.selection.provider:
            raise HFJobsError("binding_mismatch")
        return self.descriptor

    def context(self, resolved, provider):
        if resolved != self.resolved or provider != self.selection.provider:
            raise HFJobsError("binding_mismatch")
        return self.context_value

    def preflight(self, plan):
        if plan != self.plan:
            raise HFJobsError("binding_mismatch")
        now = self.clock.now()
        expiry = datetime.fromisoformat(now.replace("Z", "+00:00")) + timedelta(seconds=self.preflight_seconds)
        return TrainingPreflight(plan.plan_fingerprint, True, now, expiry.isoformat(), (self.requirement,))

    def resolve(self, provider, context):
        if provider != self.selection.provider or context != self.context_value:
            raise HFJobsError("binding_mismatch")
        return self.binding

    def prepare(self, plan, run, binding):
        if (plan != self.plan or binding != self.binding or type(run) is not TrainingRunRef
                or run.project_ref != self.resolved.project_ref
                or run.run_id != self.rich_plan.execution_source.run_id):
            raise HFJobsError("binding_mismatch")
        return CanonicalPreparationV2.build(provider=binding.provider, scope=binding.scope,
            project_ref=run.project_ref, run_id=run.run_id, plan_fingerprint=plan.plan_fingerprint,
            source_digest=plan.basis.source_digest, workload_digest=plan.basis.workload_digest,
            runtime_digest=plan.basis.runtime_digest, resource_digest=binding.resource_digest,
            artifact_contract_digest=plan.basis.artifact_policy_digest, quote_digest=binding.quote_digest,
            secret_requirements_digest=binding.secret_requirements_digest,
            execution_binding_digest=binding.binding_digest)

    def payload(self, preparation, kind):
        if kind not in {EffectKind.STAGE, EffectKind.SUBMIT}:
            raise HFJobsError("capability_unavailable")
        if (preparation.provider != self.selection.provider
                or preparation.plan_fingerprint != self.plan.plan_fingerprint
                or preparation.execution_binding_digest != self.binding.binding_digest):
            raise HFJobsError("binding_mismatch")
        return CanonicalProviderPayloadV1.build("hf_jobs", f"{kind.value}-payload/v2", self.bundle.workload_fingerprint)
