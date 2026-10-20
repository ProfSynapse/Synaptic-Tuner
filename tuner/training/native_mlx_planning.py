"""Native planning/preparation adapters for the existing public coordinator.

The consumer supplies a resolved request and independently authenticated native
qualification. This module grants neither source nor hardware authority and
does not register a provider. Candidate planning remains inspectable when the
qualification port refuses execution.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib

from synaptic_tuner.api.v1.planning import (
    ProviderPlanContextV1, ProviderPlanRef, ResolvedTrainingRequest,
    TrainingPlan, TrainingPlanBasisV1,
)
from synaptic_tuner.api.v1.providers import ProviderCapabilities, ProviderDescriptor, ProviderRef
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.training_facade import AuthorizationRequirement, TrainingPreflight
from tuner.execution.coordinator_v1.model import ProviderExecutionBindingV1
from tuner.execution.foundation_v2.canonical import digest_text, safe_ref
from tuner.execution.foundation_v2.commands import CanonicalProviderPayloadV1
from tuner.execution.foundation_v2.executors import AdapterDescriptorV1, ExecutorDescriptorV1
from tuner.execution.foundation_v2.identities import EffectKind
from tuner.execution.foundation_v2.preparation import CanonicalPreparationV2
from tuner.execution.foundation_v2.references import ExecutionScopeV1
from tuner.runtime.native_mlx_material import canonical, native_runtime_spec
from tuner.runtime.native_process import NativeProcessLimits
from tuner.training.coordinator_material import _domain, _runtime
from tuner.training.native_mlx_compilation import parse_native_workload


def _digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


@dataclass(frozen=True)
class NativeQualificationV1:
    """Evidence references, authenticated by the consumer's independent port.

    These hashes are not proof by themselves. The authority must verify retained
    hardware runs, source review, selected wheel/install provenance and the exact
    final-adapter/public-readback evidence. No default accepting authority exists.
    """

    profile_digest: str
    source_digest: str
    runtime_digest: str
    hardware_digest: str
    attempt_evidence: tuple[str, str]
    authority_ref: str
    key_ref: str
    tag: str

    def __post_init__(self):
        for name in ("profile_digest", "source_digest", "runtime_digest", "hardware_digest", "tag"):
            digest_text(getattr(self, name), name)
        if (type(self.attempt_evidence) is not tuple or len(self.attempt_evidence) != 2
                or len(set(self.attempt_evidence)) != 2):
            raise ValueError("two distinct native qualification attempts required")
        for value in self.attempt_evidence:
            digest_text(value, "attempt_evidence")
        safe_ref(self.authority_ref, "authority_ref")
        safe_ref(self.key_ref, "key_ref")


class NativePlanningV1:
    """One immutable native material selection; reusable across host restarts."""

    __slots__ = (
        "resolved", "run", "workload", "runtime_capture", "limits", "scope",
        "provider", "profile_digest", "_clock", "_qualification",
        "_qualification_authority", "descriptor", "executor_descriptor",
        "adapter_descriptor", "binding", "context_value", "plan", "_sealed",
    )

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("native planning selection is immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self, *, resolved: ResolvedTrainingRequest, run: TrainingRunRef,
        workload: bytes, runtime_capture: bytes, limits: NativeProcessLimits,
        scope: ExecutionScopeV1, clock, qualification=None, qualification_authority=None,
    ):
        if (type(resolved) is not ResolvedTrainingRequest or type(run) is not TrainingRunRef
                or type(limits) is not NativeProcessLimits or type(scope) is not ExecutionScopeV1):
            raise TypeError("exact native plan inputs required")
        projection = parse_native_workload(workload)
        runtime = native_runtime_spec(runtime_capture)
        if (run.project_ref != resolved.project_ref
                or projection.workload.fingerprint != resolved.workload_digest
                or projection.profile.source_digest != resolved.source_digest
                or projection.profile.runtime_digest != runtime.material_digest
                or _domain("synaptic-coordinator-runtime/v1", _runtime(runtime)) != resolved.runtime_digest
                or _domain("synaptic-coordinator-artifact-policy/v1",
                           projection.workload.document["training_input"]["artifacts"])
                != resolved.artifact_policy_digest):
            raise ValueError("native resolved request differs from compiled material")
        self.resolved, self.run, self.workload = resolved, run, workload
        self.runtime_capture, self.limits, self.scope = runtime_capture, limits, scope
        self.provider = ProviderRef("native-mlx", projection.profile.profile_ref)
        self.profile_digest = _digest({
            "schema": "synaptic-native-provider-profile/v1",
            "profile": asdict(projection.profile), "limits": asdict(limits),
            "scope": scope.to_dict(),
        })
        self._clock = clock
        self._qualification, self._qualification_authority = qualification, qualification_authority
        qualified = self.qualified()
        self.descriptor = ProviderDescriptor(
            "synaptic-provider-descriptor/v1", "native-mlx", "Apple native MLX", "1.0.0",
            ProviderCapabilities(qualified, False, False, False, qualified, False),
        )
        self.executor_descriptor = ExecutorDescriptorV1("native-mlx", "owned-native-process", "1.0.0")
        self.adapter_descriptor = AdapterDescriptorV1("native-mlx", "no-replay-native-reconciliation", "1.0.0")
        self.binding = ProviderExecutionBindingV1(
            self.provider, self.descriptor.descriptor_digest, self.profile_digest, scope,
            self.executor_descriptor, self.adapter_descriptor.digest, _digest(asdict(limits)),
            _digest({"schema": "synaptic-native-local-quote/v1", "amount": 0, "currency": "USD"}),
            _digest({"schema": "synaptic-native-secret-requirements/v1", "secrets": []}),
        )
        self.context_value = ProviderPlanContextV1(
            "synaptic-provider-plan-context/v1", self.provider,
            TrainingPlanBasisV1.from_resolved(resolved).basis_digest,
            self.descriptor.descriptor_digest, self.profile_digest,
        )
        self.plan = TrainingPlan(
            "synaptic-training-plan/v2", TrainingPlanBasisV1.from_resolved(resolved),
            ProviderPlanRef(self.context_value.provider_context_digest),
        )
        self._sealed = True

    def qualified(self):
        proof = self._qualification
        try:
            return (
                type(proof) is NativeQualificationV1
                and proof.profile_digest == self.profile_digest
                and proof.source_digest == self.resolved.source_digest
                and proof.runtime_digest == self.resolved.runtime_digest
                and self._qualification_authority.authenticate(proof) is True
            )
        except Exception:
            return False

    def require_qualified(self):
        if not self.qualified():
            raise ValueError("native_hardware_qualification_required")

    def describe(self, provider):
        if provider != self.provider:
            raise ValueError("native provider differs")
        return self.descriptor

    def context(self, resolved, provider):
        if resolved != self.resolved or provider != self.provider:
            raise ValueError("native resolved material differs")
        return self.context_value

    def resolve(self, provider, context):
        if provider != self.provider or context != self.context_value:
            raise ValueError("native context differs")
        return self.binding

    def for_plan(self, plan):
        if plan != self.plan:
            raise ValueError("native run plan differs")
        return self.run

    def preflight(self, plan):
        if plan != self.plan:
            raise ValueError("native plan differs")
        now = self._clock.now()
        expiry = (datetime.fromisoformat(now.replace("Z", "+00:00")) + timedelta(minutes=15))
        ready = self.qualified()
        return TrainingPreflight(
            plan.plan_fingerprint, ready, now,
            expiry.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            (AuthorizationRequirement("training.start", False),),
            () if ready else ("native_hardware_qualification_required",),
        )

    def prepare(self, plan, run, binding):
        if plan != self.plan or run != self.run or binding != self.binding:
            raise ValueError("native preparation differs")
        return CanonicalPreparationV2.build(
            provider=self.provider, scope=self.scope, project_ref=run.project_ref,
            run_id=run.run_id, plan_fingerprint=plan.plan_fingerprint,
            source_digest=self.resolved.source_digest, workload_digest=self.resolved.workload_digest,
            runtime_digest=self.resolved.runtime_digest, resource_digest=binding.resource_digest,
            artifact_contract_digest=self.resolved.artifact_policy_digest,
            quote_digest=binding.quote_digest,
            secret_requirements_digest=binding.secret_requirements_digest,
            execution_binding_digest=binding.binding_digest,
        )

    def validate_preparation(self, preparation):
        expected = self.prepare(self.plan, self.run, self.binding)
        if (type(preparation) is not CanonicalPreparationV2
                or preparation.canonical_bytes != expected.canonical_bytes):
            raise ValueError("native preparation is not the selected immutable material")
        return expected

    def payload(self, preparation, kind):
        self.validate_preparation(preparation)
        if kind not in (EffectKind.STAGE, EffectKind.SUBMIT):
            raise ValueError("native lifecycle operation is unavailable")
        return CanonicalProviderPayloadV1.build(
            "native-mlx", f"{kind.value}-payload/v2", self.resolved.workload_digest,
        )
