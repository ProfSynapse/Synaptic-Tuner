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
from tuner.runtime.native_mlx_material import canonical, native_runtime_spec, strict_json
from tuner.runtime.native_process import NativeProcessLimits
from tuner.runtime.native_lifecycle_journal import NativeLifecycleLaunchV1
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


@dataclass(frozen=True)
class NativeRestrictedAdmissionV1:
    """An independently authenticated grant for one initial hardware attempt.

    This is not a qualification receipt. Its verifier is supplied by the host;
    the engine neither issues admissions nor trusts a digest as approval.
    """

    profile_digest: str
    source_digest: str
    runtime_digest: str
    hardware_digest: str
    prepared_digest: str
    model_digest: str
    tokenizer_digest: str
    workload_digest: str
    selection_digest: str
    run_id: str
    attempt: str
    maximum_updates: int
    expires_at: str
    authority_ref: str
    key_ref: str
    tag: str

    def __post_init__(self):
        for name in ("profile_digest", "source_digest", "runtime_digest", "hardware_digest",
                     "prepared_digest", "model_digest", "tokenizer_digest",
                     "workload_digest", "selection_digest", "tag"):
            digest_text(getattr(self, name), name)
        for name in ("run_id", "attempt", "authority_ref", "key_ref"):
            safe_ref(getattr(self, name), name)
        if type(self.maximum_updates) is not int or not 1 <= self.maximum_updates <= 1000000:
            raise ValueError("restricted native update bound is invalid")
        expiry = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
        if expiry.tzinfo is None or expiry.utcoffset() is None:
            raise ValueError("restricted native admission needs a timezone")


@dataclass(frozen=True)
class NativeLifecycleGrantV1:
    """Independently authenticated lifecycle qualification or bounded test grant.

    A restricted grant is not evidence of success. No default issuer/verifier
    exists; the host must authenticate its reviewed exact operation selection.
    """
    profile_digest: str
    source_digest: str
    runtime_digest: str
    launch_digest: str
    hardware_digest: str
    mode: str
    evidence_digests: tuple[str, ...]
    selection_digest: str | None
    run_id: str | None
    attempt: str | None
    authority_ref: str
    key_ref: str
    tag: str

    def __post_init__(self):
        if type(self.mode) is not str:
            raise TypeError("native lifecycle grant mode must be exact text")
        for name in ("profile_digest", "source_digest", "runtime_digest", "launch_digest", "hardware_digest", "tag"):
            digest_text(getattr(self, name), name)
        for name in ("authority_ref", "key_ref"):
            safe_ref(getattr(self, name), name)
        if self.mode == "qualified":
            if (type(self.evidence_digests) is not tuple or len(self.evidence_digests) < 3
                    or len(self.evidence_digests) > 16 or len(set(self.evidence_digests)) != len(self.evidence_digests)
                    or any(value is not None for value in (self.selection_digest, self.run_id, self.attempt))):
                raise ValueError("native lifecycle qualification fields differ")
            for value in self.evidence_digests: digest_text(value, "evidence_digest")
        elif self.mode == "restricted":
            if type(self.evidence_digests) is not tuple or self.evidence_digests:
                raise ValueError("native lifecycle admission cannot claim qualification evidence")
            digest_text(self.selection_digest, "selection_digest")
            safe_ref(self.run_id, "run_id")
            safe_ref(self.attempt, "attempt")
        else:
            raise ValueError("native lifecycle grant mode is invalid")


class NativePlanningV1:
    """One immutable native material selection; reusable across host restarts."""

    __slots__ = (
        "resolved", "run", "workload", "runtime_capture", "limits", "scope",
        "provider", "profile_digest", "_clock", "_qualification",
        "_qualification_authority", "_restricted_admission", "_restricted_authority",
        "_selection_digest", "_retained_read",
        "descriptor", "executor_descriptor",
        "adapter_descriptor", "binding", "context_value", "plan", "_sealed",
        "lifecycle", "_lifecycle_grant", "_lifecycle_authority",
    )

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("native planning selection is immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self, *, resolved: ResolvedTrainingRequest, run: TrainingRunRef,
        workload: bytes, runtime_capture: bytes, limits: NativeProcessLimits,
        scope: ExecutionScopeV1, clock, qualification=None, qualification_authority=None,
        restricted_admission=None, restricted_authority=None,
        selection_digest=None, retained_read=False,
        lifecycle=None, lifecycle_grant=None, lifecycle_authority=None,
    ):
        if (type(resolved) is not ResolvedTrainingRequest or type(run) is not TrainingRunRef
                or type(limits) is not NativeProcessLimits or type(scope) is not ExecutionScopeV1
                or type(retained_read) is not bool):
            raise TypeError("exact native plan inputs required")
        if selection_digest is not None:
            digest_text(selection_digest, "selection_digest")
        if lifecycle is not None and (type(lifecycle) is not NativeLifecycleLaunchV1
                or lifecycle.policy.maximum_journal_bytes > limits.maximum_file_bytes):
            raise ValueError("native lifecycle selection is invalid")
        if lifecycle is None and (lifecycle_grant is not None or lifecycle_authority is not None):
            raise ValueError("native lifecycle authority needs an explicit selection")
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
        profile_document = {
            "schema": "synaptic-native-provider-profile/v1",
            "profile": asdict(projection.profile), "limits": asdict(limits),
            "scope": scope.to_dict(),
        }
        if lifecycle is not None:
            profile_document["lifecycle"] = lifecycle.to_dict()
            matching = [member["sha256"] for tree in strict_json(runtime_capture)["trees"]
                        for member in tree["members"]
                        if (tree["root"] + "/" + member["path"]).endswith("/tuner/runtime/native_lifecycle_protocol.py")]
            if matching != [lifecycle.protocol_sha256]:
                raise ValueError("native lifecycle codec is not the captured source")
        self.profile_digest = _digest(profile_document)
        self._clock = clock
        self._qualification, self._qualification_authority = qualification, qualification_authority
        self._restricted_admission, self._restricted_authority = restricted_admission, restricted_authority
        self._selection_digest, self._retained_read = selection_digest, retained_read
        self.lifecycle, self._lifecycle_grant, self._lifecycle_authority = lifecycle, lifecycle_grant, lifecycle_authority
        available = self.qualified() or self.restricted() or (retained_read and self.restricted_selection())
        lifecycle_available = available and self.lifecycle_allowed()
        self.descriptor = ProviderDescriptor(
            "synaptic-provider-descriptor/v1", "native-mlx", "Apple native MLX", "1.0.0",
            ProviderCapabilities(available, lifecycle_available, lifecycle_available, lifecycle_available, available, False),
        )
        version = "1.0.0" if lifecycle is None else "2.0.0"
        self.executor_descriptor = ExecutorDescriptorV1("native-mlx", "owned-native-process", version)
        self.adapter_descriptor = AdapterDescriptorV1("native-mlx", "no-replay-native-reconciliation", version)
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

    def restricted_selection(self):
        proof = self._restricted_admission
        try:
            workload = parse_native_workload(self.workload)
            return (
                type(proof) is NativeRestrictedAdmissionV1
                and proof.profile_digest == self.profile_digest
                and proof.source_digest == self.resolved.source_digest
                and proof.runtime_digest == self.resolved.runtime_digest
                and proof.prepared_digest == workload.source.content_digest
                and proof.workload_digest == self.resolved.workload_digest
                and self._selection_digest is not None
                and proof.selection_digest == self._selection_digest
                and proof.run_id == self.run.run_id
                and proof.maximum_updates == workload.controls.iterations
                and self._restricted_authority.authenticate(proof) is True
            )
        except Exception:
            return False

    def restricted(self):
        if not self.restricted_selection():
            return False
        try:
            now = datetime.fromisoformat(self._clock.now().replace("Z", "+00:00"))
            expires = datetime.fromisoformat(self._restricted_admission.expires_at.replace("Z", "+00:00"))
            return now < expires <= now + timedelta(hours=1)
        except Exception:
            return False

    def admitted(self):
        return self.qualified() or self.restricted()

    def require_admitted(self, *, attempt=None, model_digest=None, tokenizer_digest=None):
        if self._retained_read:
            raise ValueError("native_retained_read_has_no_dispatch_authority")
        if self.lifecycle is not None:
            self.require_lifecycle()
        if self.qualified():
            return
        proof = self._restricted_admission
        if (not self.restricted() or proof.attempt != attempt
                or proof.model_digest != model_digest
                or proof.tokenizer_digest != tokenizer_digest):
            raise ValueError("native_restricted_admission_required")

    def describe(self, provider):
        if provider != self.provider:
            raise ValueError("native provider differs")
        return self.descriptor

    def lifecycle_allowed(self):
        proof = self._lifecycle_grant
        try:
            if (self.lifecycle is None or type(proof) is not NativeLifecycleGrantV1
                    or proof.profile_digest != self.profile_digest
                    or proof.source_digest != self.resolved.source_digest
                    or proof.runtime_digest != self.resolved.runtime_digest
                    or proof.launch_digest != _digest(self.lifecycle.to_dict())
                    or self._lifecycle_authority.authenticate(proof) is not True):
                return False
            if proof.mode == "qualified":
                return self.qualified() and proof.hardware_digest == self._qualification.hardware_digest
            return (self.restricted_selection() and proof.hardware_digest == self._restricted_admission.hardware_digest
                    and proof.selection_digest == self._selection_digest and proof.run_id == self.run.run_id
                    and proof.attempt == self._restricted_admission.attempt)
        except Exception:
            return False

    def require_lifecycle(self):
        if not self.lifecycle_allowed():
            raise ValueError("native_lifecycle_authority_required")

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
        ready = not self._retained_read and self.admitted() and (self.lifecycle is None or self.lifecycle_allowed())
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
        if kind not in (EffectKind.STAGE, EffectKind.SUBMIT) and not (
                kind is EffectKind.CANCEL and self.lifecycle is not None and self.lifecycle_allowed()):
            raise ValueError("native lifecycle operation is unavailable")
        return CanonicalProviderPayloadV1.build(
            "native-mlx", f"{kind.value}-payload/v2", self.resolved.workload_digest,
        )
