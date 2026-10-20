"""Native MLX family for the existing TrainingAPI/RunsAPI coordinator.

Host-owned authenticated catalogs bind exact commands to already prepared private
inputs and invocations. A bounded synchronous submit is intentional: no background
thread touches a host's thread-affine record store. Unknown effects never replay.
Logs/cancel/reconcile are unavailable public capabilities in this version.
"""

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

from synaptic_tuner.api.v1.reference.provider_family import ProviderFamilyV1, require_methods
from tuner.execution.coordinator_v1.model import (
    ArtifactManifestV1, ArtifactVerificationContentV1, AuthenticatedArtifactVerificationReceiptV1,
    EffectIntentV1,
    ProviderReadPurposeV1, ProviderRunObservationContentV1, ProviderRunPhaseV1,
    ProviderRunReadRequestV1, VerificationVerdictV1,
)
from tuner.execution.coordinator_v1.state_machine import _derive_foundation
from tuner.execution.foundation_v2.canonical import canonical_bytes, digest_text, safe_ref
from tuner.execution.foundation_v2.commands import parse_exact_command
from tuner.execution.foundation_v2.executors import mint_resolved_executor
from tuner.execution.foundation_v2.observations import ObservationDisposition, ProviderObservationV1
from tuner.execution.foundation_v2.references import ProviderStageRefV1, ScopedProviderRunRefV1
from tuner.runtime.native_files import NativeFile, NativeTree
from tuner.runtime.native_mlx_input import NativeModelSnapshot
from tuner.runtime.native_mlx_material import canonical, strict_json
from tuner.runtime.native_mlx_publication import NativeArtifactBudget, verify_native_publication
from tuner.runtime.native_process import NativeInvocation, NativeOwnedProcess, NativeProcessLimits
from tuner.training.native_mlx_planning import NativePlanningV1


@dataclass(frozen=True)
class NativeCommandMaterialV1:
    """Content which the host catalog's independent authority authenticates.

    The host must validate its execution-source evidence, prepared source lease,
    candidate wheel/install provenance and current hardware before issuing this
    binding. Neither a matching digest nor catalog presence replaces that step.
    """

    command_bytes: bytes
    invocation: NativeInvocation
    model: NativeModelSnapshot
    tokenizer: NativeModelSnapshot
    tokenizer_members: tuple[str, ...]
    output: str
    artifact_budget: NativeArtifactBudget
    authority_ref: str
    key_ref: str
    tag: str

    def __post_init__(self):
        if (type(self.command_bytes) is not bytes or type(self.invocation) is not NativeInvocation
                or type(self.model) is not NativeModelSnapshot or type(self.tokenizer) is not NativeModelSnapshot
                or type(self.artifact_budget) is not NativeArtifactBudget
                or type(self.tokenizer_members) is not tuple or not self.tokenizer_members
                or self.tokenizer_members != tuple(sorted(set(self.tokenizer_members)))
                or any(type(name) is not str for name in self.tokenizer_members)
                or type(self.output) is not str
                or Path(self.output).parent != Path(self.invocation.private_root)):
            raise ValueError("native command material fields are invalid")
        command = parse_exact_command(self.command_bytes)
        if command.digest != self.invocation.binding_digest:
            raise ValueError("native invocation does not bind command")
        safe_ref(self.authority_ref, "authority_ref")
        safe_ref(self.key_ref, "key_ref")
        digest_text(self.tag, "tag")

    def content_bytes(self):
        return canonical({
            "schema": "synaptic-native-command-material/v1",
            "command": strict_json(self.command_bytes),
            "invocation": asdict(self.invocation), "model": asdict(self.model),
            "tokenizer": asdict(self.tokenizer), "tokenizer_members": self.tokenizer_members,
            "output": self.output, "artifact_budget": asdict(self.artifact_budget),
        })

    @property
    def canonical_bytes(self):
        return canonical({
            "content": strict_json(self.content_bytes()), "authority_ref": self.authority_ref,
            "key_ref": self.key_ref, "tag": self.tag,
        })

    @classmethod
    def parse(cls, raw):
        value = strict_json(raw, maximum=4 * 1024 * 1024)
        if type(value) is not dict or set(value) != {"content", "authority_ref", "key_ref", "tag"}:
            raise ValueError("native material envelope fields differ")
        content = value["content"]
        if (type(content) is not dict or set(content) != {
                "schema", "command", "invocation", "model", "tokenizer",
                "tokenizer_members", "output", "artifact_budget",
            } or content["schema"] != "synaptic-native-command-material/v1"):
            raise ValueError("native material content fields differ")
        def snapshot(document):
            return NativeModelSnapshot(
                document["repository"], document["revision"], document["root"],
                tuple(NativeFile(**item) for item in document["files"]),
            )
        invocation = dict(content["invocation"])
        invocation["limits"] = NativeProcessLimits(**invocation["limits"])
        result = cls(
            canonical(content["command"]), NativeInvocation(**invocation),
            snapshot(content["model"]), snapshot(content["tokenizer"]),
            tuple(content["tokenizer_members"]), content["output"],
            NativeArtifactBudget(**content["artifact_budget"]),
            value["authority_ref"], value["key_ref"], value["tag"],
        )
        if result.canonical_bytes != raw:
            raise ValueError("native material canonical bytes differ")
        return result


class NativeCommandCatalogV1:
    """Create-only authenticated command catalog over the consumer record port."""

    def __init__(self, records, authority):
        require_methods(records, "create", "read")
        require_methods(authority, "authenticate")
        self.records, self.authority = records, authority

    def resolve(self, command_digest):
        digest_text(command_digest, "command_digest")
        stored = self.records.read(partition="effects", key="native-material/" + command_digest)
        if stored is None:
            raise ValueError("native command material is absent")
        material = NativeCommandMaterialV1.parse(stored.canonical)
        if (material.invocation.binding_digest != command_digest
                or self.authority.authenticate(material) is not True):
            raise ValueError("native retained command material is unauthenticated")
        return material

    def publish_once(self, material):
        if type(material) is not NativeCommandMaterialV1 or self.authority.authenticate(material) is not True:
            raise ValueError("native command material is unauthenticated")
        raw = material.canonical_bytes
        self.records.create(partition="effects", key="native-material/" + material.invocation.binding_digest,
                            canonical=raw)
        retained = self.resolve(material.invocation.binding_digest)
        if retained.canonical_bytes != raw:
            raise ValueError("native command material collision")
        return retained


class _NativeFamily:
    effect_kinds = ("stage", "submit")
    payload_schemas = ("stage-payload/v2", "submit-payload/v2")

    def __init__(self, *, planning, catalog, material_authority, records,
                 evidence_authority, foundation_authenticator, assessment_authenticator, clock):
        if type(planning) is not NativePlanningV1:
            raise TypeError("exact native planning required")
        for port, methods in (
            (catalog, ("resolve",)), (material_authority, ("authenticate",)),
            (evidence_authority, ("observation", "verification", "authenticate_observation",
                                  "authenticate_verification", "authenticate_log_page",
                                  "process_outcome", "authenticate_process_outcome")),
        ):
            require_methods(port, *methods)
        self.planning, self.catalog, self.material_authority = planning, catalog, material_authority
        self.transport = NativeOwnedProcess(records)
        self.records = records
        self.authority, self.foundation, self.assessments, self.clock = (
            evidence_authority, foundation_authenticator, assessment_authenticator, clock,
        )
        self.descriptor = planning.executor_descriptor
        self.provider_id, self.profile_ref = planning.provider.provider_id, planning.provider.profile_ref
        self.account_ref, self.namespace_ref = planning.scope.account_ref, planning.scope.namespace_ref

    def _material(self, command_digest):
        material = self.catalog.resolve(command_digest)
        if (type(material) is not NativeCommandMaterialV1
                or self.material_authority.authenticate(material) is not True
                or type(material.invocation) is not NativeInvocation
                or type(material.model) is not NativeModelSnapshot
                or type(material.tokenizer) is not NativeModelSnapshot
                or type(material.artifact_budget) is not NativeArtifactBudget):
            raise ValueError("native_command_material_unauthenticated")
        command = parse_exact_command(material.command_bytes)
        self.planning.validate_preparation(command.preparation)
        invocation = material.invocation
        if (command.digest != command_digest or invocation.binding_digest != command_digest
                or invocation.limits != self.planning.limits
                or command.executor != self.descriptor
                or command.operation.effect.kind.value not in self.effect_kinds
                or Path(material.output).parent != Path(invocation.private_root)):
            raise ValueError("native_command_binding_differs")
        with NativeTree(Path(invocation.private_root), private=True) as root:
            raw = root.read(Path(invocation.request).name, maximum=2 * 1024 * 1024)
            request = strict_json(raw)
            if (hashlib.sha256(raw).hexdigest() != invocation.request_digest
                    or canonical(request) != raw
                    or canonical(request["workload"]) != self.planning.workload
                    or canonical(request["model_snapshot"]) != canonical(asdict(material.model))
                    or canonical(request["tokenizer_snapshot"]) != canonical(asdict(material.tokenizer))
                    or request["output"] != material.output):
                raise ValueError("native_request_material_differs")
            for key in ("runtime_capture", "dependency_lock", "dataset", "output"):
                if type(request[key]) is not str or Path(request[key]).parent != root.path:
                    raise ValueError("native_private_material_root_differs")
            capture = root.read(Path(request["runtime_capture"]).name, maximum=16 * 1024 * 1024)
            if capture != self.planning.runtime_capture:
                raise ValueError("native_runtime_capture_differs")
            runtime = strict_json(capture)
            if (invocation.executable != runtime["executable"]
                    or invocation.executable_digest != runtime["runtime"]["executable_sha256"]):
                raise ValueError("native_interpreter_binding_differs")
            lock = root.read(Path(request["dependency_lock"]).name, maximum=1024 * 1024)
            if hashlib.sha256(lock).hexdigest() != runtime["runtime"]["dependency_lock_sha256"]:
                raise ValueError("native_dependency_lock_differs")
            source = request["workload"]["prepared_input"]
            dataset = root.read(Path(request["dataset"]).name, maximum=64 * 1024 * 1024)
            if len(dataset) != source["size_bytes"] or hashlib.sha256(dataset).hexdigest() != source["content_digest"]:
                raise ValueError("native_prepared_source_differs")
            members = {
                str(Path(tree["root"]) / item["path"]): item["sha256"]
                for tree in runtime["trees"] for item in tree["members"]
            }
            for role, suffix in (
                ("worker", "/tuner/runtime/native_mlx_child.py"),
                ("guardian", "/tuner/runtime/native_process_guardian.py"),
            ):
                path = getattr(invocation, role)
                if (not path.endswith(suffix)
                        or members.get(path) != getattr(invocation, role + "_digest")):
                    raise ValueError("native_worker_source_binding_differs")
        return material, command

    def execute_once(self, payload, request):
        self.planning.require_qualified()
        material, command = self._material(request.command_digest)
        expected_payload = self.planning.payload(command.preparation, command.operation.effect.kind)
        if (payload != expected_payload or request.descriptor_digest != self.descriptor.digest
                or (request.provider_id, request.profile_ref, request.account_ref, request.namespace_ref)
                != (self.provider_id, self.profile_ref, self.account_ref, self.namespace_ref)
                or request.effect_kind != command.operation.effect.kind.value
                or request.payload_schema != payload.payload_kind or request.input_digest != payload.input_digest):
            raise ValueError("native_resolution_binding_differs")
        base = dict(
            effect_id=command.operation.effect.effect_id, command_digest=command.digest,
            executor_descriptor_digest=self.descriptor.digest, resolution_digest=request.digest,
            result_epoch=1,
        )
        if request.effect_kind == "stage":
            # Staging admits already prepared immutable host inputs. It performs
            # no install, network access, model load or provider resource creation.
            return ProviderObservationV1(
                **base, disposition=ObservationDisposition.FOUND,
                stage_ref=ProviderStageRefV1(
                    self.provider_id, self.profile_ref, self.account_ref, self.namespace_ref,
                    "native-stage-" + command.digest,
                ),
            )
        try:
            outcome = self.transport.run_once(material.invocation)
            terminal = self.authority.process_outcome(material.invocation.digest, outcome)
            key = "native-terminal/" + material.invocation.digest
            if self.records.create(partition="effects", key=key, canonical=terminal) is not True:
                raise ValueError("native_terminal_evidence_collision")
            retained = self.records.read(partition="effects", key=key)
            if (retained is None or retained.canonical != terminal
                    or self.authority.authenticate_process_outcome(terminal, material.invocation.digest) != outcome):
                raise ValueError("native_terminal_evidence_readback_differs")
        except Exception:
            # Any failure after/around claim remains indeterminate. No recovery
            # or cancellation authority is inferred from a saved PID.
            return ProviderObservationV1(**base, disposition=ObservationDisposition.INDETERMINATE)
        return ProviderObservationV1(
            **base, disposition=ObservationDisposition.FOUND,
            provider_run=self._run_ref(material),
        )

    def _run_ref(self, material):
        return ScopedProviderRunRefV1(
            self.provider_id, self.profile_ref, self.account_ref, self.namespace_ref,
            "native-run-" + material.invocation.digest,
        )

    def _read(self, request, purpose):
        if type(request) is not ProviderRunReadRequestV1 or request.purpose is not purpose:
            raise ValueError("native_read_request_differs")
        # Reconstruct the sealed read request, including its retained exact
        # Foundation assessment. Never mint a fresh timestamped assessment.
        ProviderRunReadRequestV1(**{
            name: getattr(request, name) for name in request.__dataclass_fields__
        })
        record = request.foundation_record
        if (self.foundation.authenticate_grant(record.grant, record.command_bytes) is not True
                or any(self.foundation.authenticate_receipt(value) is not True for value in record.results)
                or any(self.foundation.authenticate_invalid_evidence(value) is not True
                       for value in record.invalid_evidence)):
            raise ValueError("native_foundation_evidence_invalid")
        assessment = type(request.assessment).parse(request.assessment.canonical_bytes)
        if (assessment != request.assessment or self.assessments.authenticate(assessment) is not True
                or assessment.content.foundation_record_digest != record.record_digest
                or assessment.content.authenticated_receipt_digests
                != tuple(item.authenticated_receipt_digest for item in record.results)):
            raise ValueError("native_foundation_assessment_invalid")
        binding, outcome, bound = _derive_foundation(
            EffectIntentV1.from_command_bytes(request.submit_command_bytes),
            record, assessment, self.foundation, self.assessments, request.provider_run,
        )
        if (binding != request.foundation_binding or outcome != request.foundation_outcome
                or bound != request.provider_run
                or bound.authenticated_receipt_digest != request.found_receipt_digest):
            raise ValueError("native_foundation_read_derivation_differs")
        command = parse_exact_command(request.submit_command_bytes)
        material, owned = self._material(command.digest)
        if (owned.canonical_bytes != command.canonical_bytes
                or command.operation.effect.kind.value != "submit"
                or request.run != self.planning.run
                or request.provider_run.reference != self._run_ref(material)):
            raise ValueError("native_known_run_binding_differs")
        return material

    def _outcome(self, material):
        state = self.transport.observe(material.invocation)
        if state["state"] != "terminated":
            raise ValueError("native_termination_unconfirmed")
        outcome = state["outcome"]
        terminal = self.records.read(partition="effects", key="native-terminal/" + material.invocation.digest)
        if (terminal is None or self.authority.authenticate_process_outcome(
                terminal.canonical, material.invocation.digest) != outcome):
            raise ValueError("native_terminal_evidence_differs")
        if outcome.get("termination_confirmed") is not True:
            raise ValueError("native_termination_unconfirmed")
        return outcome

    def observe(self, request):
        material = self._read(request, ProviderReadPurposeV1.OBSERVE)
        outcome = self._outcome(material)
        succeeded = outcome["reason"] == "worker_exited" and outcome["exit_code"] == 0
        phase = ProviderRunPhaseV1.SUCCEEDED if succeeded else ProviderRunPhaseV1.FAILED
        ref = self._run_ref(material)
        content = ProviderRunObservationContentV1(
            "synaptic-provider-run-observation-content/v1", request.request_digest,
            request.source_workflow_record_digest, request.source_revision, request.run,
            request.provider_run.binding_digest, ref.provider_id, ref.profile_ref,
            ref.account_ref, ref.namespace_ref, ref.provider_job_ref, phase,
            canonical_bytes({"invocation_digest": material.invocation.digest, "outcome": outcome}),
            None if succeeded else "native_worker_failed", "native-mlx-reader", "1.0.0", self.clock.now(),
        )
        return self.authority.observation(content)

    def _artifacts(self, material):
        outcome = self._outcome(material)
        if outcome["reason"] != "worker_exited" or outcome["exit_code"] != 0:
            raise ValueError("native_worker_did_not_succeed")
        return verify_native_publication(
            output=Path(material.output), workload=self.planning.workload,
            model=material.model, tokenizer=material.tokenizer,
            tokenizer_members=material.tokenizer_members, budget=material.artifact_budget,
        )

    def _manifest(self, material, artifacts):
        return ArtifactManifestV1.build(
            run=self.planning.run, provider_run=self._run_ref(material),
            artifacts=tuple(item.descriptor for item in artifacts),
            artifact_source_digest=material.invocation.digest,
            canonical_evidence=canonical_bytes({
                "format": "mlx-lm-lora-safetensors", "invocation_digest": material.invocation.digest,
                "command_digest": material.invocation.binding_digest,
            }),
        )

    def artifacts(self, request):
        material = self._read(request, ProviderReadPurposeV1.ARTIFACTS)
        return self._manifest(material, self._artifacts(material))

    def iter_artifact_bytes(self, request, manifest, role, *, maximum_bytes):
        material = self._read(request, ProviderReadPurposeV1.ARTIFACTS)
        artifacts = self._artifacts(material)
        if manifest != self._manifest(material, artifacts):
            raise ValueError("native_artifact_manifest_changed")
        matches = [item for item in artifacts if item.descriptor.role == role]
        if (len(matches) != 1 or type(maximum_bytes) is not int
                or maximum_bytes < matches[0].descriptor.size_bytes):
            raise ValueError("native_artifact_stream_bound_invalid")
        raw = matches[0].content
        for offset in range(0, len(raw), 1024 * 1024):
            yield raw[offset:offset + 1024 * 1024]

    def verify(self, workflow, manifest):
        command = parse_exact_command(workflow.submit.canonical_command_bytes)
        material, _ = self._material(command.digest)
        if (workflow.run != self.planning.run
                or workflow.provider_run_ref.reference != self._run_ref(material)
                or manifest != self._manifest(material, self._artifacts(material))):
            raise ValueError("native_artifact_verification_binding_differs")
        content = ArtifactVerificationContentV1(
            "synaptic-artifact-verification-content/v1", workflow.record_digest,
            workflow.revision, workflow.run, workflow.provider_run_ref.binding_digest,
            manifest.manifest_digest, manifest.artifact_source_digest, manifest.artifacts,
            manifest.artifacts, VerificationVerdictV1.VERIFIED, None,
            "native-mlx-artifact-verifier", "1.0.0",
            canonical_bytes({"invocation_digest": material.invocation.digest}), self.clock.now(),
        )
        receipt = self.authority.verification(content)
        if type(receipt) is not AuthenticatedArtifactVerificationReceiptV1 or receipt.content != content:
            raise ValueError("native_artifact_receipt_differs")
        return receipt

    def replay(self, workflow, manifest, prior_receipt):
        if self.authenticate(prior_receipt) is not True:
            raise ValueError("native_prior_verification_unauthenticated")
        return self.verify(workflow, manifest)

    def authenticate(self, value):
        return self.authority.authenticate_verification(value)

    def logs(self, *args, **kwargs):
        raise ValueError("native_logs_unavailable")


class _ExecutorResolver:
    def __init__(self, executor):
        self.executor = executor

    def resolve(self, request):
        return mint_resolved_executor(request, self.executor)


class _NoReconciliation:
    def resolve(self, request):
        raise ValueError("native_reconciliation_unavailable")


class _Authenticator:
    def __init__(self, method):
        self.method = method

    def authenticate(self, value):
        return self.method(value)


def native_mlx_family(**ports):
    """Compose candidate/qualified native adapters without global registration.

    Consumer uses this result with ``compose_reference_host`` and its existing
    request/grant/store ports. Host ``GrantAuthorityPort.bind`` retains the
    independently authenticated ``NativeCommandMaterialV1`` for each command.
    """
    internal = _NativeFamily(**ports)
    return ProviderFamilyV1(
        internal.planning.descriptor, internal.planning, internal.planning, internal,
        _ExecutorResolver(internal), _NoReconciliation(), internal.authority, internal,
        _Authenticator(internal.authority.authenticate_observation),
        _Authenticator(internal.authority.authenticate_log_page),
    )
