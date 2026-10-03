"""Authenticated observations and bounded artifact streaming for packaged Modal."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterator, Protocol

from tuner.execution.foundation_v2.canonical import (
    digest_text,
    parse_canonical_object,
    safe_ref,
)

from .contracts import operation_path, provider_entry_identity
from .coordinator_producer import MODAL_TRAINING_ARTIFACT_BOUNDS_V1
from .facade import ExplicitModal154ReadFacade
from .packaged_binding import EXACT_ARTIFACT_ROLES, ModalPackagedCommandBinding
from .packaged_deployment import ModalPackagedDeploymentObserver


class ModalPackagedEvidenceVerifier(Protocol):
    def verify(self, purpose: str, payload: bytes, tag: bytes, key_ref: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class ModalPackagedArtifactMember:
    role: str
    path: str
    size: int
    sha256: str
    provider_entry_id: str

    def __post_init__(self) -> None:
        safe_ref(self.role, "artifact_role")
        if type(self.path) is not str or not self.path.startswith("operations/"):
            raise ValueError("packaged artifact path is invalid")
        if type(self.size) is not int or not 1 <= self.size <= MODAL_TRAINING_ARTIFACT_BOUNDS_V1.max_artifact_bytes:
            raise ValueError("packaged artifact size is invalid")
        digest_text(self.sha256, "artifact_sha256")
        safe_ref(self.provider_entry_id, "provider_entry_id")


@dataclass(frozen=True, slots=True)
class ModalPackagedCompletionObservation:
    effect_id: str
    command_digest: str
    provider_job_ref: str
    runtime_release_digest: str
    provider_runtime_binding_digest: str
    execution_binding_digest: str
    completion_digest: str
    members: tuple[ModalPackagedArtifactMember, ...]

    def __post_init__(self) -> None:
        safe_ref(self.effect_id, "effect_id")
        safe_ref(self.provider_job_ref, "provider_job_ref")
        for name in (
            "command_digest", "runtime_release_digest",
            "provider_runtime_binding_digest", "execution_binding_digest",
            "completion_digest",
        ):
            digest_text(getattr(self, name), name)
        if {item.role for item in self.members} != EXACT_ARTIFACT_ROLES:
            raise ValueError("packaged completion artifact roles are not exact")
        if sum(item.size for item in self.members) > MODAL_TRAINING_ARTIFACT_BOUNDS_V1.max_artifact_total_bytes:
            raise ValueError("packaged artifact inventory exceeds its total bound")


class ModalPackagedReader:
    __slots__ = ("_facade", "_observer", "_verifier", "_key_ref")

    def __init__(
        self,
        *,
        facade: ExplicitModal154ReadFacade,
        deployment_observer: ModalPackagedDeploymentObserver,
        verifier: ModalPackagedEvidenceVerifier,
        key_ref: str,
    ) -> None:
        if type(facade) is not ExplicitModal154ReadFacade \
                or type(deployment_observer) is not ModalPackagedDeploymentObserver:
            raise TypeError("exact packaged Modal reader collaborators required")
        if facade.client is not deployment_observer.client \
                or facade.binding != deployment_observer.client_binding:
            raise ValueError("packaged reader collaborators use different clients")
        if not hasattr(verifier, "verify"):
            raise TypeError("packaged evidence verifier required")
        self._facade, self._observer, self._verifier = facade, deployment_observer, verifier
        self._key_ref = safe_ref(key_ref, "key_ref")

    def observe_completion(
        self,
        binding: ModalPackagedCommandBinding,
        *,
        provider_job_ref: str,
    ) -> ModalPackagedCompletionObservation:
        if type(binding) is not ModalPackagedCommandBinding:
            raise TypeError("exact packaged command binding required")
        job_ref = safe_ref(provider_job_ref, "provider_job_ref")
        facts = self._observer.observe(
            binding.provider_facts,
            runtime_release=binding.runtime_release,
            provider_binding=binding.provider_binding,
        )
        command = binding.command
        effect_id = command.operation.effect.effect_id
        root = operation_path(effect_id, "evidence")
        completion = self._facade.read_complete(
            facts.control_volume_id,
            root + "/packaged-completion.json",
            max_bytes=64 * 1024,
        )
        tag = self._facade.read_complete(
            facts.control_volume_id,
            root + "/packaged-completion.mac",
            max_bytes=128,
        )
        try:
            valid = self._verifier.verify(
                "modal-packaged-completion/v1", completion, tag, self._key_ref,
            )
        except Exception:
            raise ValueError("packaged completion authentication unavailable") from None
        if valid is not True:
            raise ValueError("packaged completion authentication failed")
        document = parse_canonical_object(completion, name="packaged completion")
        fields = {
            "schema_version", "effect_id", "command_digest", "provider_job_ref",
            "runtime_release_digest", "provider_runtime_binding_digest",
            "execution_binding_digest", "stage_receipt_sha256", "inventory_sha256",
            "terminal_sha256", "members",
        }
        if set(document) != fields \
                or document.get("schema_version") != "synaptic-modal-packaged-completion/v1" \
                or document.get("effect_id") != effect_id \
                or document.get("command_digest") != command.digest \
                or document.get("provider_job_ref") != job_ref \
                or document.get("runtime_release_digest") != binding.runtime_release.manifest_digest \
                or document.get("provider_runtime_binding_digest") != binding.provider_binding.binding_digest \
                or document.get("execution_binding_digest") != binding.execution_binding.binding_digest:
            raise ValueError("packaged completion differs from retained binding")
        records = document["members"]
        if type(records) is not list or len(records) != 5:
            raise ValueError("packaged completion inventory is invalid")
        members = tuple(
            ModalPackagedArtifactMember(
                record["role"], record["path"], record["size"], record["sha256"],
                record["provider_entry_id"],
            )
            for record in records
            if type(record) is dict and set(record) == {
                "role", "path", "size", "sha256", "provider_entry_id",
            }
        )
        if len(members) != 5:
            raise ValueError("packaged completion inventory is invalid")
        prefix = operation_path(effect_id, "output") + "/"
        listing = self._facade.list_prefix(
            facts.artifact_volume_id, prefix, max_entries=6,
        )
        expected = tuple(sorted(
            (member.path, member.size, member.provider_entry_id) for member in members
        ))
        if tuple(sorted(listing)) != expected or any(
            member.provider_entry_id != provider_entry_identity(
                facts.artifact_volume_id, member.path, member.size,
            ) for member in members
        ):
            raise ValueError("packaged artifact inventory readback differs")
        return ModalPackagedCompletionObservation(
            effect_id, command.digest, job_ref,
            binding.runtime_release.manifest_digest,
            binding.provider_binding.binding_digest,
            binding.execution_binding.binding_digest,
            hashlib.sha256(completion).hexdigest(), members,
        )

    def iter_artifact(
        self,
        binding: ModalPackagedCommandBinding,
        observation: ModalPackagedCompletionObservation,
        *,
        role: str,
        maximum_bytes: int,
    ) -> Iterator[bytes]:
        if type(observation) is not ModalPackagedCompletionObservation \
                or observation.command_digest != binding.command_digest:
            raise ValueError("packaged artifact observation mismatch")
        matches = tuple(member for member in observation.members if member.role == role)
        if len(matches) != 1:
            raise ValueError("packaged artifact role is unavailable")
        member = matches[0]
        if type(maximum_bytes) is not int or not member.size <= maximum_bytes \
                <= MODAL_TRAINING_ARTIFACT_BOUNDS_V1.max_artifact_bytes:
            raise ValueError("packaged artifact stream bound is invalid")
        facts = binding.provider_facts
        prefix = operation_path(observation.effect_id, "output") + "/"
        expected = tuple(sorted(
            (item.path, item.size, item.provider_entry_id)
            for item in observation.members
        ))
        if tuple(sorted(self._facade.list_prefix(
            facts.artifact_volume_id, prefix, max_entries=6,
        ))) != expected:
            raise ValueError("packaged artifact changed before streaming")
        digest, size = hashlib.sha256(), 0
        for chunk in self._facade.iter_complete(
            facts.artifact_volume_id, member.path, max_bytes=maximum_bytes,
        ):
            if type(chunk) is not bytes or not chunk:
                raise ValueError("packaged artifact stream is invalid")
            size += len(chunk)
            digest.update(chunk)
            if size > member.size:
                raise ValueError("packaged artifact exceeds its inventory")
            yield chunk
        if (size, digest.hexdigest()) != (member.size, member.sha256):
            raise ValueError("packaged artifact stream is incomplete")
        if tuple(sorted(self._facade.list_prefix(
            facts.artifact_volume_id, prefix, max_entries=6,
        ))) != expected:
            raise ValueError("packaged artifact changed during streaming")

    def read_evaluation(self, binding, *, provider_job_ref: str, workload_bytes: bytes) -> bytes:
        """Authenticate one separately retained post-training phase record.

        Training completion still has exactly five roles. The opt-in record
        binds both that completed artifact set and the signed workload's config.
        """
        from tuner.training.recipes import canonical_json_bytes
        from tuner.training.packaged_compilation import compile_packaged_sft_workload
        from tuner.training.contracts import CanonicalDocument
        from tuner.training.post_training import validate_post_training_config
        from tuner.runtime.post_training_eval import (
            canonical_evaluation_document_bytes, MAX_EVALUATION_RECORD_BYTES,
            validate_evaluation_record,
        )
        workload = json.loads(workload_bytes)
        compiled = compile_packaged_sft_workload(
            resolved_config=CanonicalDocument.from_mapping(workload["configuration"]["document"]))
        if (compiled.canonical_bytes != workload_bytes
                or compiled.fingerprint != binding.execution_binding.workload_digest):
            raise ValueError("packaged evaluation workload differs")
        config = validate_post_training_config(workload["configuration"]["document"].get("post_training"))
        if config is None:
            raise ValueError("packaged evaluation was not configured")
        completion = self.observe_completion(binding, provider_job_ref=provider_job_ref)
        facts = binding.provider_facts
        relative = operation_path(completion.effect_id, "evaluation")
        raw = self._facade.read_complete(facts.artifact_volume_id, relative + "/record.json",
                                         max_bytes=MAX_EVALUATION_RECORD_BYTES)
        tag = self._facade.read_complete(facts.artifact_volume_id, relative + "/record.mac", max_bytes=128)
        if self._verifier.verify("modal-packaged-evaluation/v1", raw, tag, self._key_ref) is not True:
            raise ValueError("packaged evaluation authentication failed")
        document = json.loads(raw)
        expected = {
            "schema_version": "synaptic-modal-packaged-evaluation/v1",
            "effect_id": completion.effect_id,
            "command_digest": binding.command_digest,
            "provider_job_ref": completion.provider_job_ref,
            "execution_binding_digest": binding.execution_binding.binding_digest,
            "training_completion_sha256": completion.completion_digest,
            "post_training_sha256": hashlib.sha256(canonical_json_bytes(config)).hexdigest(),
        }
        if (type(document) is not dict or set(document) != {*expected, "evaluation"}
                or canonical_evaluation_document_bytes(document) != raw
                or any(document[key] != value for key, value in expected.items())):
            raise ValueError("packaged evaluation binding mismatch")
        record = validate_evaluation_record(document["evaluation"], config=config)
        adapter = next(member for member in completion.members if member.role == "final_model")
        if (record["bindings"]["workload_digest"] != binding.execution_binding.workload_digest
                or record["bindings"]["adapter_digest"] != adapter.sha256):
            raise ValueError("packaged evaluation model binding mismatch")
        # Revalidate the deployment and completion after the read; a signed
        # record from another generation cannot qualify this retained run.
        if self.observe_completion(binding, provider_job_ref=provider_job_ref) != completion:
            raise ValueError("packaged evaluation completion changed")
        return raw


__all__ = [
    "ModalPackagedArtifactMember",
    "ModalPackagedCompletionObservation",
    "ModalPackagedEvidenceVerifier",
    "ModalPackagedReader",
]
