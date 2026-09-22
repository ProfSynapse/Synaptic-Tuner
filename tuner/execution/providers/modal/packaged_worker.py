"""Packaged Modal worker composition around the provider-neutral trainer seam."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Protocol

from tuner.execution.foundation_v2.canonical import canonical_bytes, digest_text, safe_ref
from tuner.runtime.packaged_sft_execution import (
    PackagedSFTPaths,
    admit_packaged_sft,
    execute_admitted_packaged_sft,
)

from .contracts import operation_path, provider_entry_identity
from .coordinator_producer import MODAL_TRAINING_ARTIFACT_BOUNDS_V1
from .mounted_io import (
    claim_directory,
    hash_regular,
    list_regular_sizes,
    read_regular,
    write_exclusive,
)
from .packaged_binding import EXACT_ARTIFACT_ROLES, ModalPackagedRuntimeFactsV1
from .packaged_dispatch import (
    ModalPackagedDispatchVerifier,
    parse_modal_packaged_dispatch,
)


class PackagedTrainerExecutor(Protocol):
    """Provider-neutral trainer seam; no Modal value appears in this call."""

    def execute(
        self,
        *,
        runtime_release,
        provider_binding,
        execution_binding,
        workload_bytes: bytes,
        artifact_policy,
        paths: PackagedSFTPaths,
        environment: tuple[tuple[str, str], ...],
    ): ...


class ModalPackagedEvidenceSigner(Protocol):
    def sign(self, purpose: str, payload: bytes, key_ref: str) -> bytes: ...


class InstalledPackagedSFTTrainerExecutor:
    """Production implementation of the separate generic trainer executor."""

    __slots__ = ("_model_preparer", "_runner")

    def __init__(self, *, model_preparer, runner=None) -> None:
        if not callable(model_preparer):
            raise TypeError("packaged model preparer must be callable")
        self._model_preparer, self._runner = model_preparer, runner

    def execute(
        self,
        *,
        runtime_release,
        provider_binding,
        execution_binding,
        workload_bytes,
        artifact_policy,
        paths,
        environment,
    ):
        admitted = admit_packaged_sft(
            runtime_release=runtime_release,
            provider_binding=provider_binding,
            execution_binding=execution_binding,
            workload_bytes=workload_bytes,
            artifact_policy=artifact_policy,
            paths=paths,
            environment=environment,
        )
        return execute_admitted_packaged_sft(
            admitted, model_preparer=self._model_preparer, runner=self._runner,
        )


@dataclass(frozen=True, slots=True)
class ModalPackagedWorkerRoots:
    control: Path
    artifacts: Path
    cache: Path

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not isinstance(value, Path) or not value.is_absolute():
                raise ValueError("Modal packaged worker roots must be absolute")
        if len({self.control, self.artifacts, self.cache}) != 3:
            raise ValueError("Modal packaged worker roots must be distinct")


class ModalPackagedWorker:
    """Admit one signed dispatch, run once, and publish signed completion."""

    __slots__ = ("_facts", "_verifier", "_executor", "_signer", "_roots")

    def __init__(
        self,
        *,
        expected_facts: ModalPackagedRuntimeFactsV1,
        dispatch_verifier: ModalPackagedDispatchVerifier,
        trainer_executor: PackagedTrainerExecutor,
        evidence_signer: ModalPackagedEvidenceSigner,
        roots: ModalPackagedWorkerRoots,
    ) -> None:
        if type(expected_facts) is not ModalPackagedRuntimeFactsV1:
            raise TypeError("exact packaged Modal facts required")
        if not hasattr(dispatch_verifier, "verify") \
                or not hasattr(trainer_executor, "execute") \
                or not hasattr(evidence_signer, "sign"):
            raise TypeError("complete packaged worker collaborators required")
        if type(roots) is not ModalPackagedWorkerRoots:
            raise TypeError("exact packaged worker roots required")
        self._facts, self._verifier = expected_facts, dispatch_verifier
        self._executor, self._signer, self._roots = trainer_executor, evidence_signer, roots

    def _paths(self, dispatch) -> PackagedSFTPaths:
        effect_id = dispatch.submit_command.operation.effect.effect_id
        prepared = self._roots.artifacts / dispatch.stage_receipt.path
        paths = {
            "artifacts": self._roots.artifacts / operation_path(effect_id, "output"),
            "state": self._roots.control / operation_path(effect_id, "state"),
            "tracking": self._roots.control / operation_path(effect_id, "tracking"),
            "cache": self._roots.cache / operation_path(effect_id, "cache"),
            "tmp": self._roots.control / operation_path(effect_id, "tmp"),
        }
        for name, path in paths.items():
            root = self._roots.cache if name == "cache" else (
                self._roots.artifacts if name == "artifacts" else self._roots.control
            )
            claim_directory(root, path)
        return PackagedSFTPaths(prepared, **paths)

    def _publish_completion(self, dispatch, result, job_ref: str) -> bytes:
        effect_id = dispatch.submit_command.operation.effect.effect_id
        inventory = read_regular(
            self._roots.control,
            result.inventory_path,
            4 * 1024 * 1024,
        )
        terminal = read_regular(
            self._roots.control,
            result.terminal_path,
            4 * 1024 * 1024,
        )
        try:
            document = json.loads(inventory.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            raise ValueError("packaged trainer inventory is invalid") from None
        if type(document) is not dict or set(document) != {
            "schema_version", "workload_fingerprint", "artifacts",
        } or document.get("schema_version") != "synaptic-artifact-inventory/v1" \
                or document.get("workload_fingerprint") \
                != dispatch.execution_binding.workload_digest:
            raise ValueError("packaged trainer inventory binding is invalid")
        records = document["artifacts"]
        if type(records) is not list or len(records) != 5:
            raise ValueError("packaged trainer inventory is not exact")
        if {record.get("role") for record in records if type(record) is dict} \
                != EXACT_ARTIFACT_ROLES:
            raise ValueError("packaged trainer inventory roles are not exact")
        if not set(dispatch.artifact_policy.required_kinds) <= EXACT_ARTIFACT_ROLES:
            raise ValueError("packaged trainer artifact policy is invalid")
        members = []
        inventory_members: list[tuple[str, int, str, str]] = []
        for record in records:
            if type(record) is not dict or set(record) != {"role", "path", "sha256", "size"}:
                raise ValueError("packaged trainer inventory member is invalid")
            name, size, digest = record["path"], record["size"], record["sha256"]
            if type(name) is not str or "/" in name or "\\" in name \
                    or name in {"", ".", ".."} \
                    or type(size) is not int \
                    or not 1 <= size \
                    <= MODAL_TRAINING_ARTIFACT_BOUNDS_V1.max_artifact_bytes \
                    or type(digest) is not str:
                raise ValueError("packaged trainer inventory member is invalid")
            digest_text(digest, "artifact_sha256")
            inventory_members.append((name, size, digest, record["role"]))
        if len({item[0] for item in inventory_members}) != 5 \
                or sum(item[1] for item in inventory_members) \
                > MODAL_TRAINING_ARTIFACT_BOUNDS_V1.max_artifact_total_bytes:
            raise ValueError("packaged trainer artifact policy is exceeded")
        expected_listing = tuple(sorted(
            (name, size) for name, size, _, _ in inventory_members
        ))
        if list_regular_sizes(
            self._roots.artifacts,
            self._roots.artifacts / operation_path(effect_id, "output"),
            6,
        ) != expected_listing:
            raise ValueError("packaged trainer artifact inventory is not exact")
        for name, size, digest, role in inventory_members:
            path = self._roots.artifacts / operation_path(effect_id, "output", name)
            if hash_regular(
                self._roots.artifacts, path,
                MODAL_TRAINING_ARTIFACT_BOUNDS_V1.max_artifact_bytes,
            ) != (size, digest):
                raise ValueError("packaged trainer artifact differs from inventory")
            relative = operation_path(effect_id, "output", name)
            members.append({
                "role": role, "path": relative, "size": size,
                "sha256": digest,
                "provider_entry_id": provider_entry_identity(
                    self._facts.artifact_volume_id, relative, size,
                ),
            })
        completion = canonical_bytes({
            "schema_version": "synaptic-modal-packaged-completion/v1",
            "effect_id": effect_id,
            "command_digest": dispatch.submit_command.digest,
            "provider_job_ref": safe_ref(job_ref, "provider_job_ref"),
            "runtime_release_digest": dispatch.runtime_release.manifest_digest,
            "provider_runtime_binding_digest": dispatch.provider_binding.binding_digest,
            "execution_binding_digest": dispatch.execution_binding.binding_digest,
            "stage_receipt_sha256": hashlib.sha256(
                dispatch.stage_receipt.canonical_bytes
            ).hexdigest(),
            "inventory_sha256": hashlib.sha256(inventory).hexdigest(),
            "terminal_sha256": hashlib.sha256(terminal).hexdigest(),
            "members": members,
        })
        try:
            tag = self._signer.sign(
                "modal-packaged-completion/v1", completion, dispatch.key_ref,
            )
        except Exception:
            raise ValueError("packaged completion authentication unavailable") from None
        if type(tag) is not bytes or not tag or len(tag) > 128:
            raise ValueError("packaged completion authentication is invalid")
        evidence = self._roots.control / operation_path(effect_id, "evidence")
        claim_directory(self._roots.control, evidence)
        write_exclusive(
            self._roots.control, evidence / "packaged-completion.json", completion,
        )
        write_exclusive(
            self._roots.control, evidence / "packaged-completion.mac", tag,
        )
        return completion

    def __call__(
        self,
        dispatch_bytes: bytes,
        provider_job_ref: str,
        *,
        commit_artifacts,
        commit_control,
    ) -> dict[str, object]:
        try:
            dispatch = parse_modal_packaged_dispatch(dispatch_bytes, self._verifier)
            if dispatch.provider_facts != self._facts:
                raise ValueError("packaged dispatch targets another deployment")
            size, digest = hash_regular(
                self._roots.artifacts,
                self._roots.artifacts / dispatch.stage_receipt.path,
                dispatch.stage_receipt.size_bytes,
            )
            if (size, digest) != (
                dispatch.stage_receipt.size_bytes,
                dispatch.stage_receipt.content_digest,
            ):
                raise ValueError("staged packaged input differs")
            paths = self._paths(dispatch)
            result = self._executor.execute(
                runtime_release=dispatch.runtime_release,
                provider_binding=dispatch.provider_binding,
                execution_binding=dispatch.execution_binding,
                workload_bytes=dispatch.workload_bytes,
                artifact_policy=dispatch.artifact_policy,
                paths=paths,
                environment=dispatch.environment,
            )
            completion = self._publish_completion(dispatch, result, provider_job_ref)
            commit_artifacts()
            commit_control()
            return {
                "schema_version": "synaptic-modal-packaged-worker-result/v1",
                "effect_id": dispatch.submit_command.operation.effect.effect_id,
                "status_code": "completed",
                "completion_sha256": hashlib.sha256(completion).hexdigest(),
            }
        except BaseException:
            return {
                "schema_version": "synaptic-modal-packaged-worker-result/v1",
                "effect_id": "unavailable",
                "status_code": "failed",
                "completion_sha256": "0" * 64,
            }


__all__ = [
    "InstalledPackagedSFTTrainerExecutor",
    "ModalPackagedEvidenceSigner",
    "ModalPackagedWorker",
    "ModalPackagedWorkerRoots",
    "PackagedTrainerExecutor",
]
