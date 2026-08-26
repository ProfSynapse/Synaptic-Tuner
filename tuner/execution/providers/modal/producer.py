"""Provider-free producer for exact Modal v1 artifacts and terminal records."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ...contracts import safe_ref
from .contracts import (
    ArtifactMemberV1,
    ArtifactRole,
    BoundsPolicyV1,
    TerminalEvidenceV1,
    _object,
    canonical_json,
    operation_path,
    provider_entry_identity,
    sha,
    strict_int,
)
from .logs import StructuredLogChunkV1
from .manifest import CompletionManifestV1
from .mounted_io import copy_regular, read_regular, write_exclusive
from .remote import ProcessResultV1, RemoteInvocationV1


class RemoteEvidenceAuthenticator(Protocol):
    def sign(self, purpose: str, payload: bytes, key_ref: str) -> bytes: ...


@dataclass(frozen=True, slots=True)
class RemoteCompletionResultV1:
    status_code: str
    returncode: int
    terminal_digest: str
    log_chain_digest: str
    artifact_set_digest: str

    def __post_init__(self) -> None:
        if self.status_code not in {"completed", "failed"}:
            raise ValueError("remote completion status is invalid")
        if type(self.returncode) is not int:
            raise TypeError("returncode must be an exact integer")
        for name in ("terminal_digest", "log_chain_digest", "artifact_set_digest"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")


class MountedCompletionProducerV1:
    """Publish one operation's five artifacts and authenticated control records."""

    __slots__ = ("_auth", "_control", "_artifact", "_bounds")

    def __init__(
        self,
        authenticator: RemoteEvidenceAuthenticator,
        *,
        control_root: str = "/workspace/control",
        artifact_root: str = "/workspace/run",
        bounds: BoundsPolicyV1 = BoundsPolicyV1(),
    ) -> None:
        if not hasattr(authenticator, "sign"):
            raise TypeError("remote evidence authenticator is required")
        self._auth = authenticator
        self._control = Path(control_root)
        self._artifact = Path(artifact_root)
        self._bounds = bounds

    def _sign(self, purpose: str, payload: bytes, key_ref: str) -> bytes:
        try:
            tag = self._auth.sign(purpose, payload, key_ref)
        except Exception:
            raise ValueError("remote evidence authentication unavailable") from None
        if not isinstance(tag, bytes) or not tag or len(tag) > 128:
            raise ValueError("remote evidence authentication is invalid")
        return tag

    def _identity(self, invocation: RemoteInvocationV1, job_ref: str) -> dict[str, object]:
        command = invocation.command
        deployment = invocation.deployment
        target = command.operation.stage_target
        return {
            "account_ref": deployment.account_ref,
            "workspace_ref": deployment.workspace_ref,
            "environment_ref": deployment.environment_ref,
            "client_ref": deployment.client_ref,
            "sdk_version": deployment.sdk_version,
            "control_volume_id": target.control_volume_id,
            "artifact_volume_id": target.artifact_volume_id,
            "job_ref": safe_ref(job_ref, "job_ref"),
            "effect_id": command.effect.effect_id,
            "command_digest": command.digest,
            "plan_digest": command.plan_fingerprint,
            "deployment_attestation_digest": command.deployment_attestation_digest,
            "invocation_nonce": command.invocation_nonce,
            "generation": target.generation,
        }

    def _publish_artifacts(self, invocation: RemoteInvocationV1) -> tuple[ArtifactMemberV1, ...]:
        source_root = Path(invocation.source.roots["artifacts"])
        inventory_path = Path(invocation.source.roots["state"]) / "runtime-v1-inventory.json"
        inventory = _object(
            read_regular(
                self._artifact, inventory_path, self._bounds.max_control_bytes
            ),
            self._bounds.max_control_bytes,
        )
        if set(inventory) != {"schema_version", "workload_fingerprint", "artifacts"} or inventory["schema_version"] != "synaptic-artifact-inventory/v1":
            raise ValueError("runtime artifact inventory is invalid")
        import hashlib
        expected_workload = hashlib.sha256(
            b"synaptic-training-workload/v1\0" + invocation.workload
        ).hexdigest()
        if inventory["workload_fingerprint"] != expected_workload:
            raise ValueError("runtime artifact inventory workload mismatch")
        records = inventory["artifacts"]
        if not isinstance(records, list) or len(records) != 5:
            raise ValueError("runtime artifact inventory must contain exactly five records")
        target = invocation.command.operation.stage_target
        effect_id = invocation.command.effect.effect_id
        members: list[ArtifactMemberV1] = []
        for record in records:
            if not isinstance(record, dict) or set(record) != {"role", "path", "sha256", "size"}:
                raise ValueError("runtime artifact inventory member is invalid")
            role = ArtifactRole(record["role"])
            name = record["path"]
            if not isinstance(name, str) or "/" in name or "\\" in name or name in {"", ".", ".."}:
                raise ValueError("runtime artifact path is invalid")
            declared_size = strict_int(record["size"], "artifact size", minimum=1, maximum=self._bounds.max_artifact_bytes)
            source = source_root / name
            path = operation_path(effect_id, "output", role.value)
            size, content_digest = copy_regular(
                self._artifact,
                source,
                self._artifact,
                self._artifact / path,
                maximum=self._bounds.max_artifact_bytes,
            )
            if size != declared_size or content_digest != record["sha256"]:
                raise ValueError("runtime artifact inventory content mismatch")
            members.append(ArtifactMemberV1(role, path, size, content_digest, provider_entry_identity(target.artifact_volume_id, path, size)))
        result = tuple(members)
        if len({member.role for member in result}) != 5 or sum(member.size for member in result) > self._bounds.max_artifact_total_bytes:
            raise ValueError("runtime artifact set is invalid")
        return result

    def finalize(
        self,
        invocation: RemoteInvocationV1,
        result: ProcessResultV1,
        *,
        job_ref: str,
    ) -> RemoteCompletionResultV1:
        if type(invocation) is not RemoteInvocationV1 or type(result) is not ProcessResultV1:
            raise TypeError("canonical invocation and process result are required")
        identity = self._identity(invocation, job_ref)
        effect_id = invocation.command.effect.effect_id
        key_ref = invocation.command.operation.stage_target.key_ref
        completed = result.returncode == 0
        members = self._publish_artifacts(invocation) if completed else ()
        records = [{"code": "completed" if completed else "failed", "message": "training completed" if completed else "training failed"}]
        log_chunk = canonical_json({
            "schema": "synaptic.modal-log-chunk/v1", "generation": identity["generation"],
            "sequence": 0, "previous_digest": "0" * 64,
            "payload_digest": sha(canonical_json(records)), "job_ref": identity["job_ref"],
            "effect_id": identity["effect_id"], "plan_digest": identity["plan_digest"],
            "invocation_nonce": identity["invocation_nonce"], "records": records,
        })
        parsed_chunk = StructuredLogChunkV1.parse(log_chunk, bounds=self._bounds)
        chain_digest = parsed_chunk.chunk_digest
        chunk_path = operation_path(effect_id, "logs", "chunks", "000.json")
        log_metadata = canonical_json({
            "schema": "synaptic.modal-log-metadata/v1", **identity,
            "chain_digest": chain_digest,
            "chunks": [{"path": chunk_path, "size": len(log_chunk), "sha256": sha(log_chunk),
                        "provider_entry_id": provider_entry_identity(identity["control_volume_id"], chunk_path, len(log_chunk))}],
        })
        artifact_set_digest = "0" * 64
        if completed:
            provisional = CompletionManifestV1(tuple(members), **identity, terminal_evidence_digest="0" * 64, log_chain_digest=chain_digest)
            artifact_set_digest = provisional.artifact_set_digest
        terminal = canonical_json({
            "schema": "synaptic.modal-terminal/v1", "status_code": "completed" if completed else "failed",
            **identity, "artifact_set_digest": artifact_set_digest, "log_chain_digest": chain_digest,
        })
        TerminalEvidenceV1.parse(terminal, limit=self._bounds.max_control_bytes)
        evidence_root = self._control / operation_path(effect_id, "evidence")
        logs_root = self._control / operation_path(effect_id, "logs")
        write_exclusive(self._control, logs_root / "chunks" / "000.json", log_chunk)
        write_exclusive(self._control, logs_root / "log-metadata.v1.json", log_metadata)
        write_exclusive(self._control, logs_root / "log-metadata.v1.mac", self._sign("modal-log-metadata/v1", log_metadata, key_ref))
        write_exclusive(self._control, evidence_root / "terminal-evidence.v1.json", terminal)
        write_exclusive(self._control, evidence_root / "terminal-evidence.v1.mac", self._sign("modal-terminal/v1", terminal, key_ref))
        if completed:
            manifest = canonical_json({
                "schema": "synaptic.modal-completion/v1",
                "members": [{"role": member.role.value, "path": member.path, "size": member.size, "sha256": member.sha256, "provider_entry_id": member.provider_entry_id} for member in members],
                **identity, "terminal_evidence_digest": sha(terminal), "log_chain_digest": chain_digest,
            })
            CompletionManifestV1.parse(manifest, limit=self._bounds.max_control_bytes)
            write_exclusive(self._control, evidence_root / "completion-manifest.v1.json", manifest)
            write_exclusive(self._control, evidence_root / "completion-manifest.v1.mac", self._sign("modal-completion/v1", manifest, key_ref))
        return RemoteCompletionResultV1("completed" if completed else "failed", result.returncode, sha(terminal), chain_digest, artifact_set_digest)


__all__ = ["MountedCompletionProducerV1", "RemoteCompletionResultV1", "RemoteEvidenceAuthenticator"]
