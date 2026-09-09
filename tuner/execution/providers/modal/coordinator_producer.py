"""Foundation-native Modal completion production over admitted worker values."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol

from synaptic_tuner.api.v1.runs_facade import RunLogEntry, RunLogLevel

from ...contracts import safe_ref
from ...foundation_v2.canonical import digest_text
from .contracts import (
    EXACT_ARTIFACT_ROLES,
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
from .coordinator_worker import ModalWorkerInvocation, validate_modal_worker_invocation
from .coordinator_logs import ModalCoordinatorLogChunk
from .manifest import CompletionManifestV1
from .mounted_io import (
    claim_directory, copy_regular, hash_regular, list_regular_sizes, read_regular,
    write_exclusive,
)
from .worker_ports import ModalProcessResult


class ModalCoordinatorEvidenceSigner(Protocol):
    def sign(self, purpose: str, payload: bytes, key_ref: str) -> bytes: ...


@dataclass(frozen=True, slots=True)
class ModalCoordinatorCompletion:
    status_code: str
    returncode: int
    terminal_digest: str
    log_chain_digest: str
    artifact_set_digest: str

    def __post_init__(self) -> None:
        if self.status_code not in {"completed", "failed"}:
            raise ValueError("worker completion status is invalid")
        if type(self.returncode) is not int:
            raise TypeError("returncode must be an exact integer")
        for name in ("terminal_digest", "log_chain_digest", "artifact_set_digest"):
            digest_text(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class _InventoryMember:
    role: ArtifactRole
    name: str
    size: int
    sha256: str


class MountedModalCoordinatorProducer:
    """Publish exact artifacts and v1 evidence for one admitted submit effect."""

    __slots__ = ("_auth", "_control", "_artifact", "_bounds", "_clock")

    def __init__(
        self,
        authenticator: ModalCoordinatorEvidenceSigner,
        *,
        control_root: str = "/workspace/control",
        artifact_root: str = "/workspace/run",
        bounds: BoundsPolicyV1 = BoundsPolicyV1(),
        clock: Callable[[], str] = lambda: datetime.now(timezone.utc).isoformat(),
    ) -> None:
        if not hasattr(authenticator, "sign"):
            raise TypeError("Modal coordinator evidence signer is required")
        if type(bounds) is not BoundsPolicyV1:
            raise TypeError("exact Modal bounds are required")
        if not callable(clock):
            raise TypeError("UTC log clock must be callable")
        self._auth = authenticator
        self._control = Path(control_root)
        self._artifact = Path(artifact_root)
        self._bounds = bounds
        self._clock = clock

    def _sign(self, purpose: str, payload: bytes, key_ref: str) -> bytes:
        try:
            tag = self._auth.sign(purpose, payload, key_ref)
        except Exception:
            raise ValueError("Modal coordinator evidence authentication unavailable") from None
        if type(tag) is not bytes or not tag or len(tag) > 128:
            raise ValueError("Modal coordinator evidence authentication is invalid")
        return tag

    def _inventory(self, invocation: ModalWorkerInvocation) -> tuple[_InventoryMember, ...]:
        source = invocation.source
        inventory_path = Path(source.roots["state"]) / "runtime-v1-inventory.json"
        inventory = _object(
            read_regular(self._artifact, inventory_path, self._bounds.max_control_bytes),
            self._bounds.max_control_bytes,
        )
        if (
            set(inventory) != {"schema_version", "workload_fingerprint", "artifacts"}
            or inventory["schema_version"] != "synaptic-artifact-inventory/v1"
            or inventory["workload_fingerprint"]
            != invocation.submit_command.preparation.workload_digest
        ):
            raise ValueError("runtime artifact inventory is invalid")
        records = inventory["artifacts"]
        if type(records) is not list or len(records) != len(EXACT_ARTIFACT_ROLES):
            raise ValueError("runtime artifact inventory must contain exactly five records")
        members: list[_InventoryMember] = []
        for record in records:
            if type(record) is not dict or set(record) != {"role", "path", "sha256", "size"}:
                raise ValueError("runtime artifact inventory member is invalid")
            try:
                role = ArtifactRole(record["role"])
            except (TypeError, ValueError):
                raise ValueError("runtime artifact inventory role is invalid") from None
            name = record["path"]
            if (
                type(name) is not str
                or not name
                or "/" in name
                or "\\" in name
                or name in {".", ".."}
            ):
                raise ValueError("runtime artifact path is invalid")
            size = strict_int(
                record["size"], "artifact size", minimum=1,
                maximum=self._bounds.max_artifact_bytes,
            )
            content_digest = digest_text(record["sha256"], "artifact sha256")
            members.append(_InventoryMember(role, name, size, content_digest))
        result = tuple(members)
        if (
            frozenset(member.role for member in result) != EXACT_ARTIFACT_ROLES
            or len({member.role for member in result}) != len(result)
            or len({member.name for member in result}) != len(result)
            or sum(member.size for member in result) > self._bounds.max_artifact_total_bytes
        ):
            raise ValueError("runtime artifact set is invalid")
        return result

    def _publish_artifacts(
        self, invocation: ModalWorkerInvocation, inventory: tuple[_InventoryMember, ...],
    ) -> tuple[ArtifactMemberV1, ...]:
        source_root = Path(invocation.source.roots["artifacts"])
        effect_id = invocation.submit_command.operation.effect.effect_id
        published: list[ArtifactMemberV1] = []
        for item in inventory:
            path = operation_path(effect_id, "output", item.role.value)
            destination = self._artifact / path
            size, content_digest = copy_regular(
                self._artifact, source_root / item.name,
                self._artifact, destination,
                maximum=self._bounds.max_artifact_bytes,
            )
            if (size, content_digest) != (item.size, item.sha256):
                raise ValueError("runtime artifact inventory content mismatch")
            readback_size, readback_digest = hash_regular(
                self._artifact, destination, self._bounds.max_artifact_bytes,
            )
            if (readback_size, readback_digest) != (item.size, item.sha256):
                raise ValueError("published artifact readback mismatch")
            published.append(ArtifactMemberV1(
                item.role, path, item.size, item.sha256,
                provider_entry_identity(invocation.artifact_volume_id, path, item.size),
            ))
        return tuple(published)

    def finalize(
        self,
        invocation: ModalWorkerInvocation,
        result: ModalProcessResult,
        *,
        job_ref: str,
    ) -> ModalCoordinatorCompletion:
        validate_modal_worker_invocation(invocation)
        if type(result) is not ModalProcessResult:
            raise TypeError("exact Modal process result is required")
        if result.returncode not in {0, 120, 121, 122, 123, 124, 125}:
            raise ValueError("Modal process returncode is outside the closed remote protocol")
        completed = result.returncode == 0
        message = (
            "training completed" if completed
            else "training failed: " + (result.diagnostic_code or "trainer_nonzero")
        )
        log_entry = RunLogEntry(
            0, self._clock(),
            RunLogLevel.INFO if completed else RunLogLevel.ERROR,
            "completed" if completed else "failed", message,
            len(message.encode("utf-8")),
        )
        submit = invocation.submit_command
        deployment = invocation.deployment
        selection = deployment.selection
        policy = invocation.log_policy
        identity = {
            "account_ref": selection.account_ref,
            "workspace_ref": selection.workspace_ref,
            "environment_ref": selection.environment_ref,
            "client_ref": selection.client_ref,
            "sdk_version": selection.sdk_version,
            "control_volume_id": invocation.control_volume_id,
            "artifact_volume_id": invocation.artifact_volume_id,
            "job_ref": safe_ref(job_ref, "job_ref"),
            "effect_id": submit.operation.effect.effect_id,
            "command_digest": submit.digest,
            "plan_digest": submit.preparation.plan_fingerprint,
            "deployment_attestation_digest": deployment.attestation_digest,
            "invocation_nonce": submit.operation.invocation_nonce,
            "generation": strict_int(
                policy["generation"], "generation", minimum=1, maximum=2**31 - 1,
            ),
        }
        inventory = self._inventory(invocation) if completed else ()
        members = tuple(
            ArtifactMemberV1(
                item.role,
                operation_path(identity["effect_id"], "output", item.role.value),
                item.size, item.sha256,
                provider_entry_identity(
                    invocation.artifact_volume_id,
                    operation_path(identity["effect_id"], "output", item.role.value),
                    item.size,
                ),
            )
            for item in inventory
        )
        records = (log_entry,)
        payload_digest = sha(canonical_json([record.to_dict() for record in records]))
        parsed_chunk = ModalCoordinatorLogChunk(
            identity["generation"], 0, "0" * 64, payload_digest,
            identity["job_ref"], identity["effect_id"], identity["plan_digest"],
            identity["invocation_nonce"], records,
        )
        log_chunk = parsed_chunk.canonical_bytes
        if len(log_chunk) > min(
            self._bounds.max_log_chunk_bytes,
            strict_int(policy["max_chunk_bytes"], "max_chunk_bytes", minimum=1),
        ):
            raise ValueError("worker log chunk exceeds admitted policy")
        parsed_chunk = ModalCoordinatorLogChunk.parse(log_chunk, bounds=self._bounds)
        chain_digest = parsed_chunk.chunk_digest
        chunk_path = operation_path(identity["effect_id"], "logs", "chunks", "000.json")
        log_metadata = canonical_json({
            "schema": "synaptic.modal-log-metadata/v1", **identity,
            "chain_digest": chain_digest,
            "chunks": [{
                "path": chunk_path, "size": len(log_chunk), "sha256": sha(log_chunk),
                "provider_entry_id": provider_entry_identity(
                    identity["control_volume_id"], chunk_path, len(log_chunk),
                ),
            }],
        })
        artifact_set_digest = "0" * 64
        if completed:
            artifact_set_digest = CompletionManifestV1(
                members, **identity, terminal_evidence_digest="0" * 64,
                log_chain_digest=chain_digest,
            ).artifact_set_digest
        terminal = canonical_json({
            "schema": "synaptic.modal-terminal/v1",
            "status_code": "completed" if completed else "failed",
            **identity, "artifact_set_digest": artifact_set_digest,
            "log_chain_digest": chain_digest,
        })
        if len(terminal) > min(
            self._bounds.max_control_bytes,
            strict_int(policy["max_terminal_bytes"], "max_terminal_bytes", minimum=1),
        ):
            raise ValueError("worker terminal exceeds admitted policy")
        TerminalEvidenceV1.parse(terminal, limit=self._bounds.max_control_bytes)
        manifest = None
        if completed:
            manifest = canonical_json({
                "schema": "synaptic.modal-completion/v1",
                "members": [{
                    "role": member.role.value, "path": member.path, "size": member.size,
                    "sha256": member.sha256, "provider_entry_id": member.provider_entry_id,
                } for member in members],
                **identity, "terminal_evidence_digest": sha(terminal),
                "log_chain_digest": chain_digest,
            })
            CompletionManifestV1.parse(manifest, limit=self._bounds.max_control_bytes)
        key_ref = invocation.key_ref
        log_tag = self._sign("modal-log-metadata/v1", log_metadata, key_ref)
        terminal_tag = self._sign("modal-terminal/v1", terminal, key_ref)
        manifest_tag = (
            self._sign("modal-completion/v1", manifest, key_ref)
            if manifest is not None else None
        )
        effect_id = identity["effect_id"]
        logs_root = self._control / operation_path(effect_id, "logs")
        evidence_root = self._control / operation_path(effect_id, "evidence")
        claim_directory(self._control, logs_root)
        claim_directory(self._control, evidence_root)
        if completed:
            output_directory = self._artifact / operation_path(effect_id, "output")
            claim_directory(self._artifact, output_directory)
            published = self._publish_artifacts(invocation, inventory)
            if published != members:
                raise ValueError("published artifact set differs from admitted inventory")
            expected_listing = tuple(sorted(
                (member.role.value, member.size) for member in members
            ))
            if list_regular_sizes(
                self._artifact, output_directory,
                maximum_entries=len(EXACT_ARTIFACT_ROLES) + 1,
            ) != expected_listing:
                raise ValueError("published artifact directory is not exact")
        write_exclusive(self._control, logs_root / "chunks" / "000.json", log_chunk)
        write_exclusive(self._control, logs_root / "log-metadata.v1.json", log_metadata)
        write_exclusive(self._control, logs_root / "log-metadata.v1.mac", log_tag)
        if list_regular_sizes(
            self._control, logs_root / "chunks", maximum_entries=2,
        ) != (("000.json", len(log_chunk)),):
            raise ValueError("published coordinator log chunk set is not exact")
        write_exclusive(self._control, evidence_root / "terminal-evidence.v1.json", terminal)
        write_exclusive(self._control, evidence_root / "terminal-evidence.v1.mac", terminal_tag)
        if manifest is not None and manifest_tag is not None:
            write_exclusive(
                self._control, evidence_root / "completion-manifest.v1.json", manifest,
            )
            expected_evidence = tuple(sorted((
                ("completion-manifest.v1.json", len(manifest)),
                ("terminal-evidence.v1.json", len(terminal)),
                ("terminal-evidence.v1.mac", len(terminal_tag)),
            )))
            if list_regular_sizes(
                self._control, evidence_root, maximum_entries=4,
            ) != expected_evidence:
                raise ValueError("published coordinator evidence set is not exact")
            write_exclusive(
                self._control, evidence_root / "completion-manifest.v1.mac", manifest_tag,
            )
        elif list_regular_sizes(
            self._control, evidence_root, maximum_entries=3,
        ) != tuple(sorted((
            ("terminal-evidence.v1.json", len(terminal)),
            ("terminal-evidence.v1.mac", len(terminal_tag)),
        ))):
            raise ValueError("published coordinator evidence set is not exact")
        return ModalCoordinatorCompletion(
            "completed" if completed else "failed", result.returncode,
            sha(terminal), chain_digest, artifact_set_digest,
        )


__all__: list[str] = []
