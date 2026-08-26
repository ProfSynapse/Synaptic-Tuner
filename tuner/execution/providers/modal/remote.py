"""Provider-free core of the fixed Modal SFT worker.

Modal decorators and SDK objects belong in ``deployment_v1.py``.  Keeping this
module pure lets the complete admission and process boundary be tested without
credentials, networking, GPUs, or the optional Modal dependency.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tuner.project.execution_source import ExecutionSourceV1

from ...broker import MutationCommandV1
from .bundle import ModalExecutionBundleV1
from .contracts import BoundsPolicyV1, _object, operation_path, sha
from .mounted_io import read_regular
from .resolution import ModalDeploymentSelectionV1


class RemoteStageVerifier(Protocol):
    def verify(self, purpose: str, payload: bytes, tag: bytes, key_ref: str) -> bool: ...


class SourceMaterializer(Protocol):
    def prepare_and_verify(
        self,
        source: ExecutionSourceV1,
        deployment: ModalDeploymentSelectionV1,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class ProcessResultV1:
    returncode: int
    stdout: bytes = b""
    stderr: bytes = b""

    def __post_init__(self) -> None:
        if type(self.returncode) is not int:
            raise TypeError("returncode must be an exact integer")
        if not isinstance(self.stdout, bytes) or not isinstance(self.stderr, bytes):
            raise TypeError("process output must be bytes")


class FixedProcessRunner(Protocol):
    def run(
        self,
        argv: tuple[str, str, str],
        *,
        cwd: str,
        environment: dict[str, str],
        stdin: bytes,
    ) -> ProcessResultV1: ...


class RemoteCompletionProducer(Protocol):
    def finalize(
        self, invocation: "RemoteInvocationV1", result: ProcessResultV1, *, job_ref: str
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class RemoteInvocationV1:
    command: MutationCommandV1
    bundle: ModalExecutionBundleV1
    source: ExecutionSourceV1
    deployment: ModalDeploymentSelectionV1
    workload: bytes
    argv: tuple[str, str, str]
    cwd: str
    environment: dict[str, str]


def admit_remote_invocation(
    canonical_command: bytes,
    *,
    claim: bytes,
    claim_tag: bytes,
    bundle_transport: bytes,
    verifier: RemoteStageVerifier,
    bounds: BoundsPolicyV1 = BoundsPolicyV1(),
) -> RemoteInvocationV1:
    """Authenticate and cross-bind every staged input before source or process I/O."""
    command = MutationCommandV1.from_bytes(canonical_command)
    if len(bundle_transport) > bounds.max_bundle_bytes:
        raise ValueError("remote bundle exceeds the stage bound")
    if sha(bundle_transport) != command.bundle_digest or sha(claim) != command.stage_claim_digest:
        raise ValueError("remote stage digests do not bind the mutation command")
    if not isinstance(claim_tag, bytes) or not claim_tag or len(claim_tag) > 128:
        raise ValueError("remote stage authentication tag is invalid")
    target = command.operation.stage_target
    try:
        authenticated = verifier.verify(
            "modal-stage-claim/v1", claim, claim_tag, target.key_ref
        )
    except Exception:
        raise ValueError("remote stage authentication unavailable") from None
    if authenticated is not True:
        raise ValueError("remote stage authentication failed")
    claim_value = _object(claim, bounds.max_control_bytes)
    expected_claim = {
        "schema": "synaptic.modal-stage-claim/v1",
        "effect_provider": command.effect.scope.provider,
        "effect_account_ref": command.effect.scope.account_ref,
        "effect_namespace_ref": command.effect.scope.namespace_ref,
        "effect_id": command.effect.effect_id,
        "effect_kind": command.effect.kind.value,
        "operation_key": command.effect.effect_key,
        "operation_binding_digest": command.operation_binding_digest,
        "control_volume_id": target.control_volume_id,
        "artifact_volume_id": target.artifact_volume_id,
        "bundle_digest": command.bundle_digest,
        "bundle_size": len(bundle_transport),
        "plan_digest": command.plan_fingerprint,
        "invocation_nonce": command.invocation_nonce,
        "output_prefix": target.output_prefix,
    }
    if claim_value != expected_claim:
        raise ValueError("remote stage claim binding mismatch")
    bundle = ModalExecutionBundleV1.parse_transport(bundle_transport)
    if bundle.operation != command.operation:
        raise ValueError("remote bundle operation mismatch")
    members = {member.name: member.content for member in bundle.members}
    source = ExecutionSourceV1.from_dict(
        _object(members["execution-source.json"], 1_048_576)
    )
    deployment_document = _object(members["deployment.json"], 1_048_576)
    deployment = ModalDeploymentSelectionV1(**deployment_document["selection"])
    invocation = _object(members["invocation-intent.json"], 1_048_576)
    workload = members["workload.json"]
    environment = dict(source.environment)
    environment["SYNAPTIC_WORKLOAD_FINGERPRINT"] = hashlib.sha256(
        b"synaptic-training-workload/v1\0" + workload
    ).hexdigest()
    argv = (
        source.python_executable,
        source.roots["engine"] + "/Trainers/sft/runtime_v1.py",
        "--canonical-workload-stdin",
    )
    if (
        invocation["interpreter"] != source.python_executable
        or tuple(invocation["argv"]) != argv
        or invocation["cwd"] != source.roots["tmp"]
        or invocation["environment_digest"]
        != hashlib.sha256(
            json.dumps(environment, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    ):
        raise ValueError("remote invocation differs from the fixed runtime command")
    return RemoteInvocationV1(
        command, bundle, source, deployment, workload, argv,
        source.roots["tmp"], environment,
    )


def execute_remote_sft(
    invocation: RemoteInvocationV1,
    *,
    sources: SourceMaterializer,
    processes: FixedProcessRunner,
) -> ProcessResultV1:
    """Verify dual-clone materialization, then invoke only runtime_v1 without a shell."""
    if type(invocation) is not RemoteInvocationV1:
        raise TypeError("canonical remote invocation is required")
    sources.prepare_and_verify(invocation.source, invocation.deployment)
    result = processes.run(
        invocation.argv,
        cwd=invocation.cwd,
        environment=dict(invocation.environment),
        stdin=invocation.workload,
    )
    if type(result) is not ProcessResultV1:
        raise TypeError("process runner returned a noncanonical result")
    return result


class MountedModalWorkerV1:
    """Connect the fixed two-volume mount layout to the provider-free worker."""

    __slots__ = ("_verifier", "_sources", "_processes", "_completion", "_control", "_artifact", "_bounds")

    def __init__(
        self,
        *,
        verifier: RemoteStageVerifier,
        sources: SourceMaterializer,
        processes: FixedProcessRunner,
        completion: RemoteCompletionProducer,
        control_root: str = "/workspace/control",
        artifact_root: str = "/workspace/run",
        bounds: BoundsPolicyV1 = BoundsPolicyV1(),
    ) -> None:
        self._verifier = verifier
        self._sources = sources
        self._processes = processes
        if not hasattr(completion, "finalize"):
            raise TypeError("remote completion producer is required")
        self._completion = completion
        self._control = Path(control_root)
        self._artifact = Path(artifact_root)
        self._bounds = bounds

    def __call__(self, canonical_command: bytes, job_ref: str) -> dict[str, object]:
        command = MutationCommandV1.from_bytes(canonical_command)
        effect_id = command.effect.effect_id
        claim = read_regular(
            self._control,
            self._control / operation_path(effect_id, "control", "stage-claim.v1.json"),
            self._bounds.max_control_bytes,
        )
        claim_tag = read_regular(
            self._control,
            self._control / operation_path(effect_id, "control", "stage-claim.v1.mac"), 128
        )
        bundle = read_regular(
            self._artifact,
            self._artifact / operation_path(effect_id, "input", "bundle.bin"), self._bounds.max_bundle_bytes
        )
        invocation = admit_remote_invocation(
            canonical_command,
            claim=claim,
            claim_tag=claim_tag,
            bundle_transport=bundle,
            verifier=self._verifier,
            bounds=self._bounds,
        )
        try:
            result = execute_remote_sft(
                invocation, sources=self._sources, processes=self._processes
            )
        except Exception:
            result = ProcessResultV1(125)
        completion = self._completion.finalize(invocation, result, job_ref=job_ref)
        status_code = getattr(completion, "status_code", None)
        if status_code not in {"completed", "failed"}:
            raise ValueError("remote completion producer returned invalid status")
        return {
            "schema_version": "synaptic-modal-worker-result/v1",
            "effect_id": invocation.command.effect.effect_id,
            "returncode": result.returncode,
            "status_code": status_code,
        }


__all__ = [
    "FixedProcessRunner", "MountedModalWorkerV1", "ProcessResultV1", "RemoteCompletionProducer", "RemoteInvocationV1",
    "RemoteStageVerifier", "SourceMaterializer", "admit_remote_invocation",
    "execute_remote_sft",
]
