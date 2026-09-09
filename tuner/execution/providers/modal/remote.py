"""Provider-free core of the fixed Modal SFT worker.

Modal decorators and SDK objects belong in ``deployment_v1.py``.  Keeping this
module pure lets the complete admission and process boundary be tested without
credentials, networking, GPUs, or the optional Modal dependency.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Protocol

from tuner.project.execution_source import ExecutionSourceV1
from tuner.runtime.dispatch import WorkerControlLocationV1
from tuner.runtime.offline_sft_worker import parse_offline_sft_worker_manifest

from ...broker import MutationCommandV1
from .bundle import ModalExecutionBundleV1
from .contracts import BoundsPolicyV1, _object, operation_path, sha
from .mounted_io import read_regular
from .resolution import ModalDeploymentSelectionV1
from .worker_ports import (
    FixedProcessRunner, ModalProcessResult, ModalRemotePhaseError, SourceMaterializer,
)
from . import worker_source


class RemoteStageVerifier(Protocol):
    def verify(self, purpose: str, payload: bytes, tag: bytes, key_ref: str) -> bool: ...


class RemoteCompletionProducer(Protocol):
    def finalize(
        self, invocation: "RemoteInvocationV1", result: ModalProcessResult, *, job_ref: str
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
    closure_manifest: bytes
    closure_manifest_runtime_path: str


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
    closure_bytes = members["worker-closure-manifest.json"]
    closure = parse_offline_sft_worker_manifest(
        closure_bytes,
        source_ref="modal-bundle:worker-closure-manifest.json",
        manifest_path=Path("worker-closure-manifest.json"),
    )
    environment = dict(source.environment)
    if environment.get("PYTHONPATH") != source.roots["engine"]:
        raise ValueError("remote source PYTHONPATH does not bind the engine root")
    environment.pop("PYTHONPATH")
    if {"PYTHONHOME", "PYTHONUSERBASE", "HF_TOKEN"} & set(environment):
        raise ValueError("remote source environment contains a forbidden ambient variable")
    environment["SYNAPTIC_WORKLOAD_FINGERPRINT"] = hashlib.sha256(
        b"synaptic-training-workload/v1\0" + workload
    ).hexdigest()
    control = WorkerControlLocationV1(
        PurePosixPath("/workspace/worker-control") / operation_path(command.effect.effect_id, "input")
    )
    workload_document = _object(workload, 1_048_576)
    model = workload_document["configuration"]["document"]["model"]
    environment["SYNAPTIC_MODEL_SNAPSHOT"] = (
        source.roots["cache"] + "/model/models--"
        + str(model["ref"]).replace("/", "--") + "/snapshots/" + str(model["revision"])
    )
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["SYNAPTIC_WORKER_CLOSURE_MANIFEST"] = control.manifest_path.as_posix()
    environment["SYNAPTIC_WORKER_CLOSURE_DIGEST"] = closure.closure.closure_digest
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
        source.roots["tmp"], environment, closure_bytes,
        control.manifest_path.as_posix(),
    )


def execute_remote_sft(
    invocation: RemoteInvocationV1,
    *,
    sources: SourceMaterializer,
    processes: FixedProcessRunner,
    commit_prepared: Callable[[], None],
) -> ModalProcessResult:
    """Verify dual-clone materialization, then invoke only runtime_v1 without a shell."""
    if type(invocation) is not RemoteInvocationV1:
        raise TypeError("canonical remote invocation is required")
    sources.prepare_and_verify(invocation.source, invocation.deployment)
    try:
        locked_bytes = worker_source.read_locked_closure_manifest(invocation.source)
    except OSError:
        raise ModalRemotePhaseError(124, "locked_source_mismatch") from None
    if locked_bytes != invocation.closure_manifest:
        raise ModalRemotePhaseError(124, "locked_source_mismatch")
    try:
        worker_source.write_runtime_closure_manifest(
            invocation.closure_manifest_runtime_path, invocation.closure_manifest
        )
    except FileExistsError:
        raise ModalRemotePhaseError(122, "artifact_layout_collision") from None
    except OSError:
        raise ModalRemotePhaseError(122, "artifact_layout_failed") from None
    try:
        worker_source.stage_runtime_worker(
            invocation.source, invocation.closure_manifest_runtime_path, invocation.closure_manifest
        )
    except ModalRemotePhaseError:
        raise
    except FileExistsError:
        raise ModalRemotePhaseError(122, "artifact_layout_collision") from None
    except Exception:
        raise ModalRemotePhaseError(124, "locked_source_mismatch") from None
    result = processes.run(
        invocation.argv,
        cwd=invocation.cwd,
        environment=dict(invocation.environment),
        stdin=invocation.workload,
        commit_prepared=commit_prepared,
    )
    if type(result) is not ModalProcessResult:
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

    def __call__(self, canonical_command: bytes, job_ref: str, commit_prepared: Callable[[], None]) -> dict[str, object]:
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
                invocation, sources=self._sources, processes=self._processes,
                commit_prepared=commit_prepared,
            )
        except ModalRemotePhaseError as error:
            result = ModalProcessResult(
                error.returncode, diagnostic_code=error.diagnostic_code
            )
        except Exception:
            result = ModalProcessResult(125, diagnostic_code="generic_failure")
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
    "MountedModalWorkerV1", "RemoteCompletionProducer", "RemoteInvocationV1",
    "RemoteStageVerifier", "admit_remote_invocation", "execute_remote_sft",
]
