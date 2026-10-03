"""Packaged Modal worker composition around the provider-neutral trainer seam."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from typing import Protocol

from tuner.execution.foundation_v2.canonical import canonical_bytes, digest_text, safe_ref
from tuner.runtime.packaged_sft_execution import (
    CHILD_FAILURE_STAGES,
    PackagedSFTExecutionError,
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
    MODAL_PACKAGED_DISPATCH_V2_SCHEMA,
    ModalPackagedDispatchVerifier,
    parse_modal_packaged_dispatch,
)


PACKAGED_WORKER_FAILURE_STAGES = frozenset({
    "ENTRYPOINT_SETUP", "ENTRYPOINT_IMPORTS", "ENTRYPOINT_DISPATCH_AUTH",
    "ENTRYPOINT_PROVIDER_ID", "ENTRYPOINT_VOLUME_ID", "ENTRYPOINT_CALL_ID",
    "ENTRYPOINT_MOUNTS", "ENTRYPOINT_WORKER_SETUP",
    "ENTRYPOINT_MOUNT_CONTROL_DIR", "ENTRYPOINT_MOUNT_CONTROL_LINK",
    "ENTRYPOINT_MOUNT_ARTIFACTS_DIR", "ENTRYPOINT_MOUNT_ARTIFACTS_LINK",
    "ENTRYPOINT_MOUNT_MODEL_CACHE_DIR", "ENTRYPOINT_MOUNT_MODEL_CACHE_LINK",
    "DISPATCH_AUTH", "STAGED_INPUT", "PATH_CLAIM",
    "SFT_ADMISSION", "SFT_ADMISSION_CONTRACTS", "SFT_ADMISSION_RELEASE",
    "SFT_ADMISSION_PATHS", "SFT_ADMISSION_INPUT", "SFT_ADMISSION_ENVIRONMENT",
    "SFT_ADMISSION_INVOCATION", "SFT_ADMISSION_COMMITMENT",
    "SFT_PREPARATION", "SFT_PREPARATION_MODEL_UNAVAILABLE",
    "SFT_PREPARATION_MODEL_SDK_ADMISSION",
    "SFT_PREPARATION_MODEL_INPUT", "SFT_PREPARATION_MODEL_WORKSPACE_SETUP",
    "SFT_PREPARATION_MODEL_METADATA_FETCH", "SFT_PREPARATION_MODEL_METADATA_VALIDATION",
    "SFT_PREPARATION_MODEL_DOWNLOAD", "SFT_PREPARATION_MODEL_VERIFICATION",
    "SFT_PREPARATION_MODEL_PERSISTENT_PUBLICATION",
    "SFT_PREPARATION_MODEL_DESTINATION_COPY", "SFT_PREPARATION_MODEL_DESTINATION_VERIFICATION",
    "SFT_PREPARATION_CACHE_COMMIT", "SFT_PREPARATION_PATH",
    "SFT_PREPARATION_SNAPSHOT_INVENTORY", "SFT_REVALIDATION",
    "SFT_INVOCATION", "SFT_TRAINER", "SFT_EVIDENCE", "SFT_ARTIFACT",
    "SFT_EVIDENCE_PRIVATE_COPY", "SFT_EVIDENCE_DIRECTORIES",
    "SFT_EVIDENCE_OUTPUT_BINDING", "SFT_EVIDENCE_OUTPUT_INVENTORY",
    "SFT_EVIDENCE_DATASET_BINDING", "SFT_EVIDENCE_PROJECTION_BINDING",
    "SFT_EVIDENCE_OUTPUT_DIRECTORY", "SFT_EVIDENCE_METRICS",
    "SFT_UNKNOWN", "COMPLETION", "ARTIFACT_COMMIT", "CONTROL_COMMIT",
    "SFT_POST_TRAINING",
}) | frozenset("SFT_PREPARATION_MODEL_PERSISTENT_PUBLICATION_" + code for code in (
    "SOURCE_CHAIN_ROOT SOURCE_CHAIN_TMP SOURCE_CHAIN_TMP_OWNER "
    "SOURCE_CHAIN_TMP_MODE_NONWRITABLE SOURCE_CHAIN_TMP_MODE_WRITABLE "
    "SOURCE_CHAIN_OWNER SOURCE_CHAIN_MODE SOURCE_CHAIN_OPEN "
    "CLAIM_ROOT_ADMISSION CLAIM_PARENT CLAIM_CREATE_EXISTS CLAIM_CREATE_DENIED "
    "CLAIM_CREATE_OS CLAIM_IDENTITY CLAIM_RECHECK COPY_ROOT_ADMISSION COPY_MEMBER_PARENT "
    "COPY_SOURCE_ADMISSION COPY_SOURCE_OPEN COPY_DEST_CREATE_EXISTS COPY_DEST_CREATE_DENIED "
    "COPY_DEST_CREATE_OS COPY_DEST_IDENTITY COPY_STREAM COPY_STREAM_READ COPY_STREAM_WRITE "
    "COPY_STREAM_HASH COPY_STREAM_FSYNC COPY_SOURCE_RECHECK COPY_DEST_RECHECK "
    "COPY_MEMBER_RECHECK COPY_PRIVATE_RECHECK COPY_ROOT_RECHECK"
).split()) | frozenset("SFT_" + stage for stage in CHILD_FAILURE_STAGES)
_SFT_FAILURE_STAGES = frozenset({
    "ADMISSION", "ADMISSION_CONTRACTS", "ADMISSION_RELEASE", "ADMISSION_PATHS",
    "ADMISSION_INPUT", "ADMISSION_ENVIRONMENT", "ADMISSION_INVOCATION",
    "ADMISSION_COMMITMENT", "PREPARATION", "REVALIDATION", "INVOCATION", "TRAINER",
    "EVIDENCE", "ARTIFACT", "POST_TRAINING",
}) | frozenset(stage.removeprefix("SFT_") for stage in PACKAGED_WORKER_FAILURE_STAGES
               if stage.startswith(("SFT_PREPARATION_", "SFT_EVIDENCE_"))) | CHILD_FAILURE_STAGES

PACKAGED_PHASES = frozenset({
    "TRAINER_EXECUTE", "TRAINING_PUBLICATION", "TRAINING_ARTIFACT_COMMIT",
    "TRAINING_CONTROL_COMMIT", "EVALUATION_PREPARE", "EVALUATION_IDENTITY_VALIDATE",
    "VLLM_PREPARE", "VLLM_SPAWN", "VLLM_READINESS", "CHAT_BATCH", "CHAT_REQUEST",
    "VLLM_CLEANUP", "EVALUATION_PUBLICATION", "EVALUATION_ARTIFACT_COMMIT",
})


class _PackagedPhaseTrace:
    """Best-effort fixed diagnostic lines, not workload limits or authority."""
    def __init__(self, *, clock=time.monotonic, sink=None):
        self._clock, self._sink = clock, sink
        self._lock, self._count = threading.Lock(), 0
        self._metrics_count = 0
        try:
            self._started = clock()
        except Exception:
            self._started = None

    def emit(self, phase, edge, request_ordinal=None):
        try:
            if (type(phase) is not str or phase not in PACKAGED_PHASES
                    or type(edge) is not str or edge not in {"START", "RETURN", "ERROR"}
                    or (phase == "CHAT_REQUEST" and (type(request_ordinal) is not int
                        or not 1 <= request_ordinal <= 32))
                    or (phase != "CHAT_REQUEST" and request_ordinal is not None)):
                return
            if not self._lock.acquire(blocking=False):
                return
            try:
                if self._count >= 256 or type(self._started) not in (int, float):
                    return
                now = self._clock()
                if type(now) not in (int, float):
                    return
                elapsed = (now - self._started) * 1000
                if not math.isfinite(elapsed) or not 0 <= elapsed <= 86400000:
                    return
                line = "SYNAPTIC_PHASE " + json.dumps({
                    "schema_version": "synaptic-modal-packaged-phase/v1",
                    "phase": phase, "edge": edge, "elapsed_ms": int(elapsed),
                    "request_ordinal": request_ordinal,
                }, separators=(",", ":"), sort_keys=True)
                if len(line.encode("utf-8")) > 512:
                    return
                self._count += 1
                if self._sink is None:
                    print(line, flush=True)
                else:
                    self._sink(line)
            finally:
                self._lock.release()
        except Exception:
            pass

    def emit_serving(self, kind, values):
        try:
            metric_keys = {"running_requests", "waiting_requests", "generation_tokens", "kv_cache_usage"}
            if type(kind) is not str or type(values) is not dict:
                return
            projection = {key: None for key in metric_keys}
            projection["cleanup_resolved"] = None
            if kind == "METRICS":
                if set(values) != metric_keys:
                    return
                if any(type(values[key]) is not int or not 0 <= values[key] <= 2**53 - 1
                       for key in metric_keys - {"kv_cache_usage"}):
                    return
                fraction = values["kv_cache_usage"]
                if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
                    return
                projection.update(values)
            elif kind == "CLEANUP":
                if set(values) != {"cleanup_resolved"} or type(values["cleanup_resolved"]) is not bool:
                    return
                projection.update(values)
            else:
                return
            if not self._lock.acquire(blocking=False):
                return
            try:
                if self._count >= 256 or (kind == "METRICS" and
                        (self._count >= 128 or self._metrics_count >= 64)):
                    return
                if type(self._started) not in (int, float):
                    return
                now = self._clock()
                if type(now) not in (int, float):
                    return
                elapsed = (now - self._started) * 1000
                if not math.isfinite(elapsed) or not 0 <= elapsed <= 86400000:
                    return
                line = "SYNAPTIC_SERVING " + json.dumps({
                    "schema_version": "synaptic-modal-serving-diagnostic/v1",
                    "kind": kind, "elapsed_ms": int(elapsed), **projection,
                }, separators=(",", ":"), sort_keys=True)
                if len(line.encode("utf-8")) > 512:
                    return
                self._count += 1
                if kind == "METRICS":
                    self._metrics_count += 1
                if self._sink is None:
                    print(line, flush=True)
                else:
                    self._sink(line)
            finally:
                self._lock.release()
        except Exception:
            pass


def _phase_call(trace, phase, operation):
    trace.emit(phase, "START")
    try:
        result = operation()
    except BaseException:
        trace.emit(phase, "ERROR")
        raise
    trace.emit(phase, "RETURN")
    return result


def packaged_worker_failure(stage: str) -> dict[str, object]:
    """Return only a fixed, non-secret location for a failed worker boundary."""
    if type(stage) is not str or stage not in PACKAGED_WORKER_FAILURE_STAGES:
        stage = "SFT_UNKNOWN"
    return {
        "schema_version": "synaptic-modal-packaged-worker-result/v2",
        "effect_id": "unavailable",
        "status_code": "failed",
        "completion_sha256": "0" * 64,
        "failure_stage": stage,
    }


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
        on_training_complete=None,
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
        on_training_complete=None,
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
            on_training_complete=on_training_complete,
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

    __slots__ = ("_facts", "_verifier", "_executor", "_signer", "_roots", "_bindings", "_private_root")

    def __init__(
        self,
        *,
        expected_facts: ModalPackagedRuntimeFactsV1,
        dispatch_verifier: ModalPackagedDispatchVerifier,
        trainer_executor: PackagedTrainerExecutor,
        evidence_signer: ModalPackagedEvidenceSigner,
        roots: ModalPackagedWorkerRoots,
        volume_bindings: dict[str, object] | None = None,
        private_root: Path | None = None,
    ) -> None:
        if type(expected_facts) is not ModalPackagedRuntimeFactsV1:
            raise TypeError("exact packaged Modal facts required")
        if not hasattr(dispatch_verifier, "verify") \
                or not hasattr(trainer_executor, "execute") \
                or not hasattr(evidence_signer, "sign"):
            raise TypeError("complete packaged worker collaborators required")
        if type(roots) is not ModalPackagedWorkerRoots:
            raise TypeError("exact packaged worker roots required")
        if (volume_bindings is None) != (private_root is None):
            raise ValueError("bound packaged worker requires private scratch")
        if private_root is not None and (not isinstance(private_root, Path) or not private_root.is_absolute()):
            raise ValueError("bound packaged worker scratch is invalid")
        self._facts, self._verifier = expected_facts, dispatch_verifier
        self._executor, self._signer, self._roots = trainer_executor, evidence_signer, roots
        self._bindings, self._private_root = volume_bindings, private_root

    def _paths(self, dispatch) -> PackagedSFTPaths:
        effect_id = dispatch.submit_command.operation.effect.effect_id
        if self._bindings is not None:
            assert self._private_root is not None
            operation = self._private_root / "operation"
            operation.mkdir(mode=0o700)
            prepared = operation / "payload.bin"
            raw = self._bindings["artifacts"].read_regular(
                dispatch.stage_receipt.path, dispatch.stage_receipt.size_bytes,
            )
            if len(raw) != dispatch.stage_receipt.size_bytes or hashlib.sha256(raw).hexdigest() != dispatch.stage_receipt.content_digest:
                raise ValueError("staged packaged input differs")
            with prepared.open("xb") as stream:
                stream.write(raw)
            paths = {name: operation / name for name in ("artifacts", "state", "tracking", "cache", "tmp")}
            for path in paths.values():
                path.mkdir(mode=0o700)
            return PackagedSFTPaths(prepared, **paths)
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

    def _publish_completion(self, dispatch, result, job_ref: str, paths: PackagedSFTPaths) -> bytes:
        effect_id = dispatch.submit_command.operation.effect.effect_id
        bound = self._bindings is not None
        control_root = paths.state.parent if bound else self._roots.control
        artifact_root = paths.artifacts.parent if bound else self._roots.artifacts
        inventory = read_regular(
            control_root,
            result.inventory_path,
            4 * 1024 * 1024,
        )
        terminal = read_regular(
            control_root,
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
            artifact_root,
            paths.artifacts if bound else self._roots.artifacts / operation_path(effect_id, "output"),
            6,
        ) != expected_listing:
            raise ValueError("packaged trainer artifact inventory is not exact")
        for name, size, digest, role in inventory_members:
            path = paths.artifacts / name if bound else self._roots.artifacts / operation_path(effect_id, "output", name)
            if hash_regular(
                artifact_root, path,
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
        if bound:
            assert self._bindings is not None
            artifacts_binding = self._bindings["artifacts"]
            artifacts_binding.claim_directory(operation_path(effect_id))
            artifacts_binding.claim_directory(operation_path(effect_id, "output"))
            for name, size, digest, _ in inventory_members:
                artifacts_binding.copy_in_exclusive(
                    operation_path(effect_id, "output", name), str(paths.artifacts / name),
                    expected_size=size, expected_sha256=digest,
                    maximum=MODAL_TRAINING_ARTIFACT_BOUNDS_V1.max_artifact_bytes,
                )
            if artifacts_binding.list_regular(operation_path(effect_id, "output"), 6) != expected_listing:
                raise ValueError("published packaged artifacts differ")
            control_binding = self._bindings["control"]
            control_binding.claim_directory(operation_path(effect_id))
            control_binding.claim_directory(operation_path(effect_id, "state"))
            for name, source in (("runtime-v1-inventory.json", result.inventory_path),
                                 ("packaged-terminal.json", result.terminal_path)):
                raw = inventory if name == "runtime-v1-inventory.json" else terminal
                control_binding.write_exclusive(operation_path(effect_id, "state", name), raw, 4 * 1024 * 1024)
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
        evidence_path = operation_path(effect_id, "evidence")
        if bound:
            control_binding = self._bindings["control"]
            control_binding.claim_directory(evidence_path)
            control_binding.write_exclusive(evidence_path + "/packaged-completion.json", completion, 4 * 1024 * 1024)
            control_binding.write_exclusive(evidence_path + "/packaged-completion.mac", tag, 128)
        else:
            evidence = self._roots.control / evidence_path
            claim_directory(self._roots.control, evidence)
            write_exclusive(self._roots.control, evidence / "packaged-completion.json", completion)
            write_exclusive(self._roots.control, evidence / "packaged-completion.mac", tag)
        return completion

    def __call__(
        self,
        dispatch_bytes: bytes,
        provider_job_ref: str,
        *,
        commit_artifacts,
        commit_control,
    ) -> dict[str, object]:
        trace = _PackagedPhaseTrace()
        stage = "DISPATCH_AUTH"
        try:
            dispatch = parse_modal_packaged_dispatch(dispatch_bytes, self._verifier)
            if dispatch.provider_facts != self._facts:
                raise ValueError("packaged dispatch targets another deployment")
            if self._bindings is not None:
                if dispatch.schema_version != MODAL_PACKAGED_DISPATCH_V2_SCHEMA:
                    raise ValueError("bound packaged worker requires signed markers")
                if set(self._bindings) != {marker.role for marker in dispatch.volume_markers} or any(
                    (self._bindings[marker.role].volume_id,
                     self._bindings[marker.role].marker_name,
                     self._bindings[marker.role].marker_sha256)
                    != (marker.volume_id, marker.marker_name, marker.value_sha256)
                    for marker in dispatch.volume_markers
                ):
                    raise ValueError("bound packaged worker marker mismatch")
            elif dispatch.schema_version == MODAL_PACKAGED_DISPATCH_V2_SCHEMA:
                raise ValueError("packaged worker lacks bound Volume roots")
            stage = "STAGED_INPUT"
            if self._bindings is None:
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
            stage = "PATH_CLAIM"
            paths = self._paths(dispatch)
            stage = "SFT_ADMISSION"
            post_training = json.loads(dispatch.workload_bytes)["configuration"]["document"].get("post_training")
            retained_completion = []
            def complete_then_evaluate(result, context):
                # Save the verified adapter before serving. Later evaluation
                # failure cannot erase or require a replay of training.
                completion = _phase_call(trace, "TRAINING_PUBLICATION", lambda:
                    self._publish_completion(dispatch, result, provider_job_ref, paths))
                _phase_call(trace, "TRAINING_ARTIFACT_COMMIT", commit_artifacts)
                _phase_call(trace, "TRAINING_CONTROL_COMMIT", commit_control)
                retained_completion.append(completion)
                _phase_call(trace, "EVALUATION_IDENTITY_VALIDATE", context.validate)
                from tuner.runtime.post_training_eval import execute_post_training_evaluation
                record = execute_post_training_evaluation(
                    post_training,
                    base_model_path=context.base_model_path,
                    adapter_path=context.adapter_path,
                    tokenizer_path=context.tokenizer_path,
                    validate=context.validate,
                    environment=context.environment,
                    cwd=paths.tmp,
                    python_executable=context.python_executable,
                    bindings=context.bindings,
                    phase_callback=trace.emit,
                    serving_callback=trace.emit_serving,
                )
                _phase_call(trace, "EVALUATION_PUBLICATION", lambda:
                    self._publish_evaluation(dispatch, record, provider_job_ref, completion, paths))
                _phase_call(trace, "EVALUATION_ARTIFACT_COMMIT", commit_artifacts)
            result = _phase_call(trace, "TRAINER_EXECUTE", lambda: self._executor.execute(
                runtime_release=dispatch.runtime_release,
                provider_binding=dispatch.provider_binding,
                execution_binding=dispatch.execution_binding,
                workload_bytes=dispatch.workload_bytes,
                artifact_policy=dispatch.artifact_policy,
                paths=paths,
                environment=dispatch.environment,
                **({"on_training_complete": complete_then_evaluate} if post_training is not None else {}),
            ))
            stage = "COMPLETION"
            if post_training is not None and len(retained_completion) != 1:
                raise ValueError("packaged evaluation callback unavailable")
            completion = (retained_completion[0] if retained_completion else
                          _phase_call(trace, "TRAINING_PUBLICATION", lambda:
                              self._publish_completion(dispatch, result, provider_job_ref, paths)))
            if not retained_completion:
                stage = "ARTIFACT_COMMIT"
                _phase_call(trace, "TRAINING_ARTIFACT_COMMIT", commit_artifacts)
                stage = "CONTROL_COMMIT"
                _phase_call(trace, "TRAINING_CONTROL_COMMIT", commit_control)
            return {
                "schema_version": "synaptic-modal-packaged-worker-result/v1",
                "effect_id": dispatch.submit_command.operation.effect.effect_id,
                "status_code": "completed",
                "completion_sha256": hashlib.sha256(completion).hexdigest(),
            }
        except BaseException as exc:
            if stage == "SFT_ADMISSION":
                stage = (
                    "SFT_" + exc.stage
                    if type(exc) is PackagedSFTExecutionError
                    and exc.stage in _SFT_FAILURE_STAGES else "SFT_UNKNOWN"
                )
            return packaged_worker_failure(stage)

    def _publish_evaluation(self, dispatch, record, job_ref, completion, paths):
        """Separate signed phase output; the five training roles stay exact."""
        from tuner.training.recipes import canonical_json_bytes
        effect_id = dispatch.submit_command.operation.effect.effect_id
        configured = json.loads(dispatch.workload_bytes)["configuration"]["document"]["post_training"]
        from tuner.runtime.post_training_eval import (
            canonical_evaluation_document_bytes, validate_evaluation_record,
            MAX_EVALUATION_RECORD_BYTES,
        )
        validate_evaluation_record(record, config=configured)
        document = {
            "schema_version": "synaptic-modal-packaged-evaluation/v1",
            "effect_id": effect_id,
            "command_digest": dispatch.submit_command.digest,
            "provider_job_ref": safe_ref(job_ref, "provider_job_ref"),
            "execution_binding_digest": dispatch.execution_binding.binding_digest,
            "training_completion_sha256": hashlib.sha256(completion).hexdigest(),
            "post_training_sha256": hashlib.sha256(canonical_json_bytes(configured)).hexdigest(),
            "evaluation": record,
        }
        raw = canonical_evaluation_document_bytes(document)
        if len(raw) > MAX_EVALUATION_RECORD_BYTES:
            raise ValueError("packaged evaluation record exceeds bound")
        tag = self._signer.sign("modal-packaged-evaluation/v1", raw, dispatch.key_ref)
        if type(tag) is not bytes or not 1 <= len(tag) <= 128:
            raise ValueError("packaged evaluation authentication unavailable")
        relative = operation_path(effect_id, "evaluation")
        if self._bindings is not None:
            output = self._bindings["artifacts"]
            output.claim_directory(relative)
            output.write_exclusive(relative + "/record.json", raw, MAX_EVALUATION_RECORD_BYTES)
            output.write_exclusive(relative + "/record.mac", tag, 128)
        else:
            root = self._roots.artifacts
            directory = root / relative
            claim_directory(root, directory)
            write_exclusive(root, directory / "record.json", raw)
            write_exclusive(root, directory / "record.mac", tag)


__all__ = [
    "InstalledPackagedSFTTrainerExecutor",
    "ModalPackagedEvidenceSigner",
    "ModalPackagedWorker",
    "ModalPackagedWorkerRoots",
    "PACKAGED_WORKER_FAILURE_STAGES",
    "PackagedTrainerExecutor",
    "packaged_worker_failure",
]
