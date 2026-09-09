"""Foundation-native admission and orchestration for the fixed Modal worker."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Protocol

from tuner.execution.foundation_v2.canonical import (
    canonical_bytes, digest_text, parse_canonical_object, safe_ref,
)
from tuner.execution.foundation_v2.commands import StageCommandV2, SubmitCommandV2, parse_exact_command
from tuner.project.execution_source import ExecutionSourceV1
from tuner.runtime.dispatch import WorkerControlLocationV1
from tuner.runtime.offline_sft_worker import parse_offline_sft_worker_manifest
from tuner.training.methods.sft import SFTRecipe
from tuner.training.recipes import RecipeRegistry

from .contracts import BoundsPolicyV1, operation_path, sha, strict_int
from .coordinator_binding import ModalCommandBinding
from .coordinator_bundle import ModalCoordinatorBundle
from .coordinator_dispatch import parse_modal_worker_dispatch
from .coordinator_wire import ModalWorkerLaunchExpectation, admit_modal_launch_wire
from .mounted_io import read_regular
from .resolution import ModalDeploymentSelectionV1, VerifiedModalDeploymentIdentityV1
from .worker_ports import (
    FixedProcessRunner, ModalProcessResult, ModalRemotePhaseError, SourceMaterializer,
)
from . import worker_source


class ModalWireVerifier(Protocol):
    def verify(self, purpose: str, payload: bytes, tag: bytes, key_ref: str) -> bool: ...


class ModalCompletionProducer(Protocol):
    def finalize(
        self, invocation: "ModalWorkerInvocation", result: ModalProcessResult, *, job_ref: str,
    ) -> object: ...


def _root(value: str, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be exact text")
    path = PurePosixPath(value)
    if (not path.is_absolute() or path == PurePosixPath("/")
            or path.as_posix() != value or "//" in value or ".." in path.parts):
        raise ValueError(f"{name} must be a canonical absolute POSIX path")
    return value


@dataclass(frozen=True, slots=True)
class ModalWorkerStaticExpectation:
    deployment_selection_bytes: bytes
    provider_id: str
    profile_ref: str
    executor_id: str
    executor_implementation_version: str
    control_volume_id: str
    artifact_volume_id: str
    control_volume_ref: str
    artifact_volume_ref: str
    key_ref: str
    control_root: str
    artifact_root: str
    worker_control_root: str

    def __post_init__(self) -> None:
        if type(self.deployment_selection_bytes) is not bytes:
            raise TypeError("exact deployment selection bytes required")
        selection = ModalDeploymentSelectionV1.from_dict(
            parse_canonical_object(self.deployment_selection_bytes, name="static deployment selection")
        )
        if selection.to_dict() != parse_canonical_object(
            self.deployment_selection_bytes, name="static deployment selection",
        ):
            raise ValueError("static deployment selection does not round-trip")
        for name in (
            "provider_id", "profile_ref", "executor_id", "executor_implementation_version",
            "control_volume_id", "artifact_volume_id", "control_volume_ref",
            "artifact_volume_ref", "key_ref",
        ):
            safe_ref(getattr(self, name), name)
        if self.control_volume_id == self.artifact_volume_id:
            raise ValueError("control and artifact volumes must differ")
        for name in ("control_root", "artifact_root", "worker_control_root"):
            object.__setattr__(self, name, _root(getattr(self, name), name))
        if len({self.control_root, self.artifact_root, self.worker_control_root}) != 3:
            raise ValueError("worker mount roots must be distinct")

    @property
    def selection(self) -> ModalDeploymentSelectionV1:
        return ModalDeploymentSelectionV1.from_dict(
            parse_canonical_object(self.deployment_selection_bytes, name="static deployment selection")
        )


def _static_check(
    expectation: ModalWorkerLaunchExpectation, static: ModalWorkerStaticExpectation,
) -> None:
    if type(static) is not ModalWorkerStaticExpectation:
        raise TypeError("exact static worker expectation required")
    selection = VerifiedModalDeploymentIdentityV1.from_dict(
        parse_canonical_object(expectation.deployment_bytes, name="dispatch deployment")
    ).selection
    expected = static.selection
    if selection != expected:
        raise ValueError("dispatch deployment differs from static worker selection")
    if (
        expectation.provider_id, expectation.profile_ref,
        expectation.executor_id, expectation.executor_implementation_version,
        expectation.control_volume_id, expectation.artifact_volume_id,
        expectation.control_volume_ref, expectation.artifact_volume_ref,
        expectation.key_ref,
    ) != (
        static.provider_id, static.profile_ref,
        static.executor_id, static.executor_implementation_version,
        static.control_volume_id, static.artifact_volume_id,
        static.control_volume_ref, static.artifact_volume_ref, static.key_ref,
    ):
        raise ValueError("dispatch differs from static worker expectation")


@dataclass(frozen=True, slots=True, init=False)
class ModalWorkerInvocation:
    submit_command_bytes: bytes
    stage_command_bytes: bytes
    preparation_snapshot: bytes
    deployment_bytes: bytes
    execution_source_bytes: bytes
    workload: bytes
    log_policy_bytes: bytes
    closure_manifest: bytes
    environment_items: tuple[tuple[str, str], ...]
    argv: tuple[str, str, str]
    cwd: str
    closure_manifest_runtime_path: str
    control_volume_id: str
    artifact_volume_id: str
    key_ref: str
    launch_claim_sha256: str
    stage_claim_sha256: str
    bundle_sha256: str

    def __new__(cls, *args, **kwargs):
        raise TypeError("Modal worker invocations are admission-minted")

    @classmethod
    def _create(cls, **values) -> "ModalWorkerInvocation":
        value = object.__new__(cls)
        for name in cls.__dataclass_fields__:
            object.__setattr__(value, name, values[name])
        _validate_invocation(value)
        return value

    @property
    def submit_command(self) -> SubmitCommandV2:
        command = parse_exact_command(self.submit_command_bytes)
        if type(command) is not SubmitCommandV2:
            raise ValueError("invocation submit command is invalid")
        return command

    @property
    def stage_command(self) -> StageCommandV2:
        command = parse_exact_command(self.stage_command_bytes)
        if type(command) is not StageCommandV2:
            raise ValueError("invocation stage command is invalid")
        return command

    @property
    def deployment(self) -> VerifiedModalDeploymentIdentityV1:
        return VerifiedModalDeploymentIdentityV1.from_dict(
            parse_canonical_object(self.deployment_bytes, name="invocation deployment")
        )

    @property
    def source(self) -> ExecutionSourceV1:
        return ExecutionSourceV1.from_dict(
            parse_canonical_object(self.execution_source_bytes, name="invocation execution source")
        )

    @property
    def environment(self) -> dict[str, str]:
        return dict(self.environment_items)

    @property
    def log_policy(self) -> dict[str, object]:
        return parse_canonical_object(self.log_policy_bytes, name="invocation log policy")


def _validate_invocation(value: ModalWorkerInvocation) -> None:
    byte_fields = (
        "submit_command_bytes", "stage_command_bytes", "preparation_snapshot",
        "deployment_bytes", "execution_source_bytes", "workload", "log_policy_bytes",
        "closure_manifest",
    )
    if any(type(getattr(value, name)) is not bytes or not getattr(value, name) for name in byte_fields):
        raise TypeError("invocation canonical values must be exact nonempty bytes")
    submit, stage = value.submit_command, value.stage_command
    submit_binding = ModalCommandBinding(
        value.submit_command_bytes, value.preparation_snapshot, value.deployment_bytes,
    )
    stage_binding = ModalCommandBinding(
        value.stage_command_bytes, value.preparation_snapshot, value.deployment_bytes,
    )
    predecessor = submit.stage_predecessor
    if (
        stage.preparation != submit.preparation
        or predecessor.stage_effect_id != stage.operation.effect.effect_id
        or predecessor.preparation_digest != stage.preparation.preparation_digest
        or predecessor.workload_digest != stage.preparation.workload_digest
        or submit_binding.deployment != stage_binding.deployment
    ):
        raise ValueError("invocation stage and submit lineage differs")
    source = value.source
    if source.canonical_bytes != value.execution_source_bytes:
        raise ValueError("invocation source does not round-trip")
    workload = parse_canonical_object(value.workload, name="invocation workload")
    if canonical_bytes(workload.get("execution_source")) != source.canonical_bytes:
        raise ValueError("invocation workload differs from source")
    if source.run_id != submit.preparation.run_id:
        raise ValueError("invocation source run differs from preparation")
    if source.fingerprint != submit.preparation.source_digest:
        raise ValueError("invocation source digest differs from preparation")
    workload_fingerprint = hashlib.sha256(
        b"synaptic-training-workload/v1\0" + value.workload
    ).hexdigest()
    if workload_fingerprint != submit.preparation.workload_digest:
        raise ValueError("invocation workload digest differs from preparation")
    policy = value.log_policy
    if set(policy) != {
        "schema_version", "generation", "max_log_chunks", "max_chunk_bytes",
        "max_terminal_bytes",
    } or policy.get("schema_version") != "synaptic-modal-log-terminal-policy/v2":
        raise ValueError("invocation log policy is invalid")
    for name, maximum in (
        ("generation", 2**31 - 1), ("max_log_chunks", 1_000_000),
        ("max_chunk_bytes", 1_048_576), ("max_terminal_bytes", 1_048_576),
    ):
        strict_int(policy.get(name), name, minimum=1, maximum=maximum)
    closure = parse_offline_sft_worker_manifest(
        value.closure_manifest, source_ref="modal-worker-invocation:closure",
        manifest_path=Path("worker-closure-manifest.json"),
    )
    if (type(value.environment_items) is not tuple
            or any(type(item) is not tuple or len(item) != 2
                   or type(item[0]) is not str or type(item[1]) is not str
                   for item in value.environment_items)
            or value.environment_items != tuple(sorted(value.environment_items))
            or len(dict(value.environment_items)) != len(value.environment_items)):
        raise TypeError("invocation environment must be exact sorted unique string pairs")
    expected_environment = dict(source.environment)
    if expected_environment.pop("PYTHONPATH", None) != source.roots["engine"]:
        raise ValueError("invocation source PYTHONPATH is invalid")
    model = workload["configuration"]["document"]["model"]
    expected_environment.update({
        "SYNAPTIC_WORKLOAD_FINGERPRINT": hashlib.sha256(
            b"synaptic-training-workload/v1\0" + value.workload
        ).hexdigest(),
        "SYNAPTIC_MODEL_SNAPSHOT": source.roots["cache"] + "/model/models--"
        + str(model["ref"]).replace("/", "--") + "/snapshots/" + str(model["revision"]),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "SYNAPTIC_WORKER_CLOSURE_MANIFEST": value.closure_manifest_runtime_path,
        "SYNAPTIC_WORKER_CLOSURE_DIGEST": closure.closure.closure_digest,
    })
    expected_argv = (
        source.python_executable,
        source.roots["engine"] + "/Trainers/sft/runtime_v1.py",
        "--canonical-workload-stdin",
    )
    if (type(value.argv) is not tuple or value.argv != expected_argv
            or value.cwd != source.roots["tmp"]
            or value.environment_items != tuple(sorted(expected_environment.items()))):
        raise ValueError("invocation fixed runtime derivation differs")
    path = PurePosixPath(value.closure_manifest_runtime_path)
    expected_suffix = PurePosixPath(operation_path(
        submit.operation.effect.effect_id, "input", path.name,
    ))
    if (type(value.closure_manifest_runtime_path) is not str
            or not path.is_absolute() or path.as_posix() != value.closure_manifest_runtime_path
            or "//" in value.closure_manifest_runtime_path or ".." in path.parts
            or path.name != "offline-sft-worker-v1.json"
            or tuple(path.parts[-len(expected_suffix.parts):]) != expected_suffix.parts):
        raise ValueError("invocation closure path is not submit-scoped")
    for name in ("control_volume_id", "artifact_volume_id", "key_ref"):
        safe_ref(getattr(value, name), name)
    if value.control_volume_id == value.artifact_volume_id:
        raise ValueError("invocation volumes must differ")
    for name in ("launch_claim_sha256", "stage_claim_sha256", "bundle_sha256"):
        digest_text(getattr(value, name), name)


def validate_modal_worker_invocation(value: ModalWorkerInvocation) -> None:
    """Reconstruct and validate immutable invocation data before any consumer I/O."""
    if type(value) is not ModalWorkerInvocation:
        raise TypeError("exact Modal worker invocation required")
    _validate_invocation(value)


def fixed_sft_recipe_registry() -> RecipeRegistry:
    registry = RecipeRegistry()
    registry.register(SFTRecipe())
    return registry


def _member(bundle: ModalCoordinatorBundle, name: str) -> bytes:
    return next(member.content for member in bundle.members if member.name == name)


def _derive_invocation(
    admission, bundle: ModalCoordinatorBundle, *, stage_claim_sha256: str,
    worker_control_root: str,
) -> ModalWorkerInvocation:
    submit = parse_exact_command(admission.submit_command_bytes)
    stage = parse_exact_command(admission.stage_command_bytes)
    if type(submit) is not SubmitCommandV2 or type(stage) is not StageCommandV2:
        raise ValueError("worker admission requires exact stage and submit commands")
    source_bytes = _member(bundle, "execution-source.json")
    source = ExecutionSourceV1.from_dict(
        parse_canonical_object(source_bytes, name="worker execution source")
    )
    workload = _member(bundle, "workload.json")
    closure_bytes = _member(bundle, "worker-closure-manifest.json")
    closure = parse_offline_sft_worker_manifest(
        closure_bytes, source_ref="modal-coordinator-worker:closure",
        manifest_path=Path("worker-closure-manifest.json"),
    )
    policy_bytes = _member(bundle, "log-terminal-policy.json")
    policy = parse_canonical_object(policy_bytes, name="worker log policy")
    strict_int(policy.get("generation"), "generation", minimum=1)
    environment = dict(source.environment)
    if environment.pop("PYTHONPATH", None) != source.roots["engine"]:
        raise ValueError("worker PYTHONPATH does not bind engine root")
    if {"PYTHONHOME", "PYTHONUSERBASE", "HF_TOKEN"} & set(environment):
        raise ValueError("worker environment contains forbidden ambient authority")
    workload_document = parse_canonical_object(workload, name="worker workload")
    model = workload_document["configuration"]["document"]["model"]
    environment.update({
        "SYNAPTIC_WORKLOAD_FINGERPRINT": hashlib.sha256(
            b"synaptic-training-workload/v1\0" + workload
        ).hexdigest(),
        "SYNAPTIC_MODEL_SNAPSHOT": source.roots["cache"] + "/model/models--"
        + str(model["ref"]).replace("/", "--") + "/snapshots/" + str(model["revision"]),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    })
    control = WorkerControlLocationV1(
        PurePosixPath(worker_control_root)
        / operation_path(submit.operation.effect.effect_id, "input")
    )
    environment["SYNAPTIC_WORKER_CLOSURE_MANIFEST"] = control.manifest_path.as_posix()
    environment["SYNAPTIC_WORKER_CLOSURE_DIGEST"] = closure.closure.closure_digest
    argv = (
        source.python_executable,
        source.roots["engine"] + "/Trainers/sft/runtime_v1.py",
        "--canonical-workload-stdin",
    )
    return ModalWorkerInvocation._create(
        submit_command_bytes=submit.canonical_bytes, stage_command_bytes=stage.canonical_bytes,
        preparation_snapshot=bytes(admission.preparation_snapshot),
        deployment_bytes=bytes(admission.deployment_bytes), execution_source_bytes=source.canonical_bytes,
        workload=bytes(workload), log_policy_bytes=bytes(policy_bytes),
        closure_manifest=bytes(closure_bytes), environment_items=tuple(sorted(environment.items())),
        argv=argv, cwd=source.roots["tmp"],
        closure_manifest_runtime_path=control.manifest_path.as_posix(),
        control_volume_id=admission.control_volume_id,
        artifact_volume_id=admission.artifact_volume_id, key_ref=admission.key_ref,
        launch_claim_sha256=admission.launch_claim_sha256,
        stage_claim_sha256=stage_claim_sha256,
        bundle_sha256=bundle.sha256,
    )


def admit_modal_worker(
    dispatch_bytes: bytes, *, stage_claim: bytes, stage_claim_tag: bytes,
    bundle_transport: bytes, verifier: ModalWireVerifier, recipes: RecipeRegistry,
    static: ModalWorkerStaticExpectation, bounds: BoundsPolicyV1 = BoundsPolicyV1(),
) -> ModalWorkerInvocation:
    dispatch = parse_modal_worker_dispatch(dispatch_bytes, bounds=bounds)
    _static_check(dispatch.expectation, static)
    admission = admit_modal_launch_wire(
        dispatch.launch_claim, dispatch.launch_claim_tag, stage_claim, stage_claim_tag,
        bundle_transport, expectation=dispatch.expectation, verifier=verifier, bounds=bounds,
    )
    stage_binding = ModalCommandBinding(
        admission.stage_command_bytes, admission.preparation_snapshot, admission.deployment_bytes,
    )
    bundle = ModalCoordinatorBundle.parse_transport(
        admission.bundle, binding=stage_binding, recipes=recipes,
    )
    if bundle.transport_bytes != bundle_transport:
        raise ValueError("worker bundle differs from wire admission")
    return _derive_invocation(
        admission, bundle, stage_claim_sha256=sha(stage_claim),
        worker_control_root=static.worker_control_root,
    )


def _execute_modal_worker(
    invocation: ModalWorkerInvocation, *, sources: SourceMaterializer,
    processes: FixedProcessRunner, commit_prepared: Callable[[], None],
) -> ModalProcessResult:
    if type(invocation) is not ModalWorkerInvocation:
        raise TypeError("exact Modal worker invocation required")
    validate_modal_worker_invocation(invocation)
    source = invocation.source
    sources.prepare_and_verify(source, invocation.deployment.selection)
    try:
        locked = worker_source.read_locked_closure_manifest(source)
    except OSError:
        raise ModalRemotePhaseError(124, "locked_source_mismatch") from None
    if locked != invocation.closure_manifest:
        raise ModalRemotePhaseError(124, "locked_source_mismatch")
    try:
        worker_source.write_runtime_closure_manifest(
            invocation.closure_manifest_runtime_path, invocation.closure_manifest,
        )
        worker_source.stage_runtime_worker(
            source, invocation.closure_manifest_runtime_path, invocation.closure_manifest,
        )
    except ModalRemotePhaseError:
        raise
    except FileExistsError:
        raise ModalRemotePhaseError(122, "artifact_layout_collision") from None
    except OSError:
        raise ModalRemotePhaseError(122, "artifact_layout_failed") from None
    except Exception:
        raise ModalRemotePhaseError(124, "locked_source_mismatch") from None
    result = processes.run(
        invocation.argv, cwd=invocation.cwd, environment=invocation.environment,
        stdin=invocation.workload, commit_prepared=commit_prepared,
    )
    if type(result) is not ModalProcessResult:
        raise TypeError("process runner returned noncanonical result")
    return result


class MountedModalCoordinatorWorker:
    __slots__ = ("_verifier", "_sources", "_processes", "_completion", "_recipes", "_static", "_bounds")

    def __init__(self, *, verifier: ModalWireVerifier, sources: SourceMaterializer,
                 processes: FixedProcessRunner, completion: ModalCompletionProducer,
                 static: ModalWorkerStaticExpectation,
                 bounds: BoundsPolicyV1 = BoundsPolicyV1()) -> None:
        if type(static) is not ModalWorkerStaticExpectation:
            raise TypeError("exact static worker expectation required")
        self._verifier, self._sources, self._processes = verifier, sources, processes
        if not hasattr(completion, "finalize"):
            raise TypeError("completion producer is required")
        self._completion, self._static, self._bounds = completion, static, bounds
        self._recipes = fixed_sft_recipe_registry()

    def __call__(self, dispatch_bytes: bytes, job_ref: str,
                 commit_prepared: Callable[[], None]) -> dict[str, object]:
        safe_job_ref = safe_ref(job_ref, "job_ref")
        if not callable(commit_prepared):
            raise TypeError("commit_prepared callback is required")
        dispatch = parse_modal_worker_dispatch(dispatch_bytes, bounds=self._bounds)
        _static_check(dispatch.expectation, self._static)
        try:
            launch_valid = self._verifier.verify(
                "modal-launch-claim/v1", dispatch.launch_claim,
                dispatch.launch_claim_tag, self._static.key_ref,
            )
        except Exception:
            raise ValueError("worker launch authentication unavailable") from None
        if launch_valid is not True:
            raise ValueError("worker launch authentication failed")
        submit = parse_exact_command(dispatch.expectation.submit_command_bytes)
        if type(submit) is not SubmitCommandV2:
            raise ValueError("worker dispatch requires submit command")
        stage_effect = submit.stage_predecessor.stage_effect_id
        control = Path(self._static.control_root)
        artifact = Path(self._static.artifact_root)
        stage_claim = read_regular(control, control / operation_path(stage_effect, "control", "stage-claim.v2.json"), self._bounds.max_control_bytes)
        stage_tag = read_regular(control, control / operation_path(stage_effect, "control", "stage-claim.v2.mac"), 128)
        bundle = read_regular(artifact, artifact / operation_path(stage_effect, "input", "bundle.bin"), self._bounds.max_bundle_bytes)
        invocation = admit_modal_worker(
            dispatch_bytes, stage_claim=stage_claim, stage_claim_tag=stage_tag,
            bundle_transport=bundle, verifier=self._verifier, recipes=self._recipes,
            static=self._static, bounds=self._bounds,
        )
        try:
            result = _execute_modal_worker(
                invocation, sources=self._sources, processes=self._processes,
                commit_prepared=commit_prepared,
            )
        except ModalRemotePhaseError as error:
            result = ModalProcessResult(error.returncode, diagnostic_code=error.diagnostic_code)
        except Exception:
            result = ModalProcessResult(125, diagnostic_code="generic_failure")
        completion = self._completion.finalize(invocation, result, job_ref=safe_job_ref)
        status = getattr(completion, "status_code", None)
        if status not in {"completed", "failed"}:
            raise ValueError("completion producer returned invalid status")
        return {
            "schema_version": "synaptic-modal-worker-result/v2",
            "effect_id": invocation.submit_command.operation.effect.effect_id,
            "returncode": result.returncode, "status_code": status,
        }


__all__: list[str] = []
