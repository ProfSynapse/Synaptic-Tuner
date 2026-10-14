"""One-use installed GRPO execution; provider and credential objects stay outside."""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import time
from types import SimpleNamespace

from tuner.runtime import packaged_sft_execution as shared
from tuner.runtime.releases import (
    PackagedExecutionBindingV1, ProviderRuntimeBindingV1,
    PackagedTrainingRuntimeReleaseV1, PackagedTrainingRuntimeReleaseV2,
)
from tuner.training.contracts import ArtifactPolicy
from tuner.training.grpo_compilation import admit_grpo_snapshot
from tuner.training.packaged_compilation import packaged_artifact_policy_digest
from tuner.training.packaged_grpo_compilation import (
    ROLES, parse_packaged_grpo_workload, packaged_grpo_configuration_digest, _declared_input,
)

_TOKEN = object()
_CHILD_SCHEMA = "synaptic-packaged-grpo-child/v1"
LINEAGE_SCHEMA = "synaptic-packaged-grpo-training-lineage/v1"
TERMINAL_SCHEMA = "synaptic-packaged-grpo-terminal/v1"


class PackagedGRPOExecutionError(RuntimeError):
    """Closed consumed-execution failure; never automatic replay authority."""
    def __init__(self, stage):
        if stage not in {"ADMISSION", "REVALIDATION", "PREPARATION", "INVOCATION",
                         "TRAINER", "CHILD_DEADLINE", "CLEANUP_UNCERTAIN", "ARTIFACT"}:
            stage = "ADMISSION"
        self.stage = stage
        super().__init__("PACKAGED_GRPO_" + stage + "_REJECTED")


@dataclass(frozen=True)
class AdmittedPackagedGRPO:
    release: object
    provider_binding: ProviderRuntimeBindingV1
    execution: PackagedExecutionBindingV1
    workload_bytes: bytes
    artifact_policy: ArtifactPolicy
    paths: shared.PackagedSFTPaths
    environment: tuple
    _directories: tuple = field(repr=False)
    _seal: object = field(repr=False)
    _commitment: str = field(repr=False)
    _used: list = field(default_factory=list, repr=False)

    def close(self):
        if not self._used:
            self._used.append(True)
        if shared._close_resources(self._directories):
            raise PackagedGRPOExecutionError("REVALIDATION")


def _admit_contracts(release, provider, execution, raw, policy):
    from tuner.runtime.packaged_training_worker import admit_packaged_training_release
    if (type(release) not in (PackagedTrainingRuntimeReleaseV1, PackagedTrainingRuntimeReleaseV2)
            or type(provider) is not ProviderRuntimeBindingV1
            or type(execution) is not PackagedExecutionBindingV1
            or type(policy) is not ArtifactPolicy):
        raise TypeError
    release = admit_packaged_training_release(release.canonical_bytes(),
        expected_release_digest=execution.runtime_release_digest, method="grpo")
    provider = ProviderRuntimeBindingV1.from_dict(provider.to_dict())
    execution = PackagedExecutionBindingV1.from_dict(execution.to_dict())
    execution.validate_bindings(release, provider)
    compiled = parse_packaged_grpo_workload(raw)
    document = compiled.document
    core = document["core_workload"]
    model = core["model"]
    if (document["context"]["runtime_release"] != release.to_dict()
            or compiled.fingerprint != execution.workload_digest
            or packaged_grpo_configuration_digest(compiled) != execution.configuration_digest
            or core["prepared_input"] != execution.to_dict()["prepared_input"]
            or model["tokenizer_revision"] != model["revision"]
            or (model["model_id"], model["revision"]) not in release.compatible_models
            or set(policy.required_kinds) != set(ROLES) or policy.retain_checkpoints
            or packaged_artifact_policy_digest(policy) != execution.artifact_policy_digest):
        raise ValueError
    return compiled


def _admit_dataset(compiled, raw):
    from synaptic_tuner.api.v1.training_sources import PreparedTrainingInputIdentity
    core = compiled.document["core_workload"]
    return admit_grpo_snapshot(_declared_input(core),
        PreparedTrainingInputIdentity.from_dict(core["prepared_input"]), raw)


def _model(compiled):
    model = compiled.document["core_workload"]["model"]
    return {"ref": model["model_id"], "revision": model["revision"],
            "tokenizer_revision": model["tokenizer_revision"]}


def admit_packaged_grpo(*, runtime_release, provider_binding, execution_binding,
                        workload_bytes, artifact_policy, paths, environment=()):
    """Read-only input admission; no model preparation or ML import."""
    directories = ()
    try:
        compiled = _admit_contracts(runtime_release, provider_binding, execution_binding,
                                   workload_bytes, artifact_policy)
        shared._inspect_parent_release(runtime_release, method="grpo")
        directories = shared._hold_paths(paths)
        _admit_dataset(compiled, shared._input(paths, execution_binding))
        values = dict(environment)
        if (len(values) != len(environment)
                or not set(values) <= {"CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "LANG", "LC_ALL"}
                or any(type(value) is not str or "\0" in value for value in values.values())):
            raise ValueError
        environment = tuple(sorted(values.items()))
        return AdmittedPackagedGRPO(runtime_release, provider_binding, execution_binding,
            workload_bytes, artifact_policy, paths, environment, directories, _TOKEN,
            shared._physical_commitment(paths, environment))
    except BaseException:
        shared._close_resources(directories)
        raise PackagedGRPOExecutionError("ADMISSION") from None


def _invocation_spec(admitted, manifest, dataset_fd, metric_fd, deadline):
    paths = admitted.paths
    model = _model(parse_packaged_grpo_workload(admitted.workload_bytes))
    _, snapshot = shared._private_manifest(manifest, paths, model)
    environment = dict(admitted.environment)
    environment.setdefault("LC_ALL", "C.UTF-8")
    environment.update({
        "PATH": ":".join(dict.fromkeys((str(Path(admitted.release.python_executable).parent),
                                       "/usr/local/bin", "/usr/bin", "/bin"))),
        "PYTHONNOUSERSITE": "1", "PYTHONSAFEPATH": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HOME": str(paths.cache / "huggingface"), "TRANSFORMERS_CACHE": str(paths.cache / "transformers"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "WANDB_DISABLED": "true",
        "TMPDIR": str(paths.tmp), "HOME": str(paths.tmp), "SYNAPTIC_MODEL_SNAPSHOT": str(snapshot),
    })
    payload = {
        "schema_version": _CHILD_SCHEMA, "release": admitted.release.to_dict(),
        "provider_binding": admitted.provider_binding.to_dict(), "execution": admitted.execution.to_dict(),
        "workload": shared._document(admitted.workload_bytes, 1024 * 1024),
        "policy": {"required_kinds": list(admitted.artifact_policy.required_kinds),
                   "retain_checkpoints": admitted.artifact_policy.retain_checkpoints},
        "paths": {name: str(getattr(paths, name)) for name in paths.__dataclass_fields__},
        "environment": list(admitted.environment), "model_snapshot": manifest,
        "dataset_fd": dataset_fd, "metric_fd": metric_fd, "deadline": deadline,
    }
    raw = shared._canonical(payload)
    if len(raw) > shared._MAX_CHILD_BYTES:
        raise ValueError
    transport = paths.state / "packaged-grpo-child.json"
    argv = (admitted.release.python_executable, "-I", "-B", "-m",
            "tuner.runtime.packaged_grpo_child", "--transport", str(transport),
            "--digest", shared._digest(raw))
    invocation = SimpleNamespace(argv=argv, cwd=paths.tmp, environment=tuple(sorted(environment.items())),
        retained_fds=(dataset_fd,), stdout_path=paths.tracking / "trainer.stdout.log",
        stderr_path=paths.tracking / "trainer.stderr.log")
    return invocation, transport, raw


def _bindings(admitted):
    return shared._bindings(admitted)


def execute_admitted_packaged_grpo(admitted, *, model_preparer, on_metric_frame=None,
                                  on_metric_exhausted=None):
    """Child-only finite budget starts after model preparation; no public cancel."""
    from Trainers.sft import runtime_v1 as artifact_utils
    from tuner.runtime.packaged_grpo_verification import assemble_grpo_artifacts, verify_packaged_grpo_artifacts
    stage = "REVALIDATION"
    resources = []
    dataset_fd = None
    pipe = None
    try:
        if (type(admitted) is not AdmittedPackagedGRPO or admitted._seal is not _TOKEN
                or admitted._used or shared._physical_commitment(admitted.paths, admitted.environment)
                != admitted._commitment):
            raise ValueError
        admitted._used.append(True)
        compiled = _admit_contracts(admitted.release, admitted.provider_binding, admitted.execution,
                                   admitted.workload_bytes, admitted.artifact_policy)
        stage = "PREPARATION"
        custody = shared._prepared_execution_inputs(paths=admitted.paths, execution=admitted.execution,
            directories=admitted._directories, release=admitted.release, model=_model(compiled),
            model_preparer=model_preparer, method="grpo")
        resources.append(custody)
        custody.check()
        _admit_dataset(compiled, custody.dataset_bytes)
        stage = "INVOCATION"
        dataset_fd, _ = artifact_utils._sealed_prepared_dataset(custody.dataset_bytes,
            content_digest=admitted.execution.prepared_input_content_digest)
        pipe = os.pipe2(os.O_CLOEXEC | os.O_NONBLOCK)
        # This monotonic deadline is minted once from committed configuration.
        # Model preparation/provider lifetime are separate bounds; never reset it.
        deadline = time.monotonic() + compiled.document["limits"]["child_max_seconds"]
        invocation, transport, raw = _invocation_spec(admitted, custody.private_copy.manifest,
                                                    dataset_fd, pipe[1], deadline)
        state = next(directory for directory in admitted._directories if directory.path == admitted.paths.state)
        shared._write_terminal_exclusive(transport, raw, state, custody.check)
        custody.check()
        stage = "TRAINER"
        read_fd, write_fd = pipe
        pipe = None
        exit_code = shared._run_child_process(invocation, read_fd, write_fd,
            maximum=min(10000, compiled.document["core_workload"]["config"]["training"]["max_steps"]),
            every_n_steps=1, on_metric_frame=on_metric_frame, on_metric_exhausted=on_metric_exhausted,
            deadline=deadline)
        if exit_code != 0:
            raise ValueError
        stage = "ARTIFACT"
        custody.check()
        result = assemble_grpo_artifacts(admitted, compiled, custody.private_copy.manifest)
        inventory, files = shared._retain_result_artifacts(admitted, result)
        resources.extend(files)
        terminal = {"schema_version": TERMINAL_SCHEMA, **_bindings(admitted),
                    "status": "completed", "trainer_exit_code": exit_code,
                    "inventory_sha256": shared._digest(inventory), "artifact_roles": list(ROLES)}
        verify_packaged_grpo_artifacts(admitted=admitted, inventory_bytes=inventory,
                                      terminal_bytes=shared._canonical(terminal))
        artifact_root = next(directory for directory in admitted._directories
                             if directory.path == admitted.paths.artifacts)
        def check():
            custody.check()
            for member in files:
                member.check()
            shared._check_artifact_namespace(artifact_root, files)
        terminal_path = admitted.paths.state / "packaged-terminal.json"
        shared._write_terminal_exclusive(terminal_path, shared._canonical(terminal), state, check)
        return SimpleNamespace(workload_fingerprint=result.workload_fingerprint,
            inventory_path=result.inventory_path, terminal_path=terminal_path, artifacts=result.artifacts)
    except BaseException as error:
        if isinstance(error, (TimeoutError, shared.subprocess.TimeoutExpired)):
            stage = "CHILD_DEADLINE"
        elif type(error) is RuntimeError and error.args == ("PACKAGED_CHILD_CLEANUP_UNCERTAIN",):
            stage = "CLEANUP_UNCERTAIN"
        raise PackagedGRPOExecutionError(stage) from None
    finally:
        failed = False
        for descriptor in ((dataset_fd,) if dataset_fd is not None else ()) + (pipe or ()):
            try:
                os.close(descriptor)
            except OSError:
                failed = True
        if type(admitted) is AdmittedPackagedGRPO:
            resources.append(admitted)
        failed = shared._close_resources(resources) or failed
        if failed:
            raise PackagedGRPOExecutionError("CLEANUP_UNCERTAIN") from None
