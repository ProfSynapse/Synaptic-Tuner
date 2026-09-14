"""Consumer-owned rich training resolution for the minimal Modal chat example."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import stat

from synaptic_tuner.api.v1 import load_training_input_contract_v1
from synaptic_tuner.api.v1._contract import contract_digest
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.training_input import TrainingInputV1
from tuner.execution.foundation_v2.canonical import safe_ref
from tuner.execution.providers.modal.config import ModalRuntimeLockV1
from tuner.execution.providers.modal.resolution import (
    ModalDeploymentSelectionV1,
    ModalExecutionSourceResolutionV1,
)
from tuner.execution.evidence import canonical_utc
from tuner.project.context import ProjectContext
from tuner.project.git_verification import GitCliLocalSourceInspector
from tuner.project.source_bundle import SourceLock
from tuner.training.contracts import (
    ArtifactPolicy,
    CanonicalDocument,
    ResolvedTrainingComponents,
    ResourceSpec,
    RuntimeSpec,
    TrainingRequest,
)
from tuner.training.methods.sft import (
    SFT_CONFIG_SCHEMA,
)

_REVISION = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_MAX_DATASET_BYTES = 16 * 1024 * 1024 * 1024


class ModalChatResolutionError(RuntimeError):
    """Closed error at the example's real-input resolution boundary."""


def _directory_identity(path: Path) -> tuple[int, int, int]:
    value = path.lstat()
    if not stat.S_ISDIR(value.st_mode) or path.is_symlink():
        raise ValueError("consumer source path is invalid")
    return value.st_dev, value.st_ino, value.st_mode


def _regular_digest(root: Path, relative: Path, maximum: int) -> tuple[int, str]:
    if (
        not isinstance(root, Path)
        or not isinstance(relative, Path)
        or relative.is_absolute()
    ):
        raise ValueError("consumer source path is invalid")
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("consumer source path is invalid")
    canonical_root = root.resolve(strict=True)
    selected = canonical_root / relative
    parents = (canonical_root,)
    for component in relative.parts[:-1]:
        parents += (parents[-1] / component,)
    parent_snapshot = tuple((path, _directory_identity(path)) for path in parents)
    if selected.is_symlink() or selected.resolve(strict=True) != selected:
        raise ValueError("consumer source path is invalid")
    descriptor = -1
    try:
        flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(selected, flags)
        before = os.fstat(descriptor)
        leaf_before = selected.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > maximum
            or (before.st_dev, before.st_ino)
            != (leaf_before.st_dev, leaf_before.st_ino)
        ):
            raise ValueError("consumer source file is invalid")
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(descriptor, min(1024 * 1024, maximum + 1 - size)):
            size += len(chunk)
            if size > maximum:
                raise ValueError("consumer source file exceeds its bound")
            digest.update(chunk)
        after = os.fstat(descriptor)
        leaf_after = selected.lstat()
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if (
            size != before.st_size
            or identity(after) != identity(before)
            or identity(leaf_after) != identity(leaf_before)
            or selected.resolve(strict=True) != selected
            or any(
                _directory_identity(path) != prior for path, prior in parent_snapshot
            )
        ):
            raise ValueError("consumer source file changed during resolution")
        return size, digest.hexdigest()
    finally:
        if descriptor != -1:
            os.close(descriptor)


class ModalChatRichTrainingResolver:
    """Resolve one allocated request through real local and authenticated sources."""

    __slots__ = (
        "_context",
        "_run",
        "_created_at",
        "_dataset_path",
        "_load_in_4bit",
        "_deployment",
        "_runtime_lock",
        "_finalizer",
        "_audience_ref",
    )

    def __init__(
        self,
        *,
        context: ProjectContext,
        run: TrainingRunRef,
        created_at: str,
        dataset_project_path: Path,
        load_in_4bit: bool,
        deployment: ModalDeploymentSelectionV1,
        source_finalizer: object,
        audience_ref: str,
    ) -> None:
        if type(context) is not ProjectContext or context.mode != "host":
            raise TypeError("exact host project context required")
        if type(run) is not TrainingRunRef or type(created_at) is not str:
            raise TypeError("exact preallocated run identity required")
        if (
            not isinstance(dataset_project_path, Path)
            or dataset_project_path.is_absolute()
        ):
            raise TypeError("dataset_project_path must be a relative Path")
        if type(load_in_4bit) is not bool:
            raise TypeError("load_in_4bit must be an exact boolean")
        if type(deployment) is not ModalDeploymentSelectionV1:
            raise TypeError("exact Modal deployment selection required")
        if not callable(getattr(source_finalizer, "finalize", None)):
            raise TypeError("configured Modal source finalizer required")
        runtime_lock = ModalRuntimeLockV1.packaged()
        runtime_lock.validate_selection(deployment)
        self._context = context
        self._run = run
        self._created_at = canonical_utc(created_at, "created_at")
        self._dataset_path = dataset_project_path
        self._load_in_4bit = load_in_4bit
        self._deployment = deployment
        self._runtime_lock = runtime_lock
        self._finalizer = source_finalizer
        self._audience_ref = safe_ref(audience_ref, "audience_ref")

    def resolve(
        self, request: TrainingRequest, *, context: ProjectContext
    ) -> ResolvedTrainingComponents:
        try:
            if type(request) is not TrainingRequest or context is not self._context:
                raise TypeError("exact retained request and context required")
            input_contract = load_training_input_contract_v1()
            training_input = input_contract.parse_json(request.document.canonical_json)
            if (
                type(training_input) is not input_contract.input_type
                or CanonicalDocument(training_input.canonical_json())
                != request.document
            ):
                raise ValueError("training input is not canonical")
            if (
                _REVISION.fullmatch(training_input.model.revision) is None
                or _REVISION.fullmatch(training_input.model.tokenizer_revision) is None
            ):
                raise ValueError("model and tokenizer revisions must be immutable")
            inspected = GitCliLocalSourceInspector().inspect(context=context)
            dataset_relative = self._dataset_path
            expected_ref = "project://" + dataset_relative.as_posix()
            if training_input.dataset.ref != expected_ref:
                raise ValueError("dataset reference differs from consumer project path")
            dataset_size, dataset_sha256 = _regular_digest(
                context.project_root, dataset_relative, _MAX_DATASET_BYTES
            )
            training_source_sha256 = hashlib.sha256(
                request.document.canonical_json.encode("utf-8")
            ).hexdigest()
            input_digest = training_input.input_digest()
            contract_identity = input_contract.identity.identity_digest
            ingress_digest = contract_digest(
                "synaptic-modal-chat-training-ingress/v1",
                {
                    "dataset": {
                        "content_digest": dataset_sha256,
                        "path": dataset_relative.as_posix(),
                        "size_bytes": dataset_size,
                    },
                    "training_input_digest": input_digest,
                },
            )
            provider_policy_digest = contract_digest(
                "synaptic-modal-chat-provider-policy/v1", self._deployment.to_dict()
            )
            configuration = {
                "training_input_digest": input_digest,
                "training_contract_identity_digest": contract_identity,
                "training_source_sha256": training_source_sha256,
                "training_ingress_digest": ingress_digest,
                "provider_policy_digest": provider_policy_digest,
            }
            source_lock = SourceLock(
                run_id=self._run.run_id,
                created_at=self._created_at,
                mode="superproject",
                project_source=inspected.project_source,
                engine_source=inspected.engine_source,
                project={"project_ref": self._run.project_ref},
                configuration=configuration,
                inputs=(
                    {
                        "kind": "model",
                        "ref": training_input.model.ref,
                        "revision": training_input.model.revision,
                    },
                    {
                        "content_digest": dataset_sha256,
                        "kind": "dataset",
                        "ref": training_input.dataset.ref,
                        "revision": inspected.project_source.commit.lower(),
                        "size_bytes": dataset_size,
                    },
                ),
                runtime={
                    "deployment_ref": self._deployment.deployment_ref,
                    "image_digest": self._deployment.image_digest,
                    "selection_digest": provider_policy_digest,
                },
                outputs={"artifacts": training_input.artifacts.to_dict()},
            )
            finalized = self._finalizer.finalize(
                source_lock,
                context=context,
                deployment=self._deployment,
                audience_ref=self._audience_ref,
            )
            if type(finalized) is not ModalExecutionSourceResolutionV1:
                raise TypeError("source finalizer returned an invalid result")
            if (
                finalized.execution_source.run_id != self._run.run_id
                or finalized.execution_source.source_evidence.source_lock_binding
                != source_lock.binding
                or not finalized.execution_source.source_evidence.binds(source_lock)
            ):
                raise ValueError("source finalizer returned a substituted source")
            hyperparameters = training_input.hyperparameters.to_dict()
            hyperparameters.pop("schema_version")
            duration = hyperparameters.pop("duration")
            hyperparameters.update(
                {name: value for name, value in duration.items() if value is not None}
            )
            resolved_config = CanonicalDocument.from_mapping(
                {
                    "schema_version": SFT_CONFIG_SCHEMA,
                    "method": "sft",
                    "model": {
                        **training_input.model.to_dict(),
                        "load_in_4bit": self._load_in_4bit,
                    },
                    "dataset": {
                        "content_digest": dataset_sha256,
                        "ref": training_input.dataset.ref,
                        "revision": inspected.project_source.commit.lower(),
                    },
                    "sft": hyperparameters,
                }
            )
            return ResolvedTrainingComponents(
                execution_source=finalized.execution_source,
                execution_context=CanonicalDocument.from_mapping(
                    {
                        "schema_version": "synaptic-modal-chat-execution-context/v1",
                        "source_lock_binding": source_lock.binding.to_dict(),
                        "deployment_evidence_sha256": finalized.execution_source.deployment_member_sha256,
                    }
                ),
                resolved_config=resolved_config,
                runtime=RuntimeSpec(
                    self._runtime_lock.registry_reference,
                    self._runtime_lock.locked_digest("dependency_lock"),
                    self._runtime_lock.python_version,
                ),
                resources=ResourceSpec(
                    self._deployment.accelerator,
                    1,
                    self._deployment.timeout_seconds,
                ),
                artifact_policy=ArtifactPolicy(
                    training_input.artifacts.required_kinds,
                    training_input.artifacts.retain_checkpoints,
                ),
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            raise ModalChatResolutionError("modal_chat_resolution_invalid") from None


__all__ = ["ModalChatResolutionError", "ModalChatRichTrainingResolver"]
