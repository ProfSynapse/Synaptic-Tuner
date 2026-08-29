"""Authenticated Modal execution-source finalization for host superprojects."""

from __future__ import annotations

import hashlib
import json
import os
import re
import base64
import binascii
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Callable, Mapping, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .binding import ModalClientBinding
    from .config import ModalProviderProfileV1

from tuner.project.context import ProjectContext
from tuner.project.execution_source import (
    AuthenticatedSourceEvidenceV1,
    ExecutionSourceV1,
    LocalSourceInspectionPort,
    PushedSourceVerificationPort,
    _SealedSourceEvidenceSnapshotV1,
    _capture_source_evidence_snapshot,
    _require_source_evidence_snapshot,
)
from tuner.project.source_bundle import (
    SourceLock,
    SourceLockError,
    _SealedSourceLockSnapshotV1,
    _capture_source_lock_snapshot,
    _require_source_lock_snapshot,
)

from ...contracts import digest, safe_ref
from .deployment_identity import validate_modal_function_identity
from ...evidence import (
    DEPLOYMENT_EVIDENCE_POLICY, SOURCE_EVIDENCE_POLICY, SOURCE_EVIDENCE_PURPOSE,
    EvidenceAuthenticator,
    EvidenceReplayRepository, admit_evidence, validate_evidence_window,
)


_VERSION_RE = re.compile(r"^(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)$")


def _canonical(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _runtime_environment(value: Mapping[str, str]) -> Mapping[str, str]:
    result = dict(value)
    if any(
        not isinstance(key, str) or not key or not isinstance(member, str)
        or any(ord(char) < 0x20 for char in key + member)
        for key, member in result.items()
    ):
        raise ValueError("runtime_environment must be a closed text map")
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class ModalDeploymentSelectionV1:
    account_ref: str
    workspace_ref: str
    environment_ref: str
    client_ref: str
    app_name: str
    function_name: str
    deployment_ref: str
    image_digest: str
    dependency_lock_digest: str
    wrapper_digest: str
    runtime_digest: str
    python_version: str
    python_executable: str
    python_executable_digest: str
    secret_requirements_digest: str
    provider_runtime_requirements_digest: str
    runtime_environment: Mapping[str, str]
    accelerator: str = "A10"
    timeout_seconds: int = 3600
    max_retries: int = 0
    sdk_version: str = "1.5.4"
    schema_version: str = "synaptic-modal-deployment-selection/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "synaptic-modal-deployment-selection/v1":
            raise ValueError("unsupported Modal deployment selection schema")
        if self.sdk_version != "1.5.4" or self.accelerator != "A10" or self.max_retries != 0:
            raise ValueError("Modal v1 requires SDK 1.5.4, A10, and retries disabled")
        if type(self.timeout_seconds) is not int or not 1 <= self.timeout_seconds <= 86400:
            raise ValueError("Modal timeout must be a bounded exact integer")
        for name in (
            "account_ref", "workspace_ref", "environment_ref", "client_ref", "app_name",
            "deployment_ref",
        ):
            object.__setattr__(self, name, safe_ref(getattr(self, name), name))
        deployment_ref, function_name = validate_modal_function_identity(
            self.deployment_ref, self.function_name
        )
        object.__setattr__(self, "deployment_ref", deployment_ref)
        object.__setattr__(self, "function_name", function_name)
        for name in (
            "image_digest", "dependency_lock_digest", "wrapper_digest", "runtime_digest",
            "python_executable_digest", "secret_requirements_digest",
            "provider_runtime_requirements_digest",
        ):
            object.__setattr__(self, name, digest(getattr(self, name), name))
        if not isinstance(self.python_version, str) or _VERSION_RE.fullmatch(self.python_version) is None:
            raise ValueError("python_version must be exact major.minor.micro")
        if not isinstance(self.python_executable, str) or not self.python_executable.startswith("/"):
            raise ValueError("Modal python_executable must be an exact absolute image path")
        object.__setattr__(self, "runtime_environment", _runtime_environment(self.runtime_environment))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "account_ref": self.account_ref,
            "workspace_ref": self.workspace_ref, "environment_ref": self.environment_ref,
            "client_ref": self.client_ref, "sdk_version": self.sdk_version,
            "app_name": self.app_name, "function_name": self.function_name,
            "deployment_ref": self.deployment_ref,
            "image_digest": self.image_digest,
            "dependency_lock_digest": self.dependency_lock_digest,
            "wrapper_digest": self.wrapper_digest, "runtime_digest": self.runtime_digest,
            "python_version": self.python_version, "python_executable": self.python_executable,
            "python_executable_digest": self.python_executable_digest,
            "runtime_environment": dict(self.runtime_environment),
            "secret_requirements_digest": self.secret_requirements_digest,
            "provider_runtime_requirements_digest": self.provider_runtime_requirements_digest,
            "accelerator": self.accelerator, "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ModalDeploymentSelectionV1":
        expected = {
            "schema_version", "account_ref", "workspace_ref", "environment_ref",
            "client_ref", "sdk_version", "app_name", "function_name",
            "deployment_ref", "image_digest", "dependency_lock_digest",
            "wrapper_digest", "runtime_digest", "python_version", "python_executable",
            "python_executable_digest", "runtime_environment",
            "secret_requirements_digest", "provider_runtime_requirements_digest",
            "accelerator", "timeout_seconds", "max_retries",
        }
        if type(value) is not dict or set(value) != expected:
            raise ValueError("Modal deployment selection contains missing or unknown fields")
        if type(value["runtime_environment"]) is not dict:
            raise ValueError("Modal deployment runtime environment must be an exact object")
        string_fields=expected-{"runtime_environment","timeout_seconds","max_retries"}
        if any(type(value[name]) is not str for name in string_fields):
            raise ValueError("Modal deployment selection text fields must be exact strings")
        if type(value["timeout_seconds"]) is not int or type(value["max_retries"]) is not int:
            raise ValueError("Modal deployment selection limits must be exact integers")
        return cls(**dict(value))

    @classmethod
    def from_profile(
        cls,
        profile: "ModalProviderProfileV1",
        *,
        binding: "ModalClientBinding",
        runtime_environment: Mapping[str, str],
        timeout_seconds: int = 3600,
    ) -> "ModalDeploymentSelectionV1":
        """Resolve the exact v1 selection without duplicating engine digests."""
        from .binding import ModalClientBinding
        from .config import ModalProviderProfileV1, ModalRuntimeLockV1

        if type(profile) is not ModalProviderProfileV1:
            raise TypeError("canonical Modal provider profile is required")
        if type(binding) is not ModalClientBinding:
            raise TypeError("canonical Modal client binding is required")
        runtime_lock = ModalRuntimeLockV1.packaged()
        environment = dict(runtime_environment)
        return cls(
            account_ref=binding.account_ref,
            workspace_ref=binding.workspace_ref,
            environment_ref=binding.environment_ref,
            client_ref=binding.client_ref,
            sdk_version=binding.sdk_version,
            app_name=profile.app_name,
            function_name=profile.function_name,
            deployment_ref=profile.deployment_ref,
            image_digest=runtime_lock.image_digest,
            dependency_lock_digest=runtime_lock.locked_digest("dependency_lock"),
            wrapper_digest=runtime_lock.locked_digest("deployment_wrapper"),
            runtime_digest=runtime_lock.locked_digest("sft_runtime"),
            python_version=runtime_lock.python_version,
            python_executable=runtime_lock.python_executable,
            python_executable_digest=runtime_lock.python_executable_digest,
            secret_requirements_digest=profile.secret_requirements_digest,
            provider_runtime_requirements_digest=(
                profile.provider_runtime_requirements_digest(
                    runtime_lock,
                    runtime_environment=environment,
                    timeout_seconds=timeout_seconds,
                )
            ),
            runtime_environment=environment,
            timeout_seconds=timeout_seconds,
        )


@dataclass(frozen=True, slots=True)
class _SealedDeploymentSelectionSnapshotV1:
    selection: ModalDeploymentSelectionV1
    canonical_bytes: bytes


def _capture_deployment_selection_snapshot(
    selection: ModalDeploymentSelectionV1,
) -> _SealedDeploymentSelectionSnapshotV1:
    if type(selection) is not ModalDeploymentSelectionV1:
        raise TypeError("exact ModalDeploymentSelectionV1 is required")
    try:
        reconstructed = ModalDeploymentSelectionV1.from_dict(selection.to_dict())
        canonical_bytes = _canonical(reconstructed.to_dict())
    except BaseException:
        raise SourceLockError("Modal deployment selection is malformed") from None
    return _SealedDeploymentSelectionSnapshotV1(reconstructed, canonical_bytes)


def _require_deployment_selection_snapshot(
    selection: ModalDeploymentSelectionV1,
    baseline: _SealedDeploymentSelectionSnapshotV1,
) -> None:
    if _capture_deployment_selection_snapshot(selection) != baseline:
        raise SourceLockError("Modal deployment selection changed during processing")

@dataclass(frozen=True, slots=True)
class VerifiedModalDeploymentIdentityV1:
    selection: ModalDeploymentSelectionV1
    issuer_ref: str
    evidence_ref: str
    audience_ref: str
    challenge_nonce: str
    verified_at: str
    expires_at: str
    key_ref: str
    tag_base64: str
    attestation_digest: str
    schema_version: str = "synaptic-verified-modal-deployment/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "synaptic-verified-modal-deployment/v1":
            raise ValueError("unsupported verified Modal deployment schema")
        if not isinstance(self.selection, ModalDeploymentSelectionV1):
            raise TypeError("selection must be ModalDeploymentSelectionV1")
        object.__setattr__(self, "issuer_ref", safe_ref(self.issuer_ref, "issuer_ref"))
        object.__setattr__(self, "evidence_ref", safe_ref(self.evidence_ref, "evidence_ref"))
        for name in ("audience_ref", "challenge_nonce", "key_ref"):
            object.__setattr__(self, name, safe_ref(getattr(self, name), name))
        from tuner.execution.evidence import canonical_utc
        object.__setattr__(self, "verified_at", canonical_utc(self.verified_at, "verified_at"))
        object.__setattr__(self, "expires_at", canonical_utc(self.expires_at, "expires_at"))
        if not isinstance(self.tag_base64, str) or not self.tag_base64.isascii():
            raise ValueError("tag_base64 must be canonical Base64")
        try: tag=base64.b64decode(self.tag_base64,validate=True)
        except (ValueError,binascii.Error) as exc:raise ValueError("tag_base64 must be canonical Base64") from exc
        if not tag or base64.b64encode(tag).decode("ascii")!=self.tag_base64:raise ValueError("tag_base64 must be canonical Base64")
        object.__setattr__(
            self, "attestation_digest", digest(self.attestation_digest, "attestation_digest")
        )
        if hashlib.sha256(self.authenticated_payload).hexdigest() != self.attestation_digest:
            raise ValueError("verified Modal deployment attestation digest mismatch")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "selection": self.selection.to_dict(),
            "issuer_ref": self.issuer_ref,
            "evidence_ref": self.evidence_ref,
            "audience_ref": self.audience_ref,
            "challenge_nonce": self.challenge_nonce,
            "verified_at": self.verified_at,
            "expires_at": self.expires_at,
            "key_ref": self.key_ref,
            "tag_base64": self.tag_base64,
            "attestation_digest": self.attestation_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "VerifiedModalDeploymentIdentityV1":
        expected = {
            "schema_version", "selection", "issuer_ref", "evidence_ref",
            "audience_ref", "challenge_nonce", "verified_at", "expires_at",
            "key_ref", "tag_base64", "attestation_digest",
        }
        if type(value) is not dict or set(value) != expected:
            raise ValueError("verified Modal deployment contains missing or unknown fields")
        raw = dict(value)
        if type(raw["selection"]) is not dict:
            raise ValueError("verified Modal deployment selection is malformed")
        if any(type(raw[name]) is not str for name in expected-{"selection"}):
            raise ValueError("verified Modal deployment fields must be exact strings")
        raw["selection"] = ModalDeploymentSelectionV1.from_dict(raw["selection"])
        return cls(**raw)

    @property
    def authenticated_payload(self) -> bytes:
        value=self.to_dict();value.pop("tag_base64");value.pop("attestation_digest")
        return _canonical(value)

    @property
    def tag(self) -> bytes:
        return base64.b64decode(self.tag_base64,validate=True)


@runtime_checkable
class ModalDeploymentVerificationPort(Protocol):
    def verify(self, selection: ModalDeploymentSelectionV1) -> VerifiedModalDeploymentIdentityV1: ...


@dataclass(frozen=True, slots=True)
class ModalExecutionSourceResolutionV1:
    execution_source: ExecutionSourceV1
    deployment: VerifiedModalDeploymentIdentityV1

    def __post_init__(self) -> None:
        if not isinstance(self.execution_source, ExecutionSourceV1):
            raise TypeError("execution_source must be ExecutionSourceV1")
        if not isinstance(self.deployment, VerifiedModalDeploymentIdentityV1):
            raise TypeError("deployment must be VerifiedModalDeploymentIdentityV1")
        if self.execution_source.deployment_member_sha256 != hashlib.sha256(
            _canonical(self.deployment.to_dict())
        ).hexdigest():
            raise ValueError("execution source does not bind the verified deployment")


def _has_reparse_component(path: Path) -> bool:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            info = current.lstat()
        except OSError as exc:
            raise SourceLockError("source root could not be inspected") from exc
        attributes = getattr(info, "st_file_attributes", 0)
        if current.is_symlink() or attributes & getattr(os, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
            return True
    return False


def _same_source_identity(actual: SourceLock, supplied: SourceLock) -> bool:
    return (
        type(actual) is SourceLock
        and type(supplied) is SourceLock
        and
        actual.mode == supplied.mode == "superproject"
        and actual.project_source.location.canonical_url
        == supplied.project_source.location.canonical_url
        and actual.project_source.commit.lower() == supplied.project_source.commit.lower()
        and actual.project_source.dirty == supplied.project_source.dirty == False
        and actual.engine_source.location.canonical_url
        == supplied.engine_source.location.canonical_url
        and actual.engine_source.commit.lower() == supplied.engine_source.commit.lower()
        and actual.engine_source.dirty == supplied.engine_source.dirty == False
        and actual.engine_source.submodule_path == supplied.engine_source.submodule_path
        and actual.engine_source.gitlink_commit == supplied.engine_source.gitlink_commit
    )


@dataclass(frozen=True, slots=True)
class _SealedDeploymentEvidenceSnapshotV1:
    evidence: VerifiedModalDeploymentIdentityV1
    document_bytes: bytes
    authenticated_payload: bytes
    tag: bytes
    attestation_digest: str
    replay_identity: tuple[str, str, str, str, str, str]


def _capture_deployment_evidence_snapshot(
    evidence: VerifiedModalDeploymentIdentityV1,
) -> _SealedDeploymentEvidenceSnapshotV1:
    if type(evidence) is not VerifiedModalDeploymentIdentityV1:
        raise TypeError("exact VerifiedModalDeploymentIdentityV1 is required")
    try:
        document = evidence.to_dict()
        if type(document) is not dict:
            raise TypeError
        reconstructed = VerifiedModalDeploymentIdentityV1.from_dict(document)
        document_bytes = _canonical(reconstructed.to_dict())
        payload = reconstructed.authenticated_payload
        tag = reconstructed.tag
    except BaseException:
        raise SourceLockError("authenticated deployment evidence is malformed") from None
    return _SealedDeploymentEvidenceSnapshotV1(
        reconstructed, document_bytes, payload, tag, reconstructed.attestation_digest,
        (
            reconstructed.issuer_ref, reconstructed.evidence_ref,
            reconstructed.challenge_nonce, reconstructed.audience_ref,
            reconstructed.attestation_digest, reconstructed.expires_at,
        ),
    )


def _require_deployment_evidence_snapshot(
    evidence: VerifiedModalDeploymentIdentityV1,
    baseline: _SealedDeploymentEvidenceSnapshotV1,
) -> None:
    if _capture_deployment_evidence_snapshot(evidence) != baseline:
        raise SourceLockError("authenticated deployment evidence changed during processing")


class ModalDualCloneSourceFinalizer:
    """Turn inspected and provider-verified provenance into the sole runtime source."""

    def __init__(
        self,
        local_sources: LocalSourceInspectionPort,
        pushed_sources: PushedSourceVerificationPort,
        deployments: ModalDeploymentVerificationPort,
        *, authenticator: EvidenceAuthenticator, replay: EvidenceReplayRepository,
        clock: Callable[[], str], source_issuer_ref: str,
        deployment_issuer_ref: str, source_key_ref: str, deployment_key_ref: str,
    ) -> None:
        if not isinstance(local_sources, LocalSourceInspectionPort):
            raise TypeError("local_sources must implement LocalSourceInspectionPort")
        if not isinstance(pushed_sources, PushedSourceVerificationPort):
            raise TypeError("pushed_sources must implement PushedSourceVerificationPort")
        if not isinstance(deployments, ModalDeploymentVerificationPort):
            raise TypeError("deployments must implement ModalDeploymentVerificationPort")
        self._local_sources = local_sources
        self._pushed_sources = pushed_sources
        self._deployments = deployments
        if not isinstance(authenticator, EvidenceAuthenticator):
            raise TypeError("authenticator must implement EvidenceAuthenticator")
        if not isinstance(replay, EvidenceReplayRepository):
            raise TypeError("replay must implement EvidenceReplayRepository")
        if not callable(clock):raise TypeError("clock must be callable")
        self._authenticator=authenticator;self._replay=replay;self._clock=clock
        self._source_issuer_ref=safe_ref(source_issuer_ref,"source_issuer_ref")
        self._deployment_issuer_ref=safe_ref(deployment_issuer_ref,"deployment_issuer_ref")
        self._source_key_ref=safe_ref(source_key_ref,"source_key_ref")
        self._deployment_key_ref=safe_ref(deployment_key_ref,"deployment_key_ref")

    def _admit_evidence(self,evidence_baseline,*,purpose,policy,issuer,key_ref,audience,checkpoint):
        evidence=evidence_baseline.evidence
        if evidence.issuer_ref!=issuer or evidence.key_ref!=key_ref or evidence.audience_ref!=audience:
            raise SourceLockError("authenticated evidence issuer, key, or audience mismatch")
        failed=False
        try:
            checkpoint()
            now=self._clock()
            checkpoint()
            validate_evidence_window(verified_at=evidence.verified_at,expires_at=evidence.expires_at,now=now,policy=policy)
            if hashlib.sha256(evidence_baseline.authenticated_payload).hexdigest()!=evidence_baseline.attestation_digest:
                raise ValueError("evidence payload digest mismatch")
            checkpoint()
            if not self._authenticator.verify(purpose,evidence_baseline.authenticated_payload,evidence_baseline.tag,evidence.key_ref):
                raise ValueError("evidence authentication failed")
            checkpoint()
            admit_evidence(self._replay,purpose=purpose,issuer_ref=evidence.issuer_ref,evidence_ref=evidence.evidence_ref,challenge_nonce=evidence.challenge_nonce,audience_ref=evidence.audience_ref,payload_digest=evidence.attestation_digest,expires_at=evidence.expires_at)
            checkpoint()
        except BaseException:
            failed=True
        if failed:
            raise SourceLockError("authenticated evidence admission failed") from None

    def finalize(
        self,
        source_lock: SourceLock,
        *,
        context: ProjectContext,
        deployment: ModalDeploymentSelectionV1,
        audience_ref: str,
    ) -> ModalExecutionSourceResolutionV1:
        if type(source_lock) is not SourceLock or not isinstance(context, ProjectContext):
            raise TypeError("source_lock and context must be canonical project values")
        if type(deployment) is not ModalDeploymentSelectionV1:
            raise TypeError("deployment must be exact ModalDeploymentSelectionV1")
        lock_baseline = _capture_source_lock_snapshot(source_lock)
        locked = lock_baseline.canonical_lock
        deployment_baseline = _capture_deployment_selection_snapshot(deployment)
        locked_deployment = deployment_baseline.selection

        def checkpoint(
            source_pair: tuple[AuthenticatedSourceEvidenceV1, _SealedSourceEvidenceSnapshotV1] | None = None,
            deployment_pair: tuple[VerifiedModalDeploymentIdentityV1, _SealedDeploymentEvidenceSnapshotV1] | None = None,
        ) -> None:
            _require_source_lock_snapshot(source_lock, lock_baseline)
            _require_source_lock_snapshot(locked, lock_baseline)
            _require_deployment_selection_snapshot(deployment, deployment_baseline)
            _require_deployment_selection_snapshot(locked_deployment, deployment_baseline)
            if source_pair is not None:
                _require_source_evidence_snapshot(*source_pair)
            if deployment_pair is not None:
                _require_deployment_evidence_snapshot(*deployment_pair)

        audience_ref=safe_ref(audience_ref,"audience_ref")
        if locked.mode != "superproject" or context.mode != "host":
            raise SourceLockError("Modal v1 accepts only host superproject provenance")
        project_root = context.project_root.absolute()
        engine_root = context.engine_root.absolute()
        if _has_reparse_component(project_root) or _has_reparse_component(engine_root):
            raise SourceLockError("source roots cannot traverse symlink or reparse points")
        resolved_project = project_root.resolve(strict=True)
        resolved_engine = engine_root.resolve(strict=True)
        try:
            actual_submodule = resolved_engine.relative_to(resolved_project).as_posix()
        except ValueError as exc:
            raise SourceLockError("superproject engine root must be inside the project root") from exc
        checkpoint()
        try:
            inspected = self._local_sources.inspect(context=context)
        except BaseException:
            raise SourceLockError("authenticated local source inspection is unavailable") from None
        checkpoint()
        if not _same_source_identity(inspected, locked):
            raise SourceLockError("supplied source lock does not match the current local checkout")
        if actual_submodule != locked.engine_source.submodule_path:
            raise SourceLockError("engine root does not match the inspected submodule path")
        if locked.project_source.dirty or locked.engine_source.dirty:
            raise SourceLockError("Modal source finalization requires clean sources")
        if locked.engine_source.commit.lower() != str(locked.engine_source.gitlink_commit).lower():
            raise SourceLockError("engine commit does not match the inspected project gitlink")
        checkpoint()
        try:
            pushed = self._pushed_sources.verify(locked)
        except BaseException:
            raise SourceLockError("authenticated pushed-source evidence is unavailable") from None
        checkpoint()
        pushed_baseline = _capture_source_evidence_snapshot(pushed)
        checkpoint((pushed, pushed_baseline))
        canonical_pushed = pushed_baseline.evidence
        if not canonical_pushed.binds(locked):
            raise SourceLockError("pushed-source evidence does not bind both repositories")
        if canonical_pushed.source_lock_binding != lock_baseline.binding:
            raise SourceLockError("pushed-source evidence does not bind the complete source lock")
        source_checkpoint=lambda:checkpoint((pushed,pushed_baseline))
        self._admit_evidence(pushed_baseline,purpose=SOURCE_EVIDENCE_PURPOSE,policy=SOURCE_EVIDENCE_POLICY,issuer=self._source_issuer_ref,key_ref=self._source_key_ref,audience=audience_ref,checkpoint=source_checkpoint)
        source_checkpoint()
        try:
            verified_deployment = self._deployments.verify(locked_deployment)
        except BaseException:
            raise SourceLockError("authenticated Modal deployment evidence is unavailable") from None
        source_checkpoint()
        verified_baseline = _capture_deployment_evidence_snapshot(verified_deployment)
        checkpoint((pushed,pushed_baseline),(verified_deployment,verified_baseline))
        canonical_verified = verified_baseline.evidence
        if canonical_verified.selection != locked_deployment:
            raise SourceLockError("verified deployment evidence does not bind the selection")
        if canonical_verified.challenge_nonce==canonical_pushed.challenge_nonce:
            raise SourceLockError("source and deployment evidence require distinct challenges")
        all_checkpoint=lambda:checkpoint((pushed,pushed_baseline),(verified_deployment,verified_baseline))
        self._admit_evidence(verified_baseline,purpose="modal-deployment-evidence/v1",policy=DEPLOYMENT_EVIDENCE_POLICY,issuer=self._deployment_issuer_ref,key_ref=self._deployment_key_ref,audience=audience_ref,checkpoint=all_checkpoint)
        all_checkpoint()

        run_root = f"/workspace/run/{locked.run_id}"
        roots = {
            "engine": "/workspace/engine", "project": "/workspace/project",
            "artifacts": f"{run_root}/artifacts", "state": f"{run_root}/state",
            "tracking": f"{run_root}/tracking", "cache": f"{run_root}/cache",
            "tmp": f"{run_root}/tmp",
        }
        fixed_environment = {
            "PYTHONNOUSERSITE": "1", "PYTHONSAFEPATH": "1",
            "PYTHONPATH": roots["engine"], "SYNAPTIC_ENGINE_ROOT": roots["engine"],
            "SYNAPTIC_PROJECT_ROOT": roots["project"],
            "SYNAPTIC_ARTIFACT_ROOT": roots["artifacts"],
            "SYNAPTIC_STATE_ROOT": roots["state"],
            "SYNAPTIC_TRACKING_ROOT": roots["tracking"],
            "SYNAPTIC_CACHE_ROOT": roots["cache"], "SYNAPTIC_TMP_ROOT": roots["tmp"],
            "HF_HOME": roots["cache"] + "/huggingface",
            "TRANSFORMERS_CACHE": roots["cache"] + "/transformers",
            "WANDB_DISABLED": "true",
        }
        overlap = set(fixed_environment).intersection(locked_deployment.runtime_environment)
        if any(locked_deployment.runtime_environment[key] != fixed_environment[key] for key in overlap):
            raise SourceLockError("deployment runtime environment conflicts with fixed isolation")
        environment = {**locked_deployment.runtime_environment, **fixed_environment}
        all_checkpoint()
        execution_source = ExecutionSourceV1(
            run_id=locked.run_id, created_at=locked.created_at,
            project_source=locked.project_source, engine_source=locked.engine_source,
            engine_submodule_path=actual_submodule, source_evidence=canonical_pushed,
            deployment_member_sha256=hashlib.sha256(
                verified_baseline.document_bytes
            ).hexdigest(),
            roots=roots, writable_capability_root="/workspace/run",
            python_implementation="cpython",
            python_version=locked_deployment.python_version,
            python_executable=locked_deployment.python_executable,
            python_executable_digest=locked_deployment.python_executable_digest,
            environment=environment,
            secret_requirements_digest=locked_deployment.secret_requirements_digest,
            provider_runtime_requirements_digest=locked_deployment.provider_runtime_requirements_digest,
        )
        all_checkpoint()
        result=ModalExecutionSourceResolutionV1(execution_source, canonical_verified)
        all_checkpoint()
        return result


__all__ = [
    "ModalDeploymentSelectionV1", "ModalDeploymentVerificationPort",
    "ModalDualCloneSourceFinalizer", "ModalExecutionSourceResolutionV1",
    "VerifiedModalDeploymentIdentityV1",
]
