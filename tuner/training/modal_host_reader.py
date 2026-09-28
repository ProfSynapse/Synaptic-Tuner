"""Exact packaged Modal completion projected into the generic run reader."""

from __future__ import annotations

import time
from typing import Iterator

from synaptic_tuner.api.v1.results import VerifiedArtifact
from tuner.execution.coordinator_v1.model import (
    ArtifactManifestV1, AuthenticatedProviderRunObservationV1,
    EffectIntentV1, FoundationDispositionV1, ProviderReadPurposeV1,
    ProviderRunObservationContentV1, ProviderRunPhaseV1, ProviderRunReadRequestV1,
)
from tuner.execution.coordinator_v1.state_machine import _derive_foundation
from tuner.execution.foundation_v2.canonical import canonical_bytes, domain_digest
from tuner.execution.foundation_v2.commands import SubmitCommandV2, parse_exact_command
from tuner.execution.providers.modal.facade import ModalFunctionCallState
from tuner.execution.providers.modal.packaged_binding import ModalPackagedCommandBinding
from tuner.execution.providers.modal.packaged_reader import ModalPackagedCompletionObservation
from tuner.execution.providers.modal.runtime_build import _bounded


class ModalPackagedReadUnavailable(RuntimeError):
    """No authenticated terminal evidence yet; this is not a failed run."""


_ARTIFACT_READ_STEP_SECONDS = 60
_ARTIFACT_READ_TOTAL_SECONDS = 300
_FIXED_WORKER_FAILURE = {
    "schema_version": "synaptic-modal-packaged-worker-result/v1",
    "effect_id": "unavailable",
    "status_code": "failed",
    "completion_sha256": "0" * 64,
}
_FIXED_WORKER_FAILURE_STAGES = frozenset({
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
    "SFT_PREPARATION", "SFT_REVALIDATION",
    "SFT_INVOCATION", "SFT_TRAINER", "SFT_EVIDENCE", "SFT_ARTIFACT",
    "SFT_UNKNOWN", "COMPLETION", "ARTIFACT_COMMIT", "CONTROL_COMMIT",
})


class ModalPackagedCoordinatorReaderV1:
    """Authenticate current Foundation evidence before any Modal read.

    The first packaged path has no qualified log or scheduler-state reader.  A
    pending call is reported as unavailable, not guessed queued/running.
    """

    def __init__(self, *, bindings: object, binding_authority: object,
                 foundation_repository: object | None, foundation_authenticator: object,
                 assessment_authenticator: object, packaged_reader: object,
                 facade: object, evidence_authority: object, clock: object) -> None:
        self._bindings, self._binding_authority = bindings, binding_authority
        self._foundation, self._foundation_authenticator = (
            foundation_repository, foundation_authenticator,
        )
        self._assessments = assessment_authenticator
        self._reader, self._facade = packaged_reader, facade
        self._authority, self._clock = evidence_authority, clock

    def bind_foundation(self, repository: object) -> None:
        if self._foundation is not None or not callable(getattr(repository, "get", None)):
            raise ValueError("packaged reader Foundation binding is unavailable")
        self._foundation = repository

    def _binding(self, request: ProviderRunReadRequestV1,
                 purpose: ProviderReadPurposeV1) -> ModalPackagedCommandBinding:
        try:
            if type(request) is not ProviderRunReadRequestV1 or request.purpose is not purpose:
                raise ValueError
            command = parse_exact_command(request.submit_command_bytes)
            if type(command) is not SubmitCommandV2:
                raise ValueError
            if self._foundation is None:
                raise ValueError
            current = self._foundation.get(command.operation.effect.effect_id)
            if current != request.foundation_record:
                raise ValueError
            intent = EffectIntentV1.from_command_bytes(request.submit_command_bytes)
            derived = _derive_foundation(
                intent, current, request.assessment,
                self._foundation_authenticator, self._assessments, None,
            )
            if derived != (request.foundation_binding, request.foundation_outcome, request.provider_run):
                raise ValueError
            if (derived[1].disposition is not FoundationDispositionV1.FOUND
                    or derived[2] is None
                    or request.found_receipt_digest != derived[2].authenticated_receipt_digest
                    or request.provider_run.reference.provider_job_ref is None):
                raise ValueError
            binding = self._bindings.resolve(command.digest)
            if (type(binding) is not ModalPackagedCommandBinding
                    or binding.command_bytes != request.submit_command_bytes
                    or binding.command_digest != request.provider_run.command_digest
                    or self._binding_authority.authenticate(binding) is not True):
                raise ValueError
            return binding
        except Exception:
            raise ModalPackagedReadUnavailable("modal_packaged_read_authentication_failed") from None

    def _completion(self, binding: ModalPackagedCommandBinding,
                    provider_job_ref: str) -> ModalPackagedCompletionObservation:
        state, claimed_completion = self._poll_packaged_call(
            binding, provider_job_ref,
        )
        if state is ModalFunctionCallState.PENDING:
            raise ModalPackagedReadUnavailable("modal_packaged_call_pending")
        if state is not ModalFunctionCallState.RETURNED:
            raise ModalPackagedReadUnavailable("modal_packaged_call_unknown")
        try:
            result = _bounded(
                lambda: self._reader.observe_completion(
                    binding, provider_job_ref=provider_job_ref,
                ),
                deadline=time.monotonic() + 60,
                code="modal_packaged_completion_read_unavailable",
            )
            if (type(result) is not ModalPackagedCompletionObservation
                    or result.completion_digest != claimed_completion):
                raise ValueError
            return result
        except Exception:
            raise ModalPackagedReadUnavailable("modal_packaged_completion_invalid") from None

    def _poll_packaged_call(self, binding: ModalPackagedCommandBinding,
                            provider_job_ref: str, *,
                            deadline: float | None = None) -> tuple[ModalFunctionCallState, str | None]:
        try:
            def read_call():
                call = self._facade.sdk.FunctionCall.from_id(
                    provider_job_ref, client=self._facade.client,
                )
                hydrated = call.hydrate(self._facade.client)
                if (hydrated is not call or getattr(call, "is_hydrated", False) is not True
                        or getattr(call, "object_id", None) != provider_job_ref):
                    raise ValueError
                try:
                    return call.get(timeout=0)
                except TimeoutError:
                    return None

            result = _bounded(
                read_call, deadline=min(time.monotonic() + 30,
                                        deadline if deadline is not None else float("inf")),
                code="modal_packaged_call_read_unavailable",
            )
            if result is None:
                return ModalFunctionCallState.PENDING, None
            if type(result) is dict and result == _FIXED_WORKER_FAILURE:
                raise ModalPackagedReadUnavailable("modal_packaged_call_failed")
            if type(result) is dict and set(result) == {
                    "schema_version", "effect_id", "status_code",
                    "completion_sha256", "failure_stage",
            } and type(result.get("failure_stage")) is str:
                stage = result["failure_stage"]
                if (stage in _FIXED_WORKER_FAILURE_STAGES and result == {
                        **_FIXED_WORKER_FAILURE,
                        "schema_version": "synaptic-modal-packaged-worker-result/v2",
                        "failure_stage": stage,
                }):
                    raise ModalPackagedReadUnavailable(
                        f"modal_packaged_call_failed_{stage}",
                    )
            if (type(result) is not dict or set(result) != {
                    "schema_version", "effect_id", "status_code", "completion_sha256",
            } or result["schema_version"] != "synaptic-modal-packaged-worker-result/v1"
                    or result["effect_id"] != binding.command.operation.effect.effect_id
                    or type(result["completion_sha256"]) is not str
                    or len(result["completion_sha256"]) != 64
                    or any(c not in "0123456789abcdef" for c in result["completion_sha256"])):
                raise ValueError
            if result["status_code"] != "completed":
                raise ValueError
            return ModalFunctionCallState.RETURNED, result["completion_sha256"]
        except ModalPackagedReadUnavailable:
            raise
        except Exception:
            raise ModalPackagedReadUnavailable("modal_packaged_call_unknown") from None

    def observe(self, request: ProviderRunReadRequestV1) -> AuthenticatedProviderRunObservationV1:
        binding = self._binding(request, ProviderReadPurposeV1.OBSERVE)
        ref = request.provider_run.reference
        completion = self._completion(binding, ref.provider_job_ref)
        content = ProviderRunObservationContentV1(
            "synaptic-provider-run-observation-content/v1", request.request_digest,
            request.source_workflow_record_digest, request.source_revision,
            request.run, request.provider_run.binding_digest,
            ref.provider_id, ref.profile_ref, ref.account_ref, ref.namespace_ref,
            ref.provider_job_ref, ProviderRunPhaseV1.SUCCEEDED,
            canonical_bytes({
                "schema_version": "synaptic-modal-packaged-completion-observation/v1",
                "command_digest": completion.command_digest,
                "completion_digest": completion.completion_digest,
                "execution_binding_digest": completion.execution_binding_digest,
            }),
            None, "modal-packaged-host-reader", "0.1.0", self._clock.now(),
        )
        return self._authority.observation(content)

    def logs(self, request, query):
        raise ModalPackagedReadUnavailable("modal_packaged_logs_unsupported")

    def _manifest(self, request: ProviderRunReadRequestV1):
        binding = self._binding(request, ProviderReadPurposeV1.ARTIFACTS)
        ref = request.provider_run.reference
        completion = self._completion(binding, ref.provider_job_ref)
        artifacts = tuple(sorted(
            (VerifiedArtifact(item.role, item.sha256, item.size)
             for item in completion.members), key=lambda item: item.role,
        ))
        evidence = canonical_bytes({
            "schema_version": "synaptic-modal-packaged-artifact-evidence/v1",
            "completion_digest": completion.completion_digest,
            "members": [
                {"role": item.role, "path": item.path, "size": item.size,
                 "sha256": item.sha256, "provider_entry_id": item.provider_entry_id}
                for item in sorted(completion.members, key=lambda item: item.role)
            ],
        })
        manifest = ArtifactManifestV1.build(
            run=request.run, provider_run=ref, artifacts=artifacts,
            artifact_source_digest=domain_digest(
                "synaptic-modal-packaged-artifact-source/v1", evidence,
            ), canonical_evidence=evidence,
        )
        return binding, completion, manifest

    def artifacts(self, request: ProviderRunReadRequestV1) -> ArtifactManifestV1:
        return self._manifest(request)[2]

    def iter_artifact_bytes(self, request: ProviderRunReadRequestV1,
                            manifest: ArtifactManifestV1, role: str, *,
                            maximum_bytes: int) -> Iterator[bytes]:
        binding, completion, expected = self._manifest(request)
        if type(manifest) is not ArtifactManifestV1 or manifest != expected:
            raise ModalPackagedReadUnavailable("modal_packaged_manifest_mismatch")
        def checked() -> Iterator[bytes]:
            try:
                stream = self._reader.iter_artifact(
                    binding, completion, role=role, maximum_bytes=maximum_bytes,
                )
                deadline = time.monotonic() + _ARTIFACT_READ_TOTAL_SECONDS
                while True:
                    remaining = min(deadline, time.monotonic() + _ARTIFACT_READ_STEP_SECONDS)
                    finished, chunk = _bounded(
                        lambda: _next_chunk(stream), deadline=remaining,
                        code="modal_packaged_artifact_stream_unavailable",
                    )
                    if finished:
                        break
                    yield chunk
            except Exception:
                raise ModalPackagedReadUnavailable(
                    "modal_packaged_artifact_stream_unavailable",
                ) from None
        return checked()


def _next_chunk(stream: Iterator[bytes]) -> tuple[bool, bytes | None]:
    try:
        return False, next(stream)
    except StopIteration:
        return True, None
