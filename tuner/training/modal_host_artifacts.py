"""Host-owned verification of the exact five packaged Modal artifact streams."""

from __future__ import annotations

import hashlib
import hmac

from tuner.execution.coordinator_v1.model import (
    ArtifactManifestV1, ArtifactVerificationContentV1,
    AuthenticatedArtifactVerificationReceiptV1,
    AuthenticatedFoundationRecordAssessmentV1, ProviderReadPurposeV1,
    VerificationVerdictV1, WorkflowPhaseV1, WorkflowRecordV1,
)
from tuner.execution.coordinator_v1.state_machine import provider_run_read_request
from tuner.execution.foundation_v2.canonical import canonical_bytes, safe_ref
from tuner.execution.providers.modal.packaged_binding import EXACT_ARTIFACT_ROLES
from tuner.execution.providers.modal.coordinator_producer import MODAL_TRAINING_ARTIFACT_BOUNDS_V1


class ModalPackagedArtifactVerifierV1:
    def __init__(self, *, reader: object, foundation_authenticator: object,
                 assessment_authenticator: object, authority_ref: str,
                 key_ref: str, key: bytes, clock: object) -> None:
        if type(key) is not bytes or len(key) < 32:
            raise ValueError("artifact verification key is invalid")
        self._reader, self._foundation = reader, None
        self._foundation_authenticator = foundation_authenticator
        self._assessments = assessment_authenticator
        self._authority_ref, self._key_ref = (
            safe_ref(authority_ref, "authority_ref"), safe_ref(key_ref, "key_ref"),
        )
        self._key, self._clock = bytes(key), clock

    def bind_foundation(self, repository: object) -> None:
        if self._foundation is not None or not callable(getattr(repository, "get", None)):
            raise ValueError("artifact verifier Foundation binding is unavailable")
        self._foundation = repository

    def _request(self, workflow: WorkflowRecordV1):
        if (self._foundation is None or type(workflow) is not WorkflowRecordV1
                or workflow.provider.provider_id != "modal"
                or workflow.submit is None or workflow.provider_run_ref is None):
            raise ValueError("packaged artifact workflow is unavailable")
        record = self._foundation.get(workflow.submit.effect_id)
        if record is None:
            raise ValueError("packaged artifact Foundation record is unavailable")
        assessment = AuthenticatedFoundationRecordAssessmentV1.parse(
            workflow.submit.foundation_bindings[-1].canonical_assessment_bytes,
        )
        return provider_run_read_request(
            workflow, record, assessment,
            self._foundation_authenticator, self._assessments,
            purpose=ProviderReadPurposeV1.ARTIFACTS,
        )

    def _tag(self, content: ArtifactVerificationContentV1) -> str:
        payload = canonical_bytes({
            "schema_version": "synaptic-modal-packaged-artifact-mac/v1",
            "authority_ref": self._authority_ref, "key_ref": self._key_ref,
            "content_digest": content.content_digest,
        })
        return hmac.new(self._key, payload, hashlib.sha256).hexdigest()

    def _issue(self, workflow: WorkflowRecordV1,
               manifest: ArtifactManifestV1) -> AuthenticatedArtifactVerificationReceiptV1:
        if (type(manifest) is not ArtifactManifestV1 or manifest.run != workflow.run
                or workflow.provider_run_ref is None
                or manifest.provider_run != workflow.provider_run_ref.reference
                or {item.role for item in manifest.artifacts} != EXACT_ARTIFACT_ROLES):
            raise ValueError("packaged artifact manifest is invalid")
        request = self._request(workflow)
        if self._reader.artifacts(request) != manifest:
            raise ValueError("packaged artifact inventory changed")
        total, observed = 0, []
        for artifact in manifest.artifacts:
            total += artifact.size_bytes
            if total > MODAL_TRAINING_ARTIFACT_BOUNDS_V1.max_artifact_total_bytes:
                raise ValueError("packaged artifact inventory exceeds bound")
            digest, size = hashlib.sha256(), 0
            for chunk in self._reader.iter_artifact_bytes(
                request, manifest, artifact.role,
                maximum_bytes=max(1, artifact.size_bytes),
            ):
                if type(chunk) is not bytes or not chunk:
                    raise ValueError("packaged artifact stream is invalid")
                size += len(chunk)
                digest.update(chunk)
                if size > artifact.size_bytes:
                    raise ValueError("packaged artifact stream exceeds inventory")
            if (size, digest.hexdigest()) != (artifact.size_bytes, artifact.sha256):
                raise ValueError("packaged artifact stream differs from inventory")
            observed.append({
                "role": artifact.role, "size_bytes": size,
                "sha256": digest.hexdigest(),
            })
        evidence = canonical_bytes({
            "schema_version": "synaptic-modal-packaged-artifact-verification/v1",
            "read_request_digest": request.request_digest,
            "artifacts": observed,
        })
        content = ArtifactVerificationContentV1(
            "synaptic-artifact-verification-content/v1",
            workflow.record_digest, workflow.revision, workflow.run,
            workflow.provider_run_ref.binding_digest, manifest.manifest_digest,
            manifest.artifact_source_digest, manifest.artifacts, manifest.artifacts,
            VerificationVerdictV1.VERIFIED, None,
            "modal-packaged-artifact-verifier", "0.1.0", evidence,
            self._clock.now(),
        )
        return AuthenticatedArtifactVerificationReceiptV1(
            content, self._authority_ref, self._key_ref, self._tag(content),
        )

    def verify(self, workflow, manifest):
        try:
            if type(workflow) is not WorkflowRecordV1 or workflow.phase not in {
                WorkflowPhaseV1.SUCCEEDED_UNVERIFIED, WorkflowPhaseV1.VERIFICATION_FAILED,
            }:
                raise ValueError("packaged workflow cannot be verified")
            return self._issue(workflow, manifest)
        except Exception:
            raise ValueError("modal_packaged_artifact_verification_failed") from None

    def replay(self, workflow, manifest, prior_receipt):
        try:
            if (type(workflow) is not WorkflowRecordV1
                    or workflow.phase is not WorkflowPhaseV1.VERIFIED
                    or not workflow.verification_receipts
                    or workflow.verification_receipts[-1] != prior_receipt
                    or workflow.artifact_manifest != manifest
                    or self.authenticate(prior_receipt) is not True):
                raise ValueError("packaged reverification predecessor is invalid")
            return self._issue(workflow, manifest)
        except Exception:
            raise ValueError("modal_packaged_artifact_verification_failed") from None

    def authenticate(self, receipt) -> bool:
        try:
            if type(receipt) is not AuthenticatedArtifactVerificationReceiptV1:
                return False
            parsed = AuthenticatedArtifactVerificationReceiptV1.parse(receipt.canonical_bytes)
            return (parsed == receipt and parsed.authority_ref == self._authority_ref
                    and parsed.key_ref == self._key_ref
                    and hmac.compare_digest(parsed.tag, self._tag(parsed.content)))
        except Exception:
            return False
