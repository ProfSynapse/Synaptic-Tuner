"""Consumer-owned streamed artifact verification for the Modal chat example."""

from __future__ import annotations

import hashlib
import hmac

from tuner.execution.coordinator_v1.model import (
    ArtifactManifestV1,
    ArtifactVerificationContentV1,
    AuthenticatedArtifactVerificationReceiptV1,
    AuthenticatedFoundationRecordAssessmentV1,
    ProviderReadPurposeV1,
    VerificationVerdictV1,
    WorkflowPhaseV1,
    WorkflowRecordV1,
)
from tuner.execution.coordinator_v1.state_machine import provider_run_read_request
from tuner.execution.foundation_v2.canonical import canonical_bytes, safe_ref
from tuner.execution.providers.modal.coordinator_reader import ModalCoordinatorRunReader


class ModalChatArtifactError(RuntimeError):
    """Closed failure from consumer-owned artifact verification."""


class ModalChatArtifactVerifier:
    """Verify every retained artifact through the authenticated native reader."""

    __slots__ = (
        "_reader",
        "_foundation",
        "_foundation_authenticator",
        "_assessment_authenticator",
        "_authority_ref",
        "_key_ref",
        "_key",
        "_clock",
        "_maximum_total_bytes",
    )

    def __init__(
        self,
        *,
        foundation_authenticator: object,
        assessment_authenticator: object,
        authority_ref: str,
        key_ref: str,
        key: bytes,
        clock: object,
        maximum_total_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        for value, names in (
            (
                foundation_authenticator,
                (
                    "authenticate_grant",
                    "authenticate_receipt",
                    "authenticate_invalid_evidence",
                ),
            ),
            (assessment_authenticator, ("authenticate",)),
        ):
            if any(not callable(getattr(value, name, None)) for name in names):
                raise TypeError("artifact verification dependency is incomplete")
        if type(key) is not bytes or len(key) < 32:
            raise ValueError("artifact verification key is invalid")
        if not callable(getattr(clock, "now_iso", None)):
            raise TypeError("artifact verification clock is required")
        if (
            type(maximum_total_bytes) is not int
            or not 1 <= maximum_total_bytes <= 2**63 - 1
        ):
            raise ValueError("artifact verification bound is invalid")
        self._reader = None
        self._foundation = None
        self._foundation_authenticator = foundation_authenticator
        self._assessment_authenticator = assessment_authenticator
        self._authority_ref = safe_ref(authority_ref, "authority_ref")
        self._key_ref = safe_ref(key_ref, "key_ref")
        self._key = bytes(key)
        self._clock = clock
        self._maximum_total_bytes = maximum_total_bytes

    def bind(self, *, reader: ModalCoordinatorRunReader, foundation: object) -> None:
        """Bind the reader graph exactly once after provider composition."""
        if self._reader is not None or self._foundation is not None:
            raise ModalChatArtifactError("modal_chat_artifact_already_bound")
        if type(reader) is not ModalCoordinatorRunReader or any(
            not callable(getattr(foundation, name, None)) for name in ("get", "assess")
        ):
            raise TypeError("exact artifact reader and Foundation service required")
        self._reader, self._foundation = reader, foundation

    def _request(self, workflow: WorkflowRecordV1):
        if (
            self._reader is None
            or self._foundation is None
            or type(workflow) is not WorkflowRecordV1
            or workflow.provider.provider_id != "modal"
            or workflow.submit is None
            or workflow.provider_run_ref is None
        ):
            raise ValueError("workflow is not a submitted Modal run")
        record = self._foundation.get(workflow.submit.effect_id)
        if record is None:
            raise ValueError("retained Foundation record is unavailable")
        assessment = AuthenticatedFoundationRecordAssessmentV1.parse(
            workflow.submit.foundation_bindings[-1].canonical_assessment_bytes
        )
        return provider_run_read_request(
            workflow,
            record,
            assessment,
            self._foundation_authenticator,
            self._assessment_authenticator,
            purpose=ProviderReadPurposeV1.ARTIFACTS,
        )

    def _tag(self, content: ArtifactVerificationContentV1) -> str:
        message = canonical_bytes(
            {
                "schema_version": "synaptic-modal-chat-artifact-mac/v1",
                "authority_ref": self._authority_ref,
                "key_ref": self._key_ref,
                "content_digest": content.content_digest,
            }
        )
        return hmac.new(self._key, message, hashlib.sha256).hexdigest()

    def _issue(
        self, workflow: WorkflowRecordV1, manifest: ArtifactManifestV1
    ) -> AuthenticatedArtifactVerificationReceiptV1:
        if (
            type(manifest) is not ArtifactManifestV1
            or manifest.run != workflow.run
            or workflow.provider_run_ref is None
            or manifest.provider_run != workflow.provider_run_ref.reference
            or manifest.manifest_digest != manifest.expected_manifest_digest
        ):
            raise ValueError("artifact manifest identity is invalid")
        request = self._request(workflow)
        current = self._reader.artifacts(request)
        if current != manifest:
            raise ValueError("native artifact manifest changed")
        total = 0
        observed = []
        for artifact in manifest.artifacts:
            total += artifact.size_bytes
            if total > self._maximum_total_bytes:
                raise ValueError("artifact inventory exceeds consumer bound")
            digest = hashlib.sha256()
            size = 0
            for chunk in self._reader.iter_artifact_bytes(
                request,
                manifest,
                artifact.role,
                maximum_bytes=max(1, artifact.size_bytes),
            ):
                if type(chunk) is not bytes or not chunk:
                    raise ValueError("artifact stream is invalid")
                size += len(chunk)
                digest.update(chunk)
                if size > artifact.size_bytes:
                    raise ValueError("artifact stream exceeds manifest")
            if size != artifact.size_bytes or digest.hexdigest() != artifact.sha256:
                raise ValueError("artifact stream differs from manifest")
            observed.append(
                {
                    "role": artifact.role,
                    "size_bytes": size,
                    "sha256": digest.hexdigest(),
                }
            )
        evidence = canonical_bytes(
            {
                "schema_version": "synaptic-modal-chat-artifact-verification/v1",
                "read_request_digest": request.request_digest,
                "artifacts": observed,
            }
        )
        content = ArtifactVerificationContentV1(
            "synaptic-artifact-verification-content/v1",
            workflow.record_digest,
            workflow.revision,
            workflow.run,
            workflow.provider_run_ref.binding_digest,
            manifest.manifest_digest,
            manifest.artifact_source_digest,
            manifest.artifacts,
            manifest.artifacts,
            VerificationVerdictV1.VERIFIED,
            None,
            "modal-chat-artifact-verifier",
            "1.0.0",
            evidence,
            self._clock.now_iso(),
        )
        return AuthenticatedArtifactVerificationReceiptV1(
            content,
            self._authority_ref,
            self._key_ref,
            self._tag(content),
        )

    def verify(self, workflow, manifest):
        try:
            if type(workflow) is not WorkflowRecordV1 or workflow.phase not in {
                WorkflowPhaseV1.SUCCEEDED_UNVERIFIED,
                WorkflowPhaseV1.VERIFICATION_FAILED,
            }:
                raise ValueError("workflow is not eligible for verification")
            return self._issue(workflow, manifest)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            raise ModalChatArtifactError("modal_chat_artifact_invalid") from None

    def replay(self, workflow, manifest, prior_receipt):
        try:
            if (
                type(workflow) is not WorkflowRecordV1
                or workflow.phase is not WorkflowPhaseV1.VERIFIED
                or type(prior_receipt) is not AuthenticatedArtifactVerificationReceiptV1
                or not workflow.verification_receipts
                or workflow.verification_receipts[-1] != prior_receipt
                or workflow.artifact_manifest != manifest
                or self.authenticate(prior_receipt) is not True
            ):
                raise ValueError("reverification predecessor is invalid")
            return self._issue(workflow, manifest)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            raise ModalChatArtifactError("modal_chat_artifact_invalid") from None

    def authenticate(self, receipt) -> bool:
        try:
            if type(receipt) is not AuthenticatedArtifactVerificationReceiptV1:
                return False
            parsed = AuthenticatedArtifactVerificationReceiptV1.parse(
                receipt.canonical_bytes
            )
            return bool(
                parsed == receipt
                and parsed.authority_ref == self._authority_ref
                and parsed.key_ref == self._key_ref
                and hmac.compare_digest(parsed.tag, self._tag(parsed.content))
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            return False


__all__ = ["ModalChatArtifactError", "ModalChatArtifactVerifier"]
