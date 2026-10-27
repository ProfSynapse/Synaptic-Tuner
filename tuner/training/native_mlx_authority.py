"""Native reader evidence over the existing host HMAC implementation."""

from dataclasses import replace
import hashlib
import hmac

from tuner.execution.coordinator_v1.model import (
    ArtifactVerificationContentV1, AuthenticatedArtifactVerificationReceiptV1,
)
from tuner.training.modal_host_authority import ReaderEvidenceAuthority
from tuner.runtime.native_mlx_material import canonical, strict_json
from tuner.execution.foundation_v2.canonical import digest_text
from synaptic_tuner.api.v1._timestamps import require_rfc3339


class NativeEvidenceAuthority(ReaderEvidenceAuthority):
    """Host constructs keys; no key generation, secret resolution, or storage."""

    def lifecycle_key(self, invocation_digest, policy_digest):
        """Derive one invocation/policy-scoped capability for trusted supervision.

        Host HMAC policy must explicitly allow this new purpose. Deliver only
        through an anonymous channel to the trusted guardian, never the ML child
        or any file, argument, environment value or diagnostic. This is no start
        admission, hardware proof, or default public lifecycle qualification.
        """
        digest_text(invocation_digest, "invocation_digest")
        digest_text(policy_digest, "policy_digest")
        scope = canonical({"schema": "synaptic-native-lifecycle-capability/v1",
                           "invocation_digest": invocation_digest, "policy_digest": policy_digest})
        return bytes.fromhex(self._tag("native-lifecycle-capability/v1", hashlib.sha256(scope).hexdigest()))

    def _lifecycle_envelope(self, purpose, content):
        return canonical({"content": content, "authority_ref": self.authority_ref, "key_ref": self.key_ref,
                          "tag": self._tag(purpose, hashlib.sha256(canonical(content)).hexdigest())})

    def _lifecycle_content(self, purpose, raw):
        value = strict_json(raw, maximum=8192)
        if (type(value) is not dict or set(value) != {"content", "authority_ref", "key_ref", "tag"}
                or canonical(value) != raw or value["authority_ref"] != self.authority_ref
                or value["key_ref"] != self.key_ref or type(value["content"]) is not dict
                or type(value["tag"]) is not str or not hmac.compare_digest(value["tag"], self._tag(
                    purpose, hashlib.sha256(canonical(value["content"])).hexdigest()))):
            raise ValueError("native lifecycle evidence authentication differs")
        return value["content"]

    def cancellation_ack(self, command_digest, invocation_digest):
        digest_text(command_digest, "command_digest");digest_text(invocation_digest, "invocation_digest")
        return self._lifecycle_envelope("native-cancellation-ack/v1", {
            "schema": "synaptic-native-cancellation-ack/v1", "command_digest": command_digest,
            "invocation_digest": invocation_digest})

    def stage_ack(self, command_digest):
        digest_text(command_digest, "command_digest")
        return self._lifecycle_envelope("native-stage-ack/v1", {
            "schema": "synaptic-native-stage-ack/v1", "command_digest": command_digest})

    def authenticate_stage_ack(self, raw, command_digest):
        return canonical(self._lifecycle_content("native-stage-ack/v1", raw)) == canonical({
            "schema": "synaptic-native-stage-ack/v1", "command_digest": command_digest})

    def authenticate_cancellation_ack(self, raw, command_digest, invocation_digest):
        value = self._lifecycle_content("native-cancellation-ack/v1", raw)
        return canonical(value) == canonical({"schema": "synaptic-native-cancellation-ack/v1",
            "command_digest": command_digest, "invocation_digest": invocation_digest})

    def log_metadata(self, invocation_digest, sequence, event_digest, observed_at):
        digest_text(invocation_digest, "invocation_digest");digest_text(event_digest, "event_digest")
        require_rfc3339(observed_at, "observed_at")
        if type(sequence) is not int or not 0 <= sequence < 2**63:
            raise ValueError("native progress sequence differs")
        return self._lifecycle_envelope("native-progress-metadata/v1", {
            "schema": "synaptic-native-progress-metadata/v1", "invocation_digest": invocation_digest,
            "sequence": sequence, "event_digest": event_digest, "observed_at": observed_at})

    def authenticate_log_metadata(self, raw, invocation_digest, sequence, event_digest):
        value = self._lifecycle_content("native-progress-metadata/v1", raw)
        observed = value.get("observed_at")
        if self.log_metadata(invocation_digest, sequence, event_digest, observed) != raw:
            raise ValueError("native progress metadata scope differs")
        return observed

    def recovery_evidence(self, content):
        return self._lifecycle_envelope("native-quiescence/v1", content)

    def authenticate_recovery_evidence(self, raw):
        return self._lifecycle_content("native-quiescence/v1", raw)

    def verification(self, content):
        if type(content) is not ArtifactVerificationContentV1:
            raise TypeError("exact native verification content required")
        receipt = AuthenticatedArtifactVerificationReceiptV1(
            content, self.authority_ref, self.key_ref,
            self._tag("native-artifact-verification/v1", content.content_digest),
        )
        return AuthenticatedArtifactVerificationReceiptV1.parse(receipt.canonical_bytes)

    def authenticate_verification(self, value):
        try:
            owned = AuthenticatedArtifactVerificationReceiptV1.parse(value.canonical_bytes)
            return (
                type(value) is AuthenticatedArtifactVerificationReceiptV1 and value == owned
                and owned.authority_ref == self.authority_ref and owned.key_ref == self.key_ref
                and hmac.compare_digest(owned.tag, self._tag(
                    "native-artifact-verification/v1", owned.content.content_digest,
                ))
            )
        except Exception:
            return False

    def process_outcome(self, invocation_digest, outcome):
        content = canonical({
            "schema": "synaptic-native-terminal-evidence/v1",
            "invocation_digest": invocation_digest, "outcome": outcome,
        })
        return canonical({
            "content": strict_json(content), "authority_ref": self.authority_ref, "key_ref": self.key_ref,
            "tag": self._tag("native-process-outcome/v1", hashlib.sha256(content).hexdigest()),
        })

    def authenticate_process_outcome(self, raw, invocation_digest):
        value = strict_json(raw, maximum=8192)
        if (type(value) is not dict or set(value) != {"content", "authority_ref", "key_ref", "tag"}
                or canonical(value) != raw or value["authority_ref"] != self.authority_ref
                or value["key_ref"] != self.key_ref
                or value["content"].get("schema") != "synaptic-native-terminal-evidence/v1"
                or value["content"].get("invocation_digest") != invocation_digest
                or not hmac.compare_digest(value["tag"], self._tag(
                    "native-process-outcome/v1", hashlib.sha256(canonical(value["content"])).hexdigest(),
                ))):
            raise ValueError("native terminal evidence is unauthenticated")
        return value["content"]["outcome"]


class NativeMaterialAuthority(ReaderEvidenceAuthority):
    """Host-only issuer; call only after the host's source/admission checks."""

    def issue(self, material):
        from tuner.training.native_mlx_family import NativeCommandMaterialV1, NativeCancellationMaterialV1
        if type(material) not in (NativeCommandMaterialV1, NativeCancellationMaterialV1):
            raise TypeError("exact native command material required")
        return replace(
            material, authority_ref=self.authority_ref, key_ref=self.key_ref,
            tag=self._tag("native-command-material/v1" if type(material) is NativeCommandMaterialV1
                          else "native-cancellation-material/v1", hashlib.sha256(material.content_bytes()).hexdigest()),
        )

    def authenticate(self, material):
        try:
            from tuner.training.native_mlx_family import NativeCommandMaterialV1, NativeCancellationMaterialV1
            if type(material) not in (NativeCommandMaterialV1, NativeCancellationMaterialV1):
                return False
            owned = type(material).parse(material.canonical_bytes)
            return (
                owned == material
                and owned.authority_ref == self.authority_ref and owned.key_ref == self.key_ref
                and hmac.compare_digest(owned.tag, self._tag(
                    "native-command-material/v1" if type(material) is NativeCommandMaterialV1
                    else "native-cancellation-material/v1", hashlib.sha256(owned.content_bytes()).hexdigest(),
                ))
            )
        except Exception:
            return False
