"""Native reader evidence over the existing host HMAC implementation."""

from dataclasses import replace
import hashlib
import hmac

from tuner.execution.coordinator_v1.model import (
    ArtifactVerificationContentV1, AuthenticatedArtifactVerificationReceiptV1,
)
from tuner.training.modal_host_authority import ReaderEvidenceAuthority
from tuner.runtime.native_mlx_material import canonical, strict_json


class NativeEvidenceAuthority(ReaderEvidenceAuthority):
    """Host constructs keys; no key generation, secret resolution, or storage."""

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
        from tuner.training.native_mlx_family import NativeCommandMaterialV1
        if type(material) is not NativeCommandMaterialV1:
            raise TypeError("exact native command material required")
        return replace(
            material, authority_ref=self.authority_ref, key_ref=self.key_ref,
            tag=self._tag("native-command-material/v1", hashlib.sha256(material.content_bytes()).hexdigest()),
        )

    def authenticate(self, material):
        try:
            from tuner.training.native_mlx_family import NativeCommandMaterialV1
            owned = NativeCommandMaterialV1.parse(material.canonical_bytes)
            return (
                type(material) is NativeCommandMaterialV1 and owned == material
                and owned.authority_ref == self.authority_ref and owned.key_ref == self.key_ref
                and hmac.compare_digest(owned.tag, self._tag(
                    "native-command-material/v1", hashlib.sha256(owned.content_bytes()).hexdigest(),
                ))
            )
        except Exception:
            return False
