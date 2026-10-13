"""Host-key authentication of the existing Docker evidence contracts."""

from __future__ import annotations

from dataclasses import replace

from synaptic_tuner.api.v1.reference.repositories import _decode, _encode, _dump
from tuner.execution.coordinator_v1.model import (
    ArtifactVerificationContentV1, AuthenticatedArtifactVerificationReceiptV1,
)
from tuner.execution.providers.docker_provider_v1.model import (
    AuthenticatedDockerArtifactInventoryV1, AuthenticatedDockerCancellationEvidenceV1,
    AuthenticatedDockerCommandBindingV1, AuthenticatedDockerSourceSealV1,
    DockerArtifactEOFV1,
)
from tuner.training.modal_host_authority import HMACAuthenticator, ReaderEvidenceAuthority


class DockerHostEvidenceAuthorityV1(ReaderEvidenceAuthority):
    """Original host keys persist; no daemon result is trusted by type alone."""

    _KINDS = {
        AuthenticatedDockerCommandBindingV1: "docker-command-binding/v1",
        AuthenticatedDockerSourceSealV1: "docker-source-seal/v1",
        AuthenticatedDockerCancellationEvidenceV1: "docker-cancellation/v1",
        AuthenticatedDockerArtifactInventoryV1: "docker-artifact-inventory/v1",
        DockerArtifactEOFV1: "docker-artifact-eof/v1",
        AuthenticatedArtifactVerificationReceiptV1: "artifact-verification/v1",
    }

    def __init__(self, *, key: bytes, authority_ref: str, key_ref: str):
        super().__init__(authority_ref, key_ref, HMACAuthenticator(
            {key_ref: key}, allowed_purposes=frozenset((*self._KINDS.values(),
                "provider-run-observation/v1", "provider-log-page/v1")),
        ))

    def _signed(self, value):
        kind = self._KINDS.get(type(value))
        if kind is None:
            raise ValueError("docker_host_evidence_invalid")
        unsigned = replace(value, authority_ref=self.authority_ref,
                           key_ref=self.key_ref, tag="0" * 64)
        raw = self._bytes(unsigned)
        tag = self._hmac.sign(kind, raw, self.key_ref).hex()
        return replace(unsigned, tag=tag)

    @staticmethod
    def _bytes(value):
        raw = _dump(_encode(value, type(value)))
        if len(raw) > 1048576:
            raise ValueError('docker_host_evidence_bound')
        return raw

    def authenticate(self, value):
        try:
            kind = self._KINDS.get(type(value))
            if kind is None:
                return False
            owned = _decode(_encode(value, type(value)), type(value))
            return (owned == value and value.authority_ref == self.authority_ref
                    and value.key_ref == self.key_ref
                    and self._hmac.verify(kind,
                        self._bytes(replace(value, tag="0" * 64)),
                        bytes.fromhex(value.tag), self.key_ref))
        except Exception:
            return False

    def issue(self, binding):
        return self._signed(AuthenticatedDockerCommandBindingV1(
            binding, binding.binding_digest, self.authority_ref, self.key_ref, "0" * 64))

    def source_seal(self, content):
        return self._signed(AuthenticatedDockerSourceSealV1(
            content, self.authority_ref, self.key_ref, "0" * 64))

    def cancellation(self, content):
        return self._signed(AuthenticatedDockerCancellationEvidenceV1(
            content, self.authority_ref, self.key_ref, "0" * 64))

    def inventory(self, content):
        return self._signed(AuthenticatedDockerArtifactInventoryV1(
            content, self.authority_ref, self.key_ref, "0" * 64))

    def eof(self, value):
        return self._signed(value)

    def verification(self, content):
        if type(content) is not ArtifactVerificationContentV1:
            raise TypeError("exact verification content required")
        return self._signed(AuthenticatedArtifactVerificationReceiptV1(
            content, self.authority_ref, self.key_ref, "0" * 64))

    def authenticate_source_seal(self, value):
        return type(value) is AuthenticatedDockerSourceSealV1 and self.authenticate(value)

    def authenticate_cancellation(self, value):
        return type(value) is AuthenticatedDockerCancellationEvidenceV1 and self.authenticate(value)

    def authenticate_inventory(self, value):
        return type(value) is AuthenticatedDockerArtifactInventoryV1 and self.authenticate(value)

    def authenticate_eof(self, value):
        return type(value) is DockerArtifactEOFV1 and self.authenticate(value)

    def authenticate_absence(self, value):
        return False

    def authenticate_cancellation_absence(self, value):
        return False

    def authenticate_log_page(self, value):
        # Local artifact transport does not implement log publication.
        return False
