"""Foundation-native Modal stage material and collision-failing writer."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Protocol

from tuner.execution.foundation_v2.canonical import (
    canonical_bytes, digest_text, parse_canonical_object, safe_ref,
)
from tuner.execution.foundation_v2.commands import StageCommandV2, parse_exact_command

from .contracts import BoundsPolicyV1, operation_path, sha
from .coordinator_binding import ModalCommandBinding
from .facade import ExplicitModal154ReadFacade, ModalFacadeError


class _BindingAuthority(Protocol):
    def authenticate(self, binding: ModalCommandBinding) -> bool: ...


class _StageSigner(Protocol):
    def sign(self, purpose: str, payload: bytes, key_ref: str) -> bytes: ...


class _StageVerifier(Protocol):
    def verify(self, purpose: str, payload: bytes, tag: bytes, key_ref: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class ModalStageMaterial:
    binding: ModalCommandBinding
    control_volume_id: str
    artifact_volume_id: str
    key_ref: str
    bundle: bytes
    claim: bytes
    claim_tag: bytes

    def __post_init__(self) -> None:
        if type(self.binding) is not ModalCommandBinding:
            raise TypeError("exact Modal command binding required")
        for name in ("control_volume_id", "artifact_volume_id", "key_ref"):
            safe_ref(getattr(self, name), name)
        if self.control_volume_id == self.artifact_volume_id:
            raise ValueError("control and artifact volumes must differ")
        if type(self.bundle) is not bytes or not self.bundle:
            raise ValueError("bundle must be nonempty bytes")
        if type(self.claim) is not bytes or not self.claim:
            raise ValueError("claim must be nonempty bytes")
        if type(self.claim_tag) is not bytes or not self.claim_tag or len(self.claim_tag) > 128:
            raise ValueError("claim tag is invalid")


@dataclass(frozen=True, slots=True)
class ModalStageReceipt:
    effect_id: str
    command_digest: str
    binding_digest: str
    control_volume_id: str
    artifact_volume_id: str
    claim_digest: str
    bundle_digest: str

    def __post_init__(self) -> None:
        for name in ("effect_id", "control_volume_id", "artifact_volume_id"):
            safe_ref(getattr(self, name), name)
        for name in ("command_digest", "binding_digest", "claim_digest", "bundle_digest"):
            digest_text(getattr(self, name), name)


def _reconstruct(binding: ModalCommandBinding) -> tuple[ModalCommandBinding, StageCommandV2]:
    if type(binding) is not ModalCommandBinding:
        raise TypeError("exact Modal command binding required")
    rebuilt = ModalCommandBinding(
        binding.command_bytes, binding.preparation_snapshot, binding.deployment_bytes,
    )
    if rebuilt != binding:
        raise ValueError("Modal command binding reconstruction mismatch")
    command = parse_exact_command(rebuilt.command_bytes)
    if type(command) is not StageCommandV2:
        raise ValueError("exact Foundation stage command required")
    return rebuilt, command


def _claim_document(material: ModalStageMaterial) -> dict[str, object]:
    binding, command = _reconstruct(material.binding)
    prep = command.preparation
    effect = command.operation.effect
    return {
        "schema_version": "synaptic.modal-stage-claim/v2",
        "command": parse_canonical_object(command.canonical_bytes, name="stage command"),
        "command_digest": command.digest,
        "binding_digest": binding.authenticated_binding_digest,
        "provider_id": prep.provider.provider_id,
        "profile_ref": prep.provider.profile_ref,
        "account_ref": prep.scope.account_ref,
        "namespace_ref": prep.scope.namespace_ref,
        "project_ref": prep.project_ref,
        "run_id": prep.run_id,
        "effect_id": effect.effect_id,
        "invocation_nonce": command.operation.invocation_nonce,
        "plan_fingerprint": prep.plan_fingerprint,
        "preparation_digest": prep.preparation_digest,
        "control_volume_id": material.control_volume_id,
        "artifact_volume_id": material.artifact_volume_id,
        "key_ref": material.key_ref,
        "bundle_sha256": sha(material.bundle),
        "bundle_size": len(material.bundle),
    }


def modal_stage_provider_ref(material: ModalStageMaterial) -> str:
    """Content identity returned by a successful stage transport, not authority."""
    if type(material) is not ModalStageMaterial:
        raise TypeError("exact Modal stage material required")
    if canonical_bytes(_claim_document(material)) != material.claim:
        raise ValueError("stage claim does not bind retained material")
    return safe_ref("modal-stage-claim:" + sha(material.claim), "stage_provider_ref")


def prepare_modal_foundation_stage(
    binding: ModalCommandBinding,
    bundle: bytes,
    binding_authority: _BindingAuthority,
    signer: _StageSigner,
    *,
    control_volume_id: str,
    artifact_volume_id: str,
    key_ref: str,
    bounds: BoundsPolicyV1 = BoundsPolicyV1(),
) -> ModalStageMaterial:
    rebuilt, _ = _reconstruct(binding)
    if binding_authority.authenticate(rebuilt) is not True:
        raise ValueError("Modal command binding authentication failed")
    if type(bundle) is not bytes or not bundle or len(bundle) > bounds.max_bundle_bytes:
        raise ValueError("bundle exceeds the Modal stage bound")
    # Construct once with a placeholder solely to derive the closed claim.
    provisional = ModalStageMaterial(
        rebuilt, control_volume_id, artifact_volume_id, key_ref, bundle, b"{}", b"x",
    )
    claim = canonical_bytes(_claim_document(provisional))
    if len(claim) > bounds.max_control_bytes:
        raise ValueError("stage claim exceeds the Modal control bound")
    try:
        tag = signer.sign("modal-stage-claim/v2", claim, provisional.key_ref)
    except Exception:
        raise ValueError("stage authentication unavailable") from None
    return ModalStageMaterial(
        rebuilt, provisional.control_volume_id, provisional.artifact_volume_id,
        provisional.key_ref, bundle, claim, tag,
    )


class ModalFoundationVolumeWriter:
    """Write exactly three operation-scoped files without overwrite."""

    def __init__(
        self, facade: ExplicitModal154ReadFacade,
        binding_authority: _BindingAuthority, verifier: _StageVerifier,
        *, bounds: BoundsPolicyV1 = BoundsPolicyV1(),
    ) -> None:
        if type(facade) is not ExplicitModal154ReadFacade:
            raise TypeError("exact Modal 1.5.4 facade required")
        self._facade = facade
        self._authority = binding_authority
        self._verifier = verifier
        self._bounds = bounds

    def _missing_exact(self, volume_id, prefix, expected):
        entries = self._facade.list_prefix(volume_id, prefix, max_entries=len(expected) + 1)
        declared = {path: data for path, data in expected}
        observed = {path: size for path, size, _ in entries}
        if len(observed) != len(entries) or not set(observed).issubset(declared):
            raise ModalFacadeError("modal_stage_collision")
        for path, size in observed.items():
            if size != len(declared[path]):
                raise ModalFacadeError("modal_stage_collision")
            data = self._facade.read_complete(volume_id, path, max_bytes=max(size, 1))
            if data != declared[path]:
                raise ModalFacadeError("modal_stage_collision")
        return tuple((path, data) for path, data in expected if path not in observed)

    def _upload(self, volume_id, files):
        try:
            volume = self._facade._volume(volume_id)
            with volume.batch_upload(force=False) as batch:
                for path, data in files:
                    batch.put_file(BytesIO(data), path)
        except ModalFacadeError:
            raise
        except Exception:
            raise ModalFacadeError("modal_stage_write_failed") from None

    def stage_once(self, material: ModalStageMaterial) -> ModalStageReceipt:
        if type(material) is not ModalStageMaterial:
            raise TypeError("exact Modal stage material required")
        rebuilt, command = _reconstruct(material.binding)
        expected = ModalStageMaterial(
            rebuilt, material.control_volume_id, material.artifact_volume_id,
            material.key_ref, material.bundle, material.claim, material.claim_tag,
        )
        if len(expected.bundle) > self._bounds.max_bundle_bytes or len(expected.claim) > self._bounds.max_control_bytes:
            raise ValueError("stage material exceeds bounds")
        if canonical_bytes(_claim_document(expected)) != expected.claim:
            raise ValueError("stage claim does not bind material")
        if self._authority.authenticate(rebuilt) is not True:
            raise ValueError("Modal command binding authentication failed")
        try:
            verified = self._verifier.verify(
                "modal-stage-claim/v2", expected.claim, expected.claim_tag, expected.key_ref,
            )
        except Exception:
            raise ValueError("stage authentication unavailable") from None
        if verified is not True:
            raise ValueError("stage authentication failed")

        profile = parse_canonical_object(
            rebuilt.preparation_snapshot, name="preparation snapshot",
        )["configuration"]["profile"]
        if (
            self._facade.binding != rebuilt.client_binding
            or self._facade.volume_name(expected.control_volume_id)
            != profile["volumes"]["control_ref"]
            or self._facade.volume_name(expected.artifact_volume_id)
            != profile["volumes"]["artifact_ref"]
        ):
            raise ModalFacadeError("modal_stage_binding_mismatch")
        # Authentication and cached name checks above precede the first live
        # provider observation.  The explicit client must still prove its
        # exact retained session before any Volume is resolved.
        self._facade.bound_scope()

        effect_id = command.operation.effect.effect_id
        bundle_path = operation_path(effect_id, "input", "bundle.bin")
        claim_path = operation_path(effect_id, "control", "stage-claim.v2.json")
        tag_path = operation_path(effect_id, "control", "stage-claim.v2.mac")
        root = operation_path(effect_id) + "/"
        artifact_files = ((bundle_path, expected.bundle),)
        control_files = ((claim_path, expected.claim), (tag_path, expected.claim_tag))
        missing_artifact = self._missing_exact(
            expected.artifact_volume_id, root, artifact_files,
        )
        missing_control = self._missing_exact(
            expected.control_volume_id, root, control_files,
        )
        if missing_artifact:
            self._upload(expected.artifact_volume_id, missing_artifact)
        if missing_control:
            self._upload(expected.control_volume_id, missing_control)
        observed = (
            self._facade.read_complete(expected.artifact_volume_id, bundle_path,
                                       max_bytes=self._bounds.max_bundle_bytes),
            self._facade.read_complete(expected.control_volume_id, claim_path,
                                       max_bytes=self._bounds.max_control_bytes),
            self._facade.read_complete(expected.control_volume_id, tag_path, max_bytes=128),
        )
        if observed != (expected.bundle, expected.claim, expected.claim_tag):
            raise ModalFacadeError("modal_stage_readback_mismatch")
        return ModalStageReceipt(
            effect_id, command.digest, rebuilt.authenticated_binding_digest,
            expected.control_volume_id, expected.artifact_volume_id,
            sha(expected.claim), sha(expected.bundle),
        )


__all__: list[str] = []
