"""Foundation-native Modal stage material and collision-failing writer."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
import hashlib
from typing import Protocol

from tuner.execution.foundation_v2.canonical import (
    canonical_bytes, digest_text, parse_canonical_object, safe_ref,
)
from tuner.execution.foundation_v2.commands import StageCommandV2, parse_exact_command

from .contracts import BoundsPolicyV1, operation_path, sha
from .coordinator_binding import ModalCommandBinding
from .facade import ExplicitModal154ReadFacade, ModalFacadeError
from .prepared_input import (
    MAX_MOUNTED_PREPARED_DATASET_BYTES,
    MountedPreparedInputDescriptor,
)
from tuner.training.contracts import VerifiedTrainingInputSource


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
    bundle: bytes = field(repr=False)
    claim: bytes
    claim_tag: bytes
    prepared_input_descriptor: bytes | None = None
    prepared_input_source: VerifiedTrainingInputSource | None = field(
        default=None, repr=False, compare=False,
    )

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
        if self.prepared_input_descriptor is None:
            if self.prepared_input_source is not None:
                raise ValueError("prepared input source requires a descriptor")
        else:
            if type(self.prepared_input_descriptor) is not bytes:
                raise TypeError("prepared input descriptor must be exact bytes")
            descriptor = MountedPreparedInputDescriptor.parse(
                self.prepared_input_descriptor
            )
            command = parse_exact_command(self.binding.command_bytes)
            if descriptor != MountedPreparedInputDescriptor.create(
                descriptor.identity,
                stage_effect_id=command.operation.effect.effect_id,
            ):
                raise ValueError("prepared input descriptor is not stage scoped")
            if self.prepared_input_source is not None and (
                not isinstance(self.prepared_input_source, VerifiedTrainingInputSource)
                or self.prepared_input_source.identity != descriptor.identity
            ):
                raise ValueError("prepared input source differs from its descriptor")


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
    result = {
        "schema_version": (
            "synaptic.modal-stage-claim/v3"
            if material.prepared_input_descriptor is not None
            else "synaptic.modal-stage-claim/v2"
        ),
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
    if material.prepared_input_descriptor is not None:
        result["prepared_input"] = parse_canonical_object(
            material.prepared_input_descriptor, name="prepared input descriptor",
        )
    return result


def stage_claim_purpose(material: ModalStageMaterial) -> str:
    return (
        "modal-stage-claim/v3"
        if material.prepared_input_descriptor is not None
        else "modal-stage-claim/v2"
    )


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
    prepared_input_descriptor: bytes | None = None,
    prepared_input_source: VerifiedTrainingInputSource | None = None,
) -> ModalStageMaterial:
    rebuilt, _ = _reconstruct(binding)
    if binding_authority.authenticate(rebuilt) is not True:
        raise ValueError("Modal command binding authentication failed")
    if type(bundle) is not bytes or not bundle or len(bundle) > bounds.max_bundle_bytes:
        raise ValueError("bundle exceeds the Modal stage bound")
    # Construct once with a placeholder solely to derive the closed claim.
    provisional = ModalStageMaterial(
        rebuilt, control_volume_id, artifact_volume_id, key_ref, bundle, b"{}", b"x",
        prepared_input_descriptor, prepared_input_source,
    )
    claim = canonical_bytes(_claim_document(provisional))
    if len(claim) > bounds.max_control_bytes:
        raise ValueError("stage claim exceeds the Modal control bound")
    try:
        tag = signer.sign(stage_claim_purpose(provisional), claim, provisional.key_ref)
    except Exception:
        raise ValueError("stage authentication unavailable") from None
    return ModalStageMaterial(
        rebuilt, provisional.control_volume_id, provisional.artifact_volume_id,
        provisional.key_ref, bundle, claim, tag,
        prepared_input_descriptor, prepared_input_source,
    )


class ModalFoundationVolumeWriter:
    """Write the exact operation-scoped stage files without overwrite."""

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

    def _verify_streamed_object(self, volume_id, path, descriptor):
        digest = hashlib.sha256()
        size = 0
        for chunk in self._facade.iter_complete(
            volume_id, path, max_bytes=MAX_MOUNTED_PREPARED_DATASET_BYTES,
        ):
            size += len(chunk)
            digest.update(chunk)
        if (size, digest.hexdigest()) != (
            descriptor.identity.size_bytes,
            descriptor.identity.content_digest,
        ):
            raise ModalFacadeError("modal_stage_readback_mismatch")

    def _stage_mounted_payload(
        self, material, descriptor, root, bundle_path, *, control_present,
    ):
        expected_paths = {
            bundle_path: len(material.bundle),
            descriptor.relative_path: descriptor.identity.size_bytes,
        }
        entries = self._facade.list_prefix(
            material.artifact_volume_id, root, max_entries=3,
        )
        observed = {path: size for path, size, _ in entries}
        if len(observed) != len(entries) or not set(observed).issubset(expected_paths):
            raise ModalFacadeError("modal_stage_collision")
        if any(expected_paths[path] != size for path, size in observed.items()):
            raise ModalFacadeError("modal_stage_collision")
        if bundle_path in observed:
            if self._facade.read_complete(
                material.artifact_volume_id, bundle_path,
                max_bytes=self._bounds.max_bundle_bytes,
            ) != material.bundle:
                raise ModalFacadeError("modal_stage_collision")
        if descriptor.relative_path in observed:
            try:
                self._verify_streamed_object(
                    material.artifact_volume_id, descriptor.relative_path, descriptor,
                )
            except ModalFacadeError:
                raise ModalFacadeError("modal_stage_collision") from None
        if control_present and set(observed) != set(expected_paths):
            raise ModalFacadeError("modal_stage_collision")
        lease = None
        stream = None
        if descriptor.relative_path not in observed:
            if material.prepared_input_source is None:
                raise ModalFacadeError("modal_stage_source_unavailable")
            try:
                lease = material.prepared_input_source.open_lease()
                if lease.identity != descriptor.identity:
                    raise ValueError
                stream = lease.take_stream()
            except Exception:
                if lease is not None:
                    lease.close()
                raise ModalFacadeError("modal_stage_source_unavailable") from None
        try:
            if bundle_path not in observed:
                self._upload(
                    material.artifact_volume_id, ((bundle_path, material.bundle),),
                )
            if stream is not None:
                volume = self._facade._volume(material.artifact_volume_id)
                with volume.batch_upload(force=False) as batch:
                    batch.put_file(stream, descriptor.relative_path)
        except ModalFacadeError:
            raise
        except Exception:
            raise ModalFacadeError("modal_stage_write_failed") from None
        finally:
            if lease is not None:
                lease.close()
        # A second inventory rejects partial writes and every unexpected extra.
        entries = self._facade.list_prefix(
            material.artifact_volume_id, root, max_entries=3,
        )
        observed = {path: size for path, size, _ in entries}
        if observed != expected_paths or len(entries) != 2:
            raise ModalFacadeError("modal_stage_collision")
        if self._facade.read_complete(
            material.artifact_volume_id, bundle_path,
            max_bytes=self._bounds.max_bundle_bytes,
        ) != material.bundle:
            raise ModalFacadeError("modal_stage_readback_mismatch")
        self._verify_streamed_object(
            material.artifact_volume_id, descriptor.relative_path, descriptor,
        )

    def stage_once(self, material: ModalStageMaterial) -> ModalStageReceipt:
        if type(material) is not ModalStageMaterial:
            raise TypeError("exact Modal stage material required")
        rebuilt, command = _reconstruct(material.binding)
        expected = ModalStageMaterial(
            rebuilt, material.control_volume_id, material.artifact_volume_id,
            material.key_ref, material.bundle, material.claim, material.claim_tag,
            material.prepared_input_descriptor, material.prepared_input_source,
        )
        if len(expected.bundle) > self._bounds.max_bundle_bytes or len(expected.claim) > self._bounds.max_control_bytes:
            raise ValueError("stage material exceeds bounds")
        if canonical_bytes(_claim_document(expected)) != expected.claim:
            raise ValueError("stage claim does not bind material")
        if self._authority.authenticate(rebuilt) is not True:
            raise ValueError("Modal command binding authentication failed")
        try:
            verified = self._verifier.verify(
                stage_claim_purpose(expected), expected.claim, expected.claim_tag,
                expected.key_ref,
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
        descriptor = (
            None
            if expected.prepared_input_descriptor is None
            else MountedPreparedInputDescriptor.parse(
                expected.prepared_input_descriptor
            )
        )
        if descriptor is not None:
            control_entries = self._facade.list_prefix(
                expected.control_volume_id, root, max_entries=3,
            )
            control_observed = {path: size for path, size, _ in control_entries}
            declared_control = {path: len(data) for path, data in control_files}
            if (
                len(control_observed) != len(control_entries)
                or not set(control_observed).issubset(declared_control)
                or any(
                    control_observed[path] != declared_control[path]
                    for path in control_observed
                )
            ):
                raise ModalFacadeError("modal_stage_collision")
            self._stage_mounted_payload(
                expected, descriptor, root, bundle_path,
                control_present=bool(control_observed),
            )
            # Control material is deliberately published only after exact artifact
            # readback; exact pre-existing controls are recovery evidence.
            missing_control = self._missing_exact(
                expected.control_volume_id, root, control_files,
            )
            if missing_control:
                self._upload(expected.control_volume_id, missing_control)
            observed = (
                self._facade.read_complete(
                    expected.artifact_volume_id, bundle_path,
                    max_bytes=self._bounds.max_bundle_bytes,
                ),
                self._facade.read_complete(
                    expected.control_volume_id, claim_path,
                    max_bytes=self._bounds.max_control_bytes,
                ),
                self._facade.read_complete(
                    expected.control_volume_id, tag_path, max_bytes=128,
                ),
            )
            if observed != (expected.bundle, expected.claim, expected.claim_tag):
                raise ModalFacadeError("modal_stage_readback_mismatch")
            return ModalStageReceipt(
                effect_id, command.digest, rebuilt.authenticated_binding_digest,
                expected.control_volume_id, expected.artifact_volume_id,
                sha(expected.claim), sha(expected.bundle),
            )
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
