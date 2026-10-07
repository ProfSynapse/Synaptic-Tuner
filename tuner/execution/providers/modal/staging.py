"""Private, single-attempt Modal Volume v1 staging adapter."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from ...operation import OperationBindingV1
from .binding import ModalClientBinding, Readiness, readiness_report
from .contracts import BoundsPolicyV1, StageReceiptV1, canonical_json, operation_path, sha
from .control import StageExpectationV1
from .facade import ExplicitModal154ReadFacade, ModalFacadeError

@dataclass(frozen=True, slots=True)
class StageMaterialV1:
    expectation: StageExpectationV1
    bundle: bytes
    claim: bytes
    claim_tag: bytes

    def __post_init__(self) -> None:
        if type(self.expectation) is not StageExpectationV1:
            raise TypeError("expectation must be StageExpectationV1")
        for name in ("bundle", "claim", "claim_tag"):
            if not isinstance(getattr(self, name), bytes) or not getattr(self, name):
                raise ValueError(f"{name} must be nonempty bytes")
        if (
            sha(self.bundle) != self.expectation.bundle_digest
            or len(self.bundle) != self.expectation.bundle_size
            or sha(self.claim) != self.expectation.claim_digest
        ):
            raise ValueError("stage material does not bind its expectation")


def prepare_modal_stage(
    operation: OperationBindingV1,
    binding: ModalClientBinding,
    bundle: bytes,
    authenticator,
    *,
    bounds: BoundsPolicyV1 = BoundsPolicyV1(),
) -> StageMaterialV1:
    """Create deterministic material; the host persists expectation before I/O."""
    if type(operation) is not OperationBindingV1 or type(binding) is not ModalClientBinding:
        raise TypeError("canonical operation and Modal binding are required")
    if not isinstance(bundle, bytes) or not bundle or len(bundle) > bounds.max_bundle_bytes:
        raise ValueError("bundle exceeds the Modal v1 stage bound")
    target = operation.stage_target
    claim = canonical_json({
        "schema": "synaptic.modal-stage-claim/v1",
        "effect_provider": operation.effect.scope.provider,
        "effect_account_ref": operation.effect.scope.account_ref,
        "effect_namespace_ref": operation.effect.scope.namespace_ref,
        "effect_id": operation.effect.effect_id,
        "effect_kind": operation.effect.kind.value,
        "operation_key": operation.effect.effect_key,
        "operation_binding_digest": operation.digest,
        "control_volume_id": target.control_volume_id,
        "artifact_volume_id": target.artifact_volume_id,
        "bundle_digest": sha(bundle),
        "bundle_size": len(bundle),
        "plan_digest": operation.plan_fingerprint,
        "invocation_nonce": operation.invocation_nonce,
        "output_prefix": target.output_prefix,
    })
    if len(claim) > bounds.max_control_bytes:
        raise ValueError("stage claim exceeds the Modal v1 control bound")
    try:
        tag = authenticator.sign("modal-stage-claim/v1", claim, target.key_ref)
    except Exception:
        raise ValueError("stage authentication unavailable") from None
    if not isinstance(tag, bytes) or not tag or len(tag) > 128:
        raise ValueError("stage authentication tag is invalid")
    expectation = StageExpectationV1.from_stage(operation, binding, claim=claim, bundle=bundle)
    return StageMaterialV1(expectation, bundle, claim, tag)


class _ExplicitModal154VolumeWriter:
    """Mutate only the three reserved stage paths, once, without overwrite."""

    __slots__ = ("_facade", "_bounds")

    def __init__(self, facade: ExplicitModal154ReadFacade, *, bounds: BoundsPolicyV1 = BoundsPolicyV1()) -> None:
        if type(facade) is not ExplicitModal154ReadFacade:
            raise TypeError("exact Modal 1.5.4 facade is required")
        self._facade = facade
        self._bounds = bounds

    def _missing_exact(
        self,
        volume_id: str,
        prefix: str,
        expected: tuple[tuple[str, bytes], ...],
    ) -> tuple[tuple[str, bytes], ...]:
        entries = self._facade.list_prefix(
            volume_id, prefix, max_entries=len(expected) + 1
        )
        declared = {path: data for path, data in expected}
        observed = {path: size for path, size, _ in entries}
        if len(observed) != len(entries) or not set(observed).issubset(declared):
            raise ModalFacadeError("modal_stage_collision")
        for path, size in observed.items():
            if size != len(declared[path]):
                raise ModalFacadeError("modal_stage_collision")
            data = self._facade.read_complete(
                volume_id, path, max_bytes=max(len(declared[path]), 1)
            )
            if size != len(declared[path]) or data != declared[path]:
                raise ModalFacadeError("modal_stage_collision")
        return tuple((path, data) for path, data in expected if path not in observed)

    def _upload(self, volume_id: str, files: tuple[tuple[str, bytes], ...]) -> None:
        volume = self._facade._volume(volume_id)
        try:
            with volume.batch_upload(force=False) as batch:
                for path, data in files:
                    batch.put_file(BytesIO(data), path)
        except ModalFacadeError:
            raise
        except Exception:
            raise ModalFacadeError("modal_stage_write_failed") from None

    def stage_once(self, material: StageMaterialV1) -> StageReceiptV1:
        if type(material) is not StageMaterialV1:
            raise TypeError("stage material must be StageMaterialV1")
        expected = material.expectation
        effect_id = expected.effect.effect_id
        bundle_path = operation_path(effect_id, "input", "bundle.bin")
        claim_path = operation_path(effect_id, "control", "stage-claim.v1.json")
        claim_mac_path = operation_path(effect_id, "control", "stage-claim.v1.mac")
        report = readiness_report(expected.binding, self._facade)
        if report.status is not Readiness.READY:
            raise ModalFacadeError(report.reason_code)
        root = operation_path(effect_id) + "/"
        missing_artifact = self._missing_exact(
            expected.artifact_volume_id, root, ((bundle_path, material.bundle),)
        )
        missing_control = self._missing_exact(
            expected.control_volume_id,
            root,
            ((claim_path, material.claim), (claim_mac_path, material.claim_tag)),
        )
        if missing_artifact:
            self._upload(expected.artifact_volume_id, missing_artifact)
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
                expected.control_volume_id, claim_mac_path, max_bytes=128,
            ),
        )
        if observed != (material.bundle, material.claim, material.claim_tag):
            raise ModalFacadeError("modal_stage_readback_mismatch")
        return StageReceiptV1(
            expected.effect.effect_id,
            expected.operation_binding_digest,
            expected.control_volume_id,
            expected.artifact_volume_id,
            expected.claim_digest,
            expected.bundle_digest,
        )


__all__ = ["StageMaterialV1", "prepare_modal_stage"]
