"""Pure remote admission for a host-authenticated Modal launch claim."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from tuner.execution.foundation_v2.canonical import (
    canonical_bytes, digest_text, domain_digest, parse_canonical_object, safe_ref,
)
from tuner.execution.foundation_v2.commands import StageCommandV2, SubmitCommandV2, parse_exact_command

from .contracts import BoundsPolicyV1, sha, strict_int
from .coordinator_binding import ModalCommandBinding


class _Verifier(Protocol):
    def verify(self, purpose: str, payload: bytes, tag: bytes, key_ref: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class ModalWorkerLaunchExpectation:
    submit_command_bytes: bytes
    deployment_bytes: bytes
    provider_id: str
    profile_ref: str
    account_ref: str
    namespace_ref: str
    app_name: str
    function_name: str
    control_volume_id: str
    artifact_volume_id: str
    control_volume_ref: str
    artifact_volume_ref: str
    key_ref: str
    stage_claim_sha256: str
    stage_bundle_sha256: str
    stage_bundle_size: int
    executor_id: str
    executor_implementation_version: str

    def __post_init__(self) -> None:
        if type(self.submit_command_bytes) is not bytes or type(self.deployment_bytes) is not bytes:
            raise TypeError("exact command and deployment bytes required")
        for name in self.__dataclass_fields__:
            if name not in {"submit_command_bytes", "deployment_bytes", "stage_bundle_size"}:
                safe_ref(getattr(self, name), name)
        digest_text(self.stage_claim_sha256, "stage_claim_sha256")
        digest_text(self.stage_bundle_sha256, "stage_bundle_sha256")
        strict_int(self.stage_bundle_size, "stage_bundle_size", minimum=1)
        if self.control_volume_id == self.artifact_volume_id:
            raise ValueError("control and artifact volumes must differ")


@dataclass(frozen=True, slots=True)
class ModalWireLaunchAdmission:
    submit_command_bytes: bytes
    stage_command_bytes: bytes
    preparation_snapshot: bytes
    submit_command_digest: str
    submit_effect_id: str
    stage_effect_id: str
    deployment_bytes: bytes
    control_volume_id: str
    artifact_volume_id: str
    key_ref: str
    bundle: bytes
    launch_claim_sha256: str


_LAUNCH_FIELDS = {
    "schema_version", "submit_binding", "submit_binding_digest", "submit_command",
    "submit_command_digest", "submit_effect_id", "submit_invocation_nonce",
    "stage_binding_digest", "stage_command_digest", "stage_record_digest",
    "stage_assessment_digest", "stage_foundation_binding_digest",
    "stage_outcome_digest", "stage_bound_reference_digest", "stage_predecessor",
    "stage_claim_sha256", "stage_bundle_sha256", "stage_bundle_size",
    "control_volume_id", "artifact_volume_id", "configured_control_volume_ref",
    "configured_artifact_volume_ref", "key_ref",
}
_STAGE_FIELDS = {
    "schema_version", "command", "command_digest", "binding_digest", "provider_id",
    "profile_ref", "account_ref", "namespace_ref", "project_ref", "run_id",
    "effect_id", "invocation_nonce", "plan_fingerprint", "preparation_digest",
    "control_volume_id", "artifact_volume_id", "key_ref", "bundle_sha256",
    "bundle_size",
}


def _bounded_object(raw: bytes, maximum: int, name: str) -> dict[str, object]:
    if type(raw) is not bytes or not raw or len(raw) > maximum:
        raise ValueError(f"{name} exceeds bound")
    value = parse_canonical_object(raw, name=name)
    if type(value) is not dict or canonical_bytes(value) != raw:
        raise ValueError(f"{name} is not a canonical object")
    return value


def _verified(verifier: _Verifier, purpose: str, payload: bytes, tag: bytes, key_ref: str) -> None:
    if type(tag) is not bytes or not tag or len(tag) > 128:
        raise ValueError("authentication tag is invalid")
    try:
        valid = verifier.verify(purpose, payload, tag, key_ref)
    except Exception:
        raise ValueError("wire authentication unavailable") from None
    if valid is not True:
        raise ValueError("wire authentication failed")


def admit_modal_launch_wire(
    claim: bytes, claim_tag: bytes, stage_claim: bytes, stage_claim_tag: bytes,
    bundle: bytes, *, expectation: ModalWorkerLaunchExpectation,
    verifier: _Verifier, bounds: BoundsPolicyV1 = BoundsPolicyV1(),
) -> ModalWireLaunchAdmission:
    """Admit signed public lineage without importing host authority services."""
    if type(expectation) is not ModalWorkerLaunchExpectation:
        raise TypeError("exact worker launch expectation required")
    launch = _bounded_object(claim, bounds.max_control_bytes, "launch claim")
    if set(launch) != _LAUNCH_FIELDS or launch["schema_version"] != "synaptic.modal-launch-claim/v1":
        raise ValueError("launch claim fields are invalid")
    key_ref = safe_ref(launch["key_ref"], "key_ref")
    if key_ref != expectation.key_ref:
        raise ValueError("launch key differs from worker expectation")
    _verified(verifier, "modal-launch-claim/v1", claim, claim_tag, key_ref)

    stage_doc = _bounded_object(stage_claim, bounds.max_control_bytes, "stage claim")
    strict_int(stage_doc.get("bundle_size"), "stage bundle_size", minimum=1)
    _verified(verifier, "modal-stage-claim/v2", stage_claim, stage_claim_tag, key_ref)
    if type(bundle) is not bytes or not bundle or len(bundle) > bounds.max_bundle_bytes:
        raise ValueError("bundle exceeds bound")
    strict_int(launch["stage_bundle_size"], "stage_bundle_size", minimum=1)

    binding_doc = launch["submit_binding"]
    if type(binding_doc) is not dict or set(binding_doc) != {"command", "preparation_snapshot", "deployment"}:
        raise ValueError("submit binding is invalid")
    submit_binding = ModalCommandBinding(
        canonical_bytes(binding_doc["command"]), canonical_bytes(binding_doc["preparation_snapshot"]),
        canonical_bytes(binding_doc["deployment"]),
    )
    submit = parse_exact_command(submit_binding.command_bytes)
    if type(submit) is not SubmitCommandV2:
        raise ValueError("exact submit command required")
    stage = parse_exact_command(canonical_bytes(stage_doc.get("command")))
    if type(stage) is not StageCommandV2:
        raise ValueError("exact stage command required")
    stage_binding = ModalCommandBinding(
        stage.canonical_bytes, submit_binding.preparation_snapshot, submit_binding.deployment_bytes,
    )
    prep = submit.preparation
    predecessor = submit.stage_predecessor
    if stage.preparation != prep:
        raise ValueError("stage and submit preparations differ")

    for name in (
        "submit_binding_digest", "submit_command_digest", "stage_binding_digest",
        "stage_command_digest", "stage_record_digest", "stage_assessment_digest",
        "stage_foundation_binding_digest", "stage_outcome_digest",
        "stage_bound_reference_digest", "stage_claim_sha256", "stage_bundle_sha256",
    ):
        digest_text(launch[name], name)
    expected_projection = (
        submit_binding.authenticated_binding_digest, submit.to_dict(), submit.digest,
        submit.operation.effect.effect_id, submit.operation.invocation_nonce,
        stage_binding.authenticated_binding_digest, stage.digest, predecessor.to_dict(),
        sha(stage_claim), sha(bundle), len(bundle),
    )
    actual_projection = (
        launch["submit_binding_digest"], launch["submit_command"],
        launch["submit_command_digest"], launch["submit_effect_id"],
        launch["submit_invocation_nonce"], launch["stage_binding_digest"],
        launch["stage_command_digest"], launch["stage_predecessor"],
        launch["stage_claim_sha256"], launch["stage_bundle_sha256"],
        launch["stage_bundle_size"],
    )
    if actual_projection != expected_projection:
        raise ValueError("launch claim projection differs")

    stage_expected = {
        "command_digest": stage.digest, "binding_digest": stage_binding.authenticated_binding_digest,
        "provider_id": prep.provider.provider_id, "profile_ref": prep.provider.profile_ref,
        "account_ref": prep.scope.account_ref, "namespace_ref": prep.scope.namespace_ref,
        "project_ref": prep.project_ref, "run_id": prep.run_id,
        "effect_id": stage.operation.effect.effect_id,
        "invocation_nonce": stage.operation.invocation_nonce,
        "plan_fingerprint": prep.plan_fingerprint,
        "preparation_digest": prep.preparation_digest,
        "control_volume_id": launch["control_volume_id"],
        "artifact_volume_id": launch["artifact_volume_id"], "key_ref": key_ref,
        "bundle_sha256": sha(bundle), "bundle_size": len(bundle),
    }
    if set(stage_doc) != _STAGE_FIELDS or stage_doc.get("schema_version") != "synaptic.modal-stage-claim/v2" or any(
        stage_doc.get(name) != value for name, value in stage_expected.items()
    ):
        raise ValueError("stage claim projection differs")

    selection = submit_binding.deployment.selection
    snapshot = parse_canonical_object(
        submit_binding.preparation_snapshot, name="preparation snapshot",
    )
    profile_volumes = snapshot["configuration"]["profile"]["volumes"]
    expected_worker = (
        submit.canonical_bytes, submit_binding.deployment_bytes,
        prep.provider.provider_id, prep.provider.profile_ref,
        prep.scope.account_ref, prep.scope.namespace_ref, selection.app_name,
        selection.function_name, launch["control_volume_id"], launch["artifact_volume_id"],
        profile_volumes["control_ref"], profile_volumes["artifact_ref"], key_ref,
        sha(stage_claim), sha(bundle), len(bundle),
        submit.executor.executor_id, submit.executor.implementation_version,
    )
    supplied_worker = tuple(getattr(expectation, name) for name in expectation.__dataclass_fields__)
    if expected_worker != supplied_worker:
        raise ValueError("launch differs from worker expectation")
    if (launch["configured_control_volume_ref"], launch["configured_artifact_volume_ref"]) != (
        profile_volumes["control_ref"], profile_volumes["artifact_ref"],
    ):
        raise ValueError("launch Volume names differ from retained profile")

    reference = {
        "provider_id": prep.provider.provider_id, "profile_ref": prep.provider.profile_ref,
        "account_ref": prep.scope.account_ref, "namespace_ref": prep.scope.namespace_ref,
        "stage_ref": f"modal-stage-claim:{sha(stage_claim)}",
    }
    command_bytes_digest = domain_digest("synaptic-foundation-command-bytes/v1", stage.canonical_bytes)
    bound_doc = {
        "reference": reference, "effect_id": stage.operation.effect.effect_id,
        "command_digest": stage.digest, "command_bytes_digest": command_bytes_digest,
        "preparation_digest": prep.preparation_digest,
        "foundation_binding_digest": launch["stage_foundation_binding_digest"],
        "foundation_outcome_digest": launch["stage_outcome_digest"],
        "authenticated_receipt_digest": predecessor.authenticated_receipt_digest,
    }
    if domain_digest("synaptic-stage-evidence-binding/v1", canonical_bytes(bound_doc)) != launch["stage_bound_reference_digest"]:
        raise ValueError("stage result commitment differs")
    predecessor_projection = (
        predecessor.provider_id, predecessor.profile_ref, predecessor.account_ref,
        predecessor.namespace_ref, predecessor.project_ref, predecessor.run_id,
        predecessor.plan_fingerprint, predecessor.preparation_digest,
        predecessor.workload_digest, predecessor.stage_effect_id,
        predecessor.record_digest,
    )
    expected_predecessor = (
        prep.provider.provider_id, prep.provider.profile_ref, prep.scope.account_ref,
        prep.scope.namespace_ref, prep.project_ref, prep.run_id,
        prep.plan_fingerprint, prep.preparation_digest, prep.workload_digest,
        stage.operation.effect.effect_id, launch["stage_record_digest"],
    )
    if predecessor_projection != expected_predecessor:
        raise ValueError("stage predecessor differs")
    return ModalWireLaunchAdmission(
        submit.canonical_bytes, stage.canonical_bytes,
        submit_binding.preparation_snapshot, submit.digest,
        submit.operation.effect.effect_id, stage.operation.effect.effect_id,
        submit_binding.deployment_bytes,
        expectation.control_volume_id, expectation.artifact_volume_id,
        key_ref, bytes(bundle), sha(claim),
    )


__all__: list[str] = []
