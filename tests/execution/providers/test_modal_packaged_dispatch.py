"""Provider-free qualification of the single packaged Modal dispatch frame."""

from __future__ import annotations

from dataclasses import replace
import json

import pytest

from tuner.execution.foundation_v2.commands import (
    CanonicalProviderPayloadV1,
    build_submit_command,
)
from tuner.execution.foundation_v2.references import StagePredecessorV2
from tuner.execution.providers.modal.packaged_binding import ModalPackagedCommandBinding
from tuner.execution.providers.modal.contracts import (
    operation_path,
    provider_entry_identity,
)
from tuner.execution.providers.modal.packaged_dispatch import (
    MAX_MODAL_PACKAGED_DISPATCH_BYTES,
    ModalPackagedDispatch,
    build_modal_packaged_dispatch,
    parse_modal_packaged_dispatch,
)
from tuner.execution.providers.modal.packaged_staging import ModalPackagedStageReceipt
from tuner.training.packaged_compilation import compile_packaged_sft_workload

from tests.execution.providers.test_modal_packaged_binding import _binding


PREPARED_PAYLOAD = b"packaged-prepared-input"


class Auth:
    def __init__(self) -> None:
        self.signed: list[tuple[str, bytes, str]] = []
        self.verified: list[tuple[str, bytes, bytes, str]] = []

    def sign(self, purpose, payload, key_ref):
        self.signed.append((purpose, payload, key_ref))
        return b"authenticated-tag"

    def verify(self, purpose, payload, tag, key_ref):
        self.verified.append((purpose, payload, tag, key_ref))
        return tag == b"authenticated-tag"


def _case():
    stage_binding = _binding(PREPARED_PAYLOAD)
    stage = stage_binding.command
    predecessor = StagePredecessorV2(
        stage.preparation.provider.provider_id,
        stage.preparation.provider.profile_ref,
        stage.preparation.scope.account_ref,
        stage.preparation.scope.namespace_ref,
        stage.preparation.project_ref,
        stage.preparation.run_id,
        stage.preparation.plan_fingerprint,
        stage.preparation.preparation_digest,
        stage.preparation.workload_digest,
        stage.operation.effect.effect_id,
        "8" * 64,
        "9" * 64,
    )
    submit = build_submit_command(
        stage.preparation, "submit-nonce",
        CanonicalProviderPayloadV1.build(
            "modal", "submit-payload/v2", stage.preparation.workload_digest,
        ),
        stage.executor,
        predecessor,
    )
    binding = ModalPackagedCommandBinding(
        submit.canonical_bytes, stage_binding.runtime_release_bytes,
        stage_binding.provider_binding_bytes, stage_binding.provider_facts_bytes,
        stage_binding.execution_binding_bytes,
    )
    facts = binding.provider_facts
    receipt_path = operation_path(
        predecessor.stage_effect_id, "input", "prepared",
        binding.execution_binding.prepared_input_content_digest, "payload.bin",
    )
    receipt = ModalPackagedStageReceipt(
        predecessor.stage_effect_id, binding.execution_binding.binding_digest,
        facts.artifact_volume_id, receipt_path,
        binding.execution_binding.prepared_input_size_bytes,
        binding.execution_binding.prepared_input_content_digest,
        provider_entry_identity(
            facts.artifact_volume_id, receipt_path,
            binding.execution_binding.prepared_input_size_bytes,
        ),
    )
    _, components = __import__(
        "tests.training.test_packaged_execution_material",
        fromlist=["packaged_fixture"],
    ).packaged_fixture()
    workload = compile_packaged_sft_workload(
        resolved_config=components.resolved_config,
    ).canonical_bytes
    return binding, receipt, workload, components.artifact_policy


def test_builds_one_bounded_canonical_argument_and_round_trips_exactly() -> None:
    binding, receipt, workload, policy = _case()
    auth = Auth()
    payload = build_modal_packaged_dispatch(
        binding, receipt, workload, policy, auth,
        key_ref="dispatch-key", environment=(("PATH", "/usr/bin:/bin"),),
    )
    assert type(payload) is bytes and len(payload) <= MAX_MODAL_PACKAGED_DISPATCH_BYTES
    parsed = parse_modal_packaged_dispatch(payload, auth)
    assert type(parsed) is ModalPackagedDispatch
    assert parsed.submit_command_bytes == binding.command_bytes
    assert parsed.execution_binding == binding.execution_binding
    assert parsed.stage_receipt == receipt
    assert parsed.workload_bytes == workload
    assert parsed.environment == (("PATH", "/usr/bin:/bin"),)
    assert len(auth.signed) == len(auth.verified) == 1


def test_dispatch_binds_every_release_input_policy_and_stage_commitment() -> None:
    binding, receipt, workload, policy = _case()
    dispatch = ModalPackagedDispatch(
        binding.command_bytes, binding.runtime_release, binding.provider_binding,
        binding.provider_facts, binding.execution_binding, receipt, workload,
        policy, (), "dispatch-key",
    )
    unsigned = dispatch.unsigned_dict()
    assert unsigned["runtime_release"]["manifest_digest"] == (
        binding.runtime_release.manifest_digest
    )
    assert unsigned["provider_binding"]["binding_digest"] == (
        binding.provider_binding.binding_digest
    )
    assert unsigned["execution_binding"]["binding_digest"] == (
        binding.execution_binding.binding_digest
    )
    assert unsigned["stage_receipt"]["stage_effect_id"] == receipt.stage_effect_id


@pytest.mark.parametrize(
    "environment",
    (
        (("HF_TOKEN", "secret"),),
        (("MODAL_TOKEN_SECRET", "secret"),),
        (("PATH", "/bin"), ("PATH", "/other")),
        (("PATH", "bad\0value"),),
    ),
)
def test_secret_or_ambiguous_environment_is_rejected(environment) -> None:
    binding, receipt, workload, policy = _case()
    with pytest.raises(ValueError, match="environment"):
        ModalPackagedDispatch(
            binding.command_bytes, binding.runtime_release,
            binding.provider_binding, binding.provider_facts,
            binding.execution_binding, receipt, workload, policy,
            environment, "dispatch-key",
        )


def _rebound_receipt(receipt, *, stage=None, volume=None, size=None, digest=None):
    stage = receipt.stage_effect_id if stage is None else stage
    volume = receipt.artifact_volume_id if volume is None else volume
    size = receipt.size_bytes if size is None else size
    digest = receipt.content_digest if digest is None else digest
    path = operation_path(stage, "input", "prepared", digest, "payload.bin")
    return ModalPackagedStageReceipt(
        stage, receipt.execution_binding_digest, volume, path, size, digest,
        provider_entry_identity(volume, path, size),
    )


@pytest.mark.parametrize(
    "fault", ("workload", "stage", "volume", "size", "digest", "policy"),
)
def test_cross_binding_substitution_is_rejected_before_signing(fault: str) -> None:
    binding, receipt, workload, policy = _case()
    if fault == "workload":
        workload = workload.replace(b'"method":"sft"', b'"method":"kto"')
    elif fault == "stage":
        receipt = _rebound_receipt(receipt, stage="stage-other")
    elif fault == "volume":
        receipt = _rebound_receipt(receipt, volume="vo-other")
    elif fault == "size":
        receipt = _rebound_receipt(receipt, size=receipt.size_bytes + 1)
    elif fault == "digest":
        receipt = _rebound_receipt(receipt, digest="0" * 64)
    else:
        policy = replace(policy, retain_checkpoints=not policy.retain_checkpoints)
    auth = Auth()
    with pytest.raises(ValueError, match="cross-binding"):
        build_modal_packaged_dispatch(
            binding, receipt, workload, policy, auth, key_ref="dispatch-key",
        )
    assert auth.signed == []


def test_dispatch_contains_no_credentials_host_paths_or_launch_directives() -> None:
    binding, receipt, workload, policy = _case()
    payload = build_modal_packaged_dispatch(
        binding, receipt, workload, policy, Auth(), key_ref="dispatch-key",
    )
    lowered = payload.lower()
    for forbidden in (
        b"modal_token_id", b"modal_token_secret", b"password", b"credential",
        b"file://", b"git clone", b"add_local_python_source", b"pip install",
    ):
        assert forbidden not in lowered


def test_noncanonical_tampered_or_oversize_frames_are_rejected() -> None:
    binding, receipt, workload, policy = _case()
    payload = build_modal_packaged_dispatch(
        binding, receipt, workload, policy, Auth(), key_ref="dispatch-key",
    )
    with pytest.raises(ValueError):
        parse_modal_packaged_dispatch(payload + b" ", Auth())
    document = json.loads(payload)
    document["authentication"]["tag"] = "d3Jvbmc="
    attacked = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()
    with pytest.raises(ValueError, match="authentication failed"):
        parse_modal_packaged_dispatch(attacked, Auth())
    with pytest.raises(ValueError, match="bytes are invalid"):
        parse_modal_packaged_dispatch(
            b"x" * (MAX_MODAL_PACKAGED_DISPATCH_BYTES + 1), Auth(),
        )
