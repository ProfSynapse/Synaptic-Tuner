"""One bounded canonical dispatch frame for a packaged Modal worker."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import json
from typing import Protocol

from tuner.execution.foundation_v2.canonical import canonical_bytes, parse_canonical_object, safe_ref
from tuner.execution.foundation_v2.commands import SubmitCommandV2, parse_exact_command
from tuner.runtime.releases import (
    PackagedExecutionBindingV1,
    PackagedTrainingRuntimeReleaseV1,
    ProviderRuntimeBindingV1,
)
from tuner.training.contracts import ArtifactPolicy, CanonicalDocument
from tuner.training.recipes import CompiledWorkload
from tuner.training.packaged_compilation import (
    compile_packaged_sft_workload,
    packaged_artifact_policy_digest,
)

from .packaged_binding import (
    ModalPackagedCommandBinding,
    ModalPackagedRuntimeFactsV1,
)
from .contracts import operation_path, provider_entry_identity
from .packaged_staging import ModalPackagedStageReceipt


MODAL_PACKAGED_DISPATCH_SCHEMA = "synaptic-modal-packaged-dispatch/v1"
MAX_MODAL_PACKAGED_DISPATCH_BYTES = 1024 * 1024
_PURPOSE = "modal-packaged-dispatch/v1"


def _canonical(value: object) -> bytes:
    try:
        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        raise ValueError("packaged dispatch contains non-canonical JSON") from None
    if len(payload) > MAX_MODAL_PACKAGED_DISPATCH_BYTES:
        raise ValueError("packaged dispatch exceeds its byte bound")
    return payload


def _object(payload: bytes) -> dict[str, object]:
    if type(payload) is not bytes or not 0 < len(payload) <= MAX_MODAL_PACKAGED_DISPATCH_BYTES:
        raise ValueError("packaged dispatch bytes are invalid")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("packaged dispatch contains duplicate keys")
            result[key] = value
        return result
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
            parse_float=lambda _: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError):
        raise ValueError("packaged dispatch is not strict canonical JSON") from None
    if type(value) is not dict or _canonical(value) != payload:
        raise ValueError("packaged dispatch is not canonical")
    return value


def _packaged_workload(payload: bytes) -> CompiledWorkload:
    """Recompile the exact A-D workload; finite floats remain valid."""
    if type(payload) is not bytes or not payload:
        raise ValueError("packaged workload bytes are invalid")
    try:
        document = CanonicalDocument(payload.decode("utf-8"))
        value = document.to_dict()
        configuration = value["configuration"]["document"]
        compiled = compile_packaged_sft_workload(
            resolved_config=CanonicalDocument.from_mapping(configuration),
        )
    except Exception:
        raise ValueError("packaged workload is invalid") from None
    if compiled.canonical_bytes != payload:
        raise ValueError("packaged workload differs from deterministic compilation")
    return compiled


class ModalPackagedDispatchSigner(Protocol):
    def sign(self, purpose: str, payload: bytes, key_ref: str) -> bytes: ...


class ModalPackagedDispatchVerifier(Protocol):
    def verify(self, purpose: str, payload: bytes, tag: bytes, key_ref: str) -> bool: ...


def _policy_dict(policy: ArtifactPolicy) -> dict[str, object]:
    if type(policy) is not ArtifactPolicy:
        raise TypeError("exact artifact policy required")
    return {
        "required_kinds": list(policy.required_kinds),
        "retain_checkpoints": policy.retain_checkpoints,
    }


@dataclass(frozen=True, slots=True)
class ModalPackagedDispatch:
    submit_command_bytes: bytes = field(repr=False)
    runtime_release: PackagedTrainingRuntimeReleaseV1
    provider_binding: ProviderRuntimeBindingV1
    provider_facts: ModalPackagedRuntimeFactsV1
    execution_binding: PackagedExecutionBindingV1
    stage_receipt: ModalPackagedStageReceipt
    workload_bytes: bytes = field(repr=False)
    artifact_policy: ArtifactPolicy
    environment: tuple[tuple[str, str], ...]
    key_ref: str

    def __post_init__(self) -> None:
        command = parse_exact_command(self.submit_command_bytes)
        if type(command) is not SubmitCommandV2:
            raise ValueError("packaged dispatch requires an exact submit command")
        if type(self.runtime_release) is not PackagedTrainingRuntimeReleaseV1:
            raise TypeError("exact packaged runtime release required")
        if type(self.provider_binding) is not ProviderRuntimeBindingV1:
            raise TypeError("exact provider runtime binding required")
        if type(self.provider_facts) is not ModalPackagedRuntimeFactsV1:
            raise TypeError("exact Modal packaged runtime facts required")
        if type(self.execution_binding) is not PackagedExecutionBindingV1:
            raise TypeError("exact packaged execution binding required")
        if type(self.stage_receipt) is not ModalPackagedStageReceipt:
            raise TypeError("exact packaged stage receipt required")
        if type(self.artifact_policy) is not ArtifactPolicy:
            raise TypeError("exact artifact policy required")
        if type(self.workload_bytes) is not bytes or not self.workload_bytes:
            raise TypeError("exact canonical workload bytes required")
        safe_ref(self.key_ref, "key_ref")
        values = dict(self.environment)
        allowed = {
            "PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES", "LANG", "LC_ALL",
        }
        if len(values) != len(self.environment) or not set(values) <= allowed:
            raise ValueError("packaged worker environment is not closed")
        if any(type(key) is not str or type(value) is not str or "\0" in value
               for key, value in self.environment):
            raise ValueError("packaged worker environment is invalid")
        object.__setattr__(self, "environment", tuple(sorted(values.items())))
        try:
            compiled_workload = _packaged_workload(self.workload_bytes)
        except ValueError:
            raise ValueError("packaged dispatch has a cross-binding mismatch") from None
        self.execution_binding.validate_bindings(
            self.runtime_release, self.provider_binding,
        )
        self.provider_facts.validate_release(self.runtime_release)
        predecessor = command.stage_predecessor
        expected_stage_path = operation_path(
            predecessor.stage_effect_id, "input", "prepared",
            self.execution_binding.prepared_input_content_digest, "payload.bin",
        )
        expected_stage_entry = provider_entry_identity(
            self.provider_facts.artifact_volume_id,
            expected_stage_path,
            self.execution_binding.prepared_input_size_bytes,
        )
        if (
            self.provider_binding
            != self.provider_facts.build_provider_binding(self.runtime_release)
            or compiled_workload.fingerprint != self.execution_binding.workload_digest
            or packaged_artifact_policy_digest(self.artifact_policy)
            != self.execution_binding.artifact_policy_digest
            or command.preparation.source_digest
            != self.execution_binding.binding_digest
            or command.preparation.run_id != self.execution_binding.run_ref
            or predecessor.stage_effect_id != self.stage_receipt.stage_effect_id
            or self.stage_receipt.execution_binding_digest
            != self.execution_binding.binding_digest
            or self.stage_receipt.artifact_volume_id
            != self.provider_facts.artifact_volume_id
            or self.stage_receipt.path != expected_stage_path
            or self.stage_receipt.size_bytes
            != self.execution_binding.prepared_input_size_bytes
            or self.stage_receipt.content_digest
            != self.execution_binding.prepared_input_content_digest
            or self.stage_receipt.provider_entry_id != expected_stage_entry
        ):
            raise ValueError("packaged dispatch has a cross-binding mismatch")

    @property
    def submit_command(self) -> SubmitCommandV2:
        command = parse_exact_command(self.submit_command_bytes)
        assert type(command) is SubmitCommandV2
        return command

    def unsigned_dict(self) -> dict[str, object]:
        return {
            "schema_version": MODAL_PACKAGED_DISPATCH_SCHEMA,
            "submit_command": self.submit_command.to_dict(),
            "runtime_release": self.runtime_release.to_dict(),
            "provider_binding": self.provider_binding.to_dict(),
            "provider_facts": self.provider_facts.to_dict(),
            "execution_binding": self.execution_binding.to_dict(),
            "stage_receipt": self.stage_receipt.to_dict(),
            "workload_b64": base64.urlsafe_b64encode(self.workload_bytes).decode("ascii"),
            "artifact_policy": _policy_dict(self.artifact_policy),
            "environment": dict(self.environment),
            "key_ref": self.key_ref,
        }

    @property
    def unsigned_bytes(self) -> bytes:
        # Reparse at every projection boundary so a caller cannot smuggle a
        # non-finite or non-canonical numeric workload through retained bytes.
        _packaged_workload(self.workload_bytes)
        return _canonical(self.unsigned_dict())


def build_modal_packaged_dispatch(
    binding: ModalPackagedCommandBinding,
    stage_receipt: ModalPackagedStageReceipt,
    workload_bytes: bytes,
    artifact_policy: ArtifactPolicy,
    signer: ModalPackagedDispatchSigner,
    *,
    key_ref: str,
    environment: tuple[tuple[str, str], ...] = (),
) -> bytes:
    if type(binding) is not ModalPackagedCommandBinding:
        raise TypeError("exact Modal packaged command binding required")
    if not hasattr(signer, "sign"):
        raise TypeError("packaged dispatch signer required")
    dispatch = ModalPackagedDispatch(
        binding.command_bytes, binding.runtime_release, binding.provider_binding,
        binding.provider_facts, binding.execution_binding, stage_receipt,
        workload_bytes, artifact_policy, environment, key_ref,
    )
    unsigned = dispatch.unsigned_bytes
    try:
        tag = signer.sign(_PURPOSE, unsigned, dispatch.key_ref)
    except Exception:
        raise ValueError("packaged dispatch authentication unavailable") from None
    if type(tag) is not bytes or not tag or len(tag) > 128:
        raise ValueError("packaged dispatch tag is invalid")
    payload = _canonical({
        **dispatch.unsigned_dict(),
        "authentication": {
            "purpose": _PURPOSE,
            "tag": base64.urlsafe_b64encode(tag).decode("ascii"),
        },
    })
    return payload


def parse_modal_packaged_dispatch(
    payload: bytes,
    verifier: ModalPackagedDispatchVerifier,
) -> ModalPackagedDispatch:
    if type(payload) is not bytes or not 0 < len(payload) <= MAX_MODAL_PACKAGED_DISPATCH_BYTES:
        raise ValueError("packaged dispatch bytes are invalid")
    if not hasattr(verifier, "verify"):
        raise TypeError("packaged dispatch verifier required")
    document = _object(payload)
    fields = {
        "schema_version", "submit_command", "runtime_release", "provider_binding",
        "provider_facts", "execution_binding", "stage_receipt", "workload_b64",
        "artifact_policy", "environment", "key_ref", "authentication",
    }
    if set(document) != fields or document.get("schema_version") != MODAL_PACKAGED_DISPATCH_SCHEMA:
        raise ValueError("packaged dispatch has unknown fields")
    authentication = document.pop("authentication")
    if type(authentication) is not dict or set(authentication) != {"purpose", "tag"} \
            or authentication.get("purpose") != _PURPOSE:
        raise ValueError("packaged dispatch authentication is invalid")
    try:
        tag_text = authentication["tag"]
        if type(tag_text) is not str:
            raise ValueError
        tag = base64.b64decode(tag_text, altchars=b"-_", validate=True)
        unsigned = _canonical(document)
        valid = verifier.verify(_PURPOSE, unsigned, tag, document["key_ref"])
    except Exception:
        raise ValueError("packaged dispatch authentication unavailable") from None
    if valid is not True:
        raise ValueError("packaged dispatch authentication failed")
    policy = document["artifact_policy"]
    if type(policy) is not dict or set(policy) != {"required_kinds", "retain_checkpoints"} \
            or type(policy["required_kinds"]) is not list:
        raise ValueError("packaged artifact policy is invalid")
    environment = document["environment"]
    if type(environment) is not dict:
        raise ValueError("packaged environment is invalid")
    workload_b64 = document["workload_b64"]
    try:
        if type(workload_b64) is not str:
            raise ValueError
        workload_bytes = base64.b64decode(workload_b64, altchars=b"-_", validate=True)
    except Exception:
        raise ValueError("packaged workload encoding is invalid") from None
    result = ModalPackagedDispatch(
        canonical_bytes(document["submit_command"]),
        PackagedTrainingRuntimeReleaseV1.from_dict(document["runtime_release"]),
        ProviderRuntimeBindingV1.from_dict(document["provider_binding"]),
        ModalPackagedRuntimeFactsV1.from_dict(document["provider_facts"]),
        PackagedExecutionBindingV1.from_dict(document["execution_binding"]),
        ModalPackagedStageReceipt.from_dict(document["stage_receipt"]),
        workload_bytes,
        ArtifactPolicy(tuple(policy["required_kinds"]), policy["retain_checkpoints"]),
        tuple(environment.items()),
        document["key_ref"],
    )
    _packaged_workload(result.workload_bytes)
    if _canonical({
        **result.unsigned_dict(),
        "authentication": authentication,
    }) != payload:
        raise ValueError("packaged dispatch is not canonical")
    return result


__all__ = [
    "MAX_MODAL_PACKAGED_DISPATCH_BYTES",
    "MODAL_PACKAGED_DISPATCH_SCHEMA",
    "ModalPackagedDispatch",
    "ModalPackagedDispatchSigner",
    "ModalPackagedDispatchVerifier",
    "build_modal_packaged_dispatch",
    "parse_modal_packaged_dispatch",
]
