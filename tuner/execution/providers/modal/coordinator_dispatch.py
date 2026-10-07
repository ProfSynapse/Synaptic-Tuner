"""Canonical one-argument dispatch codec for a Modal worker invocation."""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import json

from tuner.execution.foundation_v2.canonical import canonical_bytes, parse_canonical_object
from tuner.execution.foundation_v2.commands import SubmitCommandV2, parse_exact_command

from .contracts import BoundsPolicyV1, strict_int
from .coordinator_binding import ModalCommandBinding
from .coordinator_wire import ModalWorkerLaunchExpectation


_SCHEMA = "synaptic.modal-worker-dispatch/v1"
_MAX_DISPATCH_BYTES = 192 * 1024
_FIELDS = {"schema_version", "expectation", "launch_claim", "launch_claim_tag_base64"}
_EXPECTATION_FIELDS = frozenset(ModalWorkerLaunchExpectation.__dataclass_fields__)
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
_FORBIDDEN_KEYS = {
    "token", "token_id", "token_secret", "password", "api_key", "credential",
    "credentials", "secret_value", "private_key",
}


def _object(raw: bytes, maximum: int, name: str) -> dict[str, object]:
    if type(raw) is not bytes or not raw or len(raw) > maximum:
        raise ValueError(f"{name} exceeds bound")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is invalid JSON") from exc
    if type(value) is not dict or _encode(value) != raw:
        raise ValueError(f"{name} is not canonical")
    return value


def _encode(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("dispatch contains noncanonical JSON") from exc


def _reject_secret_fields(value: object) -> None:
    if type(value) is dict:
        for key, member in value.items():
            if type(key) is not str or key.casefold() in _FORBIDDEN_KEYS:
                raise ValueError("dispatch contains a forbidden credential field")
            _reject_secret_fields(member)
    elif type(value) is list:
        for member in value:
            _reject_secret_fields(member)


def _expectation_document(value: ModalWorkerLaunchExpectation) -> dict[str, object]:
    if type(value) is not ModalWorkerLaunchExpectation:
        raise TypeError("exact worker launch expectation required")
    document = {name: getattr(value, name) for name in value.__dataclass_fields__}
    document["submit_command_bytes"] = parse_canonical_object(
        value.submit_command_bytes, name="submit command",
    )
    document["deployment_bytes"] = parse_canonical_object(
        value.deployment_bytes, name="deployment",
    )
    return document


def _expectation(value: object) -> ModalWorkerLaunchExpectation:
    if type(value) is not dict or frozenset(value) != _EXPECTATION_FIELDS:
        raise ValueError("dispatch expectation fields are invalid")
    document = dict(value)
    document["submit_command_bytes"] = canonical_bytes(document["submit_command_bytes"])
    document["deployment_bytes"] = canonical_bytes(document["deployment_bytes"])
    return ModalWorkerLaunchExpectation(**document)


def _check_projection(
    expectation: ModalWorkerLaunchExpectation, launch: dict[str, object],
    bounds: BoundsPolicyV1,
) -> None:
    if set(launch) != _LAUNCH_FIELDS or launch.get("schema_version") != "synaptic.modal-launch-claim/v1":
        raise ValueError("launch claim fields are invalid")
    binding_doc = launch.get("submit_binding")
    if type(binding_doc) is not dict or set(binding_doc) != {
        "command", "preparation_snapshot", "deployment",
    }:
        raise ValueError("launch submit binding is invalid")
    binding = ModalCommandBinding(
        canonical_bytes(binding_doc["command"]),
        canonical_bytes(binding_doc["preparation_snapshot"]),
        canonical_bytes(binding_doc["deployment"]),
    )
    command = parse_exact_command(binding.command_bytes)
    if type(command) is not SubmitCommandV2:
        raise ValueError("launch requires exact submit command")
    prep = command.preparation
    selection = binding.deployment.selection
    snapshot = parse_canonical_object(binding.preparation_snapshot, name="preparation snapshot")
    profile_volumes = snapshot["configuration"]["profile"]["volumes"]
    if (launch["submit_binding_digest"], launch["submit_command"],
            launch["submit_command_digest"], launch["submit_effect_id"],
            launch["submit_invocation_nonce"], launch["stage_predecessor"]) != (
            binding.authenticated_binding_digest, command.to_dict(), command.digest,
            command.operation.effect.effect_id, command.operation.invocation_nonce,
            command.stage_predecessor.to_dict()):
        raise ValueError("dispatch launch command projection differs")
    if (launch["configured_control_volume_ref"],
            launch["configured_artifact_volume_ref"]) != (
            profile_volumes["control_ref"], profile_volumes["artifact_ref"]):
        raise ValueError("dispatch launch Volume profile differs")
    projection = (
        binding.command_bytes, binding.deployment_bytes,
        prep.provider.provider_id, prep.provider.profile_ref,
        prep.scope.account_ref, prep.scope.namespace_ref,
        selection.app_name, selection.function_name,
        launch.get("control_volume_id"), launch.get("artifact_volume_id"),
        launch.get("configured_control_volume_ref"),
        launch.get("configured_artifact_volume_ref"), launch.get("key_ref"),
        launch.get("stage_claim_sha256"), launch.get("stage_bundle_sha256"),
        launch.get("stage_bundle_size"), command.executor.executor_id,
        command.executor.implementation_version,
    )
    expected = tuple(getattr(expectation, name) for name in expectation.__dataclass_fields__)
    if projection != expected:
        raise ValueError("dispatch launch claim differs from expectation")
    strict_int(
        launch.get("stage_bundle_size"), "stage_bundle_size", minimum=1,
        maximum=bounds.max_bundle_bytes,
    )


@dataclass(frozen=True, slots=True, init=False)
class ModalWorkerDispatch:
    _canonical_bytes: bytes

    @classmethod
    def _from_document(cls, document: dict[str, object]) -> "ModalWorkerDispatch":
        value = object.__new__(cls)
        object.__setattr__(value, "_canonical_bytes", _encode(document))
        return value

    @property
    def canonical_bytes(self) -> bytes:
        return bytes(self._canonical_bytes)

    @property
    def expectation(self) -> ModalWorkerLaunchExpectation:
        return _expectation(json.loads(self._canonical_bytes)["expectation"])

    @property
    def launch_claim(self) -> bytes:
        return canonical_bytes(json.loads(self._canonical_bytes)["launch_claim"])

    @property
    def launch_claim_tag(self) -> bytes:
        return base64.b64decode(
            json.loads(self._canonical_bytes)["launch_claim_tag_base64"], validate=True,
        )


def build_modal_worker_dispatch(
    expectation: ModalWorkerLaunchExpectation, launch_claim: bytes,
    launch_claim_tag: bytes, *, bounds: BoundsPolicyV1 = BoundsPolicyV1(),
) -> ModalWorkerDispatch:
    if type(expectation) is not ModalWorkerLaunchExpectation:
        raise TypeError("exact worker launch expectation required")
    expectation_document = _expectation_document(expectation)
    rebuilt_expectation = _expectation(
        json.loads(_encode(expectation_document).decode("utf-8")),
    )
    if rebuilt_expectation != expectation:
        raise ValueError("worker launch expectation reconstruction differs")
    launch = _object(launch_claim, bounds.max_control_bytes, "launch claim")
    if type(launch_claim_tag) is not bytes or not launch_claim_tag or len(launch_claim_tag) > 128:
        raise ValueError("launch claim tag is invalid")
    if rebuilt_expectation.stage_bundle_size > bounds.max_bundle_bytes:
        raise ValueError("stage bundle size exceeds bound")
    _check_projection(rebuilt_expectation, launch, bounds)
    document = {
        "schema_version": _SCHEMA,
        "expectation": expectation_document,
        "launch_claim": launch,
        "launch_claim_tag_base64": base64.b64encode(launch_claim_tag).decode("ascii"),
    }
    _reject_secret_fields(document)
    encoded = _encode(document)
    if len(encoded) > _MAX_DISPATCH_BYTES:
        raise ValueError("worker dispatch exceeds bound")
    return ModalWorkerDispatch._from_document(document)


def parse_modal_worker_dispatch(
    payload: bytes, *, bounds: BoundsPolicyV1 = BoundsPolicyV1(),
) -> ModalWorkerDispatch:
    document = _object(payload, _MAX_DISPATCH_BYTES, "worker dispatch")
    if set(document) != _FIELDS or document.get("schema_version") != _SCHEMA:
        raise ValueError("worker dispatch fields are invalid")
    _reject_secret_fields(document)
    expectation = _expectation(document["expectation"])
    launch = document["launch_claim"]
    if type(launch) is not dict:
        raise ValueError("launch claim must be an exact object")
    tag_text = document["launch_claim_tag_base64"]
    if type(tag_text) is not str:
        raise ValueError("launch claim tag is invalid")
    try:
        tag = base64.b64decode(tag_text, validate=True)
    except (ValueError, binascii.Error):
        raise ValueError("launch claim tag is invalid") from None
    if not tag or len(tag) > 128 or base64.b64encode(tag).decode("ascii") != tag_text:
        raise ValueError("launch claim tag is invalid")
    if expectation.stage_bundle_size > bounds.max_bundle_bytes:
        raise ValueError("stage bundle size exceeds bound")
    _check_projection(expectation, launch, bounds)
    rebuilt = build_modal_worker_dispatch(
        expectation, canonical_bytes(launch), tag, bounds=bounds,
    )
    if rebuilt.canonical_bytes != payload:
        raise ValueError("worker dispatch reconstruction differs")
    return rebuilt


__all__: list[str] = []
