"""Pure, closed contracts for at-most-once HF source provisioning."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator, FormatChecker

from tuner.core.exceptions import CloudProviderError


SCHEMA_VERSION = "synaptic-hf-provisioning-claim/v1"
_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "schemas"
    / "synaptic-hf-provisioning-claim-v1.schema.json"
)


class HFProvisioningState(str, Enum):
    CLAIMED = "CLAIMED"
    SUCCEEDED = "SUCCEEDED"
    AMBIGUOUS = "AMBIGUOUS"


HF_PROVISIONING_AMBIGUITY_EFFECTS = {
    "CREDENTIAL_REJECTED": False,
    "LOCAL_POSTCLAIM_FAILURE": False,
    "PROVIDER_OUTCOME_AMBIGUOUS": True,
    "INTERRUPTED_AFTER_CLAIM": True,
    "RECOVERY_EVIDENCE_INVALID": True,
}


def canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CloudProviderError("HF provisioning event is not canonical JSON data.") from exc


def _canonical_utc(value: datetime | str | None) -> str:
    if value is None:
        parsed = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.removesuffix("Z") + ("+00:00" if value.endswith("Z") else ""))
        except ValueError as exc:
            raise CloudProviderError("HF provisioning event timestamp is invalid.") from exc
    else:
        raise CloudProviderError("HF provisioning event timestamp is invalid.")
    if parsed.tzinfo is None:
        raise CloudProviderError("HF provisioning event timestamp must include a timezone.")
    parsed = parsed.astimezone(timezone.utc)
    return parsed.isoformat(
        timespec="microseconds" if parsed.microsecond else "seconds"
    ).replace("+00:00", "Z")


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CloudProviderError(f"HF provisioning {label} must be an object.")
    return value


def _ref(uri: str, sha256: str) -> dict[str, str]:
    return {"uri": uri, "sha256": sha256}


def _identity_from_descriptor(
    *,
    experiment_id: str,
    descriptor_uri: str,
    descriptor_sha256: str,
    descriptor: Mapping[str, object],
) -> dict[str, object]:
    source_lock = _mapping(descriptor.get("source_lock"), label="descriptor source_lock")
    capsule = _mapping(descriptor.get("capsule"), label="descriptor capsule")
    manifest = _mapping(capsule.get("manifest"), label="descriptor capsule manifest")
    checkout_policy = _mapping(
        descriptor.get("checkout_policy"), label="descriptor checkout_policy"
    )
    bundle = _mapping(descriptor.get("bundle"), label="descriptor bundle")
    volume = _mapping(descriptor.get("volume"), label="descriptor volume")
    if descriptor.get("run_id") != experiment_id:
        raise CloudProviderError("HF provisioning descriptor belongs to another experiment.")
    return {
        "experiment_id": experiment_id,
        "run_id": descriptor.get("run_id"),
        "descriptor": _ref(descriptor_uri, descriptor_sha256),
        "source_lock": _ref(str(source_lock.get("uri")), str(source_lock.get("sha256"))),
        "volume": {
            "source": volume.get("source"),
            "path": volume.get("path"),
            "type": volume.get("type"),
            "read_only": volume.get("read_only"),
        },
        "bundle_sha256": bundle.get("content_sha256"),
        "capsule_manifest_sha256": manifest.get("sha256"),
        "checkout_policy_sha256": checkout_policy.get("sha256"),
    }


def _event_id(document: Mapping[str, object]) -> str:
    body = {key: value for key, value in document.items() if key != "event_id"}
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def build_hf_provisioning_claim(
    *,
    experiment_id: str,
    descriptor_uri: str,
    descriptor_sha256: str,
    descriptor: Mapping[str, object],
    actor: str,
    authority: str = "operator",
    occurred_at: datetime | str | None = None,
) -> dict[str, object]:
    """Build a deterministic identity-bound CLAIMED event."""

    document: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "event_id": "0" * 64,
        **_identity_from_descriptor(
            experiment_id=experiment_id,
            descriptor_uri=descriptor_uri,
            descriptor_sha256=descriptor_sha256,
            descriptor=descriptor,
        ),
        "state": HFProvisioningState.CLAIMED.value,
        "sequence": 1,
        "authority": authority,
        "actor": actor,
        "occurred_at": _canonical_utc(occurred_at),
        "previous_event": None,
        "evidence": None,
        "reason_code": None,
        "provider_effect_possible": False,
    }
    document["event_id"] = _event_id(document)
    return validate_hf_provisioning_event(document)


def _build_terminal_event(
    claim: Mapping[str, object],
    *,
    state: HFProvisioningState,
    claim_uri: str,
    claim_sha256: str,
    evidence_uri: str | None,
    evidence_sha256: str | None,
    reason_code: str | None,
    occurred_at: datetime | str | None,
) -> dict[str, object]:
    accepted = validate_hf_provisioning_event(claim)
    if accepted["state"] != HFProvisioningState.CLAIMED.value:
        raise CloudProviderError("HF provisioning terminal event requires a CLAIMED predecessor.")
    actual_claim_sha256 = hashlib.sha256(canonical_json_bytes(accepted)).hexdigest()
    if claim_sha256 != actual_claim_sha256:
        raise CloudProviderError("HF provisioning terminal event claim digest is invalid.")
    document = {
        **accepted,
        "event_id": "0" * 64,
        "state": state.value,
        "sequence": 2,
        "occurred_at": _canonical_utc(occurred_at),
        "previous_event": _ref(claim_uri, claim_sha256),
        "evidence": (
            _ref(evidence_uri, evidence_sha256)
            if evidence_uri is not None and evidence_sha256 is not None
            else None
        ),
        "reason_code": reason_code,
        "provider_effect_possible": (
            True
            if state is HFProvisioningState.SUCCEEDED
            else HF_PROVISIONING_AMBIGUITY_EFFECTS.get(str(reason_code))
        ),
    }
    document["event_id"] = _event_id(document)
    return validate_hf_provisioning_event(document, previous_event=accepted)


def build_hf_provisioning_succeeded_event(
    claim: Mapping[str, object],
    *,
    claim_uri: str,
    claim_sha256: str,
    evidence_uri: str,
    evidence_sha256: str,
    occurred_at: datetime | str | None = None,
) -> dict[str, object]:
    return _build_terminal_event(
        claim,
        state=HFProvisioningState.SUCCEEDED,
        claim_uri=claim_uri,
        claim_sha256=claim_sha256,
        evidence_uri=evidence_uri,
        evidence_sha256=evidence_sha256,
        reason_code=None,
        occurred_at=occurred_at,
    )


def build_hf_provisioning_ambiguous_event(
    claim: Mapping[str, object],
    *,
    claim_uri: str,
    claim_sha256: str,
    reason_code: str,
    occurred_at: datetime | str | None = None,
) -> dict[str, object]:
    return _build_terminal_event(
        claim,
        state=HFProvisioningState.AMBIGUOUS,
        claim_uri=claim_uri,
        claim_sha256=claim_sha256,
        evidence_uri=None,
        evidence_sha256=None,
        reason_code=reason_code,
        occurred_at=occurred_at,
    )


def validate_hf_provisioning_event(
    value: Mapping[str, object],
    *,
    previous_event: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate closure, content identity, and the optional state transition."""

    if not isinstance(value, Mapping):
        raise CloudProviderError("HF provisioning event must be an object.")
    try:
        document = json.loads(canonical_json_bytes(value))
        schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(document)
    except Exception as exc:
        if isinstance(exc, CloudProviderError):
            raise
        raise CloudProviderError("HF provisioning event does not match its exact schema.") from exc
    if document["event_id"] != _event_id(document):
        raise CloudProviderError("HF provisioning event ID does not match its canonical document.")
    occurred = _canonical_utc(str(document["occurred_at"]))
    if occurred != document["occurred_at"]:
        raise CloudProviderError("HF provisioning event timestamp is not canonical UTC.")
    state = HFProvisioningState(str(document["state"]))
    if document["run_id"] != document["experiment_id"]:
        raise CloudProviderError("HF provisioning run ID must equal its experiment ID.")
    expected_effect = (
        False
        if state is HFProvisioningState.CLAIMED
        else True
        if state is HFProvisioningState.SUCCEEDED
        else HF_PROVISIONING_AMBIGUITY_EFFECTS.get(str(document["reason_code"]))
    )
    if expected_effect is None or document["provider_effect_possible"] is not expected_effect:
        raise CloudProviderError("HF provisioning reason/effect mapping is invalid.")
    if previous_event is None:
        if state is not HFProvisioningState.CLAIMED:
            raise CloudProviderError("HF provisioning terminal event requires its CLAIMED predecessor.")
    else:
        previous = validate_hf_provisioning_event(previous_event)
        if state not in {HFProvisioningState.SUCCEEDED, HFProvisioningState.AMBIGUOUS}:
            raise CloudProviderError("HF provisioning transition must be terminal.")
        previous_reference = document["previous_event"]
        if not isinstance(previous_reference, dict) or previous_reference["sha256"] != hashlib.sha256(
            canonical_json_bytes(previous)
        ).hexdigest():
            raise CloudProviderError(
                "HF provisioning terminal event predecessor digest is invalid."
            )
        immutable = {
            "experiment_id", "run_id", "descriptor", "source_lock", "volume",
            "bundle_sha256", "capsule_manifest_sha256", "checkout_policy_sha256",
            "authority", "actor",
        }
        if any(document[key] != previous[key] for key in immutable):
            raise CloudProviderError("HF provisioning terminal event changed immutable claim identity.")
        if datetime.fromisoformat(str(document["occurred_at"]).replace("Z", "+00:00")) < datetime.fromisoformat(str(previous["occurred_at"]).replace("Z", "+00:00")):
            raise CloudProviderError("HF provisioning terminal event predates its claim.")
    return document


__all__ = [
    "HFProvisioningState",
    "HF_PROVISIONING_AMBIGUITY_EFFECTS",
    "build_hf_provisioning_ambiguous_event",
    "build_hf_provisioning_claim",
    "build_hf_provisioning_succeeded_event",
    "canonical_json_bytes",
    "validate_hf_provisioning_event",
]
