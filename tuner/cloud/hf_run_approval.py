"""Provider-free exact-run approval and immutable submission-event contracts.

This module is deliberately pure: it validates and authenticates canonical JSON
documents but never imports a provider SDK, resolves a credential, or performs
network, filesystem, tracking, or submission effects.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import Mapping
from urllib.parse import unquote

from jsonschema import Draft202012Validator, FormatChecker

from tuner.cloud.hf_bootstrap_smoke import (
    WORKLOAD_KIND as FIXED_WORKLOAD_KIND,
    canonical_workload_bytes,
    workload_sha256,
)
from tuner.core.exceptions import CloudProviderError


APPROVAL_SCHEMA_VERSION = "synaptic-hf-run-approval/v1"
CLAIM_SCHEMA_VERSION = "synaptic-hf-submission-claim/v1"
PROVIDER = "hf_jobs"
ACTION = "hf.bootstrap.verify"
WORKLOAD_KIND = "bootstrap_verification"
RUNTIME = "python:3.12"
IMAGE = "python:3.12"
PROVIDER_TIMEOUT_SECONDS = 600
CANCEL_AFTER_SECONDS = 720
OBSERVE_UNTIL_SECONDS = 900
MAXIMUM_PROJECTED_COST_USD = Decimal("0.01")
HARD_CAP_USD = Decimal("2")

_SCHEMA_ROOT = Path(__file__).resolve().parents[2] / "schemas"
_APPROVAL_SCHEMA = _SCHEMA_ROOT / "synaptic-hf-run-approval-v1.schema.json"
_CLAIM_SCHEMA = _SCHEMA_ROOT / "synaptic-hf-submission-claim-v1.schema.json"
_EXPECTED_WORKLOAD = json.loads(canonical_workload_bytes().decode("ascii"))
_EXPECTED_WORKLOAD_SHA256 = workload_sha256()
_SECRET_VALUE_RE = re.compile(
    r"(?:hf_[A-Za-z0-9]{8,}|sk-[A-Za-z0-9_-]{8,}|ghp_[A-Za-z0-9]{8,}|"
    r"github_pat_[A-Za-z0-9_]{8,}|xox[baprs]-[A-Za-z0-9-]{8,}|"
    r"AKIA[A-Z0-9]{16}|Bearer\s+[A-Za-z0-9._~+/-]{8,}|"
    r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,})",
    re.IGNORECASE,
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?:^|[?&;,\s])(?:token|secret|password|passwd|api[_-]?key|authorization|"
    r"credential|hf[_-]?token)\s*(?:=|:|%3[dD])\s*[^\s&;,]+",
    re.IGNORECASE,
)


class HFSubmissionState(str, Enum):
    """Tracking projection and immutable provider-call event states."""

    APPROVED = "APPROVED"
    SUBMITTING = "SUBMITTING"
    SUBMITTED = "SUBMITTED"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class HFRunApproval:
    """Validated canonical exact-run approval."""

    document: Mapping[str, object]
    approval_id: str
    authorization_id: str

    def to_dict(self) -> dict[str, object]:
        return copy.deepcopy(dict(self.document))


@dataclass(frozen=True)
class HFSubmissionClaim:
    """Validated immutable submission claim event."""

    document: Mapping[str, object]
    event_id: str
    authorization_id: str
    state: HFSubmissionState

    def to_dict(self) -> dict[str, object]:
        return copy.deepcopy(dict(self.document))


def canonical_json_bytes(value: object) -> bytes:
    """Encode one canonical, ASCII JSON representation."""

    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise CloudProviderError("HF run authorization is not canonical JSON data.") from exc


def approval_document_sha256(value: Mapping[str, object] | HFRunApproval) -> str:
    """Digest the complete persisted approval document."""

    return hashlib.sha256(canonical_json_bytes(_document(value))).hexdigest()


def submission_event_sha256(value: Mapping[str, object] | HFSubmissionClaim) -> str:
    """Digest the complete persisted submission-event document."""

    return hashlib.sha256(canonical_json_bytes(_document(value))).hexdigest()


def build_hf_run_approval(
    *,
    experiment_id: str,
    run_id: str,
    descriptor_uri: str,
    descriptor_sha256: str,
    provisioning_evidence_uri: str,
    provisioning_evidence_sha256: str,
    source_lock_uri: str,
    source_lock_sha256: str,
    bundle_sha256: str,
    capsule_manifest_sha256: str,
    checkout_policy_sha256: str,
    hardware_flavor: str,
    user_authorization_reference: str,
    issued_at: str | datetime,
    expires_at: str | datetime,
    hourly_price_usd: str | int | Decimal,
    projected_cost_usd: str | int | Decimal,
    quoted_at: str | datetime,
) -> HFRunApproval:
    """Build Joseph's one-shot, bootstrap-only HF authorization envelope."""

    document: dict[str, object] = {
        "schema_version": APPROVAL_SCHEMA_VERSION,
        "approval_id": "0" * 64,
        "authorization_id": "0" * 64,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "provider": PROVIDER,
        "action": ACTION,
        "descriptor": {"uri": descriptor_uri, "sha256": descriptor_sha256},
        "provisioning_evidence": {
            "uri": provisioning_evidence_uri,
            "sha256": provisioning_evidence_sha256,
        },
        "source_lock": {"uri": source_lock_uri, "sha256": source_lock_sha256},
        "bundle_sha256": bundle_sha256,
        "capsule_manifest_sha256": capsule_manifest_sha256,
        "checkout_policy_sha256": checkout_policy_sha256,
        "workload": {
            "kind": FIXED_WORKLOAD_KIND,
            "sha256": _EXPECTED_WORKLOAD_SHA256,
            "declaration": copy.deepcopy(_EXPECTED_WORKLOAD),
        },
        "execution": {
            "runtime": RUNTIME,
            "image": IMAGE,
            "hardware": {"flavor": hardware_flavor},
            "bootstrap_only": True,
            "training": False,
            "publication": False,
            "ssh": False,
            "ports": [],
            "maximum_submissions": 1,
            "retry_count": 0,
        },
        "timeouts": {
            "provider_seconds": PROVIDER_TIMEOUT_SECONDS,
            "cancel_after_seconds": CANCEL_AFTER_SECONDS,
            "observe_until_seconds": OBSERVE_UNTIL_SECONDS,
        },
        "cost": {
            "currency": "USD",
            "maximum_projected_cost_usd": _decimal_text(MAXIMUM_PROJECTED_COST_USD),
            "hard_cap_usd": "2.00",
            "quote": {
                "hourly_price_usd": _decimal_text(hourly_price_usd),
                "projected_cost_usd": _decimal_text(projected_cost_usd),
                "billing_increment_seconds": 60,
                "quoted_at": _timestamp_text(quoted_at),
            },
        },
        "authorization": {
            "subject": "joseph-rosenbaum",
            "reference": user_authorization_reference,
        },
        "secrets": [{"provider": "env", "name": "HF_TOKEN"}],
        "issued_at": _timestamp_text(issued_at),
        "expires_at": _timestamp_text(expires_at),
    }
    document["authorization_id"] = _authorization_id(document)
    document["approval_id"] = _identity(document, omitted="approval_id")
    return validate_hf_run_approval(document)


def validate_hf_run_approval(
    value: Mapping[str, object] | HFRunApproval,
    *,
    at: str | datetime | None = None,
) -> HFRunApproval:
    """Validate closed shape, identities, envelope, prices, and optional liveness."""

    document = _document(value)
    _validate_schema(document, _APPROVAL_SCHEMA, "run approval")
    _reject_secret_values(document)
    workload = _mapping(document["workload"])
    if (
        workload["kind"] != FIXED_WORKLOAD_KIND
        or workload["sha256"] != _EXPECTED_WORKLOAD_SHA256
        or canonical_json_bytes(workload["declaration"]) != canonical_workload_bytes()
    ):
        raise CloudProviderError("HF run approval does not bind the fixed bootstrap workload.")
    if document["authorization_id"] != _authorization_id(document):
        raise CloudProviderError("HF run approval authorization_id does not match its exact scope.")
    if document["approval_id"] != _identity(document, omitted="approval_id"):
        raise CloudProviderError("HF run approval approval_id does not match its canonical document.")

    issued = _parse_timestamp(document["issued_at"], label="issued_at")
    expires = _parse_timestamp(document["expires_at"], label="expires_at")
    quote = _mapping(_mapping(document["cost"])["quote"])
    quoted = _parse_timestamp(quote["quoted_at"], label="quoted_at")
    if expires <= issued:
        raise CloudProviderError("HF run approval must expire after it is issued.")
    if quoted > issued:
        raise CloudProviderError("HF run approval price quote cannot postdate issuance.")

    hourly = _decimal(quote["hourly_price_usd"], label="hourly price")
    projected = _decimal(quote["projected_cost_usd"], label="projected cost")
    if hourly > Decimal("0.01"):
        raise CloudProviderError("HF run approval hardware is not CPU Basic or cheaper.")
    if projected > MAXIMUM_PROJECTED_COST_USD or projected > HARD_CAP_USD:
        raise CloudProviderError("HF run approval projected cost exceeds the authorized envelope.")

    if at is not None:
        instant = _parse_timestamp(_timestamp_text(at), label="validation time")
        if instant < issued:
            raise CloudProviderError("HF run approval is not active yet.")
        if instant >= expires:
            raise CloudProviderError("HF run approval has expired.")
    return HFRunApproval(
        document=document,
        approval_id=str(document["approval_id"]),
        authorization_id=str(document["authorization_id"]),
    )


def refresh_hf_run_approval(
    value: Mapping[str, object] | HFRunApproval,
    *,
    issued_at: str | datetime,
    expires_at: str | datetime,
    hourly_price_usd: str | int | Decimal,
    projected_cost_usd: str | int | Decimal,
    quoted_at: str | datetime,
) -> HFRunApproval:
    """Refresh only time/price metadata, preserving exact authorization scope."""

    previous = validate_hf_run_approval(value)
    document = previous.to_dict()
    old_issued = _parse_timestamp(document["issued_at"], label="issued_at")
    old_expires = _parse_timestamp(document["expires_at"], label="expires_at")
    old_quote = _mapping(_mapping(document["cost"])["quote"])
    new_issued_text = _timestamp_text(issued_at)
    new_expires_text = _timestamp_text(expires_at)
    new_quoted_text = _timestamp_text(quoted_at)
    new_issued = _parse_timestamp(new_issued_text, label="issued_at")
    new_expires = _parse_timestamp(new_expires_text, label="expires_at")
    new_quoted = _parse_timestamp(new_quoted_text, label="quoted_at")
    if new_issued < old_issued or new_issued >= old_expires:
        raise CloudProviderError("HF run approval may only be refreshed while still active.")
    if new_expires <= old_expires or new_expires <= new_issued:
        raise CloudProviderError("HF run approval refresh must extend expiry.")
    if new_quoted < _parse_timestamp(old_quote["quoted_at"], label="quoted_at"):
        raise CloudProviderError("HF run approval refresh cannot roll price time backward.")

    new_hourly = _decimal(hourly_price_usd, label="hourly price")
    new_projected = _decimal(projected_cost_usd, label="projected cost")
    if new_hourly > _decimal(old_quote["hourly_price_usd"], label="hourly price"):
        raise CloudProviderError("HF run approval refresh cannot increase hourly price.")
    if new_projected > _decimal(old_quote["projected_cost_usd"], label="projected cost"):
        raise CloudProviderError("HF run approval refresh cannot increase projected cost.")

    quote = _mapping(_mapping(document["cost"])["quote"])
    mutable_quote = dict(quote)
    mutable_quote.update(
        hourly_price_usd=_decimal_text(new_hourly),
        projected_cost_usd=_decimal_text(new_projected),
        quoted_at=new_quoted_text,
    )
    mutable_cost = dict(_mapping(document["cost"]))
    mutable_cost["quote"] = mutable_quote
    document["cost"] = mutable_cost
    document["issued_at"] = new_issued_text
    document["expires_at"] = new_expires_text
    document["approval_id"] = _identity(document, omitted="approval_id")
    refreshed = validate_hf_run_approval(document)
    if refreshed.authorization_id != previous.authorization_id:
        raise CloudProviderError("HF run approval refresh changed authorization scope.")
    return refreshed


def build_hf_submitting_event(
    approval: Mapping[str, object] | HFRunApproval,
    *,
    approval_uri: str,
    occurred_at: str | datetime,
) -> HFSubmissionClaim:
    accepted = validate_hf_run_approval(approval, at=occurred_at)
    document = _event_base(accepted, approval_uri=approval_uri, occurred_at=occurred_at)
    document.update(state=HFSubmissionState.SUBMITTING.value, sequence=1)
    document["event_id"] = _identity(document, omitted="event_id")
    return validate_hf_submission_claim(document, approval=accepted)


def build_hf_submitted_event(
    approval: Mapping[str, object] | HFRunApproval,
    *,
    approval_uri: str,
    previous_event: Mapping[str, object] | HFSubmissionClaim,
    previous_event_uri: str,
    occurred_at: str | datetime,
    provider_namespace: str,
    provider_job_id: str,
) -> HFSubmissionClaim:
    accepted = validate_hf_run_approval(approval)
    previous = validate_hf_submission_claim(previous_event, approval=accepted)
    document = _event_base(accepted, approval_uri=approval_uri, occurred_at=occurred_at)
    document.update(
        state=HFSubmissionState.SUBMITTED.value,
        sequence=2,
        previous_event={"uri": previous_event_uri, "sha256": submission_event_sha256(previous)},
        provider_job={"namespace": provider_namespace, "job_id": provider_job_id},
    )
    document["event_id"] = _identity(document, omitted="event_id")
    return validate_hf_submission_claim(document, approval=accepted, previous_event=previous)


def build_hf_ambiguous_event(
    approval: Mapping[str, object] | HFRunApproval,
    *,
    approval_uri: str,
    previous_event: Mapping[str, object] | HFSubmissionClaim,
    previous_event_uri: str,
    occurred_at: str | datetime,
    reason_code: str,
) -> HFSubmissionClaim:
    accepted = validate_hf_run_approval(approval)
    previous = validate_hf_submission_claim(previous_event, approval=accepted)
    document = _event_base(accepted, approval_uri=approval_uri, occurred_at=occurred_at)
    document.update(
        state=HFSubmissionState.AMBIGUOUS.value,
        sequence=2,
        previous_event={"uri": previous_event_uri, "sha256": submission_event_sha256(previous)},
        reason_code=reason_code,
    )
    document["event_id"] = _identity(document, omitted="event_id")
    return validate_hf_submission_claim(document, approval=accepted, previous_event=previous)


def validate_hf_submission_claim(
    value: Mapping[str, object] | HFSubmissionClaim,
    *,
    approval: Mapping[str, object] | HFRunApproval | None = None,
    previous_event: Mapping[str, object] | HFSubmissionClaim | None = None,
) -> HFSubmissionClaim:
    """Validate one append-only submission event and any supplied predecessors."""

    document = _document(value)
    _validate_schema(document, _CLAIM_SCHEMA, "submission claim event")
    _reject_secret_values(document)
    if document["event_id"] != _identity(document, omitted="event_id"):
        raise CloudProviderError("HF submission event_id does not match its canonical event.")
    state = HFSubmissionState(str(document["state"]))

    accepted_approval: HFRunApproval | None = None
    if approval is not None:
        accepted_approval = validate_hf_run_approval(
            approval,
            at=document["occurred_at"] if state is HFSubmissionState.SUBMITTING else None,
        )
        approval_ref = _mapping(document["approval"])
        expected = {
            "authorization_id": accepted_approval.authorization_id,
            "approval.sha256": approval_document_sha256(accepted_approval),
            "experiment_id": accepted_approval.document["experiment_id"],
            "run_id": accepted_approval.document["run_id"],
        }
        actual = {
            "authorization_id": document["authorization_id"],
            "approval.sha256": approval_ref["sha256"],
            "experiment_id": document["experiment_id"],
            "run_id": document["run_id"],
        }
        mismatch = next((key for key in expected if actual[key] != expected[key]), None)
        if mismatch:
            raise CloudProviderError(f"HF submission event approval binding mismatch: {mismatch}.")

    if state in {HFSubmissionState.SUBMITTED, HFSubmissionState.AMBIGUOUS}:
        if previous_event is not None:
            previous = validate_hf_submission_claim(previous_event, approval=accepted_approval)
            if previous.state is not HFSubmissionState.SUBMITTING:
                raise CloudProviderError("HF terminal submission event must follow SUBMITTING.")
            previous_ref = _mapping(document["previous_event"])
            expected_previous = {
                "sha256": submission_event_sha256(previous),
                "authorization_id": previous.authorization_id,
                "approval": previous.document["approval"],
                "experiment_id": previous.document["experiment_id"],
                "run_id": previous.document["run_id"],
            }
            if previous_ref["sha256"] != expected_previous["sha256"]:
                raise CloudProviderError("HF terminal submission event has the wrong predecessor digest.")
            for field in ("authorization_id", "approval", "experiment_id", "run_id"):
                if document[field] != expected_previous[field]:
                    raise CloudProviderError(
                        f"HF terminal submission event changed predecessor binding: {field}."
                    )
            if _parse_timestamp(document["occurred_at"], label="occurred_at") < _parse_timestamp(
                previous.document["occurred_at"], label="previous occurred_at"
            ):
                raise CloudProviderError("HF terminal submission event predates SUBMITTING.")
    elif previous_event is not None:
        raise CloudProviderError("HF SUBMITTING event cannot have a predecessor.")

    return HFSubmissionClaim(
        document=document,
        event_id=str(document["event_id"]),
        authorization_id=str(document["authorization_id"]),
        state=state,
    )


def _event_base(
    approval: HFRunApproval,
    *,
    approval_uri: str,
    occurred_at: str | datetime,
) -> dict[str, object]:
    return {
        "schema_version": CLAIM_SCHEMA_VERSION,
        "event_id": "0" * 64,
        "authorization_id": approval.authorization_id,
        "approval": {
            "uri": approval_uri,
            "sha256": approval_document_sha256(approval),
        },
        "experiment_id": approval.document["experiment_id"],
        "run_id": approval.document["run_id"],
        "occurred_at": _timestamp_text(occurred_at),
    }


def _authorization_id(document: Mapping[str, object]) -> str:
    cost = _mapping(document["cost"])
    scope = {
        "experiment_id": document["experiment_id"],
        "run_id": document["run_id"],
        "provider": document["provider"],
        "action": document["action"],
        "descriptor": document["descriptor"],
        "provisioning_evidence": document["provisioning_evidence"],
        "source_lock": document["source_lock"],
        "bundle_sha256": document["bundle_sha256"],
        "capsule_manifest_sha256": document["capsule_manifest_sha256"],
        "checkout_policy_sha256": document["checkout_policy_sha256"],
        "workload": document["workload"],
        "execution": document["execution"],
        "timeouts": document["timeouts"],
        "cost_envelope": {
            "currency": cost["currency"],
            "maximum_projected_cost_usd": cost["maximum_projected_cost_usd"],
            "hard_cap_usd": cost["hard_cap_usd"],
        },
        "authorization": document["authorization"],
        "secrets": document["secrets"],
    }
    return hashlib.sha256(canonical_json_bytes(scope)).hexdigest()


def _identity(document: Mapping[str, object], *, omitted: str) -> str:
    body = {key: value for key, value in document.items() if key != omitted}
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def _document(value: Mapping[str, object] | HFRunApproval | HFSubmissionClaim) -> dict[str, object]:
    if isinstance(value, (HFRunApproval, HFSubmissionClaim)):
        value = value.document
    if not isinstance(value, Mapping):
        raise CloudProviderError("HF run authorization must be an object.")
    try:
        return copy.deepcopy(dict(value))
    except Exception as exc:
        raise CloudProviderError("HF run authorization contains unsupported values.") from exc


def _validate_schema(value: object, path: Path, label: str) -> None:
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
        errors = sorted(
            Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(value),
            key=lambda item: tuple(str(part) for part in item.absolute_path),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise CloudProviderError(f"HF {label} schema is unavailable.") from exc
    if errors:
        raise CloudProviderError(f"HF {label} is invalid: {errors[0].message}")


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CloudProviderError("HF run authorization contains an invalid object.")
    return value


def _timestamp_text(value: str | datetime) -> str:
    if isinstance(value, str):
        parsed = _parse_timestamp(value, label="timestamp")
    elif isinstance(value, datetime):
        if value.tzinfo is None:
            raise CloudProviderError("HF run authorization timestamps require UTC.")
        parsed = value.astimezone(timezone.utc)
    else:
        raise CloudProviderError("HF run authorization timestamp is invalid.")
    text = parsed.isoformat(timespec="microseconds" if parsed.microsecond else "seconds")
    return text.replace("+00:00", "Z")


def _parse_timestamp(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise CloudProviderError(f"HF run authorization {label} must be canonical UTC.")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise CloudProviderError(f"HF run authorization {label} is invalid.") from exc
    if _timestamp_text_unchecked(parsed) != value:
        raise CloudProviderError(f"HF run authorization {label} is not canonically encoded.")
    return parsed


def _timestamp_text_unchecked(value: datetime) -> str:
    text = value.astimezone(timezone.utc).isoformat(
        timespec="microseconds" if value.microsecond else "seconds"
    )
    return text.replace("+00:00", "Z")


def _decimal_text(value: str | int | Decimal) -> str:
    decimal_value = _decimal(value, label="decimal")
    text = format(decimal_value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def _decimal(value: object, *, label: str) -> Decimal:
    if isinstance(value, bool) or isinstance(value, float):
        raise CloudProviderError(f"HF run authorization {label} must use exact decimal text.")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise CloudProviderError(f"HF run authorization {label} is invalid.") from exc
    if not parsed.is_finite() or parsed < 0:
        raise CloudProviderError(f"HF run authorization {label} must be finite and non-negative.")
    if isinstance(value, str) and _decimal_text_unchecked(parsed) != value:
        raise CloudProviderError(f"HF run authorization {label} is not canonically encoded.")
    return parsed


def _decimal_text_unchecked(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def _reject_secret_values(value: object) -> None:
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_secret_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_secret_values(item)
    elif isinstance(value, str) and _resembles_known_secret(value):
        raise CloudProviderError("HF run authorization contains a prohibited secret value.")


def _resembles_known_secret(value: str) -> bool:
    candidates = {value}
    current = value
    for _ in range(2):
        decoded = unquote(current)
        if decoded == current:
            break
        candidates.add(decoded)
        current = decoded
    for candidate in tuple(candidates):
        compact = candidate.strip()
        if 8 <= len(compact) <= 512 and re.fullmatch(r"[A-Za-z0-9+/=_-]+", compact):
            padded = compact.replace("-", "+").replace("_", "/")
            padded += "=" * (-len(padded) % 4)
            try:
                decoded = base64.b64decode(padded, validate=True).decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                continue
            if len(decoded) <= 512:
                candidates.add(decoded)
    return any(
        _SECRET_VALUE_RE.search(candidate) or _SECRET_ASSIGNMENT_RE.search(candidate)
        for candidate in candidates
    )


__all__ = [
    "ACTION",
    "APPROVAL_SCHEMA_VERSION",
    "CANCEL_AFTER_SECONDS",
    "CLAIM_SCHEMA_VERSION",
    "HARD_CAP_USD",
    "HFRunApproval",
    "HFSubmissionClaim",
    "HFSubmissionState",
    "IMAGE",
    "MAXIMUM_PROJECTED_COST_USD",
    "OBSERVE_UNTIL_SECONDS",
    "PROVIDER_TIMEOUT_SECONDS",
    "RUNTIME",
    "approval_document_sha256",
    "build_hf_ambiguous_event",
    "build_hf_run_approval",
    "build_hf_submitted_event",
    "build_hf_submitting_event",
    "canonical_json_bytes",
    "refresh_hf_run_approval",
    "submission_event_sha256",
    "validate_hf_run_approval",
    "validate_hf_submission_claim",
]
