from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from tuner.cloud.hf_provisioning_claim import (
    HF_PROVISIONING_AMBIGUITY_EFFECTS,
    HFProvisioningState,
    build_hf_provisioning_ambiguous_event,
    build_hf_provisioning_claim,
    build_hf_provisioning_succeeded_event,
    canonical_json_bytes,
    validate_hf_provisioning_event,
)
from tuner.core.exceptions import CloudProviderError


def _descriptor() -> dict[str, object]:
    return {
        "schema_version": "synaptic-hf-source-transport/v1",
        "run_id": "exp-1",
        "profile": "C",
        "provider": "hf_jobs",
        "source_lock": {
            "uri": "tracking://experiments/exp-1/source-lock.json",
            "sha256": "1" * 64,
            "path": "source-lock.json",
        },
        "capsule": {
            "engine_commit": "2" * 40,
            "uri": "tracking://experiments/exp-1/bundle/capsule",
            "root": "capsule",
            "manifest": {
                "path": "capsule/synaptic-bootstrap-capsule.json",
                "sha256": "3" * 64,
            },
        },
        "checkout_policy": {
            "uri": "tracking://experiments/exp-1/bundle/checkout-policy.json",
            "path": "checkout-policy.json",
            "sha256": "4" * 64,
        },
        "bundle": {
            "uri": "tracking://experiments/exp-1/bundle",
            "content_sha256": "5" * 64,
        },
        "volume": {
            "type": "bucket",
            "source": "org/bucket",
            "path": f"bootstrap/exp-1/{'5' * 64}",
            "mount_path": "/workspace/synaptic-bootstrap-input",
            "read_only": True,
        },
    }


def _claim() -> dict[str, object]:
    return build_hf_provisioning_claim(
        experiment_id="exp-1",
        descriptor_uri="tracking://experiments/exp-1/descriptor.json",
        descriptor_sha256="6" * 64,
        descriptor=_descriptor(),
        actor="operator-1",
        authority="operator",
        occurred_at=datetime(2026, 8, 20, 12, tzinfo=timezone.utc),
    )


def _claim_sha256(claim: dict[str, object]) -> str:
    return __import__("hashlib").sha256(canonical_json_bytes(claim)).hexdigest()


def test_schema_is_closed_and_accepts_all_three_exact_states() -> None:
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "schemas"
        / "synaptic-hf-provisioning-claim-v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    claim = _claim()
    succeeded = build_hf_provisioning_succeeded_event(
        claim,
        claim_uri=f"tracking://events/{claim['event_id']}.json",
        claim_sha256=_claim_sha256(claim),
        evidence_uri="tracking://evidence.json",
        evidence_sha256="8" * 64,
        occurred_at="2026-08-20T12:01:00Z",
    )
    ambiguous = build_hf_provisioning_ambiguous_event(
        claim,
        claim_uri=f"tracking://events/{claim['event_id']}.json",
        claim_sha256=_claim_sha256(claim),
        reason_code="PROVIDER_OUTCOME_AMBIGUOUS",
        occurred_at="2026-08-20T12:01:00Z",
    )
    assert HFProvisioningState(str(claim["state"])) is HFProvisioningState.CLAIMED
    assert claim["run_id"] == claim["experiment_id"] == "exp-1"
    assert claim["sequence"] == 1
    assert claim["reason_code"] is None
    assert claim["provider_effect_possible"] is False
    assert succeeded["state"] == "SUCCEEDED"
    assert succeeded["sequence"] == 2
    assert succeeded["reason_code"] is None
    assert succeeded["provider_effect_possible"] is True
    assert succeeded["evidence"] == {
        "uri": "tracking://evidence.json",
        "sha256": "8" * 64,
    }
    assert ambiguous["state"] == "AMBIGUOUS"
    assert ambiguous["evidence"] is None
    assert ambiguous["sequence"] == 2
    assert ambiguous["provider_effect_possible"] is True


@pytest.mark.parametrize(
    ("reason_code", "effect"), HF_PROVISIONING_AMBIGUITY_EFFECTS.items()
)
def test_ambiguous_reason_codes_have_closed_provider_effect_mapping(
    reason_code: str, effect: bool
) -> None:
    claim = _claim()
    event = build_hf_provisioning_ambiguous_event(
        claim,
        claim_uri=f"tracking://events/{claim['event_id']}.json",
        claim_sha256=_claim_sha256(claim),
        reason_code=reason_code,
        occurred_at="2026-08-20T12:01:00Z",
    )
    assert event["reason_code"] == reason_code
    assert event["provider_effect_possible"] is effect

    tampered = copy.deepcopy(event)
    tampered["provider_effect_possible"] = not effect
    tampered["event_id"] = __import__("hashlib").sha256(
        canonical_json_bytes(
            {key: value for key, value in tampered.items() if key != "event_id"}
        )
    ).hexdigest()
    with pytest.raises(CloudProviderError, match="schema|reason/effect"):
        validate_hf_provisioning_event(tampered, previous_event=claim)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(extension=True),
        lambda value: value.update(state="SUCCEEDED"),
        lambda value: value.update(bundle_sha256="9" * 64),
        lambda value: value.update(event_id="9" * 64),
        lambda value: value.update(occurred_at="2026-08-20T08:00:00-04:00"),
    ],
)
def test_claim_rejects_extensions_invalid_transition_and_identity_tampering(mutation) -> None:
    value = copy.deepcopy(_claim())
    mutation(value)
    with pytest.raises(CloudProviderError):
        validate_hf_provisioning_event(value)


def test_terminal_requires_exact_claim_identity_and_monotonic_time() -> None:
    claim = _claim()
    terminal = build_hf_provisioning_ambiguous_event(
        claim,
        claim_uri=f"tracking://events/{claim['event_id']}.json",
        claim_sha256=_claim_sha256(claim),
        reason_code="PROVIDER_OUTCOME_AMBIGUOUS",
        occurred_at="2026-08-20T12:01:00Z",
    )
    other = copy.deepcopy(claim)
    other["actor"] = "operator-2"
    other["event_id"] = "0" * 64
    other["event_id"] = __import__("hashlib").sha256(
        canonical_json_bytes({key: value for key, value in other.items() if key != "event_id"})
    ).hexdigest()
    with pytest.raises(CloudProviderError, match="predecessor digest|immutable claim identity"):
        validate_hf_provisioning_event(terminal, previous_event=other)
    with pytest.raises(CloudProviderError, match="predates"):
        build_hf_provisioning_ambiguous_event(
            claim,
            claim_uri=f"tracking://events/{claim['event_id']}.json",
            claim_sha256=_claim_sha256(claim),
            reason_code="PROVIDER_OUTCOME_AMBIGUOUS",
            occurred_at="2026-08-20T11:59:59Z",
        )
