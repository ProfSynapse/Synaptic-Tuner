from __future__ import annotations

import copy
import importlib
import json
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from tuner.cloud.hf_bootstrap_smoke import canonical_workload_bytes, workload_sha256
from tuner.cloud.hf_run_approval import (
    HFSubmissionState,
    approval_document_sha256,
    build_hf_ambiguous_event,
    build_hf_run_approval,
    build_hf_submitted_event,
    build_hf_submitting_event,
    canonical_json_bytes,
    refresh_hf_run_approval,
    submission_event_sha256,
    validate_hf_run_approval,
    validate_hf_submission_claim,
)
from tuner.core.exceptions import CloudProviderError


ROOT = Path(__file__).resolve().parents[2]
APPROVAL_SCHEMA = ROOT / "schemas" / "synaptic-hf-run-approval-v1.schema.json"
CLAIM_SCHEMA = ROOT / "schemas" / "synaptic-hf-submission-claim-v1.schema.json"
T0 = "2026-08-20T12:00:00Z"
T1 = "2026-08-20T12:01:00Z"
T2 = "2026-08-20T12:02:00Z"


def _approval(**overrides):
    values = {
        "experiment_id": "jp-smoke",
        "run_id": "jp-smoke-001",
        "descriptor_uri": "tracking://runs/jp-smoke-001/descriptor.json",
        "descriptor_sha256": "1" * 64,
        "provisioning_evidence_uri": "tracking://runs/jp-smoke-001/provisioning.json",
        "provisioning_evidence_sha256": "2" * 64,
        "source_lock_uri": "tracking://runs/jp-smoke-001/source-lock.json",
        "source_lock_sha256": "3" * 64,
        "bundle_sha256": "4" * 64,
        "capsule_manifest_sha256": "5" * 64,
        "checkout_policy_sha256": "6" * 64,
        "hardware_flavor": "cpu-basic",
        "user_authorization_reference": "codex-thread:2026-08-20#hf-smoke-1",
        "issued_at": T0,
        "expires_at": "2026-08-20T13:00:00Z",
        "hourly_price_usd": "0.01",
        "projected_cost_usd": "0.002",
        "quoted_at": T0,
    }
    values.update(overrides)
    return build_hf_run_approval(**values)


def test_schemas_are_closed_and_valid_draft_2020_12():
    for path in (APPROVAL_SCHEMA, CLAIM_SCHEMA):
        schema = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        assert schema["additionalProperties"] is False


def test_approval_is_deterministic_canonical_and_exactly_bounded():
    left = _approval()
    right = _approval()
    assert left == right
    assert canonical_json_bytes(left.to_dict()) == canonical_json_bytes(right.to_dict())
    assert len(left.authorization_id) == 64
    assert len(left.approval_id) == 64
    document = left.document
    assert document["provider"] == "hf_jobs"
    assert document["action"] == "hf.bootstrap.verify"
    assert document["execution"] == {
        "runtime": "python:3.12",
        "image": "python:3.12",
        "hardware": {"flavor": "cpu-basic"},
        "bootstrap_only": True,
        "training": False,
        "publication": False,
        "ssh": False,
        "ports": [],
        "maximum_submissions": 1,
        "retry_count": 0,
    }
    assert document["timeouts"] == {
        "provider_seconds": 600,
        "cancel_after_seconds": 720,
        "observe_until_seconds": 900,
    }
    assert document["cost"]["maximum_projected_cost_usd"] == "0.01"
    assert document["cost"]["hard_cap_usd"] == "2.00"
    assert document["secrets"] == [{"provider": "env", "name": "HF_TOKEN"}]
    assert document["workload"] == {
        "kind": "bootstrap_verification",
        "sha256": "0d1d3454d079ea994a1e3a24b59b772bd4adb40cb441e00cc5801faf5d220841",
        "declaration": {
            "schema_version": "synaptic-hf-bootstrap-smoke-workload/v1",
            "kind": "bootstrap_verification",
            "runtime": {"image": "python:3.12"},
            "hardware": {"flavor": "cpu-basic"},
            "limits": {
                "provider_timeout_seconds": 600,
                "cancel_after_seconds": 720,
                "outer_observation_seconds": 900,
                "projected_compute_usd": "0.01",
                "hard_total_usd": "2.00",
            },
            "network": {"ports": [], "ssh": False},
            "retries": 0,
            "effects": {"training": False, "publication": False},
        },
    }
    assert canonical_json_bytes(document["workload"]["declaration"]) == canonical_workload_bytes()
    assert document["workload"]["sha256"] == workload_sha256()


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("experiment_id", "other-experiment"),
        ("run_id", "other-run"),
        ("descriptor_sha256", "8" * 64),
        ("provisioning_evidence_sha256", "8" * 64),
        ("source_lock_sha256", "8" * 64),
        ("bundle_sha256", "8" * 64),
        ("capsule_manifest_sha256", "8" * 64),
        ("checkout_policy_sha256", "8" * 64),
        ("user_authorization_reference", "codex-thread:2026-08-20#hf-smoke-2"),
    ],
)
def test_every_exact_scope_binding_changes_authorization_id(field, replacement):
    baseline = _approval()
    changed = _approval(**{field: replacement})
    assert changed.authorization_id != baseline.authorization_id


def test_expensive_hardware_with_fake_cheap_quote_is_rejected():
    with pytest.raises(CloudProviderError, match="cpu-basic"):
        _approval(
            hardware_flavor="a100-large",
            hourly_price_usd="0.01",
            projected_cost_usd="0.001",
        )


@pytest.mark.parametrize("mutation", ["digest", "declaration"])
def test_mismatched_fixed_workload_binding_is_rejected(mutation):
    document = _approval().to_dict()
    if mutation == "digest":
        document["workload"]["sha256"] = "8" * 64
    else:
        document["workload"]["declaration"]["effects"]["training"] = True
    with pytest.raises(CloudProviderError):
        validate_hf_run_approval(document)


def test_refresh_changes_approval_id_but_preserves_authorization_id():
    original = _approval()
    refreshed = refresh_hf_run_approval(
        original,
        issued_at=T1,
        expires_at="2026-08-20T14:00:00Z",
        hourly_price_usd=Decimal("0.009"),
        projected_cost_usd="0.0015",
        quoted_at=T1,
    )
    assert refreshed.authorization_id == original.authorization_id
    assert refreshed.approval_id != original.approval_id
    assert approval_document_sha256(refreshed) != approval_document_sha256(original)


@pytest.mark.parametrize(
    ("hourly", "projected"),
    [("0.011", "0.002"), ("0.01", "0.011"), ("NaN", "0.002"), ("Infinity", "0.002")],
)
def test_pricing_must_be_finite_and_within_envelope(hourly, projected):
    with pytest.raises(CloudProviderError):
        _approval(hourly_price_usd=hourly, projected_cost_usd=projected)


def test_binary_float_pricing_is_rejected():
    with pytest.raises(CloudProviderError, match="exact decimal"):
        _approval(hourly_price_usd=0.01)


def test_refresh_cannot_increase_price_or_revive_expired_approval():
    original = _approval()
    with pytest.raises(CloudProviderError, match="increase hourly"):
        refresh_hf_run_approval(
            original,
            issued_at=T1,
            expires_at="2026-08-20T14:00:00Z",
            hourly_price_usd="0.011",
            projected_cost_usd="0.002",
            quoted_at=T1,
        )
    with pytest.raises(CloudProviderError, match="still active"):
        refresh_hf_run_approval(
            original,
            issued_at="2026-08-20T13:00:00Z",
            expires_at="2026-08-20T14:00:00Z",
            hourly_price_usd="0.01",
            projected_cost_usd="0.002",
            quoted_at="2026-08-20T13:00:00Z",
        )


def test_expiry_is_fail_closed_for_submitting_claim():
    approval = _approval()
    with pytest.raises(CloudProviderError, match="expired"):
        build_hf_submitting_event(
            approval,
            approval_uri="tracking://runs/jp-smoke-001/approval.json",
            occurred_at="2026-08-20T13:00:00Z",
        )


def test_secret_values_are_rejected_but_secret_name_is_retained():
    approval = _approval()
    tampered = approval.to_dict()
    tampered["authorization"]["reference"] = "hf_abcdefghijklmnopqrstuvwxyz"
    with pytest.raises(CloudProviderError, match="secret value"):
        validate_hf_run_approval(tampered)


def test_closed_approval_rejects_literal_token_extension():
    document = _approval().to_dict()
    document["token"] = "not-even-a-real-token"
    with pytest.raises(CloudProviderError, match="Additional properties"):
        validate_hf_run_approval(document)


def test_submitting_event_is_deterministic_and_approved_is_projection_only():
    approval = _approval()
    kwargs = {
        "approval_uri": "tracking://runs/jp-smoke-001/approval.json",
        "occurred_at": T1,
    }
    left = build_hf_submitting_event(approval, **kwargs)
    right = build_hf_submitting_event(approval, **kwargs)
    assert left == right
    assert left.state is HFSubmissionState.SUBMITTING
    assert left.document["sequence"] == 1
    assert HFSubmissionState.APPROVED.value == "APPROVED"
    invalid = left.to_dict()
    invalid["state"] = "APPROVED"
    with pytest.raises(CloudProviderError):
        validate_hf_submission_claim(invalid)


def test_submitted_event_binds_exact_predecessor_and_normalized_job_identity():
    approval = _approval()
    submitting = build_hf_submitting_event(
        approval,
        approval_uri="tracking://runs/jp-smoke-001/approval.json",
        occurred_at=T1,
    )
    submitted = build_hf_submitted_event(
        approval,
        approval_uri="tracking://runs/jp-smoke-001/approval.json",
        previous_event=submitting,
        previous_event_uri="tracking://runs/jp-smoke-001/0001-submitting.json",
        occurred_at=T2,
        provider_namespace="professorsynapse",
        provider_job_id="job-123",
    )
    assert submitted.state is HFSubmissionState.SUBMITTED
    assert submitted.document["sequence"] == 2
    assert submitted.document["previous_event"]["sha256"] == submission_event_sha256(submitting)
    assert submitted.document["provider_job"] == {
        "namespace": "professorsynapse",
        "job_id": "job-123",
    }
    validate_hf_submission_claim(submitted, approval=approval, previous_event=submitting)


def test_ambiguous_event_uses_reason_code_and_cannot_store_raw_response():
    approval = _approval()
    submitting = build_hf_submitting_event(
        approval,
        approval_uri="tracking://runs/jp-smoke-001/approval.json",
        occurred_at=T1,
    )
    ambiguous = build_hf_ambiguous_event(
        approval,
        approval_uri="tracking://runs/jp-smoke-001/approval.json",
        previous_event=submitting,
        previous_event_uri="tracking://runs/jp-smoke-001/0001-submitting.json",
        occurred_at=T2,
        reason_code="PROVIDER_RESPONSE_UNKNOWN",
    )
    assert ambiguous.state is HFSubmissionState.AMBIGUOUS
    assert "provider_job" not in ambiguous.document
    extended = ambiguous.to_dict()
    extended["raw_response"] = "sensitive provider body"
    with pytest.raises(CloudProviderError, match="Additional properties"):
        validate_hf_submission_claim(extended)


def test_terminal_event_rejects_wrong_predecessor_and_time_rollback():
    approval = _approval()
    submitting = build_hf_submitting_event(
        approval,
        approval_uri="tracking://runs/jp-smoke-001/approval.json",
        occurred_at=T1,
    )
    other = build_hf_submitting_event(
        approval,
        approval_uri="tracking://runs/jp-smoke-001/approval.json",
        occurred_at=T0,
    )
    submitted = build_hf_submitted_event(
        approval,
        approval_uri="tracking://runs/jp-smoke-001/approval.json",
        previous_event=submitting,
        previous_event_uri="tracking://runs/jp-smoke-001/0001-submitting.json",
        occurred_at=T2,
        provider_namespace="professorsynapse",
        provider_job_id="job-123",
    )
    with pytest.raises(CloudProviderError, match="predecessor digest"):
        validate_hf_submission_claim(submitted, approval=approval, previous_event=other)
    with pytest.raises(CloudProviderError, match="predates"):
        build_hf_submitted_event(
            approval,
            approval_uri="tracking://runs/jp-smoke-001/approval.json",
            previous_event=submitting,
            previous_event_uri="tracking://runs/jp-smoke-001/0001-submitting.json",
            occurred_at=T0,
            provider_namespace="professorsynapse",
            provider_job_id="job-123",
        )


def test_event_tampering_and_cross_approval_replay_are_rejected():
    approval = _approval()
    submitting = build_hf_submitting_event(
        approval,
        approval_uri="tracking://runs/jp-smoke-001/approval.json",
        occurred_at=T1,
    )
    tampered = submitting.to_dict()
    tampered["run_id"] = "other-run"
    with pytest.raises(CloudProviderError, match="event_id"):
        validate_hf_submission_claim(tampered)

    other_approval = _approval(run_id="jp-smoke-002")
    with pytest.raises(CloudProviderError, match="approval binding"):
        validate_hf_submission_claim(submitting, approval=other_approval)


def test_datetime_inputs_are_normalized_to_utc_seconds():
    approval = _approval(
        issued_at=datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc),
        quoted_at=datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc),
        expires_at=datetime(2026, 8, 20, 13, 0, tzinfo=timezone.utc),
    )
    assert approval.document["issued_at"] == T0


def test_module_import_has_no_hugging_face_provider_import_or_effect(monkeypatch):
    before = {name for name in sys.modules if name.startswith("huggingface_hub")}
    module = sys.modules.pop("tuner.cloud.hf_run_approval")
    try:
        importlib.import_module("tuner.cloud.hf_run_approval")
        after = {name for name in sys.modules if name.startswith("huggingface_hub")}
        assert after == before
    finally:
        sys.modules["tuner.cloud.hf_run_approval"] = module


def test_to_dict_is_defensive():
    approval = _approval()
    copied = approval.to_dict()
    copied["execution"]["ports"].append(22)
    assert approval.document["execution"]["ports"] == []
