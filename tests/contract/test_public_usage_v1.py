"""LLM usage and spend-reference contracts (api-facade slice 1)."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from synaptic_tuner.api import v1
from synaptic_tuner.api.v1 import usage
from synaptic_tuner.api.v1.usage import SpendRef, UsageAvailability, UsageRecordV1


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((ROOT / "schemas/synaptic-usage-v1.schema.json").read_text(encoding="utf-8"))


def _measured(**changes: object) -> UsageRecordV1:
    values: dict[str, object] = {
        "schema_version": "synaptic-usage/v1",
        "availability": UsageAvailability.MEASURED,
        "input_tokens": 412334,
        "output_tokens": 88120,
        "cost_minor_units": 1874,
        "currency": "USD",
        "spend": SpendRef("openrouter", "acct-main", "spend-01J"),
    }
    values.update(changes)
    return UsageRecordV1(**values)  # type: ignore[arg-type]


def test_root_exports_have_only_canonical_contract_identities() -> None:
    for name in usage.__all__:
        assert getattr(v1, name) is getattr(usage, name)
        assert name in v1.__all__
    assert tuple(item.value for item in UsageAvailability) == ("measured", "unavailable")
    assert set(SCHEMA["properties"]["availability"]["enum"]) == {item.value for item in UsageAvailability}


def test_measured_usage_round_trips_and_matches_the_architecture_document() -> None:
    record = _measured()
    document = record.to_dict()
    assert document == {
        "schema_version": "synaptic-usage/v1", "availability": "measured",
        "input_tokens": 412334, "output_tokens": 88120,
        "cost_minor_units": 1874, "currency": "USD",
        "spend": {"provider": "openrouter", "account_ref": "acct-main", "spend_id": "spend-01J"},
    }
    jsonschema.Draft202012Validator.check_schema(SCHEMA)
    jsonschema.validate(document, SCHEMA)
    assert UsageRecordV1.from_dict(json.loads(json.dumps(document))) == record

    local = _measured(cost_minor_units=None, currency=None, spend=None)
    jsonschema.validate(local.to_dict(), SCHEMA)
    assert UsageRecordV1.from_dict(local.to_dict()) == local


def test_unavailable_usage_carries_nothing() -> None:
    record = UsageRecordV1("synaptic-usage/v1", UsageAvailability.UNAVAILABLE)
    document = record.to_dict()
    assert document == {
        "schema_version": "synaptic-usage/v1", "availability": "unavailable",
        "input_tokens": None, "output_tokens": None, "cost_minor_units": None,
        "currency": None, "spend": None,
    }
    jsonschema.validate(document, SCHEMA)
    assert UsageRecordV1.from_dict(document) == record
    for field in ("input_tokens", "output_tokens"):
        with pytest.raises(ValueError, match="unavailable usage carries no counters"):
            UsageRecordV1("synaptic-usage/v1", UsageAvailability.UNAVAILABLE, **{field: 1})  # type: ignore[arg-type]
        bad = dict(document, **{field: 1})
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, SCHEMA)
    with pytest.raises(ValueError, match="unavailable usage carries no counters"):
        UsageRecordV1("synaptic-usage/v1", UsageAvailability.UNAVAILABLE, cost_minor_units=1, currency="USD")
    with pytest.raises(ValueError, match="unavailable usage carries no counters"):
        UsageRecordV1("synaptic-usage/v1", UsageAvailability.UNAVAILABLE, spend=SpendRef("p", "a", "s"))


def test_measured_usage_requires_token_counts_and_paired_cost() -> None:
    with pytest.raises(ValueError, match="measured usage requires input_tokens and output_tokens"):
        _measured(input_tokens=None)
    with pytest.raises(ValueError, match="measured usage requires input_tokens and output_tokens"):
        _measured(output_tokens=None)
    with pytest.raises(ValueError, match="present together"):
        _measured(currency=None)
    with pytest.raises(ValueError, match="present together"):
        _measured(cost_minor_units=None)
    with pytest.raises(ValueError, match="ISO 4217"):
        _measured(currency="usd")
    with pytest.raises(ValueError, match="ISO 4217"):
        _measured(currency="USDT")
    with pytest.raises(TypeError, match="input_tokens must be an exact integer"):
        _measured(input_tokens=1.0)
    with pytest.raises(TypeError, match="input_tokens must be an exact integer"):
        _measured(input_tokens=True)
    with pytest.raises(ValueError, match="cost_minor_units must be at least 0"):
        _measured(cost_minor_units=-1)
    for bad in (
        dict(_measured().to_dict(), input_tokens=None),
        dict(_measured().to_dict(), currency=None),
        dict(_measured().to_dict(), cost_minor_units=None),
        dict(_measured().to_dict(), currency="usd"),
    ):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, SCHEMA)


def test_usage_record_fields_availability_and_schema_version_are_exact() -> None:
    document = _measured().to_dict()
    unknown = dict(document, calls=17)
    with pytest.raises(ValueError, match="unknown fields"):
        UsageRecordV1.from_dict(unknown)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(unknown, SCHEMA)
    missing = dict(document)
    del missing["spend"]
    with pytest.raises(ValueError, match="missing fields"):
        UsageRecordV1.from_dict(missing)
    with pytest.raises(ValueError, match="unknown usage availability") as captured:
        UsageRecordV1.from_dict(dict(document, availability="estimated"))
    assert captured.value.__cause__ is None
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(dict(document, availability="estimated"), SCHEMA)
    with pytest.raises(TypeError, match="availability must be exact UsageAvailability"):
        _measured(availability="measured")
    with pytest.raises(ValueError, match="unsupported usage schema version"):
        _measured(schema_version="synaptic-usage/v2")
    with pytest.raises(TypeError, match="spend must be exact SpendRef or None"):
        _measured(spend={"provider": "p", "account_ref": "a", "spend_id": "s"})
    with pytest.raises(TypeError, match="usage_record must be an exact object"):
        UsageRecordV1.from_dict([])  # type: ignore[arg-type]


def test_spend_ref_is_exact_and_round_trips() -> None:
    spend = SpendRef("openrouter", "acct-main", "spend-01J")
    assert spend.to_dict() == {"provider": "openrouter", "account_ref": "acct-main", "spend_id": "spend-01J"}
    assert SpendRef.from_dict(spend.to_dict()) == spend
    with pytest.raises(ValueError, match="unknown fields"):
        SpendRef.from_dict(dict(spend.to_dict(), namespace_ref="ns"))
    with pytest.raises(ValueError, match="spend_id is required"):
        SpendRef("openrouter", "acct-main", "")
    with pytest.raises(ValueError, match="leading or trailing whitespace"):
        SpendRef(" openrouter", "acct-main", "spend-01J")
    with pytest.raises(TypeError, match="provider must be an exact string"):
        SpendRef(None, "acct-main", "spend-01J")  # type: ignore[arg-type]
    bad = dict(_measured().to_dict(), spend=dict(spend.to_dict(), namespace_ref="ns"))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, SCHEMA)
