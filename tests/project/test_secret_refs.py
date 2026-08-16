import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from tuner.project.errors import SecretReferenceError, SecretUnavailableError
from tuner.project.secrets import (
    SecretRef,
    redact_secrets,
    reject_literal_secrets,
    resolve_secret,
)


def test_secret_ref_rejects_literal_value_fields() -> None:
    with pytest.raises(SecretReferenceError):
        SecretRef.from_dict({"provider": "env", "name": "HF_TOKEN", "value": "secret"})


def test_resolution_returns_value_without_storing_or_echoing_it() -> None:
    reference = SecretRef("env", "HF_TOKEN")
    secret = "sensitive-value"
    assert resolve_secret(reference, environment={"HF_TOKEN": secret}) == secret
    assert secret not in repr(reference)
    assert secret not in json.dumps(redact_secrets({"token": reference}))


def test_missing_secret_error_contains_identifier_not_value() -> None:
    with pytest.raises(SecretUnavailableError) as error:
        resolve_secret(SecretRef("env", "HF_TOKEN"), environment={})
    assert error.value.details == {"provider": "env", "name": "HF_TOKEN"}


def test_secret_reference_validates_against_schema() -> None:
    payload = SecretRef("provider_secret", "production/hf").to_dict()
    schema = json.loads(
        (Path(__file__).resolve().parents[2] / "schemas" / "synaptic-secret-ref-v1.schema.json").read_text()
    )
    Draft202012Validator(schema).validate(payload)


def test_literal_secret_detection_does_not_echo_value() -> None:
    literal = "should-not-appear"
    with pytest.raises(SecretReferenceError) as error:
        reject_literal_secrets({"nested": {"password": literal}})
    assert literal not in str(error.value)
