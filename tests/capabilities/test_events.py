from __future__ import annotations

import base64
import io
import json
import math
from urllib.parse import quote

import pytest
from jsonschema import ValidationError

from synaptic_tuner.api.v1 import EventEnvelope, ResultEnvelope
from tuner.capabilities.events import emit_diagnostic, redact, write_event, write_result
from tuner.capabilities.schema import validate_event, validate_result


def _result(**overrides) -> ResultEnvelope:
    values = {
        "success": True,
        "capability": "evaluation.run",
        "run_id": "run_test",
        "data": {"score": 0.9},
    }
    values.update(overrides)
    return ResultEnvelope(**values)


def test_result_writer_emits_exactly_one_valid_json_envelope() -> None:
    stream = io.StringIO()
    write_result(_result(), stream)
    lines = stream.getvalue().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert validate_result(payload) == payload
    assert payload["schema_version"] == "synaptic-result/v1"


def test_event_writer_emits_only_valid_jsonl_envelopes_with_final_result() -> None:
    stream = io.StringIO()
    result = _result()
    write_event(
        EventEnvelope(
            event="evaluation.started",
            capability=result.capability,
            run_id=result.run_id,
            sequence=0,
        ),
        stream,
    )
    write_event(
        EventEnvelope(
            event="evaluation.completed",
            capability=result.capability,
            run_id=result.run_id,
            sequence=1,
            final=True,
            result=result,
        ),
        stream,
    )
    payloads = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert len(payloads) == 2
    assert all(validate_event(payload) == payload for payload in payloads)
    assert payloads[-1]["final"] is True
    assert payloads[-1]["result"]["schema_version"] == "synaptic-result/v1"


def test_nested_secrets_are_redacted_before_stdout_serialization() -> None:
    stream = io.StringIO()
    write_result(
        _result(
            data={
                "api_key": "secret-value",
                "nested": {"authorization_token": "bearer-value", "safe": "visible"},
            }
        ),
        stream,
    )
    payload = json.loads(stream.getvalue())
    assert payload["data"]["api_key"] == "[REDACTED]"
    assert payload["data"]["nested"]["authorization_token"] == "[REDACTED]"
    assert payload["data"]["nested"]["safe"] == "visible"
    assert "secret-value" not in stream.getvalue()


def test_redaction_preserves_benign_access_client_and_private_metadata() -> None:
    payload = redact(
        {
            "access": "read",
            "permission": "read-only",
            "private": False,
            "privateMaterial": "public descriptor metadata",
            "client": "local CLI",
            "clientMetadata": {"kind": "agent"},
        }
    )
    assert payload == {
        "access": "read",
        "permission": "read-only",
        "private": False,
        "privateMaterial": "public descriptor metadata",
        "client": "local CLI",
        "clientMetadata": {"kind": "agent"},
    }


@pytest.mark.parametrize(
    "key",
    [
        "x-api-key",
        "clientSecret",
        "private_key",
        "authorization",
        "access_token",
        "access-key",
        "accessToken",
        "access_key_id",
    ],
)
def test_redaction_classifies_explicit_credential_key_variants(key) -> None:
    assert redact({key: "Q"}) == {key: "[REDACTED]"}


def test_result_writer_round_trips_descriptor_access_without_weakening_secrets() -> None:
    stdout = io.StringIO()
    write_result(
        _result(
            data={
                "input": {"name": "config", "access": "read", "private": False},
                "accessToken": "Q",
            }
        ),
        stdout,
    )
    payload = json.loads(stdout.getvalue())
    assert validate_result(payload) == payload
    assert payload["data"]["input"] == {
        "name": "config",
        "access": "read",
        "private": False,
    }
    assert payload["data"]["accessToken"] == "[REDACTED]"


def test_diagnostics_are_json_on_stderr_stream_and_redacted() -> None:
    stream = io.StringIO()
    emit_diagnostic("failed", details={"HF_TOKEN": "private", "stage": "preflight"}, stream=stream)
    payload = json.loads(stream.getvalue())
    assert payload == {
        "message": "failed",
        "details": {"HF_TOKEN": "[REDACTED]", "stage": "preflight"},
    }


def test_schema_rejects_invalid_timestamp_and_nonfinal_result() -> None:
    invalid_result = _result().to_dict()
    invalid_result["timestamp"] = "not-a-timestamp"
    with pytest.raises(ValidationError):
        validate_result(invalid_result)

    event = EventEnvelope(
        event="evaluation.completed",
        capability="evaluation.run",
        run_id="run_test",
        sequence=1,
        final=True,
        result=_result(),
    ).to_dict()
    event["final"] = False
    with pytest.raises(ValidationError):
        validate_event(event)


def test_writer_schema_failure_has_no_partial_stdout(capsys) -> None:
    stdout = io.StringIO()
    with pytest.raises(ValidationError):
        write_result(_result(timestamp="not-a-timestamp"), stdout)
    captured = capsys.readouterr()
    assert stdout.getvalue() == ""
    assert captured.out == ""
    assert json.loads(captured.err)["details"] == {"error_type": "ValidationError"}


def test_public_dataclass_enforces_final_event_invariant() -> None:
    with pytest.raises(ValueError, match="final event"):
        EventEnvelope(
            event="evaluation.completed",
            capability="evaluation.run",
            run_id="run_test",
            sequence=1,
            final=True,
        )


@pytest.mark.parametrize(
    "bad_value",
    [math.nan, math.inf, -math.inf, object(), {1: "bad-key"}, range(3)],
)
def test_writer_rejects_nonstandard_json_without_partial_stdout(bad_value, capsys) -> None:
    stdout = io.StringIO()
    with pytest.raises((TypeError, ValueError)):
        write_result(_result(data={"nested": [True, {"bad": bad_value}]}), stdout)
    captured = capsys.readouterr()
    assert stdout.getvalue() == ""
    assert captured.out == ""
    diagnostic = json.loads(captured.err)
    assert diagnostic["message"] == "Machine output rejected before stdout write."
    assert diagnostic["details"]["error_type"] in {"TypeError", "ValueError"}


def test_writer_preserves_booleans_as_json_booleans() -> None:
    stdout = io.StringIO()
    write_result(_result(data={"enabled": True, "count": 1}), stdout)
    payload = json.loads(stdout.getvalue())
    assert payload["data"] == {"enabled": True, "count": 1}
    assert type(payload["data"]["enabled"]) is bool
    assert type(payload["data"]["count"]) is int


def test_adversarial_nested_redaction_covers_headers_urls_messages_and_transforms() -> None:
    explicit = "v@ult secret/value"
    encoded = quote(explicit, safe="")
    b64 = base64.b64encode(explicit.encode()).decode()
    stdout = io.StringIO()
    write_result(
        _result(
            data={
                "password": "one",
                "passPhrase": "two",
                "accessToken": "three",
                "clientSecret": "four",
                "private_key": "five",
                "privateMaterial": "five-b",
                "x-api-key": "six",
                "Cookie": "seven",
                "nested": [
                    "https://alice:hunter2@example.test/path",
                    "Authorization: Bearer bearer-value",
                    "Basic dXNlcjpwYXNz",
                    "token=message-token",
                    "auth auth-value",
                    f"plain {explicit}; encoded {encoded}; b64 {b64}",
                ],
            }
        ),
        stdout,
        sensitive_values={explicit},
    )
    rendered = stdout.getvalue()
    payload = json.loads(rendered)
    for key in (
        "password",
        "passPhrase",
        "accessToken",
        "clientSecret",
        "private_key",
        "x-api-key",
        "Cookie",
    ):
        assert payload["data"][key] == "[REDACTED]"
    assert payload["data"]["privateMaterial"] == "five-b"
    for forbidden in (
        "hunter2",
        "bearer-value",
        "dXNlcjpwYXNz",
        "message-token",
        "auth-value",
        explicit,
        encoded,
        b64,
    ):
        assert forbidden not in rendered
    assert "https://[REDACTED]@example.test/path" in rendered


def test_diagnostic_redacts_message_details_and_explicit_values_exactly() -> None:
    secret = "diagnostic-secret/value"
    stderr = io.StringIO()
    emit_diagnostic(
        f"Request https://user:pass@example.test failed; token={secret}; X-Api-Key: literal-header",
        details={
            "x-api-key": "header-value",
            "privateKey": "pem-value",
            "note": f"Authorization: Basic {base64.b64encode(secret.encode()).decode()}",
        },
        stream=stderr,
        sensitive_values={secret},
    )
    lines = stderr.getvalue().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["details"]["x-api-key"] == "[REDACTED]"
    assert payload["details"]["privateKey"] == "[REDACTED]"
    assert secret not in stderr.getvalue()
    assert "header-value" not in stderr.getvalue()
    assert "pem-value" not in stderr.getvalue()
    assert "literal-header" not in stderr.getvalue()
    assert "https://[REDACTED]@example.test" in payload["message"]


def test_explicit_short_secrets_redact_at_unicode_alphanumeric_boundaries() -> None:
    stdout = io.StringIO()
    write_result(
        _result(
            data={
                "direct": ["123", "987", "XY", "Q"],
                "nested": {
                    "message": "PIN 123; CVV=987; code (XY); marker Q.",
                    "encoded": "PIN64 MTIz; code64 WFk=; marker64 UQ==.",
                    "safe": "A123B 59876 WXYZ opaqueQvalue é123界 éQ界",
                },
            }
        ),
        stdout,
        sensitive_values={"123", "987", "XY", "Q"},
    )

    lines = stdout.getvalue().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert validate_result(payload) == payload
    assert payload["data"]["direct"] == ["[REDACTED]"] * 4
    assert payload["data"]["nested"]["message"] == (
        "PIN [REDACTED]; CVV=[REDACTED]; code ([REDACTED]); marker [REDACTED]."
    )
    assert payload["data"]["nested"]["encoded"] == (
        "PIN64 [REDACTED]; code64 [REDACTED]; marker64 [REDACTED]."
    )
    assert payload["data"]["nested"]["safe"] == (
        "A123B 59876 WXYZ opaqueQvalue é123界 éQ界"
    )


def test_explicit_whole_string_precedes_generic_pattern_normalization() -> None:
    stdout = io.StringIO()
    write_result(
        _result(data={"direct": "Bearer Q"}),
        stdout,
        sensitive_values={"Bearer Q"},
    )

    payload = json.loads(stdout.getvalue())
    assert payload["data"]["direct"] == "[REDACTED]"


def test_short_secrets_redact_in_event_result_and_diagnostic_without_extra_lines() -> None:
    event_stdout = io.StringIO()
    result = _result(data={"pin": "123", "note": "approval code: XY"})
    write_event(
        EventEnvelope(
            event="evaluation.completed",
            capability=result.capability,
            run_id=result.run_id,
            sequence=0,
            final=True,
            result=result,
        ),
        event_stdout,
        sensitive_values={"123", "XY"},
    )
    event_lines = event_stdout.getvalue().splitlines()
    assert len(event_lines) == 1
    event_payload = json.loads(event_lines[0])
    assert validate_event(event_payload) == event_payload
    assert event_payload["result"]["data"] == {
        "pin": "[REDACTED]",
        "note": "approval code: [REDACTED]",
    }

    diagnostic_stderr = io.StringIO()
    emit_diagnostic(
        "Rejected PIN 987 and marker Q; preserve opaqueQvalue.",
        details={"code": "XY", "nested": ["123", "A123B"]},
        stream=diagnostic_stderr,
        sensitive_values={"123", "987", "XY", "Q"},
    )
    diagnostic_lines = diagnostic_stderr.getvalue().splitlines()
    assert len(diagnostic_lines) == 1
    diagnostic = json.loads(diagnostic_lines[0])
    assert diagnostic == {
        "message": (
            "Rejected PIN [REDACTED] and marker [REDACTED]; preserve opaqueQvalue."
        ),
        "details": {
            "code": "[REDACTED]",
            "nested": ["[REDACTED]", "A123B"],
        },
    }


def test_short_secret_writer_failure_has_no_partial_machine_output(capsys) -> None:
    stdout = io.StringIO()
    with pytest.raises(TypeError):
        write_result(
            _result(data={"pin": "123", "bad": object()}),
            stdout,
            sensitive_values={"123"},
        )

    captured = capsys.readouterr()
    assert stdout.getvalue() == ""
    assert captured.out == ""
    diagnostic_lines = captured.err.splitlines()
    assert len(diagnostic_lines) == 1
    assert "123" not in diagnostic_lines[0]
    assert json.loads(diagnostic_lines[0]) == {
        "message": "Machine output rejected before stdout write.",
        "details": {"error_type": "TypeError"},
    }
