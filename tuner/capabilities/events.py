"""Strict JSON/JSONL writers for public v1 result and event envelopes."""

from __future__ import annotations

import base64
import json
import math
import re
import sys
from collections.abc import Mapping
from typing import Any, Iterable, TextIO
from urllib.parse import quote, quote_plus

from synaptic_tuner.api.v1 import EventEnvelope, ResultEnvelope

from .schema import validate_event, validate_result

_REDACTED = "[REDACTED]"
_URL_USERINFO = re.compile(r"(?i)([a-z][a-z0-9+.-]*://)[^/@\s]+(?::[^/@\s]*)?@")
_AUTH_VALUE = re.compile(r"(?i)\b(bearer|basic)\s+[a-z0-9._~+/=-]+")
_ASSIGNED_SECRET = re.compile(
    r"(?i)\b(api[-_ ]?key|authorization|auth(?:entication)?|access[-_ ]?(?:key|token)|"
    r"client[-_ ]?(?:id|key|secret|token|credential)|cookie|password|passphrase|"
    r"private[-_ ]?key|secret|token)(?:\s*[:=]\s*|\s+)[^\s,;]+"
)


def _normalized_key(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _key_tokens(value: str) -> tuple[str, ...]:
    """Split separators and camel case without treating metadata prefixes as secrets."""

    separated = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", value)
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", separated)
    return tuple(re.findall(r"[a-z0-9]+", separated.casefold()))


def _is_sensitive_key(key: str) -> bool:
    normalized = _normalized_key(key)
    tokens = _key_tokens(key)
    unambiguous_tokens = {
        "password",
        "passphrase",
        "token",
        "secret",
        "cookie",
        "credential",
        "authorization",
        "authentication",
        "auth",
    }
    if any(token in unambiguous_tokens for token in tokens):
        return True

    credential_pairs = {
        ("pass", "word"),
        ("pass", "phrase"),
        ("api", "key"),
        ("private", "key"),
        ("access", "key"),
        ("access", "token"),
        ("client", "id"),
        ("client", "key"),
        ("client", "secret"),
        ("client", "token"),
        ("client", "credential"),
        ("auth", "key"),
    }
    if any(pair in credential_pairs for pair in zip(tokens, tokens[1:])):
        return True

    # Preserve compatibility with compact lowercase spellings that contain no
    # separator or camel-case boundary, while keeping bare access/client/private
    # metadata keys truthful.
    compact_credentials = {
        "apikey",
        "privatekey",
        "accesskey",
        "accesstoken",
        "clientid",
        "clientkey",
        "clientsecret",
        "clienttoken",
        "clientcredential",
        "authkey",
    }
    return normalized in compact_credentials


def _sensitive_forms(values: Iterable[str] | None) -> tuple[tuple[str, bool], ...]:
    forms: dict[str, bool] = {}
    for value in values or ():
        if not isinstance(value, str):
            raise TypeError("Sensitive values must be strings")
        if not value:
            continue
        encoded = value.encode("utf-8")
        bounded = len(value) < 4
        for form in {
            value,
            quote(value, safe=""),
            quote_plus(value, safe=""),
            base64.b64encode(encoded).decode("ascii"),
            base64.urlsafe_b64encode(encoded).decode("ascii"),
        }:
            if form:
                # If the same form came from both a short and a long explicit
                # value, the long value's exact classification is authoritative.
                forms[form] = forms.get(form, True) and bounded
    return tuple(
        sorted(forms.items(), key=lambda item: (-len(item[0]), item[0]))
    )


def _replace_bounded(value: str, secret: str) -> str:
    """Replace a short explicit secret only at Unicode-alphanumeric boundaries."""

    if value == secret:
        return _REDACTED
    pieces: list[str] = []
    cursor = 0
    while True:
        start = value.find(secret, cursor)
        if start < 0:
            pieces.append(value[cursor:])
            return "".join(pieces)
        end = start + len(secret)
        left_boundary = start == 0 or not value[start - 1].isalnum()
        right_boundary = end == len(value) or not value[end].isalnum()
        if left_boundary and right_boundary:
            pieces.extend((value[cursor:start], _REDACTED))
            cursor = end
        else:
            pieces.append(value[cursor:end])
            cursor = end


def _redact_string(
    value: str, sensitive_forms: tuple[tuple[str, bool], ...]
) -> str:
    if any(value == secret for secret, _bounded in sensitive_forms):
        return _REDACTED
    redacted = _URL_USERINFO.sub(r"\1[REDACTED]@", value)
    redacted = _AUTH_VALUE.sub(lambda match: f"{match.group(1)} {_REDACTED}", redacted)
    redacted = _ASSIGNED_SECRET.sub(
        lambda match: f"{match.group(1)}={_REDACTED}", redacted
    )
    for secret, bounded in sensitive_forms:
        redacted = (
            _replace_bounded(redacted, secret)
            if bounded
            else redacted.replace(secret, _REDACTED)
        )
    return redacted


def _json_value(
    value: Any, *, sensitive_forms: tuple[tuple[str, bool], ...]
) -> Any:
    if value is None:
        return None
    # bool is a subclass of int and must be handled first.
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Machine JSON cannot contain non-finite floats")
        return value
    if isinstance(value, str):
        return _redact_string(value, sensitive_forms)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Machine JSON object keys must be strings")
            # Validate the original value even when its rendered form is redacted.
            normalized_item = _json_value(item, sensitive_forms=sensitive_forms)
            result[key] = _REDACTED if _is_sensitive_key(key) else normalized_item
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item, sensitive_forms=sensitive_forms) for item in value]
    raise TypeError(f"Unsupported machine JSON type: {type(value).__name__}")


def redact(value: Any, *, sensitive_values: Iterable[str] | None = None) -> Any:
    return _json_value(value, sensitive_forms=_sensitive_forms(sensitive_values))


def _line(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _writer_failure(exc: Exception, *, sensitive_values: Iterable[str] | None) -> None:
    emit_diagnostic(
        "Machine output rejected before stdout write.",
        details={"error_type": type(exc).__name__},
        sensitive_values=sensitive_values,
    )


def write_result(
    result: ResultEnvelope,
    stream: TextIO | None = None,
    *,
    sensitive_values: Iterable[str] | None = None,
) -> None:
    target = stream or sys.stdout
    try:
        payload = redact(result.to_dict(), sensitive_values=sensitive_values)
        validate_result(payload)
        line = _line(payload)
    except Exception as exc:
        _writer_failure(exc, sensitive_values=sensitive_values)
        raise
    target.write(line + "\n")
    target.flush()


def write_event(
    event: EventEnvelope,
    stream: TextIO | None = None,
    *,
    sensitive_values: Iterable[str] | None = None,
) -> None:
    target = stream or sys.stdout
    try:
        payload = redact(event.to_dict(), sensitive_values=sensitive_values)
        validate_event(payload)
        line = _line(payload)
    except Exception as exc:
        _writer_failure(exc, sensitive_values=sensitive_values)
        raise
    target.write(line + "\n")
    target.flush()


def emit_diagnostic(
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
    stream: TextIO | None = None,
    sensitive_values: Iterable[str] | None = None,
) -> None:
    target = stream or sys.stderr
    payload = {"message": message}
    if details:
        payload["details"] = details
    try:
        prepared = redact(payload, sensitive_values=sensitive_values)
        line = _line(prepared)
    except Exception:
        line = '{"message":"Diagnostic rejected during safe serialization."}'
    target.write(line + "\n")
    target.flush()


__all__ = ["emit_diagnostic", "redact", "write_event", "write_result"]
