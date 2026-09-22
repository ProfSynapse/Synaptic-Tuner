#!/usr/bin/env python3
"""Validate the sanitized JSON emitted by ``tuner.py ingest --json``.

This validates the public result envelope, aggregate relationships, closed
diagnostic codes, and verified-success evidence. It deliberately does not open
or independently verify normalized bundle members.

Usage:
  python validate_ingestion_result.py RESULT [--require-verified]
         [--expected-source-count N]

Exit codes:
  0  valid
  1  violations found
  2  usage or I/O error
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import NoReturn


MAX_RESULT_BYTES = 1_048_576
HEX64 = re.compile(r"^[0-9a-f]{64}$")
REFERENCE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
DIAGNOSTICS = {
    "unmatched_source",
    "ambiguous_source",
    "source_changed",
    "structure_invalid",
    "parse_failed",
    "metadata_invalid",
    "relationship_invalid",
    "output_conflict",
    "interrupted",
    "effect_uncertain",
    "execution_failed",
    "bundle_invalid",
}
VERIFIED_FIELDS = {
    "success",
    "status",
    "source_count",
    "matched_sources",
    "unmatched_sources",
    "ambiguous_sources",
    "sources_processed",
    "documents_written",
    "structure_set_digest",
    "snapshot_ref",
    "manifest_digest",
    "plan_fingerprint",
    "run_ref",
    "outcome_digest",
    "bundle_ref",
    "bundle_digest",
    "diagnostic_codes",
}
BLOCKED_FIELDS = {
    "success",
    "status",
    "error_code",
    "source_count",
    "matched_sources",
    "unmatched_sources",
    "ambiguous_sources",
    "plan_fingerprint",
    "diagnostic_codes",
}
FAILED_MINIMAL_FIELDS = {"success", "status", "error_code"}
FAILED_RUN_FIELDS = FAILED_MINIMAL_FIELDS | {
    "source_count",
    "sources_processed",
    "documents_written",
    "run_ref",
    "outcome_digest",
    "diagnostic_codes",
}


class ResultViolation(ValueError):
    """A sanitized result-envelope violation."""


def _fail(message: str) -> NoReturn:
    raise ResultViolation(message)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(_: str) -> NoReturn:
    _fail("non-finite JSON numbers are forbidden")


def _load(path: Path) -> dict[str, object]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ResultViolation("result could not be read") from exc
    if not raw or len(raw) > MAX_RESULT_BYTES:
        _fail(f"result must contain 1..{MAX_RESULT_BYTES} bytes")
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise ResultViolation("result must be UTF-8") from exc
    lines = text.splitlines()
    if len(lines) != 1 or not lines[0]:
        _fail("result must be exactly one non-empty JSON line")
    try:
        value = json.loads(
            lines[0], object_pairs_hook=_strict_object, parse_constant=_reject_constant
        )
    except json.JSONDecodeError as exc:
        raise ResultViolation("result must be strict JSON") from exc
    if type(value) is not dict:
        _fail("result must be a JSON object")
    return value


def _exact_fields(value: dict[str, object], fields: set[str], label: str) -> None:
    if set(value) != fields:
        _fail(f"{label} result has an unexpected field set")


def _count(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        _fail(f"{label} must be a non-negative integer")
    return value


def _digest(value: object, label: str) -> None:
    if type(value) is not str or HEX64.fullmatch(value) is None:
        _fail(f"{label} must be a lowercase SHA-256 digest")


def _reference(value: object, label: str) -> None:
    if type(value) is not str or REFERENCE.fullmatch(value) is None:
        _fail(f"{label} must be a path-free public reference")


def _diagnostics(value: object) -> list[str]:
    if type(value) is not list or any(type(item) is not str for item in value):
        _fail("diagnostic_codes must be a string array")
    if len(value) != len(set(value)) or any(item not in DIAGNOSTICS for item in value):
        _fail("diagnostic_codes must be unique closed public codes")
    return value


def validate(
    value: dict[str, object], *, require_verified: bool, expected_source_count: int | None
) -> None:
    if type(value.get("success")) is not bool or type(value.get("status")) is not str:
        _fail("success and status have invalid types")

    if value["status"] == "verified":
        _exact_fields(value, VERIFIED_FIELDS, "verified")
        if value["success"] is not True:
            _fail("verified result must have success=true")
        source_count = _count(value["source_count"], "source_count")
        matched = _count(value["matched_sources"], "matched_sources")
        unmatched = _count(value["unmatched_sources"], "unmatched_sources")
        ambiguous = _count(value["ambiguous_sources"], "ambiguous_sources")
        processed = _count(value["sources_processed"], "sources_processed")
        written = _count(value["documents_written"], "documents_written")
        if (matched, unmatched, ambiguous, processed, written) != (
            source_count,
            0,
            0,
            source_count,
            source_count,
        ):
            _fail("verified Markdown counts are inconsistent")
        if _diagnostics(value["diagnostic_codes"]):
            _fail("verified result must have no diagnostics")
        for label in (
            "structure_set_digest",
            "manifest_digest",
            "plan_fingerprint",
            "outcome_digest",
            "bundle_digest",
        ):
            _digest(value[label], label)
        for label in ("snapshot_ref", "run_ref", "bundle_ref"):
            _reference(value[label], label)
        if value["bundle_ref"] != "bundle-" + value["bundle_digest"]:
            _fail("bundle_ref must bind bundle_digest")
    elif value["status"] == "blocked":
        _exact_fields(value, BLOCKED_FIELDS, "blocked")
        if value["success"] is not False or value["error_code"] != "preflight_blocked":
            _fail("blocked result has invalid success or error_code")
        source_count = _count(value["source_count"], "source_count")
        for label in ("matched_sources", "unmatched_sources", "ambiguous_sources"):
            if _count(value[label], label) > source_count:
                _fail(f"{label} exceeds source_count")
        _digest(value["plan_fingerprint"], "plan_fingerprint")
        if not _diagnostics(value["diagnostic_codes"]):
            _fail("blocked result must include a diagnostic")
    else:
        fields = set(value)
        if fields not in (FAILED_MINIMAL_FIELDS, FAILED_RUN_FIELDS):
            _fail("failed result has an unexpected field set")
        if value["success"] is not False or not value["error_code"]:
            _fail("failed result has invalid success or error_code")
        _reference(value["error_code"], "error_code")
        if fields == FAILED_RUN_FIELDS:
            source_count = _count(value["source_count"], "source_count")
            processed = _count(value["sources_processed"], "sources_processed")
            written = _count(value["documents_written"], "documents_written")
            if processed > source_count or written > processed:
                _fail("failed run counts are inconsistent")
            _reference(value["run_ref"], "run_ref")
            _digest(value["outcome_digest"], "outcome_digest")
            _diagnostics(value["diagnostic_codes"])

    if require_verified and value["status"] != "verified":
        _fail("verified result required")
    if expected_source_count is not None:
        if "source_count" not in value or value["source_count"] != expected_source_count:
            _fail("source_count does not match --expected-source-count")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="captured one-line CLI JSON result")
    parser.add_argument("--require-verified", action="store_true")
    parser.add_argument("--expected-source-count", type=int)
    args = parser.parse_args()
    if not args.result.is_file():
        print("error: result must be an existing regular file", file=sys.stderr)
        return 2
    if args.expected_source_count is not None and args.expected_source_count < 0:
        print("error: --expected-source-count must be non-negative", file=sys.stderr)
        return 2
    try:
        validate(
            _load(args.result),
            require_verified=args.require_verified,
            expected_source_count=args.expected_source_count,
        )
    except ResultViolation as exc:
        print(f"INVALID: {exc}")
        return 1
    print("VALID: sanitized ingestion result")
    return 0


if __name__ == "__main__":
    sys.exit(main())
