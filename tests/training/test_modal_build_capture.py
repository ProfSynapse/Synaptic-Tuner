"""Provider-free checks for the Modal inspector report size and identity contract."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

from tuner.execution.providers.modal import runtime_build
from tuner.execution.providers.modal.runtime_build import (
    ModalBuildCandidateV1,
    _CaptureOutputFailure,
    _capture_output,
)


IMAGE_ID = "im-CaptureTest"
INSPECTOR_DIGEST = "a" * 64
INPUTS_DIGEST = "b" * 64
WHEEL_DIGEST = "c" * 64
MATERIAL = {
    "kind": "modal_build",
    "build_inputs": {"wheels": [{"sha256": WHEEL_DIGEST}]},
}


def _report(body_size: int | None = None) -> bytes:
    document = {
        "schema_version": "synaptic-modal-build-capture/v1",
        "image_id": IMAGE_ID,
        "inspector_sha256": INSPECTOR_DIGEST,
        "measured": {
            "build_inputs_digest": INPUTS_DIGEST,
            "package": {"digest": WHEEL_DIGEST},
            "installed_distributions": {"inventory": ""},
        },
    }

    def encode() -> bytes:
        return json.dumps(
            document, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")

    if body_size is not None:
        document["measured"]["installed_distributions"]["inventory"] = "x" * (body_size - len(encode()))
    raw = encode()
    assert body_size is None or len(raw) == body_size
    return raw + b"\n"


def _admit(raw: bytes, **overrides: str) -> ModalBuildCandidateV1:
    kwargs = {
        "material": MATERIAL,
        "raw": raw,
        "expected_image_id": IMAGE_ID,
        "expected_inspector_sha256": INSPECTOR_DIGEST,
        "expected_build_inputs_digest": INPUTS_DIGEST,
    }
    kwargs.update(overrides)
    return ModalBuildCandidateV1.from_capture(**kwargs)


@pytest.mark.parametrize("body_size", [16 * 1024 + 1, 128 * 1024])
def test_canonical_inspector_report_above_foundation_limit_and_at_output_ceiling(body_size):
    raw = _report(body_size)
    candidate = _admit(raw)
    assert candidate.capture_digest == hashlib.sha256(raw).hexdigest()
    candidate._validate_capture()


def test_report_above_inspector_output_ceiling_is_rejected():
    with pytest.raises(ValueError, match="capture is invalid"):
        _admit(_report(128 * 1024 + 1))


@pytest.mark.parametrize("raw", [
    _report()[:-1],
    _report() + b"\n",
    b" " + _report(),
])
def test_report_requires_one_canonical_json_body_and_newline(raw):
    with pytest.raises(ValueError):
        _admit(raw)


@pytest.mark.parametrize("field,expected", [
    ("expected_image_id", "im-Other"),
    ("expected_inspector_sha256", "d" * 64),
    ("expected_build_inputs_digest", "e" * 64),
])
def test_report_must_match_bound_image_inspector_and_inputs(field, expected):
    with pytest.raises(ValueError):
        _admit(_report(), **{field: expected})


def test_revalidation_rejects_changed_report_bytes():
    candidate = _admit(_report(16 * 1024 + 1))
    changed = replace(candidate, _capture_raw=candidate._capture_raw.replace(b"x", b"y", 1))
    with pytest.raises(ValueError, match="capture changed"):
        changed._validate_capture()


@pytest.mark.parametrize("raw", [
    b"x" * (128 * 1024 + 1) + b"\n",
    b"{}",
])
def test_direct_candidate_rejects_unbounded_or_unframed_bytes_before_json_parse(monkeypatch, raw):
    candidate = ModalBuildCandidateV1(
        IMAGE_ID, b"{}", raw, INSPECTOR_DIGEST, hashlib.sha256(raw).hexdigest(),
    )
    monkeypatch.setattr(runtime_build.json, "loads", lambda *_args: pytest.fail("JSON parsed before capture bound"))
    with pytest.raises(ValueError, match="capture changed"):
        candidate._validate_capture()


class _Sandbox:
    returncode = 0

    def __init__(self, raw: bytes):
        self.stdout = (raw[:-1], raw[-1:])

    def wait(self, *, raise_on_termination: bool) -> None:
        assert raise_on_termination is False


def test_output_reader_accepts_exact_inspector_body_ceiling_plus_newline():
    raw = _report(128 * 1024)
    assert _capture_output(_Sandbox(raw)) == raw


@pytest.mark.parametrize("raw", [
    _report(128 * 1024 + 1),
    b"x" * (128 * 1024 + 1),
])
def test_output_reader_rejects_bytes_beyond_inspector_contract(raw):
    with pytest.raises(_CaptureOutputFailure) as caught:
        _capture_output(_Sandbox(raw))
    assert caught.value.reason == "OUTPUT_INVALID"
