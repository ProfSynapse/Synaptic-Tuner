"""Provider-free canonical Modal worker dispatch codec tests."""

from dataclasses import replace
import json

import pytest

from tests.execution.providers.test_modal_coordinator_wire import wire_case
from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.execution.providers.modal.coordinator_dispatch import (
    build_modal_worker_dispatch, parse_modal_worker_dispatch,
)
from tuner.execution.providers.modal.coordinator_wire import ModalWorkerLaunchExpectation


def encoded(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()


def dispatch_case(monkeypatch):
    values = wire_case(monkeypatch)
    envelope, expectation = values[1], values[3]
    dispatch = build_modal_worker_dispatch(
        expectation, envelope.claim, envelope.claim_tag,
    )
    return values, dispatch


def test_dispatch_is_one_canonical_argument_and_roundtrips_exactly(monkeypatch):
    values, dispatch = dispatch_case(monkeypatch)
    restored = parse_modal_worker_dispatch(dispatch.canonical_bytes)
    assert restored.canonical_bytes == dispatch.canonical_bytes
    assert restored.expectation == values[3]
    assert restored.launch_claim == values[1].claim
    assert restored.launch_claim_tag == values[1].claim_tag


@pytest.mark.parametrize("fault", ["unknown", "noncanonical", "base64", "secret"])
def test_dispatch_parser_rejects_nonclosed_or_credential_payloads(monkeypatch, fault):
    _, dispatch = dispatch_case(monkeypatch)
    document = json.loads(dispatch.canonical_bytes)
    if fault == "unknown":
        document["unknown"] = True
    elif fault == "base64":
        document["launch_claim_tag_base64"] = "***"
    elif fault == "secret":
        document["expectation"]["token_secret"] = "credential-value"
    else:
        with pytest.raises(ValueError):
            parse_modal_worker_dispatch(dispatch.canonical_bytes + b" ")
        return
    with pytest.raises(ValueError):
        parse_modal_worker_dispatch(encoded(document))


def test_builder_rejects_launch_projection_not_fixed_by_expectation(monkeypatch):
    values = wire_case(monkeypatch)
    envelope, expectation = values[1], values[3]
    with pytest.raises(ValueError, match="differs from expectation"):
        build_modal_worker_dispatch(
            replace(expectation, stage_bundle_sha256="f" * 64),
            envelope.claim, envelope.claim_tag,
        )


def test_parser_reconstructs_expectation_instead_of_caching_object(monkeypatch):
    values, dispatch = dispatch_case(monkeypatch)
    first = parse_modal_worker_dispatch(dispatch.canonical_bytes)
    object.__setattr__(values[3], "key_ref", "poisoned")
    second = parse_modal_worker_dispatch(dispatch.canonical_bytes)
    assert first.expectation == second.expectation
    assert second.expectation.key_ref == "stage-key"


def test_builder_reconstructs_and_rejects_mutated_expectation(monkeypatch):
    values = wire_case(monkeypatch)
    expectation = values[3]
    object.__setattr__(expectation, "stage_bundle_size", 8_388_609)
    with pytest.raises(ValueError, match="bound"):
        build_modal_worker_dispatch(
            expectation, values[1].claim, values[1].claim_tag,
        )


def test_builder_rejects_expectation_subclass_without_observing_it(monkeypatch):
    values = wire_case(monkeypatch)
    observed = []

    class HostileExpectation(ModalWorkerLaunchExpectation):
        def __getattribute__(self, name):
            observed.append(name)
            return super().__getattribute__(name)

    hostile = object.__new__(HostileExpectation)
    with pytest.raises(TypeError, match="exact worker launch expectation"):
        build_modal_worker_dispatch(hostile, values[1].claim, values[1].claim_tag)
    assert observed == []
