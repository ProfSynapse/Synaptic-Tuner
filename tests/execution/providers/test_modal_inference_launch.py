"""Provider-free unit conformance for Modal chat launch admission."""

from __future__ import annotations

import base64
import json
import pytest

from tests.execution.providers.test_modal_inference_launch_integration import (
    _launch_case,
)
from tuner.execution.foundation_v2.canonical import (
    canonical_bytes,
    parse_canonical_object,
)
from tuner.execution.providers.modal.inference_launch import (
    ModalChatLaunchEnvelope,
    prepare_modal_chat_launch,
)
from tuner.execution.providers.modal.inference_wire import (
    ModalChatWorkerAdmission,
    ModalChatWorkerExpectation,
    admit_modal_chat_launch,
)


def _envelope(monkeypatch):
    case = _launch_case(monkeypatch)
    envelope = prepare_modal_chat_launch(**case.arguments)
    return case, envelope


def _retag(case, envelope, document):
    claim = canonical_bytes(document)
    tag = case.authenticator.sign(
        "modal-inference-launch/v1", claim, case.expectation.key_ref
    )
    return ModalChatLaunchEnvelope(
        claim,
        tag,
        envelope.stage_command_bytes,
        envelope.submit_command_bytes,
        envelope.preparation_snapshot,
    )


def _expectation(value, **changes):
    fields = (
        "configuration_bytes",
        "submit_command_digest",
        "executor_version",
        "issuer_ref",
        "audience_ref",
        "key_ref",
        "challenge_nonce",
        "artifact_root",
        "control_root",
        "cache_root",
    )
    document = {name: getattr(value, name) for name in fields}
    document.update(changes)
    return ModalChatWorkerExpectation(**document)


def test_signed_launch_roundtrip_returns_fresh_immutable_evidence(monkeypatch):
    case, envelope = _envelope(monkeypatch)
    admitted = admit_modal_chat_launch(
        envelope.argument_bytes,
        expectation=case.expectation,
        verifier=case.authenticator,
        clock=case.clock,
    )
    assert type(admitted) is ModalChatWorkerAdmission
    assert admitted.argument_bytes == envelope.argument_bytes
    assert admitted.stage_command_bytes == envelope.stage_command_bytes
    assert admitted.submit_command_bytes == envelope.submit_command_bytes
    assert admitted.preparation_snapshot == envelope.preparation_snapshot
    assert admitted.stage_command.operation.effect.kind.value == "stage"
    assert admitted.submit_command.operation.effect.kind.value == "submit"
    assert len(case.authenticator.sign_calls) == 1
    assert len(case.authenticator.verify_calls) == 1
    with pytest.raises(AttributeError):
        admitted._claim = b"changed"
    with pytest.raises(TypeError):
        ModalChatWorkerAdmission()


@pytest.mark.parametrize(
    "field",
    (
        "submit_command_digest",
        "executor_version",
        "issuer_ref",
        "audience_ref",
        "key_ref",
        "challenge_nonce",
        "artifact_root",
        "control_root",
        "cache_root",
    ),
)
def test_each_static_worker_expectation_is_bound(monkeypatch, field):
    case, envelope = _envelope(monkeypatch)
    value = (
        "0" * 64
        if field.endswith("digest")
        else ("/worker/other" if field.endswith("root") else "other")
    )
    expectation = _expectation(case.expectation, **{field: value})
    with pytest.raises(ValueError, match="Modal chat launch admission invalid"):
        admit_modal_chat_launch(
            envelope.argument_bytes,
            expectation=expectation,
            verifier=case.authenticator,
            clock=case.clock,
        )
    assert len(case.authenticator.verify_calls) <= 1


@pytest.mark.parametrize(
    "field",
    (
        "stage_ref",
        "stage_binding_digest",
        "stage_command_digest",
        "stage_effect_id",
        "stage_invocation_nonce",
        "submit_binding_digest",
        "submit_effect_id",
        "submit_invocation_nonce",
        "preparation_snapshot_sha256",
        "configuration_digest",
        "session_id",
        "stage_authenticated_receipt_digest",
        "stage_record_digest",
    ),
)
def test_validly_signed_changed_claim_projection_is_rejected(monkeypatch, field):
    case, envelope = _envelope(monkeypatch)
    document = parse_canonical_object(envelope.claim, name="claim")
    document[field] = (
        "0" * 64 if field.endswith("digest") or field.endswith("sha256") else "other"
    )
    changed = _retag(case, envelope, document)
    case.authenticator.verify_calls.clear()
    with pytest.raises(ValueError, match="Modal chat launch admission invalid"):
        admit_modal_chat_launch(
            changed.argument_bytes,
            expectation=case.expectation,
            verifier=case.authenticator,
            clock=case.clock,
        )
    assert len(case.authenticator.verify_calls) == 1


@pytest.mark.parametrize(
    "shape", ("trailing", "duplicate", "nan", "bad-base64", "oversized")
)
def test_argument_parser_rejects_noncanonical_or_unbounded_input(monkeypatch, shape):
    case, envelope = _envelope(monkeypatch)
    argument = envelope.argument_bytes
    if shape == "trailing":
        argument += b" "
    elif shape == "duplicate":
        argument = argument.replace(b'{"claim":', b'{"claim":"AA==","claim":', 1)
    elif shape == "nan":
        argument = argument[:-1] + b',"unknown":NaN}'
    elif shape == "oversized":
        argument = b"{" + b" " * (96 * 1024)
    else:
        document = json.loads(argument)
        document["claim"] = "***"
        argument = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError, match="Modal chat launch admission invalid"):
        admit_modal_chat_launch(
            argument,
            expectation=case.expectation,
            verifier=case.authenticator,
            clock=case.clock,
        )
    assert case.authenticator.verify_calls == []


@pytest.mark.parametrize(
    "roots",
    (
        ("/worker", "/worker/control", "/cache"),
        ("/same", "/same", "/cache"),
        ("relative", "/control", "/cache"),
    ),
)
def test_worker_expectation_rejects_overlapping_or_noncanonical_roots(
    monkeypatch, roots
):
    case = _launch_case(monkeypatch)
    with pytest.raises(ValueError):
        ModalChatWorkerExpectation(
            configuration_bytes=case.expectation.configuration_bytes,
            submit_command_digest=case.expectation.submit_command_digest,
            executor_version=case.expectation.executor_version,
            issuer_ref=case.expectation.issuer_ref,
            audience_ref=case.expectation.audience_ref,
            key_ref=case.expectation.key_ref,
            challenge_nonce=case.expectation.challenge_nonce,
            artifact_root=roots[0],
            control_root=roots[1],
            cache_root=roots[2],
        )


def test_wrong_signature_stops_after_one_verification(monkeypatch):
    case, envelope = _envelope(monkeypatch)
    document = json.loads(envelope.argument_bytes)
    document["claim_tag"] = base64.b64encode(b"wrong").decode()
    argument = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError, match="Modal chat launch admission invalid"):
        admit_modal_chat_launch(
            argument,
            expectation=case.expectation,
            verifier=case.authenticator,
            clock=case.clock,
        )
    assert len(case.authenticator.verify_calls) == 1


@pytest.mark.parametrize("control", (KeyboardInterrupt, SystemExit))
def test_verifier_control_flow_is_preserved(monkeypatch, control):
    case, envelope = _envelope(monkeypatch)

    class Verifier:
        def verify(self, purpose, payload, tag, key_ref):
            raise control()

    with pytest.raises(control):
        admit_modal_chat_launch(
            envelope.argument_bytes,
            expectation=case.expectation,
            verifier=Verifier(),
            clock=case.clock,
        )


def test_duck_argument_and_expectation_are_rejected_without_property_access():
    reads = []

    class Duck:
        @property
        def argument_bytes(self):
            reads.append("read")
            raise AssertionError

    with pytest.raises(ValueError, match="Modal chat launch admission invalid"):
        admit_modal_chat_launch(
            Duck(), expectation=Duck(), verifier=Duck(), clock=Duck()
        )
    assert reads == []


def test_signer_cannot_mutate_submit_binding_after_authentication(monkeypatch):
    case = _launch_case(monkeypatch)
    replacement = case.case.stage.command_bytes

    class MutatingSigner:
        calls = 0

        def sign(self, purpose, payload, key_ref):
            self.calls += 1
            object.__setattr__(case.submit_binding, "_command_bytes", replacement)
            return case.authenticator.sign(purpose, payload, key_ref)

    signer = MutatingSigner()
    with pytest.raises(ValueError, match="Modal chat launch preparation invalid"):
        prepare_modal_chat_launch(**(case.arguments | {"signer": signer}))
    assert signer.calls == 1


@pytest.mark.parametrize("control", (KeyboardInterrupt, SystemExit))
def test_signer_control_flow_is_preserved(monkeypatch, control):
    case = _launch_case(monkeypatch)

    class Signer:
        def sign(self, purpose, payload, key_ref):
            raise control()

    with pytest.raises(control):
        prepare_modal_chat_launch(**(case.arguments | {"signer": Signer()}))


@pytest.mark.parametrize("control", (KeyboardInterrupt, SystemExit))
def test_host_clock_control_flow_is_preserved_before_auth_or_sign(monkeypatch, control):
    case = _launch_case(monkeypatch)

    class Clock:
        def now_iso(self):
            raise control()

    case.case.authority.calls.clear()
    with pytest.raises(control):
        prepare_modal_chat_launch(**(case.arguments | {"clock": Clock()}))
    assert case.case.authority.calls == []
    assert case.authenticator.sign_calls == []


def test_host_clock_callback_cannot_mutate_launch_input(monkeypatch):
    case = _launch_case(monkeypatch)

    class Clock:
        def now_iso(self):
            object.__setattr__(
                case.arguments["stage_binding"], "_command_bytes", b"changed"
            )
            return case.clock.now_iso()

    case.case.authority.calls.clear()
    with pytest.raises(ValueError, match="Modal chat launch preparation invalid"):
        prepare_modal_chat_launch(**(case.arguments | {"clock": Clock()}))
    assert case.case.authority.calls == []
    assert case.authenticator.sign_calls == []
