from __future__ import annotations

import hashlib
from decimal import Decimal
from pathlib import Path
import json
import sys

import pytest

from examples.modal_chat.authority import HMACAuthenticator
from examples.modal_chat.configuration import (
    ModalChatConfigurationError,
    RATE_CALCULATION_PURPOSE,
    build_authenticated_inference_configuration,
    issue_authenticated_modal_quote,
    _packaged,
)
from examples.modal_chat.deployment import ModalChatScope
from tuner.execution.providers.modal.inference_preparation import (
    CONFIG_EVIDENCE_PURPOSE,
    ModalInferencePreparationConfig,
)
from tuner.execution.providers.modal.coordinator_preflight import QUOTE_PURPOSE
from tuner.execution.foundation_v2.canonical import canonical_bytes

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = ROOT / "docs/review/evidence/modal-inference-qualified-d6e29ed.json"
READBACK = ROOT / "docs/review/evidence/modal-inference-qualified-d6e29ed-readback.json"


@pytest.fixture(autouse=True)
def _synthetic_reviewed_evidence(tmp_path, monkeypatch):
    """Provider-free evidence that still traverses every production parser."""
    historical_capture = json.loads(CAPTURE.read_text(encoding="utf-8"))
    historical_readback = json.loads(READBACK.read_text(encoding="utf-8"))
    manifest, packaged = _packaged()
    candidate = historical_capture["candidate"]
    candidate["operator_selection"] = {
        "image": manifest["base_registry_reference"],
        "source_commit": "1" * 40,
    }
    candidate["packaged_runtime"] = {
        **{
            name: packaged[name]
            for name in (
                "dependency_lock_digest",
                "runtime_lock_digest",
                "source_lock_digest",
                "worker_closure_digest",
            )
        },
        "status": "PACKAGED_RUNTIME_VERIFIED",
    }
    candidate["python"] = {
        "implementation": manifest["python"]["implementation"],
        "version": packaged["python_version"],
        "executable": packaged["python_executable"],
        "executable_sha256": packaged["python_executable_digest"],
    }
    candidate["distributions"] = dict(manifest["distributions"])
    candidate["requirements"]["modal"] = {
        "present": True,
        "version": manifest["sdk_version"],
    }
    historical_capture["provider_image_id"] = "im-fixture"
    historical_readback["candidate"] = candidate
    capture_path = tmp_path / "synthetic-capture.json"
    readback_path = tmp_path / "synthetic-readback.json"
    capture_path.write_bytes(canonical_bytes(historical_capture) + b"\n")
    readback_path.write_bytes(canonical_bytes(historical_readback) + b"\n")
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "CAPTURE", capture_path)
    monkeypatch.setattr(module, "READBACK", readback_path)


class _Hydrated:
    def __init__(self, *, name=None, object_id=None):
        self.name, self.object_id, self.is_hydrated = name, object_id, False

    def hydrate(self, client):
        self.is_hydrated = True


class _Billing:
    def rates(self):
        return {
            "gpu_hour_cost_a10g": Decimal("1.10000"),
            "cpu_hour_cost": Decimal("0.04730"),
            "cpu_hour_cost_sandbox": Decimal("0.141900"),
            "mem_gib_hour_cost": Decimal("0.00800"),
            "mem_gib_hour_cost_sandbox": Decimal("0.024000"),
        }


class _Clock:
    def now_iso(self):
        return "2026-09-14T13:36:30Z"


class _SDK:
    __version__ = "1.5.4"
    workspace_name = "workspace-a"

    class Workspace:
        @staticmethod
        def from_context(*, client):
            value = _Hydrated(name=_SDK.workspace_name, object_id="ws-a")
            value.billing = _Billing()
            return value

    class Environment:
        @staticmethod
        def from_name(name, *, create_if_missing, client):
            assert create_if_missing is False
            return _Hydrated(name=name, object_id="en-a")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _authority() -> HMACAuthenticator:
    return HMACAuthenticator(
        {"config-key": b"k" * 32},
        allowed_purposes=frozenset(
            {CONFIG_EVIDENCE_PURPOSE, QUOTE_PURPOSE, RATE_CALCULATION_PURPOSE}
        ),
    )


def _arguments(scope, authority):
    return dict(
        scope=scope,
        reviewed_capture_path=CAPTURE,
        reviewed_capture_sha256=_sha(CAPTURE),
        reviewed_readback_path=READBACK,
        reviewed_readback_sha256=_sha(READBACK),
        provider={"provider_id": "modal", "profile_ref": "chat-a10g"},
        application={
            "app_name": "synaptic-chat-v1",
            "app_ref": "chat-app",
            "sandbox_entrypoint": "chat-entry",
            "worker_ref": "chat-worker",
        },
        volumes={
            "source_artifact_volume_ref": "source-artifacts",
            "source_artifact_volume_id": "vo-source",
            "chat_control_volume_ref": "chat-control",
            "chat_control_volume_id": "vo-control",
            "model_cache_volume_ref": "model-cache",
            "model_cache_volume_id": "vo-model",
            "key_ref": "chat-evidence-key",
        },
        resources={
            "accelerator": "A10G",
            "accelerator_count": 1,
            "cpu_millicores": 4000,
            "memory_mb": 16384,
            "service_port": 8000,
            "provider_timeout_seconds": 900,
            "provider_idle_timeout_seconds": 300,
            "max_retries": 0,
        },
        serving={
            "served_model_name": "reviewed-chat",
            "gpu_memory_utilization_milli": 730,
            "enforce_eager": False,
            "tokenizer_mode": "mistral",
            "max_lora_rank": 32,
            "readiness_request_timeout_milliseconds": 750,
            "max_tokens": 256,
            "temperature_milli": 250,
            "top_p_milli": 875,
        },
        policy={
            "startup_timeout_seconds": 600,
            "request_timeout_seconds": 60,
            "idle_timeout_seconds": 300,
            "absolute_lifetime_seconds": 900,
            "max_turns": 32,
            "max_history_bytes": 65536,
            "max_request_bytes": 8192,
            "max_response_bytes": 65536,
        },
        secrets=[{"name": "model-download", "required_keys": ["HF_TOKEN"]}],
        evidence={
            "issuer_ref": "host-config",
            "evidence_ref": "config-1",
            "audience_ref": "chat-session",
            "challenge_nonce": "config-nonce",
            "key_ref": "config-key",
            "verified_at": "2026-09-14T13:36:00Z",
            "expires_at": "2026-09-14T13:41:00Z",
        },
        authenticator=authority,
    )


def test_builds_exact_reviewed_packaged_configuration_and_hmac() -> None:
    scope = ModalChatScope(
        sdk=_SDK(),
        client=object(),
        environment_name="environment-a",
        client_ref="host-a",
    )
    authority = _authority()
    authenticated, trust = build_authenticated_inference_configuration(
        **_arguments(scope, authority)
    )
    config = ModalInferencePreparationConfig.parse(authenticated.body_bytes)
    document = config.document

    assert document["image"]["provider_image_id"] == "im-fixture"
    assert document["client"]["workspace_ref"] == "workspace-a"
    assert document["resources"]["accelerator"] == "A10G"
    assert trust.key_ref == "config-key"
    assert authority.verify(
        CONFIG_EVIDENCE_PURPOSE,
        authenticated.body_bytes,
        authenticated.tag,
        trust.key_ref,
    )


def test_rejects_unreviewed_evidence_digest() -> None:
    scope = ModalChatScope(
        sdk=_SDK(),
        client=object(),
        environment_name="environment-a",
        client_ref="host-a",
    )
    authority = _authority()
    arguments = _arguments(scope, authority)
    arguments["reviewed_capture_sha256"] = "0" * 64

    with pytest.raises(ModalChatConfigurationError, match="configuration_invalid"):
        build_authenticated_inference_configuration(**arguments)


def test_rejects_candidate_distribution_map_substitution(tmp_path) -> None:
    scope = ModalChatScope(
        sdk=_SDK(),
        client=object(),
        environment_name="environment-a",
        client_ref="host-a",
    )
    authority = _authority()
    capture = json.loads(CAPTURE.read_text(encoding="utf-8"))
    readback = json.loads(READBACK.read_text(encoding="utf-8"))
    changed = dict(capture["candidate"]["distributions"])
    changed["modal"] = "1.5.3"
    capture["candidate"]["distributions"] = changed
    readback["candidate"]["distributions"] = changed
    capture_path, readback_path = tmp_path / "capture.json", tmp_path / "readback.json"
    capture_path.write_bytes(canonical_bytes(capture) + b"\n")
    readback_path.write_bytes(canonical_bytes(readback) + b"\n")
    arguments = _arguments(scope, authority)
    arguments.update(
        reviewed_capture_path=capture_path,
        reviewed_capture_sha256=_sha(capture_path),
        reviewed_readback_path=readback_path,
        reviewed_readback_sha256=_sha(readback_path),
    )
    with pytest.raises(ModalChatConfigurationError, match="configuration_invalid"):
        build_authenticated_inference_configuration(**arguments)


def test_rejects_capture_readback_source_commitment_substitution(tmp_path) -> None:
    scope = ModalChatScope(
        sdk=_SDK(),
        client=object(),
        environment_name="environment-a",
        client_ref="host-a",
    )
    authority = _authority()
    capture = json.loads(CAPTURE.read_text(encoding="utf-8"))
    capture["candidate"]["operator_selection"]["source_commit"] = "2" * 40
    capture_path = tmp_path / "stale-capture.json"
    capture_path.write_bytes(canonical_bytes(capture) + b"\n")
    arguments = _arguments(scope, authority)
    arguments.update(
        reviewed_capture_path=capture_path,
        reviewed_capture_sha256=_sha(capture_path),
    )
    with pytest.raises(ModalChatConfigurationError, match="configuration_invalid"):
        build_authenticated_inference_configuration(**arguments)


def test_cpu_qualification_does_not_override_selected_bounded_resources() -> None:
    scope = ModalChatScope(
        sdk=_SDK(),
        client=object(),
        environment_name="environment-a",
        client_ref="host-a",
    )
    authority = _authority()
    arguments = _arguments(scope, authority)
    arguments["resources"] = {**arguments["resources"], "accelerator": "CPU"}
    authenticated, _ = build_authenticated_inference_configuration(**arguments)

    assert (
        ModalInferencePreparationConfig.parse(authenticated.body_bytes).document[
            "resources"
        ]["accelerator"]
        == "CPU"
    )


def test_configuration_rejects_resources_outside_existing_policy() -> None:
    scope = ModalChatScope(
        sdk=_SDK(),
        client=object(),
        environment_name="environment-a",
        client_ref="host-a",
    )
    authority = _authority()
    arguments = _arguments(scope, authority)
    arguments["resources"] = {**arguments["resources"], "max_retries": 1}

    with pytest.raises(ModalChatConfigurationError, match="configuration_invalid"):
        build_authenticated_inference_configuration(**arguments)


def test_fresh_rates_issue_bound_quote_and_signed_calculation() -> None:
    scope = ModalChatScope(
        sdk=_SDK(),
        client=object(),
        environment_name="environment-a",
        client_ref="host-a",
    )
    authority = _authority()
    configuration, _ = build_authenticated_inference_configuration(
        **_arguments(scope, authority)
    )

    quote, trust, calculation, calculation_tag = issue_authenticated_modal_quote(
        scope=scope,
        configuration=configuration,
        namespace_ref="chat-namespace",
        maximum_cost_minor_units=50,
        execution_kind="function",
        issued_at="2026-09-14T13:36:30Z",
        expires_at="2026-09-14T13:41:30Z",
        issuer_ref="host-quoter",
        audience_ref="project-run",
        challenge_nonce="quote-nonce",
        key_ref="config-key",
        authenticator=authority,
        clock=_Clock(),
    )

    assert quote.body.maximum_cost_minor_units == 50
    assert quote.body.evidence_ref == "rate-" + hashlib.sha256(calculation).hexdigest()
    assert authority.verify(QUOTE_PURPOSE, quote.body_bytes, quote.tag, trust.key_ref)
    assert authority.verify(
        RATE_CALCULATION_PURPOSE, calculation, calculation_tag, trust.key_ref
    )
    assert not authority.verify(
        CONFIG_EVIDENCE_PURPOSE, calculation, calculation_tag, trust.key_ref
    )


@pytest.mark.parametrize(
    "bad_rate",
    [None, "", Decimal("NaN"), Decimal("Infinity"), Decimal("-0.1"), Decimal("0")],
)
def test_quote_rejects_missing_malformed_or_nonpositive_rate(bad_rate) -> None:
    scope = ModalChatScope(
        sdk=_SDK(),
        client=object(),
        environment_name="environment-a",
        client_ref="host-a",
    )
    authority = _authority()
    configuration, _ = build_authenticated_inference_configuration(
        **_arguments(scope, authority)
    )
    original = _Billing.rates

    def rates(self):
        value = original(self)
        if bad_rate is None:
            value.pop("gpu_hour_cost_a10g")
        else:
            value["gpu_hour_cost_a10g"] = bad_rate
        return value

    _Billing.rates = rates
    try:
        with pytest.raises(ModalChatConfigurationError, match="quote_invalid"):
            issue_authenticated_modal_quote(
                scope=scope,
                configuration=configuration,
                namespace_ref="chat-namespace",
                maximum_cost_minor_units=50,
                execution_kind="function",
                issued_at="2026-09-14T13:36:30Z",
                expires_at="2026-09-14T13:41:30Z",
                issuer_ref="host-quoter",
                audience_ref="project-run",
                challenge_nonce="quote-nonce",
                key_ref="config-key",
                authenticator=authority,
                clock=_Clock(),
            )
    finally:
        _Billing.rates = original


def test_quote_rejects_budget_below_nominal_estimate() -> None:
    scope = ModalChatScope(
        sdk=_SDK(),
        client=object(),
        environment_name="environment-a",
        client_ref="host-a",
    )
    authority = _authority()
    configuration, _ = build_authenticated_inference_configuration(
        **_arguments(scope, authority)
    )
    with pytest.raises(ModalChatConfigurationError, match="quote_invalid"):
        issue_authenticated_modal_quote(
            scope=scope,
            configuration=configuration,
            namespace_ref="chat-namespace",
            maximum_cost_minor_units=1,
            execution_kind="sandbox",
            issued_at="2026-09-14T13:36:30Z",
            expires_at="2026-09-14T13:41:30Z",
            issuer_ref="host-quoter",
            audience_ref="project-run",
            challenge_nonce="quote-nonce",
            key_ref="config-key",
            authenticator=authority,
            clock=_Clock(),
        )


def test_quote_rejects_configuration_wrapper_with_invalid_tag_before_rates() -> None:
    scope = ModalChatScope(
        sdk=_SDK(),
        client=object(),
        environment_name="environment-a",
        client_ref="host-a",
    )
    authority = _authority()
    configuration, _ = build_authenticated_inference_configuration(
        **_arguments(scope, authority)
    )
    invalid = type(configuration)(configuration.body_bytes, b"wrong-tag")
    with pytest.raises(ModalChatConfigurationError, match="quote_invalid"):
        issue_authenticated_modal_quote(
            scope=scope,
            configuration=invalid,
            namespace_ref="chat-namespace",
            maximum_cost_minor_units=50,
            execution_kind="function",
            issued_at="2026-09-14T13:36:30Z",
            expires_at="2026-09-14T13:41:30Z",
            issuer_ref="host-quoter",
            audience_ref="project-run",
            challenge_nonce="quote-nonce",
            key_ref="config-key",
            authenticator=authority,
            clock=_Clock(),
        )


@pytest.mark.parametrize("failure", ["expired", "scope-changed"])
def test_quote_rejects_invalid_window_or_changed_live_scope(failure) -> None:
    sdk = _SDK()
    scope = ModalChatScope(
        sdk=sdk, client=object(), environment_name="environment-a", client_ref="host-a"
    )
    authority = _authority()
    configuration, _ = build_authenticated_inference_configuration(
        **_arguments(scope, authority)
    )
    expires_at = (
        "2026-09-14T13:46:30Z" if failure == "expired" else "2026-09-14T13:41:30Z"
    )
    if failure == "scope-changed":
        _SDK.workspace_name = "workspace-b"
    try:
        with pytest.raises(ModalChatConfigurationError, match="quote_invalid"):
            issue_authenticated_modal_quote(
                scope=scope,
                configuration=configuration,
                namespace_ref="chat-namespace",
                maximum_cost_minor_units=50,
                execution_kind="function",
                issued_at="2026-09-14T13:36:30Z",
                expires_at=expires_at,
                issuer_ref="host-quoter",
                audience_ref="project-run",
                challenge_nonce="quote-nonce",
                key_ref="config-key",
                authenticator=authority,
                clock=_Clock(),
            )
    finally:
        _SDK.workspace_name = "workspace-a"
