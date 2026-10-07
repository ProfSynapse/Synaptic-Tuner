from __future__ import annotations

import pytest

from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.execution.providers.modal.coordinator_preflight import (
    AuthenticatedModalQuote, ModalOperationalPreflightAdapter, ModalQuoteBody,
    QUOTE_PURPOSE, TrustedEvidenceIdentity,
)


def evidence_tag(purpose, payload, key_ref):
    return __import__("hashlib").sha256(canonical_bytes({
        "purpose": purpose, "payload_hex": payload.hex(), "key_ref": key_ref,
    })).digest()


def body(**changes):
    value = {
        "schema_version": "synaptic-modal-quote-body/v1",
        "provider_id": "modal", "profile_ref": "a10", "account_ref": "acct",
        "namespace_ref": "namespace", "resource_digest": "1" * 64,
        "maximum_cost_minor_units": 125, "currency": "USD",
        "issued_at": "2026-09-09T12:00:00Z", "expires_at": "2026-09-09T12:05:00Z",
        "issuer_ref": "host-quoter", "evidence_ref": "quote-1",
        "audience_ref": "project-run", "challenge_nonce": "quote-nonce",
        "key_ref": "quote-key",
    }
    value.update(changes)
    return canonical_bytes(value)


def test_quote_digest_has_one_non_circular_direction():
    parsed = ModalQuoteBody.parse(body())
    assert parsed.quote_digest == __import__(
        "tuner.execution.foundation_v2.canonical", fromlist=["domain_digest"]
    ).domain_digest("synaptic-modal-quote-body/v1", parsed.canonical_bytes)
    assert "quote_digest" not in __import__("json").loads(parsed.canonical_bytes)
    quote = AuthenticatedModalQuote(parsed.canonical_bytes, b"tag")
    assert quote.body.quote_digest == parsed.quote_digest
    assert QUOTE_PURPOSE == "modal-quote-evidence/v1"


@pytest.mark.parametrize("value", [True, 1.5, -1])
def test_quote_cost_is_an_exact_nonnegative_integer(value):
    with pytest.raises(ValueError):
        ModalQuoteBody.parse(body(maximum_cost_minor_units=value))


@pytest.mark.parametrize("currency", ["EUR", "usd", "US", "USDX"])
def test_initial_quote_policy_is_exact_usd(currency):
    with pytest.raises(ValueError, match="USD"):
        ModalQuoteBody.parse(body(currency=currency))


def test_quote_time_and_canonical_shape_are_strict():
    with pytest.raises(ValueError):
        ModalQuoteBody.parse(body(expires_at="2026-09-09T12:00:00Z"))
    with pytest.raises(ValueError):
        ModalQuoteBody.parse(body(issued_at="2026-09-09T12:00:00+00:00"))
    document = __import__("json").loads(body())
    document["quote_digest"] = "0" * 64
    with pytest.raises(ValueError):
        ModalQuoteBody.parse(canonical_bytes(document))


def test_quote_values_are_reparsed_from_immutable_bytes():
    raw = bytearray(body())
    quote = AuthenticatedModalQuote(bytes(raw), b"tag")
    raw[:] = b"{}"
    assert quote.body.provider_id == "modal"
    with pytest.raises(TypeError):
        ModalQuoteBody()


@pytest.mark.parametrize("mode", [
    "ready", "source-lock-evidence/v1", "modal-deployment-evidence/v1", QUOTE_PURPOSE,
    "source_trust", "deployment_trust", "quote_trust", "resource",
    "quote_stale", "quote_lifetime", "plan_spoof",
])
def test_operational_preflight_can_be_ready_from_exact_current_facts(mode, monkeypatch):
    import base64
    import hashlib
    from dataclasses import replace
    from tuner.execution.foundation_v2.canonical import parse_canonical_object
    from tuner.execution.providers.modal.coordinator_adapter import ModalPreparationAdapter
    from tuner.execution.providers.modal.facade import ExplicitModal154ReadFacade
    from tuner.project.execution_source import ExecutionSourceV1
    from tests.execution.providers.test_modal_coordinator_adapter import inputs
    from tests.execution.providers.test_modal_coordinator_bundle import _fixture
    from tests.execution.providers.test_modal_sdk154_adapter import FakeVolume, SDK

    import tests.execution.providers.test_modal_coordinator_bundle as bundle_tests
    original_source = bundle_tests._execution_source
    original_verified = bundle_tests.verified

    def authenticated_source():
        source = original_source()
        evidence = source.source_evidence
        evidence = replace(
            evidence, tag_base64=base64.b64encode(evidence_tag(
                "source-lock-evidence/v1", evidence.authenticated_payload, evidence.key_ref,
            )).decode("ascii"),
            attestation_digest=hashlib.sha256(evidence.authenticated_payload).hexdigest(),
        )
        return replace(source, source_evidence=evidence)

    def authenticated_deployment(selection):
        evidence = original_verified(selection)
        document = evidence.to_dict()
        document["expires_at"] = "2026-08-25T12:05:00Z"
        unsigned = dict(document); unsigned.pop("tag_base64"); unsigned.pop("attestation_digest")
        payload = canonical_bytes(unsigned)
        document["tag_base64"] = base64.b64encode(evidence_tag(
            "modal-deployment-evidence/v1", payload, document["key_ref"],
        )).decode("ascii")
        document["attestation_digest"] = hashlib.sha256(canonical_bytes(unsigned)).hexdigest()
        return type(evidence).from_dict(document)

    with monkeypatch.context() as fixture_patch:
        fixture_patch.setattr(bundle_tests, "_execution_source", authenticated_source)
        fixture_patch.setattr(bundle_tests, "verified", authenticated_deployment)
        binding_template, material, *_ = _fixture()
    deployment = binding_template.deployment
    source = ExecutionSourceV1.from_dict(
        parse_canonical_object(material.execution_source_bytes, name="source")
    )

    class Clock:
        value = "2026-08-25T12:02:00Z"
        def now_iso(self): return type(self).value

    values = inputs()
    first = ModalPreparationAdapter(
        profile=values["profile"], binding=values["binding"],
        resolved=material.planning_request, runtime_environment=values["runtime_environment"],
        quote_digest="0" * 64, timeout_seconds=values["timeout_seconds"], clock=Clock(),
    )
    _, execution, _ = first._snapshot()
    quote_changes = {
        "profile_ref": values["profile"].profile,
        "account_ref": execution.scope.account_ref,
        "namespace_ref": execution.scope.namespace_ref,
        "resource_digest": execution.resource_digest,
        "issued_at": "2026-08-25T12:01:00Z",
        "expires_at": "2026-08-25T12:05:00Z",
    }
    if mode == "resource": quote_changes["resource_digest"] = "f" * 64
    if mode == "quote_stale": quote_changes["expires_at"] = "2026-08-25T12:01:30Z"
    if mode == "quote_lifetime":
        quote_changes.update(issued_at="2026-08-25T12:00:00Z", expires_at="2026-08-25T12:06:00Z")
    quote_bytes = body(**quote_changes)
    quote = AuthenticatedModalQuote(
        quote_bytes, evidence_tag(QUOTE_PURPOSE, quote_bytes, "quote-key"),
    )
    adapter = ModalPreparationAdapter(
        profile=values["profile"], binding=values["binding"],
        resolved=material.planning_request, runtime_environment=values["runtime_environment"],
        quote_digest=quote.body.quote_digest, timeout_seconds=values["timeout_seconds"], clock=Clock(),
    )
    provider = adapter._snapshot()[0].provider
    plan = adapter._snapshot()[2]
    selection = deployment.selection
    monkeypatch.setattr(FakeVolume, "calls", [])
    monkeypatch.setattr(FakeVolume, "registry", {
        values["profile"].control_volume_ref: FakeVolume("cv"),
        values["profile"].artifact_volume_ref: FakeVolume("av"),
    })

    class Secret:
        fail = False
        calls = []
        @classmethod
        def from_name(cls, name, **kwargs):
            cls.calls.append((name, kwargs))
            value = cls(); value.name = name; value.object_id = "st-one"; value.is_hydrated = False
            return value
        def hydrate(self, client):
            if type(self).fail: raise RuntimeError("missing required key")
            self.is_hydrated = True; return self

    sdk = type("PreflightSDK", (), {
        "__version__": "1.5.4", "Volume": FakeVolume, "Function": SDK.Function,
        "FunctionCall": SDK.FunctionCall, "Secret": Secret, "exception": SDK.exception,
    })
    provider_calls = []
    facade = ExplicitModal154ReadFacade(
        values["binding"], sdk=sdk, client=object(),
        scope_observer=lambda client: provider_calls.append("scope") or (
            values["binding"].account_ref, values["binding"].workspace_ref,
            values["binding"].environment_ref, values["binding"].client_ref,
        ), deployment_observer=lambda **kwargs: provider_calls.append("deployment") or selection,
        volume_names={"cv": values["profile"].control_volume_ref, "av": values["profile"].artifact_volume_ref},
    )

    class Auth:
        denied = None
        calls = []
        def sign(self, *args): return b"tag"
        def verify(self, purpose, payload, tag, key_ref):
            type(self).calls.append((purpose, payload, tag, key_ref))
            return (purpose != type(self).denied
                    and tag == evidence_tag(purpose, payload, key_ref))

    operational = ModalOperationalPreflightAdapter(
        adapter, execution_source_bytes=material.execution_source_bytes,
        deployment_bytes=binding_template.deployment_bytes, quote=quote,
        control_volume_id="cv", artifact_volume_id="av", facade=facade,
        authenticator=Auth(), clock=Clock(),
        source_trust=TrustedEvidenceIdentity(
            "wrong-issuer" if mode == "source_trust" else source.source_evidence.issuer_ref,
            source.source_evidence.key_ref, source.source_evidence.audience_ref,
        ),
        deployment_trust=TrustedEvidenceIdentity(
            "wrong-issuer" if mode == "deployment_trust" else deployment.issuer_ref,
            deployment.key_ref, deployment.audience_ref,
        ),
        quote_trust=TrustedEvidenceIdentity(
            "wrong-issuer" if mode == "quote_trust" else quote.body.issuer_ref,
            quote.body.key_ref, quote.body.audience_ref,
        ),
    )
    Auth.denied = mode if mode in {
        "source-lock-evidence/v1", "modal-deployment-evidence/v1", QUOTE_PURPOSE,
    } else None
    Auth.calls.clear(); provider_calls.clear(); FakeVolume.calls.clear(); Secret.calls.clear()
    if mode == "plan_spoof":
        from tuner.execution.foundation_v2.canonical import FoundationError

        class ForgedPlan:
            def __eq__(self, other): return True
            def __ne__(self, other): return False
            def __getattr__(self, name): return getattr(plan, name)

        with pytest.raises(FoundationError):
            operational.preflight(ForgedPlan())
        assert Auth.calls == []
        assert provider_calls == [] and FakeVolume.calls == [] and Secret.calls == []
        return
    result = operational.preflight(plan)
    if mode != "ready":
        assert result.ready is False
        assert provider_calls == [] and FakeVolume.calls == [] and Secret.calls == []
        Auth.denied = None
        return
    assert result.ready is True and result.authorization[0].maximum_cost_minor_units == 125
    assert operational.describe(provider) == adapter.describe(provider)
    assert Auth.calls == [
        ("source-lock-evidence/v1", source.source_evidence.authenticated_payload,
         source.source_evidence.tag, source.source_evidence.key_ref),
        ("modal-deployment-evidence/v1", deployment.authenticated_payload,
         deployment.tag, deployment.key_ref),
        (QUOTE_PURPOSE, quote.body.canonical_bytes, quote.tag, quote.body.key_ref),
    ]
    assert Secret.calls and all(
        call[1]["environment_name"] == values["binding"].environment_ref
        and call[1]["client"] is facade.client
        and call[1]["required_keys"] == list(requirement.required_keys)
        for call, requirement in zip(Secret.calls, values["profile"].secrets, strict=True)
    )
    Clock.value = quote.body.expires_at
    provider_calls.clear()
    assert operational.preflight(plan).ready is False and provider_calls == []
    Clock.value = "2026-08-25T12:02:00Z"
    FakeVolume.registry[values["profile"].control_volume_ref].object_id = "alien"
    assert operational.preflight(plan).ready is False
    FakeVolume.registry[values["profile"].control_volume_ref].object_id = "cv"
    Secret.fail = True
    assert operational.preflight(plan).ready is False
