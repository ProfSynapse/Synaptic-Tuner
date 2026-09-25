"""Provider-free host wiring of the existing CPU release qualification lane."""

from __future__ import annotations

import hashlib
import threading
from types import SimpleNamespace

import pytest

from tuner.cloud.modal_runtime_qualification_operator import ModalRuntimeQualificationOutcome
from tuner.execution.providers.modal.packaged_binding import ModalPackagedRuntimeFactsV1
from tuner.execution.providers.modal.runtime_release_qualification import (
    ModalRuntimeReleaseFixtureReceiptV1, ModalRuntimeReleaseQualificationReceiptV1,
)
from tuner.execution.providers.modal.runtime_release_qualification_reader import (
    ModalRuntimeReleaseQualificationObservation,
)
from tuner.execution.providers.modal.contracts import operation_path
from tuner.training import modal_host_qualification as qualification
from tuner.training.modal_host_runtime import ModalHostRuntimeV1
from tests.execution.providers.test_modal_runtime_release_deployment import (
    SDK, Reader, _deployer, _observation, _plan, packaged_training_entry,
)


class _Journal:
    def __init__(self, events):
        self.events = events
        self.attempts = self
        self.calls = {}

    def claim(self, ref, evidence):
        assert type(evidence) is bytes and b"qualification_key" not in evidence
        self.events.append("claim")

    def catalog(self, name, *, encode, decode):
        assert name == "modal-runtime-qualification-calls"
        owner = self

        class _Catalog:
            def resolve(self, ref):
                return owner.calls.get(ref)

            def publish_if_absent(self, ref, value):
                if ref in owner.calls:
                    return False
                owner.calls[ref] = decode(encode(value))
                return True

        return _Catalog()


class _Call:
    is_hydrated = True
    object_id = "fc-cpu123"

    def hydrate(self, client):
        return self

    def get(self, timeout):
        assert timeout == 0
        return {
            "schema_version": "synaptic-modal-runtime-release-qualification-result/v1",
            "status_code": "completed",
        }


class _FunctionCall:
    @classmethod
    def from_id(cls, call_id, **kwargs):
        assert call_id == "fc-cpu123"
        return _Call()


def _runtime():
    plan = _plan()
    deployer, client = _deployer(Reader([None, _observation(1)]))
    deployment = deployer.deploy_once(plan, entrypoints={
        "training": packaged_training_entry,
        "self_check": __import__(
            "tuner.runtime.runtime_release_modal_self_check", fromlist=["run_runtime_release_self_check"],
        ).run_runtime_release_self_check,
    })
    packaged = ModalPackagedRuntimeFactsV1.from_release_deployment(deployment)
    names = tuple(sorted((item.volume_id, item.spec.name) for item in deployment.volumes))
    current = SimpleNamespace(observe=lambda **kwargs: packaged)
    runtime = ModalHostRuntimeV1(
        plan.release, deployment, packaged, None, names, b"q" * 32, current,
    )
    return runtime, client


def test_cpu_gate_claims_before_fixture_and_spawn_and_returns_authenticated_identity(monkeypatch):
    runtime, client = _runtime()
    monkeypatch.setattr(SDK, "FunctionCall", _FunctionCall, raising=False)
    events = []
    journal = _Journal(events)
    owner_thread = threading.get_ident()

    class _Observer:
        def __init__(self, **kwargs):
            pass

        def observe(self, facts):
            return facts

    class _Operator:
        def __init__(self, **kwargs):
            pass

        def stage_fixture_once(self, *, effect_id, deployment_facts):
            events.append("fixture")
            artifact = next(item.volume_id for item in deployment_facts.volumes
                            if item.spec.role == "artifacts")
            return ModalRuntimeReleaseFixtureReceiptV1.create(
                effect_id=effect_id, artifact_volume_id=artifact,
            )

        def submit_once(self, payload, *, expected_facts, provider_invoker=None):
            assert type(payload) is bytes and payload
            assert threading.get_ident() == owner_thread
            assert callable(provider_invoker)
            assert provider_invoker(lambda: threading.get_ident()) != owner_thread
            events.append("spawn")
            return ModalRuntimeQualificationOutcome("found", "fc-cpu123")

    class _Reader:
        def __init__(self, **kwargs):
            pass

        def observe(self, dispatch, *, provider_call_id):
            assert provider_call_id == "fc-cpu123"
            events.append("receipt")
            output = b'{"verified":true}\n'
            receipt = ModalRuntimeReleaseQualificationReceiptV1(
                effect_id=dispatch.effect_id,
                dispatch_digest=dispatch.dispatch_digest,
                provider_call_id=provider_call_id,
                deployment_facts_digest=dispatch.deployment_facts.facts_digest,
                current_observation_digest="a" * 64,
                output_path=operation_path(
                    dispatch.effect_id, "runtime-release-qualification", "output", "evidence.json",
                ),
                output_size=len(output),
                output_sha256=hashlib.sha256(output).hexdigest(),
                output_provider_entry_id="fixture-entry",
            )
            return ModalRuntimeReleaseQualificationObservation(receipt, output)

    monkeypatch.setattr(qualification, "_CurrentQualificationObserver", _Observer)
    monkeypatch.setattr(qualification, "ModalRuntimeQualificationOperator", _Operator)
    monkeypatch.setattr(qualification, "ModalRuntimeReleaseQualificationReader", _Reader)
    result = qualification.qualify_modal_runtime_for_host(
        sdk=SDK, client=client, client_binding=runtime.facts.client_binding,
        runtime=runtime, private_storage=journal,
        effect_id="cpu-qual-test", environment_name="production",
    )
    assert events == ["claim", "fixture", "spawn", "receipt"]
    assert result.provider_call_id == "fc-cpu123"
    assert result.runtime_release_digest == runtime.release.manifest_digest
    assert result.deployment_facts_digest == runtime.deployment_facts.facts_digest
    assert result.training_executed is False and result.gpu_qualified is False


def test_cpu_gate_does_not_stage_when_claim_fails(monkeypatch):
    runtime, client = _runtime()
    events = []
    journal = _Journal(events)

    def fail_claim(ref, evidence):
        events.append("claim")
        raise RuntimeError("already claimed")

    journal.claim = fail_claim
    monkeypatch.setattr(qualification, "_CurrentQualificationObserver",
                        lambda **kwargs: SimpleNamespace(observe=lambda facts: facts))
    with pytest.raises(RuntimeError, match="already claimed"):
        qualification.qualify_modal_runtime_for_host(
            sdk=SDK, client=client, client_binding=runtime.facts.client_binding,
            runtime=runtime, private_storage=journal,
            effect_id="cpu-qual-test", environment_name="production",
        )
    assert events == ["claim"]


@pytest.mark.parametrize("failure_stage,expected_phase", (
    ("fixture", "FIXTURE_STAGE"),
    ("dispatch", "DISPATCH_SUBMIT"),
    ("call", "CALL_OBSERVE"),
    ("receipt", "RECEIPT_VERIFY"),
))
def test_cpu_gate_reports_only_closed_post_claim_stage(
        monkeypatch, failure_stage, expected_phase):
    runtime, client = _runtime()
    events = []

    class _Observer:
        def __init__(self, **_kwargs):
            pass

    class _Operator:
        def __init__(self, **_kwargs):
            pass

        def stage_fixture_once(self, *, effect_id, deployment_facts):
            events.append("fixture")
            if failure_stage == "fixture":
                raise RuntimeError("HF_TOKEN=private")
            artifact = next(item.volume_id for item in deployment_facts.volumes
                            if item.spec.role == "artifacts")
            return ModalRuntimeReleaseFixtureReceiptV1.create(
                effect_id=effect_id, artifact_volume_id=artifact,
            )

        def submit_once(self, payload, *, expected_facts, provider_invoker=None):
            events.append("dispatch")
            if failure_stage == "dispatch":
                raise RuntimeError("HF_TOKEN=private")
            return ModalRuntimeQualificationOutcome("found", "fc-cpu123")

    class _Call:
        is_hydrated = True
        object_id = "fc-cpu123"

        def hydrate(self, _client):
            return self

        def get(self, timeout):
            events.append("call")
            if failure_stage == "call":
                raise RuntimeError("HF_TOKEN=private")
            return {
                "schema_version": "synaptic-modal-runtime-release-qualification-result/v1",
                "status_code": "completed",
            }

    class _FunctionCall:
        @staticmethod
        def from_id(_call_id, **_kwargs):
            return _Call()

    class _Reader:
        def __init__(self, **_kwargs):
            pass

        def observe(self, _dispatch, *, provider_call_id):
            events.append("receipt")
            raise RuntimeError("HF_TOKEN=private")

    monkeypatch.setattr(SDK, "FunctionCall", _FunctionCall, raising=False)
    monkeypatch.setattr(qualification, "_CurrentQualificationObserver", _Observer)
    monkeypatch.setattr(qualification, "ModalRuntimeQualificationOperator", _Operator)
    monkeypatch.setattr(qualification, "ModalRuntimeReleaseQualificationReader", _Reader)
    with pytest.raises(qualification.ModalHostQualificationUnavailable) as caught:
        qualification.qualify_modal_runtime_for_host(
            sdk=SDK, client=client, client_binding=runtime.facts.client_binding,
            runtime=runtime, private_storage=_Journal(events),
            effect_id="cpu-qual-test", environment_name="production",
        )
    error = caught.value
    assert error.phase == expected_phase
    assert error.retry_authorized is False
    assert "HF_TOKEN" not in str(error)
    assert error.__cause__ is None
    assert events[0] == "claim"


@pytest.mark.parametrize("operator_stage,host_phase", (
    ("FUNCTION_IDENTITY", "DISPATCH_FUNCTION_IDENTITY"),
    ("SPAWN_INDETERMINATE", "DISPATCH_SPAWN_INDETERMINATE"),
    ("CATALOG_INDETERMINATE", "DISPATCH_CATALOG_INDETERMINATE"),
))
def test_cpu_gate_projects_closed_operator_stage_only(
        monkeypatch, operator_stage, host_phase):
    runtime, client = _runtime()
    events = []

    class _Operator:
        def __init__(self, **_kwargs):
            pass

        def stage_fixture_once(self, *, effect_id, deployment_facts):
            artifact = next(item.volume_id for item in deployment_facts.volumes
                            if item.spec.role == "artifacts")
            return ModalRuntimeReleaseFixtureReceiptV1.create(
                effect_id=effect_id, artifact_volume_id=artifact,
            )

        def submit_once(self, _payload, *, expected_facts, provider_invoker=None):
            events.append("submit")
            return ModalRuntimeQualificationOutcome(
                "indeterminate", failure_stage=operator_stage,
            )

    monkeypatch.setattr(qualification, "_CurrentQualificationObserver",
                        lambda **_kwargs: SimpleNamespace(observe=lambda facts: facts))
    monkeypatch.setattr(qualification, "ModalRuntimeQualificationOperator", _Operator)
    with pytest.raises(qualification.ModalHostQualificationUnavailable) as caught:
        qualification.qualify_modal_runtime_for_host(
            sdk=SDK, client=client, client_binding=runtime.facts.client_binding,
            runtime=runtime, private_storage=_Journal(events),
            effect_id="cpu-qual-test", environment_name="production",
        )
    assert caught.value.phase == host_phase
    assert caught.value.retry_authorized is False
    assert events == ["claim", "submit"]
