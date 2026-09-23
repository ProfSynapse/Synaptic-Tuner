"""Provider-free checks for one-shot Modal host runtime bootstrap."""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
import os
import signal

import pytest

from synaptic_tuner.api.v1.secrets import SecretRef
from tuner.execution.providers.modal.binding import ModalClientBinding
from tuner.execution.providers.modal.runtime_release_deployment import ModalRuntimeReleaseDeploymentError
from tuner.training.contracts import ResourceSpec
from tuner.training import modal_host_runtime as host


class _Hydrated:
    is_hydrated = True

    def __init__(self, name: str, object_id: str = ""):
        self.name, self.object_id = name, object_id

    def hydrate(self, client):
        return self


class _Workspace(_Hydrated):
    rates = {"gpu_hour_cost_a100_80gb_fixture": Decimal("2.50")}

    def __init__(self):
        super().__init__("workspace")
        self.billing = SimpleNamespace(rates=lambda: self.rates)

    @classmethod
    def from_context(cls, **kwargs):
        return cls()


class _Environment(_Hydrated):
    @classmethod
    def from_name(cls, name, **kwargs):
        assert kwargs["create_if_missing"] is False
        return cls(name)


class _Volume:
    created = []
    identities = {}
    objects = None

    @classmethod
    def create(cls, name, **kwargs):
        assert kwargs["allow_existing"] is False
        cls.created.append(name)
        cls.identities[name] = f"vo-{len(cls.created)}"

    @classmethod
    def from_name(cls, name, **kwargs):
        assert kwargs["create_if_missing"] is False
        return _Hydrated(name, cls.identities[name])


_Volume.objects = _Volume


class _Secret:
    created = []
    identities = {}
    objects = None

    @classmethod
    def create(cls, name, values, **kwargs):
        assert kwargs["allow_existing"] is False
        cls.created.append((name, tuple(values)))
        cls.identities[name] = f"st-{len(cls.created)}"

    @classmethod
    def from_name(cls, name, **kwargs):
        assert tuple(kwargs["required_keys"]) in (
            ("SYNAPTIC_MODAL_QUALIFICATION_HMAC_KEY",), ("HF_TOKEN",),
        )
        return _Hydrated(name, cls.identities[name])


_Secret.objects = _Secret


class _App:
    created = []

    def __init__(self, name, **kwargs):
        assert kwargs == {"include_source": False}
        self.name = name
        self.created.append(name)

    def run(self, **kwargs):
        class _Context:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False
        assert kwargs["name"] == self.name + "-build"
        return _Context()


class _SDK:
    __version__ = "1.5.4"
    Workspace, Environment = _Workspace, _Environment
    Volume, Secret, App = _Volume, _Secret, _App


@dataclass
class _Attempts:
    refs: list[str]

    def claim(self, ref, evidence):
        assert type(evidence) is bytes and b"HF_TOKEN" not in evidence
        self.refs.append(ref)


@dataclass
class _Storage:
    attempts: _Attempts


class _Resolver:
    def resolve(self, reference):
        assert reference == SecretRef("env", "HF_TOKEN")
        return "test-token-never-persist"


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    if os.name == "nt":
        monkeypatch.setattr(host, "require_bounded_modal_deployment_host", lambda: None)
    _Workspace.rates = {"gpu_hour_cost_a100_80gb_fixture": Decimal("2.50")}
    _Volume.created.clear()
    _Volume.identities.clear()
    _Secret.created.clear()
    _Secret.identities.clear()
    _App.created.clear()


def _binding():
    return ModalClientBinding("workspace", "workspace", "production", "client", "1.5.4")


def test_scoped_quote_reports_exclusions_and_enforces_ceiling(monkeypatch):
    monkeypatch.setattr(host, "_A100_80GB_RATE_KEY", "gpu_hour_cost_a100_80gb_fixture")
    client = object()
    rates = host.observe_scoped_modal_gpu_rates(sdk=_SDK, client=client, client_binding=_binding())
    assert rates == {"gpu_hour_cost_a100_80gb_fixture": "2.50"}
    resource = ResourceSpec("A100-80GB", 1, 3600)
    quote = host.quote_modal_runtime_for_host(
        sdk=_SDK, client=client, client_binding=_binding(),
        recipe_resource=resource, maximum_cost_minor_units=250,
    )
    assert quote.gpu_only_timeout_estimate_minor_units == 250
    assert quote.authorization_semantics == "operator-maximum-not-provider-billing-cap"
    assert "storage" in quote.excluded_billing_dimensions
    with pytest.raises(ValueError, match="ceiling"):
        host.quote_modal_runtime_for_host(
            sdk=_SDK, client=client, client_binding=_binding(),
            recipe_resource=resource, maximum_cost_minor_units=249,
        )
    with pytest.raises(ValueError, match="unsupported"):
        host.quote_modal_runtime_for_host(
            sdk=_SDK, client=client, client_binding=_binding(),
            recipe_resource=ResourceSpec("A100-80GB", 2, 3600),
            maximum_cost_minor_units=1000,
        )
    assert not _Volume.created and not _Secret.created and not _App.created


def test_observed_a100_key_quotes_gpu_only_initial_smoke():
    _Workspace.rates = {"gpu_hour_cost_a100_80gb": Decimal("2.50000")}
    quote = host.quote_modal_runtime_for_host(
        sdk=_SDK, client=object(), client_binding=_binding(),
        recipe_resource=ResourceSpec("A100-80GB", 1, 1800),
        maximum_cost_minor_units=200,
    )
    assert quote.rate_key == "gpu_hour_cost_a100_80gb"
    assert quote.gpu_only_timeout_estimate_minor_units == 125
    assert quote.maximum_cost_minor_units == 200


def test_scoped_quote_rejects_extreme_decimal_rate(monkeypatch):
    _Workspace.rates = {"gpu_hour_cost_a100_80gb": Decimal("1E-999999")}
    with pytest.raises(ValueError, match="scoped rate is invalid"):
        host.quote_modal_runtime_for_host(
            sdk=_SDK, client=object(), client_binding=_binding(),
            recipe_resource=ResourceSpec("A100-80GB", 1, 1800),
            maximum_cost_minor_units=200,
        )
    _Workspace.rates = {"gpu_hour_cost_a100_80gb_fixture": Decimal("2.50")}


def test_unverified_rate_key_fails_closed_before_effect(monkeypatch):
    monkeypatch.setattr(host, "_A100_80GB_RATE_KEY", None)
    monkeypatch.setattr(host, "plan_modal_build_material", lambda path: {"intent_digest": "a" * 64})
    storage = _Storage(_Attempts([]))
    with pytest.raises(ValueError, match="not independently verified"):
        host.prepare_modal_runtime_for_host(
            sdk=_SDK, client=object(), client_binding=_binding(), profile_path=Path("profile.json"),
            runtime_material_intent_digest="a" * 64, recipe_resource=ResourceSpec("A100-80GB"),
            maximum_cost_minor_units=1000, private_storage=storage, secret_resolver=_Resolver(),
            hf_token_ref=SecretRef("env", "HF_TOKEN"), qualification_secret_name="qualify",
            hf_token_secret_name="hf", app_name="training", environment_name="production",
        )
    assert storage.attempts.refs == []
    assert not _Volume.created and not _Secret.created and not _App.created


def test_provider_rate_error_is_closed(monkeypatch):
    def fail():
        raise RuntimeError("credential=must-not-escape")

    def init(self):
        _Hydrated.__init__(self, "workspace")
        self.billing = SimpleNamespace(rates=fail)

    monkeypatch.setattr(_Workspace, "__init__", init)
    with pytest.raises(RuntimeError, match="modal_rate_read_failed") as caught:
        host.observe_scoped_modal_gpu_rates(
            sdk=_SDK, client=object(), client_binding=_binding(),
        )
    assert "must-not-escape" not in str(caught.value)


def test_preparation_claims_before_provision_and_uses_exact_three_volumes(monkeypatch):
    monkeypatch.setattr(host, "_A100_80GB_RATE_KEY", "gpu_hour_cost_a100_80gb_fixture")
    monkeypatch.setattr(host, "plan_modal_build_material", lambda path: {"intent_digest": "a" * 64})
    events = []
    original_create = _Volume.create.__func__

    def recorded_create(cls, name, **kwargs):
        events.append("volume")
        original_create(cls, name, **kwargs)

    monkeypatch.setattr(_Volume, "create", classmethod(recorded_create))
    candidate = SimpleNamespace(capture_digest="b" * 64)
    release = SimpleNamespace(manifest_digest="c" * 64)
    facts = SimpleNamespace(app_name="training-fake", function_name="packaged-training",
                            app_id="ap-fake", deployment_generation=1,
                            function_id="fu-training", self_check_function_name="packaged-self-check",
                            self_check_function_id="fu-check", control_volume_id="vo-1",
                            artifact_volume_id="vo-2", model_cache_volume_id="vo-3")
    monkeypatch.setattr(host, "capture_modal_build_candidate", lambda **kw: candidate)
    monkeypatch.setattr(host, "build_modal_runtime_release_v2", lambda *args, **kw: release)
    monkeypatch.setattr(host, "ModalRuntimeReleaseDeploymentPlanV1", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(host.ModalPackagedRuntimeFactsV1, "from_release_deployment", lambda value: facts)

    class _Reader:
        def __init__(self, **kwargs):
            pass

    class _Deployer:
        def __init__(self, **kwargs):
            pass

        def observe(self, plan):
            return None

        def deploy_once(self, plan, **kwargs):
            assert tuple(v.role for v in plan.volumes) == ("control", "artifacts", "model_cache")
            assert plan.functions[0].volume_roles == ("control", "artifacts", "model_cache")
            assert plan.functions[0].gpu == "A100-80GB"
            assert kwargs["entrypoints"]["training"] is host.run_modal_packaged_training
            events.append("deploy")
            return SimpleNamespace(secrets=(
                SimpleNamespace(spec=SimpleNamespace(name=plan.secrets[0].name,
                                                     required_keys=plan.secrets[0].required_keys),
                                secret_id="st-1"),
                SimpleNamespace(spec=SimpleNamespace(name=plan.secrets[1].name,
                                                     required_keys=plan.secrets[1].required_keys),
                                secret_id="st-2"),
            ))

    monkeypatch.setattr(host, "ExplicitModal154ReleaseDeploymentReader", _Reader)
    monkeypatch.setattr(host, "ModalRuntimeReleaseDeployer", _Deployer)
    monkeypatch.setattr(host, "_CurrentPackagedDeploymentReader", lambda **kw: SimpleNamespace(observe=lambda **kw: facts))
    storage = _Storage(_Attempts([]))
    original_claim = storage.attempts.claim

    def claim(ref, evidence):
        events.append("claim")
        original_claim(ref, evidence)

    storage.attempts.claim = claim
    result = host.prepare_modal_runtime_for_host(
        sdk=_SDK, client=object(), client_binding=_binding(), profile_path=Path("profile.json"),
        runtime_material_intent_digest="a" * 64, recipe_resource=ResourceSpec("A100-80GB"),
        maximum_cost_minor_units=1000, private_storage=storage, secret_resolver=_Resolver(),
        hf_token_ref=SecretRef("env", "HF_TOKEN"), qualification_secret_name="qualify",
        hf_token_secret_name="hf", app_name="training", environment_name="production",
    )
    assert events == ["claim", "volume", "volume", "volume", "claim", "deploy"]
    assert storage.attempts.refs == ["build-" + "a" * 64, "deploy-" + "c" * 64]
    assert len(result.volume_names_by_id) == 3 and len(result.qualification_key) == 32
    assert len(result.deployment_facts.secrets) == 2
    assert len(_Secret.created) == 2 and len(_App.created) == 1


def test_current_reader_rechecks_layout_and_cache_identity(monkeypatch):
    facts = SimpleNamespace(
        app_name="training-fake", function_name="packaged-training", app_id="ap-fake",
        deployment_generation=2, function_id="fu-training",
        self_check_function_name="packaged-self-check", self_check_function_id="fu-check",
        control_volume_id="vo-1", artifact_volume_id="vo-2", model_cache_volume_id="vo-3",
    )
    names = {"control": "control", "artifacts": "artifacts", "model_cache": "cache"}
    _Volume.identities.update({"control": "vo-1", "artifacts": "vo-2", "cache": "vo-3"})
    _Secret.identities.update({"qualify": "st-1", "hf": "st-2"})
    observation = SimpleNamespace(
        deployed=True, app_id="ap-fake", generation=2, class_ids=(),
        function_ids=(("packaged-self-check", "fu-check"),
                      ("packaged-training", "fu-training")),
    )

    class _Reader:
        def __init__(self, **kwargs):
            pass

        def observe(self, **kwargs):
            return observation

    monkeypatch.setattr(host, "ExplicitModal154ReleaseDeploymentReader", _Reader)
    client = object()
    reader = host._CurrentPackagedDeploymentReader(
        sdk=_SDK, client=client, binding=_binding(), facts=facts, names=names,
        secret_ids=(("hf", "st-2", ("HF_TOKEN",)),
                    ("qualify", "st-1", ("SYNAPTIC_MODAL_QUALIFICATION_HMAC_KEY",))),
    )
    assert reader.observe(client=client, app_name="training-fake",
                          function_name="packaged-training", environment_name="production") is facts
    _Volume.identities["cache"] = "vo-replaced"
    with pytest.raises(ValueError, match="Volume identity differs"):
        reader.observe(client=client, app_name="training-fake",
                       function_name="packaged-training", environment_name="production")
    _Volume.identities["cache"] = "vo-3"
    _Secret.identities["hf"] = "st-replaced"
    with pytest.raises(ValueError, match="Secret identity differs"):
        reader.observe(client=client, app_name="training-fake",
                       function_name="packaged-training", environment_name="production")


def _paid_factory_probe(storage):
    return host.prepare_modal_runtime_for_host(
        sdk=_SDK, client=object(), client_binding=_binding(), profile_path=Path("profile.json"),
        runtime_material_intent_digest="a" * 64,
        recipe_resource=ResourceSpec("A100-80GB", 1, 1800),
        maximum_cost_minor_units=200, private_storage=storage, secret_resolver=_Resolver(),
        hf_token_ref=SecretRef("env", "HF_TOKEN"), qualification_secret_name="qualify",
        hf_token_secret_name="hf", app_name="training", environment_name="production",
    )


@pytest.mark.skipif(os.name == "nt", reason="Linux signal precondition")
def test_paid_factory_wrong_thread_has_zero_claims_or_provider_reads():
    storage = _Storage(_Attempts([]))
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(ModalRuntimeReleaseDeploymentError, match="bounded_deploy_unavailable"):
            pool.submit(_paid_factory_probe, storage).result()
    assert storage.attempts.refs == []
    assert not _Volume.created and not _Secret.created and not _App.created


@pytest.mark.skipif(os.name == "nt", reason="Linux signal precondition")
def test_paid_factory_active_alarm_interval_has_zero_claims_or_provider_reads(monkeypatch):
    monkeypatch.setattr(signal, "getitimer", lambda which: (0.0, 1.0))
    storage = _Storage(_Attempts([]))
    with pytest.raises(ModalRuntimeReleaseDeploymentError, match="bounded_deploy_unavailable"):
        _paid_factory_probe(storage)
    assert storage.attempts.refs == []
    assert not _Volume.created and not _Secret.created and not _App.created
