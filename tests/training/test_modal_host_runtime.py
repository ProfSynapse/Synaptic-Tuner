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
from tuner.training import modal_standalone_runner as runner


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
    monkeypatch.setattr(host, "MODAL_SFT_ACCELERATOR_RATE_KEYS", {
        "A100-80GB": "gpu_hour_cost_a100_80gb_fixture", "L40S": "gpu_hour_cost_l40s",
    })
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


def test_l40s_quote_uses_exact_live_key_and_never_falls_back_to_a100():
    _Workspace.rates = {
        "gpu_hour_cost_l40s": Decimal("1.95000"),
        "gpu_hour_cost_a100_80gb": Decimal("2.50000"),
    }
    resource = ResourceSpec("L40S", 1, 1800)
    quote = host.quote_modal_runtime_for_host(
        sdk=_SDK, client=object(), client_binding=_binding(),
        recipe_resource=resource, maximum_cost_minor_units=200,
    )
    assert quote.rate_key == "gpu_hour_cost_l40s"
    assert quote.gpu_only_timeout_estimate_minor_units == 98
    assert quote.resource == resource
    with pytest.raises(ValueError, match="ceiling"):
        host.quote_modal_runtime_for_host(
            sdk=_SDK, client=object(), client_binding=_binding(),
            recipe_resource=resource, maximum_cost_minor_units=97,
        )
    _Workspace.rates.pop("gpu_hour_cost_l40s")
    with pytest.raises(ValueError, match="scoped rate is invalid"):
        host.quote_modal_runtime_for_host(
            sdk=_SDK, client=object(), client_binding=_binding(),
            recipe_resource=resource, maximum_cost_minor_units=200,
        )
    assert not _Volume.created and not _Secret.created and not _App.created


def test_four_hour_l40s_quote_retains_cost_gate_without_provisioning():
    _Workspace.rates = {"gpu_hour_cost_l40s": Decimal("1.95000")}
    resource = ResourceSpec("L40S", 1, 14400)
    quote = host.quote_modal_runtime_for_host(
        sdk=_SDK, client=object(), client_binding=_binding(),
        recipe_resource=resource, maximum_cost_minor_units=800,
    )
    assert quote.gpu_only_timeout_estimate_minor_units == 780
    assert quote.maximum_cost_minor_units == 800 and quote.resource == resource
    assert quote.authorization_semantics == "operator-maximum-not-provider-billing-cap"
    assert {"build", "cpu", "memory", "storage", "usage-beyond-timeout"}.issubset(quote.excluded_billing_dimensions)
    with pytest.raises(ValueError, match="ceiling"):
        host.quote_modal_runtime_for_host(
            sdk=_SDK, client=object(), client_binding=_binding(),
            recipe_resource=resource, maximum_cost_minor_units=400,
        )
    assert not _Volume.created and not _Secret.created and not _App.created


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
    monkeypatch.setattr(host, "MODAL_SFT_ACCELERATOR_RATE_KEYS", {
        "A100-80GB": None, "L40S": "gpu_hour_cost_l40s",
    })
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


def test_runner_default_resource_names_pass_real_preclaim_validator(monkeypatch):
    monkeypatch.setattr(host, "MODAL_SFT_ACCELERATOR_RATE_KEYS", {
        "A100-80GB": "gpu_hour_cost_a100_80gb_fixture", "L40S": "gpu_hour_cost_l40s",
    })
    monkeypatch.setattr(host, "plan_modal_build_material", lambda path: {"intent_digest": "a" * 64})
    storage = _Storage(_Attempts([]))

    class ClaimReached(RuntimeError):
        pass

    def claim(_ref, _evidence):
        raise ClaimReached

    storage.attempts.claim = claim

    def prepare(qualification_name):
        return host.prepare_modal_runtime_for_host(
            sdk=_SDK, client=object(), client_binding=_binding(), profile_path=Path("profile.json"),
            runtime_material_intent_digest="a" * 64,
            recipe_resource=ResourceSpec("A100-80GB", 1, 1800),
            maximum_cost_minor_units=200, private_storage=storage, secret_resolver=_Resolver(),
            hf_token_ref=SecretRef("env", "HF_TOKEN"),
            qualification_secret_name=qualification_name,
            hf_token_secret_name=runner._HF_TOKEN_SECRET_NAME,
            app_name=runner._APP_NAME, environment_name="production",
        )

    with pytest.raises(ClaimReached):
        prepare(runner._QUALIFICATION_SECRET_NAME)
    with pytest.raises(ValueError, match="resource name is invalid"):
        prepare("synaptic-training-qualification")
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


@pytest.mark.parametrize("accelerator", ["A100-80GB", "L40S"])
def test_preparation_claims_before_provision_and_uses_exact_three_volumes(monkeypatch, accelerator):
    monkeypatch.setattr(host, "MODAL_SFT_ACCELERATOR_RATE_KEYS", {
        "A100-80GB": "gpu_hour_cost_a100_80gb_fixture", "L40S": "gpu_hour_cost_l40s",
    })
    _Workspace.rates["gpu_hour_cost_l40s"] = Decimal("1.95")
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
            assert plan.functions[0].gpu == accelerator
            assert plan.functions[0].restrict_modal_access is False
            assert plan.functions[1].restrict_modal_access is False
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
        runtime_material_intent_digest="a" * 64, recipe_resource=ResourceSpec(accelerator),
        maximum_cost_minor_units=1000, private_storage=storage, secret_resolver=_Resolver(),
        hf_token_ref=SecretRef("env", "HF_TOKEN"), qualification_secret_name="qualify",
        hf_token_secret_name="hf", app_name="training", environment_name="production",
    )
    assert events == ["claim", "volume", "volume", "volume", "claim", "deploy"]
    assert storage.attempts.refs == ["build-" + "a" * 64, "deploy-" + "c" * 64]
    assert len(result.volume_names_by_id) == 3 and len(result.qualification_key) == 32
    assert len(result.deployment_facts.secrets) == 2
    assert len(_Secret.created) == 2 and len(_App.created) == 1


def test_source_archive_failure_projects_only_fixed_nonretryable_diagnosis(monkeypatch):
    from tuner.execution.providers.modal.runtime_build import SourceArchiveInvalid

    monkeypatch.setattr(host, "MODAL_SFT_ACCELERATOR_RATE_KEYS", {
        "A100-80GB": "gpu_hour_cost_a100_80gb_fixture", "L40S": "gpu_hour_cost_l40s",
    })
    monkeypatch.setattr(host, "plan_modal_build_material", lambda path: {"intent_digest": "a" * 64})

    def fail_capture(**_kwargs):
        raise SourceArchiveInvalid("private provider detail must not escape")

    monkeypatch.setattr(host, "capture_modal_build_candidate", fail_capture)
    storage = _Storage(_Attempts([]))
    with pytest.raises(host.ModalHostBootstrapUnavailable) as caught:
        _paid_factory_probe(storage)
    error = caught.value
    assert (error.phase, error.failure_class, error.location, error.retry_authorized) == (
        "SOURCE_WHEEL", "SOURCE_ARCHIVE_INVALID",
        "runtime_build.prepare_current_source_wheel", False,
    )
    assert str(error) == "modal_host_bootstrap_unavailable"
    assert "private provider detail" not in str(error)
    assert error.__cause__ is None
    with pytest.raises(AttributeError, match="immutable"):
        error.phase = "OTHER"
    assert len(storage.attempts.refs) == 1


@pytest.mark.parametrize("stage,reason,location", [
    ("SOURCE_WHEEL", "SOURCE_STATE_INVALID", "runtime_build.prepare_current_source_wheel"),
    ("SOURCE_WHEEL", "INPUT_INVALID", "runtime_build.prepare_current_source_wheel"),
    ("SOURCE_WHEEL", "HEAD_BEFORE_UNAVAILABLE", "runtime_build.prepare_current_source_wheel"),
    ("SOURCE_WHEEL", "HEAD_BEFORE_MISMATCH", "runtime_build.prepare_current_source_wheel"),
    ("SOURCE_WHEEL", "STATUS_BEFORE_TIMEOUT", "runtime_build.prepare_current_source_wheel"),
    ("SOURCE_WHEEL", "STATUS_BEFORE_UNAVAILABLE", "runtime_build.prepare_current_source_wheel"),
    ("SOURCE_WHEEL", "STATUS_BEFORE_DIRTY", "runtime_build.prepare_current_source_wheel"),
    ("SOURCE_WHEEL", "HEAD_AFTER_UNAVAILABLE", "runtime_build.prepare_current_source_wheel"),
    ("SOURCE_WHEEL", "HEAD_AFTER_MISMATCH", "runtime_build.prepare_current_source_wheel"),
    ("SOURCE_WHEEL", "STATUS_AFTER_TIMEOUT", "runtime_build.prepare_current_source_wheel"),
    ("SOURCE_WHEEL", "STATUS_AFTER_UNAVAILABLE", "runtime_build.prepare_current_source_wheel"),
    ("SOURCE_WHEEL", "STATUS_AFTER_DIRTY", "runtime_build.prepare_current_source_wheel"),
    ("SOURCE_WHEEL", "BUILDER_SETUP_FAILED", "runtime_build.prepare_current_source_wheel"),
    ("SOURCE_WHEEL", "OFFLINE_WHEEL_TIMEOUT", "runtime_build.prepare_current_source_wheel"),
    ("BUILD_INPUTS", "INVALID", "runtime_build.prepare_build_inputs"),
    ("IMAGE_BUILD", "OPERATION_FAILED", "runtime_build.build_image"),
    ("CAPTURE_OUTPUT", "INSPECTOR_REJECTED", "runtime_build.capture_output"),
    ("CAPTURE_CLEANUP", "TIMEOUT", "runtime_build.cleanup_capture_sandbox"),
])
def test_closed_build_stage_projection_never_leaks_hostile_error(
        monkeypatch, stage, reason, location):
    from tuner.execution.providers.modal.runtime_build import ModalBuildStageFailure

    monkeypatch.setattr(host, "MODAL_SFT_ACCELERATOR_RATE_KEYS", {
        "A100-80GB": "gpu_hour_cost_a100_80gb_fixture", "L40S": "gpu_hour_cost_l40s",
    })
    monkeypatch.setattr(host, "plan_modal_build_material", lambda path: {"intent_digest": "a" * 64})

    def fail_capture(**_kwargs):
        raise ModalBuildStageFailure(stage, reason) from ValueError("token=must-not-escape")

    monkeypatch.setattr(host, "capture_modal_build_candidate", fail_capture)
    storage = _Storage(_Attempts([]))
    with pytest.raises(host.ModalHostBootstrapUnavailable) as caught:
        _paid_factory_probe(storage)
    error = caught.value
    assert (error.phase, error.failure_class, error.location, error.retry_authorized) == (
        stage, reason, location, False,
    )
    assert str(error) == "modal_host_bootstrap_unavailable"
    assert "must-not-escape" not in str(error)
    assert error.__cause__ is None
    assert len(storage.attempts.refs) == 1


def test_closed_build_stage_constructor_rejects_unreviewed_reason():
    with pytest.raises(ValueError, match="diagnosis is invalid"):
        host.ModalHostBootstrapUnavailable("IMAGE_BUILD_HOSTILE_PROVIDER_TEXT")


@pytest.mark.parametrize("mutated_image_id,expected_image_id", [
    (None, "im-Failed123"),
    ("im-../HF_TOKEN=private", None),
])
def test_failed_image_identity_is_claim_bound_and_retained_on_owner_thread(
        monkeypatch, mutated_image_id, expected_image_id):
    import hashlib
    import json
    import threading
    from tuner.execution.providers.modal.runtime_build import ModalBuildStageFailure

    monkeypatch.setattr(host, "MODAL_SFT_ACCELERATOR_RATE_KEYS", {
        "A100-80GB": "gpu_hour_cost_a100_80gb_fixture", "L40S": "gpu_hour_cost_l40s",
    })
    monkeypatch.setattr(host, "plan_modal_build_material", lambda path: {"intent_digest": "a" * 64})

    class Context:
        app_id = "ap-Build123"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(_App, "run", lambda self, **kwargs: Context())
    def fail_capture(**kwargs):
        error = ModalBuildStageFailure(
            "IMAGE_BUILD", "OPERATION_FAILED", image_id="im-Failed123",
        )
        if mutated_image_id is not None:
            error.image_id = mutated_image_id
        raise error

    monkeypatch.setattr(host, "capture_modal_build_candidate", fail_capture)
    owner_thread = threading.get_ident()

    class Attempts:
        def __init__(self):
            self.entries = {}

        def claim(self, ref, evidence):
            self.entries[ref] = evidence
            return hashlib.sha256(evidence).hexdigest()

        def resolve(self, ref):
            assert threading.get_ident() == owner_thread
            return self.entries.get(ref)

    class Catalog:
        def __init__(self):
            self.entries = {}

        def publish_if_absent(self, ref, payload):
            assert threading.get_ident() == owner_thread
            assert ref not in self.entries
            self.entries[ref] = payload
            return True

        def resolve(self, ref):
            assert threading.get_ident() == owner_thread
            return self.entries.get(ref)

    class Storage:
        def __init__(self):
            self.attempts = Attempts()
            self.build_catalog = Catalog()

        def catalog(self, name, *, encode, decode):
            assert threading.get_ident() == owner_thread
            assert name == "host-build-diagnostics-v1" and encode is bytes and decode is bytes
            return self.build_catalog

    storage = Storage()
    with pytest.raises(host.ModalHostBootstrapUnavailable) as caught:
        _paid_factory_probe(storage)
    error = caught.value
    assert (error.phase, error.failure_class, error.build_app_id, error.image_id,
            error.retry_authorized) == (
        "IMAGE_BUILD", "OPERATION_FAILED", "ap-Build123", expected_image_id, False,
    )
    ref = "build-" + "a" * 64
    record = json.loads(storage.build_catalog.entries[ref])
    assert record == {
        "schema_version": "synaptic-modal-host-build-diagnostic/v1",
        "build_claim_ref": ref,
        "build_claim_sha256": hashlib.sha256(storage.attempts.entries[ref]).hexdigest(),
        "intent_digest": "a" * 64,
        "failure_class": "OPERATION_FAILED",
        "build_app_id": "ap-Build123",
        "image_id": expected_image_id,
        "retry_authorized": False,
    }
    assert b"HF_TOKEN" not in storage.build_catalog.entries[ref]


def test_unretained_image_identity_does_not_project_from_failed_catalog(monkeypatch):
    from tuner.execution.providers.modal.runtime_build import ModalBuildStageFailure

    monkeypatch.setattr(host, "MODAL_SFT_ACCELERATOR_RATE_KEYS", {
        "A100-80GB": "gpu_hour_cost_a100_80gb_fixture", "L40S": "gpu_hour_cost_l40s",
    })
    monkeypatch.setattr(host, "plan_modal_build_material", lambda path: {"intent_digest": "a" * 64})

    class Context:
        app_id = "ap-Build123"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(_App, "run", lambda self, **kwargs: Context())
    monkeypatch.setattr(host, "capture_modal_build_candidate", lambda **kwargs: (
        (_ for _ in ()).throw(ModalBuildStageFailure(
            "IMAGE_BUILD", "OPERATION_FAILED", image_id="im-Failed123"))))
    storage = _Storage(_Attempts([]))  # No catalog: the original claim remains consumed.
    with pytest.raises(host.ModalHostBootstrapUnavailable) as caught:
        _paid_factory_probe(storage)
    assert caught.value.build_app_id is None and caught.value.image_id is None
    assert caught.value.failure_class == "OPERATION_FAILED"
    assert tuple(storage.attempts.refs) == ("build-" + "a" * 64,)


def test_inflight_image_timeout_cannot_publish_late_image_identity(monkeypatch):
    import threading
    import time
    from tuner.execution.providers.modal.runtime_build import (
        ModalBoundedOperationFailure, ModalBuildStageFailure, _FailedImageBuild, _bounded,
    )

    monkeypatch.setattr(host, "MODAL_SFT_ACCELERATOR_RATE_KEYS", {
        "A100-80GB": "gpu_hour_cost_a100_80gb_fixture", "L40S": "gpu_hour_cost_l40s",
    })
    monkeypatch.setattr(host, "plan_modal_build_material", lambda path: {"intent_digest": "a" * 64})

    class Context:
        app_id = "ap-Build123"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(_App, "run", lambda self, **kwargs: Context())
    release, finished = threading.Event(), threading.Event()

    def late_build():
        assert release.wait(2)
        finished.set()
        return _FailedImageBuild("im-Late123")

    def capture(**kwargs):
        try:
            _bounded(late_build, deadline=time.monotonic() + 0.01,
                     code="modal_training_image_build_failed")
        except ModalBoundedOperationFailure as error:
            raise ModalBuildStageFailure("IMAGE_BUILD", error.reason) from None

    monkeypatch.setattr(host, "capture_modal_build_candidate", capture)

    class Storage:
        def __init__(self):
            self.attempts = _Attempts([])
            self.catalog_calls = 0

        def catalog(self, *args, **kwargs):
            self.catalog_calls += 1
            raise AssertionError("timeout must not publish a diagnostic")

    storage = Storage()
    try:
        with pytest.raises(host.ModalHostBootstrapUnavailable) as caught:
            _paid_factory_probe(storage)
        assert (caught.value.phase, caught.value.failure_class,
                caught.value.build_app_id, caught.value.image_id) == (
            "IMAGE_BUILD", "TIMEOUT", None, None,
        )
    finally:
        release.set()
    assert finished.wait(2)
    assert storage.catalog_calls == 0
    assert storage.attempts.refs == ["build-" + "a" * 64]


@pytest.mark.parametrize("boundary,code,phase,failure_class", [
    ("observe", "modal_release_scope_unavailable", "RELEASE_OBSERVE", "SCOPE_UNAVAILABLE"),
    ("observe", "modal_release_observation_unavailable", "RELEASE_OBSERVE", "OBSERVATION_UNAVAILABLE"),
    ("deploy", "modal_release_construction_failed", "RELEASE_ATTEMPT", "CONSTRUCTION_FAILED"),
    ("deploy", "modal_release_deployment_indeterminate", "RELEASE_ATTEMPT", "DEPLOYMENT_INDETERMINATE"),
    ("deploy", "modal_release_acknowledgement_invalid", "RELEASE_ATTEMPT", "ACKNOWLEDGEMENT_INVALID"),
])
def test_release_failure_after_claim_is_closed_without_second_deploy(
        monkeypatch, boundary, code, phase, failure_class):
    monkeypatch.setattr(host, "MODAL_SFT_ACCELERATOR_RATE_KEYS", {
        "A100-80GB": "gpu_hour_cost_a100_80gb_fixture", "L40S": "gpu_hour_cost_l40s",
    })
    monkeypatch.setattr(host, "plan_modal_build_material", lambda path: {"intent_digest": "a" * 64})
    monkeypatch.setattr(host, "capture_modal_build_candidate",
                        lambda **kw: SimpleNamespace(capture_digest="b" * 64))
    monkeypatch.setattr(host, "build_modal_runtime_release_v2",
                        lambda *args, **kw: SimpleNamespace(manifest_digest="c" * 64))
    monkeypatch.setattr(host, "ModalRuntimeReleaseDeploymentPlanV1",
                        lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(host, "ExplicitModal154ReleaseDeploymentReader",
                        lambda **kw: object())
    calls = []

    class _FailingDeployer:
        def __init__(self, **kwargs):
            pass

        def observe(self, plan):
            calls.append("observe")
            if boundary == "observe":
                raise ModalRuntimeReleaseDeploymentError(code) from ValueError(
                    "HF_TOKEN=private /home/owner/dataset.jsonl")
            return None

        def deploy_once(self, plan, **kwargs):
            calls.append("deploy")
            raise ModalRuntimeReleaseDeploymentError(code) from ValueError(
                "HF_TOKEN=private /home/owner/dataset.jsonl")

    monkeypatch.setattr(host, "ModalRuntimeReleaseDeployer", _FailingDeployer)
    storage = _Storage(_Attempts([]))
    with pytest.raises(host.ModalHostBootstrapUnavailable) as caught:
        _paid_factory_probe(storage)
    error = caught.value
    assert (error.phase, error.failure_class, error.location, error.retry_authorized) == (
        phase, failure_class,
        "modal_host_runtime.observe_release" if boundary == "observe"
        else "modal_host_runtime.deploy_release", False,
    )
    assert str(error) == "modal_host_bootstrap_unavailable"
    assert error.__cause__ is None
    assert "HF_TOKEN" not in str(error)
    assert storage.attempts.refs == ["build-" + "a" * 64, "deploy-" + "c" * 64]
    assert calls == (["observe"] if boundary == "observe" else ["observe", "deploy"])


def test_release_error_requires_exact_class_single_known_code():
    class HostileReleaseError(ModalRuntimeReleaseDeploymentError):
        pass

    for error in (
        HostileReleaseError("modal_release_deployment_indeterminate"),
        ModalRuntimeReleaseDeploymentError("modal_release_deployment_indeterminate", "secret"),
        ModalRuntimeReleaseDeploymentError("modal_release_unknown_private_detail"),
    ):
        assert host._closed_release_diagnosis(error, before_deploy=False) is None


@pytest.mark.parametrize("capture_fails", [False, True])
def test_build_app_cleanup_does_not_hide_prior_capture_failure(monkeypatch, capture_fails):
    from tuner.execution.providers.modal.runtime_build import ModalBuildStageFailure

    monkeypatch.setattr(host, "MODAL_SFT_ACCELERATOR_RATE_KEYS", {
        "A100-80GB": "gpu_hour_cost_a100_80gb_fixture", "L40S": "gpu_hour_cost_l40s",
    })
    monkeypatch.setattr(host, "plan_modal_build_material", lambda path: {"intent_digest": "a" * 64})

    class BrokenContext:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            raise ValueError("token=must-not-escape")

    monkeypatch.setattr(_App, "run", lambda self, **kwargs: BrokenContext())

    def capture(**_kwargs):
        if capture_fails:
            raise ModalBuildStageFailure("CAPTURE_OUTPUT", "INSPECTOR_REJECTED")
        return object()

    monkeypatch.setattr(host, "capture_modal_build_candidate", capture)
    storage = _Storage(_Attempts([]))
    with pytest.raises(host.ModalHostBootstrapUnavailable) as caught:
        _paid_factory_probe(storage)
    expected = ("CAPTURE_OUTPUT", "INSPECTOR_REJECTED") if capture_fails else (
        "APP_CLEANUP", "OPERATION_FAILED",
    )
    assert (caught.value.phase, caught.value.failure_class) == expected
    assert "must-not-escape" not in str(caught.value)
    assert len(storage.attempts.refs) == 1


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
