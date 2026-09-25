"""Provider-free tests for protected packaged-runtime deployment."""

from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import ModuleType, SimpleNamespace
import asyncio
import inspect
import os
import signal
import sys

import pytest

from tuner.execution.providers.modal.binding import ModalClientBinding
from tuner.execution.providers.modal.packaged_binding import ModalPackagedRuntimeFactsV1
from tuner.execution.providers.modal.runtime_release_deployment import (
    EXACT_SELF_CHECK_MODULE,
    EXACT_SELF_CHECK_QUALNAME,
    ExplicitModal154ReleaseDeploymentReader,
    ModalRuntimeReleaseDeployer,
    ModalRuntimeReleaseDeploymentError,
    ModalRuntimeReleaseDeploymentFactsV1,
    ModalRuntimeReleaseDeploymentObservationV1,
    ModalRuntimeReleaseDeploymentPlanV1,
    ModalRuntimeReleaseFunctionSpecV1,
    ModalRuntimeReleaseSecretSpecV1,
    ModalRuntimeReleaseVolumeSpecV1,
    EXACT_QUALIFICATION_SECRET_REQUIRED_KEYS,
)
from tuner.runtime.runtime_release_modal_self_check import (
    run_runtime_release_self_check,
)
from tuner.runtime.runtime_release_modal_training import run_modal_packaged_training
from tuner.runtime.releases import PackagedTrainingRuntimeReleaseV2
from tuner.execution.providers.modal import runtime_release_deployment as deployment_module
from tuner.execution.providers.modal.runtime_build import build_modal_runtime_release_v2

from tests.execution.providers.test_modal_packaged_binding import _release_and_execution
from tests.execution.providers.test_modal_runtime_build import _candidate


def _read_deployment_response(monkeypatch, response, *, through_public_reader=False):
    """Exercise the pinned read without importing or contacting the real SDK."""
    modal = ModuleType("modal")
    modal.__path__ = []
    exception = ModuleType("modal.exception")
    exception.NotFoundError = type("NotFoundError", (Exception,), {})
    proto = ModuleType("modal_proto")
    proto.__path__ = []
    api_pb2 = ModuleType("modal_proto.api_pb2")
    api_pb2.APP_STATE_DEPLOYED = 3
    api_pb2.APP_STATE_STOPPED = 5
    api_pb2.AppLifecycle = lambda: SimpleNamespace(app_state=0, version=0, created_by="")
    api_pb2.AppGetByDeploymentNameRequest = lambda **kwargs: SimpleNamespace(**kwargs)
    async_utils = ModuleType("modal._utils.async_utils")

    class StrictSynchronizer:
        @staticmethod
        def create_blocking(value):
            # Modal 1.5.4 wraps functions, not bound classmethod objects.
            if not inspect.isfunction(value):
                raise TypeError("not a function")
            return lambda *args: asyncio.run(value(*args))

    async_utils.synchronizer = StrictSynchronizer()
    monkeypatch.setitem(sys.modules, "modal", modal)
    monkeypatch.setitem(sys.modules, "modal.exception", exception)
    monkeypatch.setitem(sys.modules, "modal._utils", ModuleType("modal._utils"))
    monkeypatch.setitem(sys.modules, "modal._utils.async_utils", async_utils)
    monkeypatch.setitem(sys.modules, "modal_proto", proto)
    monkeypatch.setitem(sys.modules, "modal_proto.api_pb2", api_pb2)

    class Stub:
        lookup_count = 0

        async def AppGetByDeploymentName(self, request):
            self.lookup_count += 1
            assert (request.name, request.environment_name) == ("fresh-app", "production")
            return response

        async def AppGetLayout(self, request):
            raise AssertionError("absent or malformed app must not request layout")

    stub = Stub()
    client = SimpleNamespace(stub=stub)
    if through_public_reader:
        result = ExplicitModal154ReleaseDeploymentReader(sdk=SimpleNamespace(
            __version__="1.5.4",
        )).observe(client=client, app_name="fresh-app", environment_name="production")
    else:
        result = asyncio.run(
            ExplicitModal154ReleaseDeploymentReader._read(client, "fresh-app", "production")
        )
    return result, stub.lookup_count


def test_pinned_readback_accepts_exact_empty_response(monkeypatch):
    response = SimpleNamespace(
        environment_name="production", app_id="", previous_app_id="",
        lifecycle=SimpleNamespace(app_state=0, version=0, created_by=""),
    )
    result, calls = _read_deployment_response(monkeypatch, response)
    assert result is None
    assert calls == 1


def test_public_reader_wraps_unbound_async_function_before_lookup(monkeypatch):
    response = SimpleNamespace(
        environment_name="production", app_id="", previous_app_id="",
        lifecycle=SimpleNamespace(app_state=0, version=0, created_by=""),
    )
    result, calls = _read_deployment_response(
        monkeypatch, response, through_public_reader=True,
    )
    assert result is None
    assert calls == 1


@pytest.mark.parametrize("change", [
    {"environment_name": "other"},
    {"environment_name": ""},
    {"app_id": "ap-partial"},
    {"app_id": None},
    {"previous_app_id": "ap-partial"},
    {"previous_app_id": None},
    {"lifecycle": SimpleNamespace(app_state=3, version=0, created_by="")},
    {"lifecycle": SimpleNamespace(app_state=0, version=1, created_by="")},
    {"lifecycle": SimpleNamespace(app_state=0, version=0, created_by="someone")},
])
def test_pinned_readback_rejects_partial_or_inconsistent_empty_response(monkeypatch, change):
    fields = {
        "environment_name": "production", "app_id": "", "previous_app_id": "",
        "lifecycle": SimpleNamespace(app_state=0, version=0, created_by=""),
    }
    fields.update(change)
    with pytest.raises(ValueError):
        _read_deployment_response(monkeypatch, SimpleNamespace(**fields))


def packaged_training_entry(payload: bytes):
    raise AssertionError("deployment must not invoke the training function")


def _plan() -> ModalRuntimeReleaseDeploymentPlanV1:
    release, _, _, _, _ = _release_and_execution()
    module = __name__
    volumes = (
        ModalRuntimeReleaseVolumeSpecV1("control", "control-volume", "/mnt/control"),
        ModalRuntimeReleaseVolumeSpecV1("artifacts", "artifact-volume", "/mnt/artifacts"),
    )
    secret = ModalRuntimeReleaseSecretSpecV1(
        "release-evidence", EXACT_QUALIFICATION_SECRET_REQUIRED_KEYS,
    )
    return ModalRuntimeReleaseDeploymentPlanV1(
        release=release,
        app_name="synaptic-packaged-release-v1",
        environment_name="production",
        functions=(
            ModalRuntimeReleaseFunctionSpecV1(
                "training", "packaged-training", module,
                "packaged_training_entry", ("control", "artifacts"),
                (secret.name,), 1000, 4096, 3600, "A10G", False,
            ),
            ModalRuntimeReleaseFunctionSpecV1(
                "self_check", "packaged-self-check", EXACT_SELF_CHECK_MODULE,
                EXACT_SELF_CHECK_QUALNAME, ("control", "artifacts"),
                (secret.name,), 1000, 512, 120, None, True,
                restrict_modal_access=False,
            ),
        ),
        volumes=volumes,
        secrets=(secret,),
    )


class FakeObject:
    is_hydrated = True

    def __init__(self, object_id: str):
        self.object_id = object_id

    def hydrate(self, client):
        self.hydrate_client = client
        return self


class FakeVolume:
    calls: list[tuple[str, dict[str, object]]] = []

    @classmethod
    def from_name(cls, name, **kwargs):
        cls.calls.append((name, kwargs))
        return FakeObject({
            "control-volume": "vo-control",
            "artifact-volume": "vo-artifact",
            "cache-volume": "vo-cache",
        }[name])


class FakeSecret:
    calls: list[tuple[str, dict[str, object]]] = []

    @classmethod
    def from_name(cls, name, **kwargs):
        cls.calls.append((name, kwargs))
        return FakeObject("st-evidence")


class FakeWorkspace(FakeObject):
    name = "workspace"

    def __init__(self):
        super().__init__("ws-workspace")

    @classmethod
    def from_context(cls, **kwargs):
        return cls()


class FakeEnvironment(FakeObject):
    name = "production"

    def __init__(self):
        super().__init__("en-production")

    @classmethod
    def from_name(cls, name, **kwargs):
        value = cls()
        value.name = name
        return value


class FakeImage(FakeObject):
    calls: list[str] = []

    def __init__(self, reference: str):
        super().__init__("im-release")
        self.reference = reference
        self.entrypoints: list[list[str]] = []

    @classmethod
    def from_registry(cls, reference):
        cls.calls.append(reference)
        return cls(reference)

    def entrypoint(self, value):
        self.entrypoints.append(value)
        return self


class FakeFunction(FakeObject):
    def __init__(self, name: str, options: dict[str, object]):
        suffix = "training" if name == "packaged-training" else "selfcheck"
        super().__init__(f"fu-{suffix}")
        self.name, self.options = name, options


class FakeApp:
    instances: list["FakeApp"] = []
    fail_deploy = False

    def __init__(self, name, **kwargs):
        self.name, self.kwargs = name, kwargs
        self.app_id = "ap-release"
        self.is_hydrated = True
        self.functions: list[tuple[FakeFunction, object]] = []
        self.deploy_calls: list[dict[str, object]] = []
        self.deploy_cwd: Path | None = None
        type(self).instances.append(self)

    def function(self, **kwargs):
        def decorate(value):
            function = FakeFunction(kwargs["name"], kwargs)
            self.functions.append((function, value))
            return function
        return decorate

    def deploy(self, **kwargs):
        self.deploy_cwd = Path.cwd()
        self.deploy_calls.append(kwargs)
        if self.fail_deploy:
            raise RuntimeError("credential=must-not-escape")
        return self


class SDK:
    __version__ = "1.5.4"
    Image = FakeImage
    App = FakeApp
    Volume = FakeVolume
    Secret = FakeSecret
    Workspace = FakeWorkspace
    Environment = FakeEnvironment


class Reader:
    def __init__(self, values):
        self.values = list(values)
        self.calls: list[dict[str, object]] = []

    def observe(self, **kwargs):
        self.calls.append(kwargs)
        return self.values.pop(0)


def _observation(generation: int, *, app_id: str = "ap-release"):
    return ModalRuntimeReleaseDeploymentObservationV1(
        "synaptic-packaged-release-v1", "production", True,
        app_id, "", generation,
        (("packaged-self-check", "fu-selfcheck"),
         ("packaged-training", "fu-training")),
    )


@pytest.fixture(autouse=True)
def _reset_fakes(monkeypatch):
    if os.name == "nt":
        monkeypatch.setattr(deployment_module, "require_bounded_modal_deployment_host", lambda: None)
    FakeApp.instances.clear()
    FakeApp.fail_deploy = False
    FakeImage.calls.clear()
    FakeVolume.calls.clear()
    FakeSecret.calls.clear()


def _deployer(reader: Reader):
    client = object()
    binding = ModalClientBinding(
        "workspace", "workspace", "production", "release-client", "1.5.4",
    )
    return ModalRuntimeReleaseDeployer(
        sdk=SDK, client=client, client_binding=binding, reader=reader,
    ), client


def test_plan_and_acknowledged_facts_round_trip_and_project_to_adapter() -> None:
    plan = _plan()
    assert ModalRuntimeReleaseDeploymentPlanV1.parse(plan.canonical_bytes) == plan
    deployer, client = _deployer(Reader([None, _observation(1)]))

    facts = deployer.deploy_once(
        plan,
        entrypoints={
            "training": packaged_training_entry,
            "self_check": run_runtime_release_self_check,
        },
    )

    assert ModalRuntimeReleaseDeploymentFactsV1.parse(facts.canonical_bytes) == facts
    assert facts.deployment_spec_digest == plan.deployment_spec_digest
    assert facts.image_id == "im-release"
    assert tuple(item.spec.role for item in facts.functions) == ("training", "self_check")
    assert facts.current_state_version_pinned is False
    assert facts.function_image_link_readable is False
    assert ModalPackagedRuntimeFactsV1.from_release_deployment(
        facts
    ).deployment_spec_digest == plan.deployment_spec_digest

    app = FakeApp.instances[-1]
    assert app.kwargs == {"image": app.kwargs["image"], "include_source": False}
    assert app.deploy_calls == [{
        "environment_name": "production", "client": client, "strategy": "rolling",
    }]
    assert app.deploy_cwd is not None and app.deploy_cwd != Path.cwd()
    assert not (app.deploy_cwd / ".git").exists()
    assert FakeImage.calls == [plan.release.image_ref]
    assert [item[0].options["serialized"] for item in app.functions] == [False, False]
    assert [item[0].options["include_source"] for item in app.functions] == [False, False]
    assert all(item[0].options["retries"] == 0 for item in app.functions)
    assert [item[0].options["restrict_modal_access"] for item in app.functions] \
        == [True, False]

    assert FakeVolume.calls == [
        ("control-volume", {
            "environment_name": "production", "create_if_missing": False,
            "version": 1, "client": client,
        }),
        ("artifact-volume", {
            "environment_name": "production", "create_if_missing": False,
            "version": 1, "client": client,
        }),
    ]
    assert FakeSecret.calls == [("release-evidence", {
        "environment_name": "production",
        "required_keys": ["SYNAPTIC_MODAL_QUALIFICATION_HMAC_KEY"],
        "client": client,
    })]


def test_v2_published_oci_uses_exact_registry_branch_without_build_candidate() -> None:
    from tuner.execution.providers.modal.runtime_release_deployment import (
        MODAL_RUNTIME_RELEASE_DEPLOYMENT_PLAN_V2_SCHEMA,
        MODAL_RUNTIME_RELEASE_DEPLOYMENT_FACTS_V2_SCHEMA,
    )
    original = _plan()
    source = original.release
    fields = {
        name: getattr(source, name)
        for name in (
            "release_ref", "package_name", "package_version", "package_digest",
            "source_provenance_digest", "worker_entrypoint", "worker_closure_digest",
            "python_implementation", "python_version", "python_executable",
            "python_executable_digest", "installed_distributions_digest",
            "installed_distribution_count", "platform_system", "platform_machine",
            "cuda_version", "runtime_facts", "compatible_methods", "compatible_models",
            "compatible_dataset_formats", "workload_schema", "prepared_input_schema",
            "artifact_contract_schema",
        )
    }
    release = PackagedTrainingRuntimeReleaseV2.build(
        **fields,
        material={"kind": "published_oci", "image": {"reference": source.image_ref, "digest": source.image_digest}},
    )
    training = replace(original.functions[0], volume_roles=("control", "artifacts", "model_cache"),
                       module="tuner.runtime.runtime_release_modal_training",
                       qualname="run_modal_packaged_training")
    cache = ModalRuntimeReleaseVolumeSpecV1("model_cache", "cache-volume", "/mnt/model-cache")
    plan = replace(original, release=release, functions=(training, original.functions[1]),
                   volumes=(*original.volumes, cache), schema_version=MODAL_RUNTIME_RELEASE_DEPLOYMENT_PLAN_V2_SCHEMA)
    deployer, _ = _deployer(Reader([None, _observation(1)]))
    facts = deployer.deploy_once(plan, entrypoints={
        "training": run_modal_packaged_training,
        "self_check": run_runtime_release_self_check,
    })
    assert facts.schema_version == MODAL_RUNTIME_RELEASE_DEPLOYMENT_FACTS_V2_SCHEMA
    assert facts.image_id == "im-release"
    assert facts.material_digest == release.material_digest
    assert {item.spec.role: item.volume_id for item in facts.volumes}["model_cache"] == "vo-cache"
    assert FakeImage.calls == [source.image_ref]
    assert ModalRuntimeReleaseDeploymentFactsV1.parse(facts.canonical_bytes) == facts
    packaged = ModalPackagedRuntimeFactsV1.from_release_deployment(facts)
    assert packaged.model_cache_volume_id == "vo-cache"
    assert ModalPackagedRuntimeFactsV1.parse(packaged.canonical_bytes) == packaged


def _modal_build_plan_and_candidate():
    from tuner.execution.providers.modal.runtime_release_deployment import (
        MODAL_RUNTIME_RELEASE_DEPLOYMENT_PLAN_V2_SCHEMA,
    )

    candidate = _candidate()
    object.__setattr__(candidate, "_builder_attested", True)
    release = build_modal_runtime_release_v2(candidate, release_ref="runtime:modal-test")
    original = _plan()
    training = replace(
        original.functions[0], volume_roles=("control", "artifacts", "model_cache"),
        module="tuner.runtime.runtime_release_modal_training",
        qualname="run_modal_packaged_training",
    )
    cache = ModalRuntimeReleaseVolumeSpecV1("model_cache", "cache-volume", "/mnt/model-cache")
    plan = replace(
        original, release=release, functions=(training, original.functions[1]),
        volumes=(*original.volumes, cache),
        schema_version=MODAL_RUNTIME_RELEASE_DEPLOYMENT_PLAN_V2_SCHEMA,
    )
    return plan, candidate


@pytest.mark.parametrize("hydrated_image_id", ["im-exact", "im-other", None])
def test_modal_build_binds_image_during_deploy_and_checks_captured_id(
        monkeypatch, hydrated_image_id) -> None:
    class BuildImage:
        instances = []

        def __init__(self, image_id):
            self.object_id = image_id
            self.is_hydrated = False
            self.build_calls = 0
            type(self).instances.append(self)

        @classmethod
        def from_id(cls, image_id, *, client):
            assert image_id == "im-exact"
            assert client is not None
            return cls(image_id)

        def build(self, app):
            self.build_calls += 1
            raise AssertionError("Image.build requires a deployed app")

    class BuildApp(FakeApp):
        def __init__(self, name, **kwargs):
            super().__init__(name, **kwargs)
            self.app_id = None
            self.is_hydrated = False

        def deploy(self, **kwargs):
            super().deploy(**kwargs)
            self.app_id = "ap-release"
            self.is_hydrated = True
            image = self.functions[0][0].options["image"]
            if hydrated_image_id is not None:
                image.object_id = hydrated_image_id
                image.is_hydrated = True
            return self

    monkeypatch.setattr(SDK, "Image", BuildImage)
    monkeypatch.setattr(SDK, "App", BuildApp)
    plan, candidate = _modal_build_plan_and_candidate()
    deployer, _ = _deployer(Reader([None, _observation(1)]))
    entrypoints = {
        "training": run_modal_packaged_training,
        "self_check": run_runtime_release_self_check,
    }

    if hydrated_image_id == candidate.image_id:
        facts = deployer.deploy_once(plan, entrypoints=entrypoints, candidate=candidate)
        assert facts.image_id == candidate.image_id
    else:
        with pytest.raises(ModalRuntimeReleaseDeploymentError) as caught:
            deployer.deploy_once(plan, entrypoints=entrypoints, candidate=candidate)
        assert str(caught.value) == "modal_release_acknowledgement_invalid"
        assert caught.value.__cause__ is None

    app = FakeApp.instances[-1]
    assert app.app_id == "ap-release"
    assert len(app.deploy_calls) == 1
    assert app.kwargs == {"include_source": False}
    assert all(function.options["image"] is BuildImage.instances[-1]
               for function, _ in app.functions)
    if hydrated_image_id is None:
        assert BuildImage.instances[-1].object_id == candidate.image_id
        assert BuildImage.instances[-1].is_hydrated is False
    assert BuildImage.instances[-1].build_calls == 0


def test_v2_plan_rejects_missing_or_reused_cache_volume() -> None:
    from tuner.execution.providers.modal.runtime_release_deployment import MODAL_RUNTIME_RELEASE_DEPLOYMENT_PLAN_V2_SCHEMA
    original = _plan()
    source = original.release
    fields = {name: getattr(source, name) for name in (
        "release_ref", "package_name", "package_version", "package_digest",
        "source_provenance_digest", "worker_entrypoint", "worker_closure_digest",
        "python_implementation", "python_version", "python_executable",
        "python_executable_digest", "installed_distributions_digest",
        "installed_distribution_count", "platform_system", "platform_machine",
        "cuda_version", "runtime_facts", "compatible_methods", "compatible_models",
        "compatible_dataset_formats", "workload_schema", "prepared_input_schema",
        "artifact_contract_schema",
    )}
    release = PackagedTrainingRuntimeReleaseV2.build(
        **fields, material={"kind": "published_oci", "image": {
            "reference": source.image_ref, "digest": source.image_digest,
        }},
    )
    with pytest.raises(ValueError, match="model-cache Volume"):
        replace(original, release=release,
                schema_version=MODAL_RUNTIME_RELEASE_DEPLOYMENT_PLAN_V2_SCHEMA)
    cache = ModalRuntimeReleaseVolumeSpecV1("model_cache", "cache-volume", "/mnt/model-cache")
    training = replace(original.functions[0], volume_roles=("control", "artifacts", "model_cache"),
                       module="tuner.runtime.runtime_release_modal_training",
                       qualname="run_modal_packaged_training")
    plan = replace(original, release=release, functions=(training, original.functions[1]),
                   volumes=(*original.volumes, cache),
                   schema_version=MODAL_RUNTIME_RELEASE_DEPLOYMENT_PLAN_V2_SCHEMA)
    with pytest.raises(ValueError, match="distinct model-cache Volume"):
        replace(plan, functions=(replace(training, module=__name__,
                                         qualname="packaged_training_entry"), original.functions[1]))
    facts = _deployer(Reader([None, _observation(1)]))[0].deploy_once(plan, entrypoints={
        "training": run_modal_packaged_training,
        "self_check": run_runtime_release_self_check,
    })
    cache_fact = next(item for item in facts.volumes if item.spec.role == "model_cache")
    with pytest.raises(ValueError, match="distinct cache Volume"):
        replace(facts, volumes=(*facts.volumes[:2], replace(cache_fact, volume_id=facts.volumes[0].volume_id)))


@pytest.mark.skipif(os.name == "nt", reason="POSIX main-thread alarm boundary")
def test_v2_deploy_timeout_restores_process_cwd(monkeypatch) -> None:
    v2_type = type("_V2", (), {})
    monkeypatch.setattr(deployment_module, "PackagedTrainingRuntimeReleaseV2", v2_type)
    original = Path.cwd()
    real_setitimer = signal.setitimer

    def fire_immediately(which, seconds):
        if seconds:
            signal.getsignal(signal.SIGALRM)(signal.SIGALRM, None)
        return real_setitimer(which, seconds)

    monkeypatch.setattr(signal, "setitimer", fire_immediately)
    plan = SimpleNamespace(release=v2_type(), environment_name="production", strategy="rolling")
    with pytest.raises(ModalRuntimeReleaseDeploymentError, match="indeterminate"):
        ModalRuntimeReleaseDeployer._deploy_from_private_nonrepo(
            SimpleNamespace(deploy=lambda **kwargs: pytest.fail("timed-out deploy called")),
            client=object(), plan=plan,
        )
    assert Path.cwd() == original
    assert signal.getsignal(signal.SIGALRM) is signal.SIG_DFL


@pytest.mark.skipif(os.name == "nt", reason="Linux signal precondition")
def test_v2_deploy_entry_rejects_wrong_thread_before_sdk_reads(monkeypatch) -> None:
    plan = _plan()
    monkeypatch.setattr(deployment_module, "PackagedTrainingRuntimeReleaseV2", type(plan.release))
    reader = Reader([])
    deployer, _ = _deployer(reader)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(ModalRuntimeReleaseDeploymentError, match="bounded_deploy_unavailable"):
            pool.submit(deployer.deploy_once, plan, entrypoints={}).result()
    assert reader.calls == []
    assert not FakeApp.instances and not FakeVolume.calls and not FakeSecret.calls


@pytest.mark.skipif(os.name == "nt", reason="Linux signal precondition")
def test_v2_deploy_entry_rejects_active_interval_before_sdk_reads(monkeypatch) -> None:
    plan = _plan()
    monkeypatch.setattr(deployment_module, "PackagedTrainingRuntimeReleaseV2", type(plan.release))
    monkeypatch.setattr(signal, "getitimer", lambda which: (0.0, 1.0))
    reader = Reader([])
    deployer, _ = _deployer(reader)
    with pytest.raises(ModalRuntimeReleaseDeploymentError, match="bounded_deploy_unavailable"):
        deployer.deploy_once(plan, entrypoints={})
    assert reader.calls == []
    assert not FakeApp.instances and not FakeVolume.calls and not FakeSecret.calls


def test_redeploy_requires_exact_single_generation_advance_and_layout() -> None:
    plan = _plan()
    deployer, _ = _deployer(Reader([_observation(7), _observation(8)]))
    facts = deployer.deploy_once(plan, entrypoints={
        "training": packaged_training_entry,
        "self_check": run_runtime_release_self_check,
    })
    assert facts.generation == 8


def test_provider_failure_after_deploy_boundary_is_indeterminate_and_secret_free() -> None:
    FakeApp.fail_deploy = True
    plan = _plan()
    deployer, _ = _deployer(Reader([None]))
    with pytest.raises(ModalRuntimeReleaseDeploymentError) as caught:
        deployer.deploy_once(plan, entrypoints={
            "training": packaged_training_entry,
            "self_check": run_runtime_release_self_check,
        })
    assert str(caught.value) == "modal_release_deployment_indeterminate"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_unknown_current_app_layout_is_rejected_before_provider_construction() -> None:
    plan = _plan()
    prior = ModalRuntimeReleaseDeploymentObservationV1(
        plan.app_name, plan.environment_name, True, "ap-other", "", 4,
        (("unrelated", "fu-unrelated"),),
    )
    deployer, _ = _deployer(Reader([prior]))
    with pytest.raises(ValueError, match="unowned layout"):
        deployer.deploy_once(plan, entrypoints={
            "training": packaged_training_entry,
            "self_check": run_runtime_release_self_check,
        })
    assert FakeImage.calls == []
    assert FakeVolume.calls == []


def test_entrypoint_substitution_stops_before_any_provider_read() -> None:
    plan = _plan()
    reader = Reader([None])
    deployer, _ = _deployer(reader)
    with pytest.raises(ValueError, match="installed package"):
        deployer.deploy_once(plan, entrypoints={
            "training": lambda value: value,
            "self_check": run_runtime_release_self_check,
        })
    assert reader.calls == []
    assert FakeVolume.calls == []
