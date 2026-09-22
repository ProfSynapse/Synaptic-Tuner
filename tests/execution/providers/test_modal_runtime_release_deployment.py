"""Provider-free tests for protected packaged-runtime deployment."""

from __future__ import annotations

from pathlib import Path

import pytest

from tuner.execution.providers.modal.binding import ModalClientBinding
from tuner.execution.providers.modal.packaged_binding import ModalPackagedRuntimeFactsV1
from tuner.execution.providers.modal.runtime_release_deployment import (
    EXACT_SELF_CHECK_MODULE,
    EXACT_SELF_CHECK_QUALNAME,
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

from tests.execution.providers.test_modal_packaged_binding import _release_and_execution


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
        return FakeObject("vo-control" if name == "control-volume" else "vo-artifact")


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
def _reset_fakes():
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
