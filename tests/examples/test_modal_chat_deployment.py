from __future__ import annotations

import inspect
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.modal_chat import deployment
from examples.modal_chat.storage import ModalChatStorage
from tuner.execution.foundation_v2.canonical import parse_canonical_object
from tuner.execution.providers.modal.config import (
    ModalProviderProfileV1,
    ModalSecretProfileV1,
)


class _Object:
    def __init__(self, object_id: str):
        self.object_id = object_id
        self.is_hydrated = False

    def hydrate(self, client):
        self.is_hydrated = True


class _Function(_Object):
    def __init__(self, sdk, name):
        super().__init__("fu-1")
        self._sdk, self._name = sdk, name

    def _get_metadata(self):
        return SimpleNamespace(
            web_url=self._sdk.web_url,
            function_name=self._sdk.remote_function_name,
            definition_id=self._sdk.definition_id,
            app_id=self._sdk.app_id,
        )


class _Image(_Object):
    def entrypoint(self, value):
        return self

    def env(self, value):
        self.environment = dict(value)
        return self

    def add_local_python_source(self, *names, **options):
        self.sources = (names, options)
        return self


class _App:
    def __init__(self, sdk, name, **kwargs):
        self._sdk, self.name = sdk, name
        self.app_id = sdk.app_id

    def function(self, **kwargs):
        self._sdk.function_options = kwargs

        def decorate(entry):
            function = _Function(self._sdk, kwargs["name"])
            self._sdk.built_function = function
            return function

        return decorate

    def deploy(self, **kwargs):
        self._sdk.deploy_calls.append(kwargs)
        self._sdk.current_function = self._sdk.built_function
        self._sdk.current_function.hydrate(kwargs["client"])


class _SDK:
    __version__ = "1.5.4"

    def __init__(self):
        self.calls, self.deploy_calls = [], []
        self.workspace = "workspace-a"
        self.environment_id = "en-1"
        self.app_id, self.definition_id = "ap-1", "df-1"
        self.remote_function_name = "run_sft_v1_" + "1" * 32
        self.web_url = None
        self.current_function = None
        sdk = self

        class Workspace:
            @staticmethod
            def from_context(*, client):
                sdk.calls.append(("workspace", client))
                item = _Object("ws-1")
                item.name = sdk.workspace
                return item

        class Environment:
            @staticmethod
            def from_name(name, *, create_if_missing, client):
                sdk.calls.append(("environment", name, create_if_missing, client))
                item = _Object(sdk.environment_id)
                item.name = name
                return item

        class Volume:
            @staticmethod
            def from_name(name, **kwargs):
                sdk.calls.append(("volume", name, kwargs))
                return _Object("vo-control" if name == "control" else "vo-artifact")

        class Secret:
            @staticmethod
            def from_name(name, **kwargs):
                sdk.calls.append(("secret", name, kwargs))
                return _Object("st-1")

        class Image:
            @staticmethod
            def from_registry(reference):
                sdk.calls.append(("image", reference))
                return _Image("im-1")

        class App:
            def __new__(cls, name, **kwargs):
                sdk.calls.append(("app", name, kwargs))
                return _App(sdk, name, **kwargs)

        class Function:
            @staticmethod
            def from_name(app_name, function_name, **kwargs):
                sdk.calls.append(("function", app_name, function_name, kwargs))
                return sdk.current_function

        self.Workspace, self.Environment = Workspace, Environment
        self.Volume, self.Secret, self.Image = Volume, Secret, Image
        self.App, self.Function = App, Function

    @staticmethod
    def current_function_call_id():
        return "fc-unused"


def _profile() -> ModalProviderProfileV1:
    return ModalProviderProfileV1(
        "profile-a",
        "synaptic-training-v1",
        "run_sft_v1_" + "1" * 32,
        "modal-deployment-" + "1" * 32,
        "engine://tuner/execution/providers/modal/modal-runtime-v1.lock.json",
        "control",
        "artifact",
        (ModalSecretProfileV1("runtime-secret", ("EVIDENCE_KEY", "MODEL_TOKEN")),),
    )


@pytest.fixture
def case(tmp_path: Path):
    os.chmod(tmp_path, 0o700)
    sdk, client = _SDK(), object()
    storage = ModalChatStorage(tmp_path / "consumer.sqlite3", "namespace-a")
    scope = deployment.ModalChatScope(
        sdk=sdk, client=client, environment_name="environment-a", client_ref="host-a"
    )
    owner = deployment.ModalChatOwnedDeployment(
        scope=scope,
        storage=storage,
        profile=_profile(),
        runtime_environment={"ENGINE_MODE": "coordinator"},
        timeout_seconds=600,
        evidence_environment_key="EVIDENCE_KEY",
        evidence_key_ref="key-a",
        model_token_key="MODEL_TOKEN",
    )
    try:
        yield sdk, client, storage, scope, owner
    finally:
        storage.close()


def test_deploys_fixed_builder_once_and_persists_exact_receipt(case) -> None:
    sdk, client, storage, scope, owner = case
    payload = owner.deploy_once(attempt_ref="attempt-a")
    receipt = parse_canonical_object(payload, name="receipt")
    assert receipt["function_id"] == "fu-1"
    assert receipt["definition_id"] == "df-1"
    assert receipt["app_id"] == "ap-1"
    assert receipt["image_id"] == "im-1"
    assert (
        storage.catalog("deployment-results", encode=bytes, decode=bytes).resolve(
            "attempt-a"
        )
        == payload
    )
    assert sdk.deploy_calls == [{"client": client, "environment_name": "environment-a"}]
    assert sdk.function_options["retries"] == 0
    assert sdk.function_options["restrict_modal_access"] is True
    assert sdk.function_options["single_use_containers"] is True
    assert sdk.function_options["include_source"] is False
    assert all(
        call[2]["create_if_missing"] is False
        for call in sdk.calls
        if call[0] == "volume"
    )
    assert not any(call[0] == "list" for call in sdk.calls)
    assert (
        owner.observe(
            client=client,
            app_name="synaptic-training-v1",
            function_name="run_sft_v1_" + "1" * 32,
            environment_name="environment-a",
        )
        == owner.selection
    )


def test_attempt_is_permanent_and_failure_has_no_result_or_cleanup(case) -> None:
    sdk, _, storage, _, owner = case
    sdk.web_url = "https://public.invalid"
    with pytest.raises(deployment.ModalChatDeploymentError, match="failed"):
        owner.deploy_once(attempt_ref="attempt-a")
    assert storage.attempts.resolve("attempt-a") is not None
    assert (
        storage.catalog("deployment-results", encode=bytes, decode=bytes).resolve(
            "attempt-a"
        )
        is None
    )
    with pytest.raises(deployment.ModalChatDeploymentError, match="already_attempted"):
        owner.deploy_once(attempt_ref="attempt-b")
    assert len(sdk.deploy_calls) == 1


@pytest.mark.parametrize("failure", ["publish", "readback", "interrupt"])
def test_failed_durable_ack_cannot_authorize_observer_or_facade(
    case, monkeypatch, failure
):
    _, client, storage, _, owner = case
    original = storage.catalog

    class FailedAcknowledgement:
        def publish_if_absent(self, key, payload):
            if failure == "interrupt":
                raise KeyboardInterrupt
            if failure == "publish":
                raise OSError("local storage unavailable")
            return True

        def resolve(self, key):
            return b"{}"

    def catalog(self, name, **kwargs):
        if name == "deployment-acknowledgements":
            return FailedAcknowledgement()
        return original(name, **kwargs)

    monkeypatch.setattr(ModalChatStorage, "catalog", catalog)
    error = (
        KeyboardInterrupt
        if failure == "interrupt"
        else deployment.ModalChatDeploymentError
    )
    with pytest.raises(error):
        owner.deploy_once(attempt_ref="failed-ack")
    assert owner.candidate_receipt is not None
    assert storage.attempts.resolve("failed-ack") is not None
    with pytest.raises(deployment.ModalChatDeploymentError, match="not_owned"):
        owner.facade()
    with pytest.raises(deployment.ModalChatDeploymentError, match="not_owned"):
        owner.observe(
            client=client,
            app_name="synaptic-training-v1",
            function_name=owner.selection.function_name,
            environment_name="environment-a",
        )
    assert (
        original("deployment-results", encode=bytes, decode=bytes).resolve("failed-ack")
        is None
    )


def test_reconstructed_owner_cannot_redeploy_same_app_with_renamed_attempt(case):
    from examples.modal_chat.storage import AttemptAlreadyClaimed

    sdk, _, storage, scope, owner = case
    sdk.web_url = "https://public.invalid"
    with pytest.raises(deployment.ModalChatDeploymentError):
        owner.deploy_once(attempt_ref="first")
    reconstructed = deployment.ModalChatOwnedDeployment(
        scope=scope,
        storage=storage,
        profile=_profile(),
        runtime_environment={"ENGINE_MODE": "coordinator"},
        timeout_seconds=600,
        evidence_environment_key="EVIDENCE_KEY",
        evidence_key_ref="key-a",
        model_token_key="MODEL_TOKEN",
    )
    with pytest.raises(AttemptAlreadyClaimed):
        reconstructed.deploy_once(attempt_ref="renamed")
    assert len(sdk.deploy_calls) == 1
    acknowledged = storage.catalog(
        "deployment-returned", encode=bytes, decode=bytes
    ).resolve("first")
    assert (
        parse_canonical_object(acknowledged, name="ack")["deployment_returned"] is True
    )
    assert (
        storage.catalog("deployment-results", encode=bytes, decode=bytes).resolve(
            "first"
        )
        is None
    )


def test_provider_readback_failure_keeps_durable_ack_but_no_success(case, monkeypatch):
    sdk, _, storage, _, owner = case

    def unavailable(*args, **kwargs):
        raise OSError("provider read unavailable")

    monkeypatch.setattr(sdk.Function, "from_name", unavailable)
    with pytest.raises(deployment.ModalChatDeploymentError, match="failed"):
        owner.deploy_once(attempt_ref="failed-readback")
    acknowledged = storage.catalog(
        "deployment-acknowledgements", encode=bytes, decode=bytes
    ).resolve("failed-readback")
    assert acknowledged == owner.candidate_receipt
    assert parse_canonical_object(acknowledged, name="ack")["function_id"] == "fu-1"
    assert (
        storage.catalog("deployment-results", encode=bytes, decode=bytes).resolve(
            "failed-readback"
        )
        is None
    )
    assert len(sdk.deploy_calls) == 1


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("definition_id", "df-changed"),
        ("app_id", "ap-changed"),
        ("remote_function_name", "other-function"),
        ("web_url", "https://public.invalid"),
    ],
)
def test_observer_rejects_changed_or_public_definition(case, attribute, value) -> None:
    sdk, client, _, _, owner = case
    owner.deploy_once(attempt_ref="attempt-a")
    setattr(sdk, attribute, value)
    with pytest.raises(deployment.ModalChatDeploymentError, match="changed"):
        owner.observe(
            client=client,
            app_name="synaptic-training-v1",
            function_name="run_sft_v1_" + "1" * 32,
            environment_name="environment-a",
        )


def test_scope_and_observer_reject_substitution_and_predeployment_adoption(
    case,
) -> None:
    sdk, client, _, scope, owner = case
    with pytest.raises(deployment.ModalChatDeploymentError, match="not_owned"):
        owner.observe(
            client=client,
            app_name="synaptic-training-v1",
            function_name="run_sft_v1_" + "1" * 32,
            environment_name="environment-a",
        )
    with pytest.raises(deployment.ModalChatDeploymentError, match="scope_mismatch"):
        scope.observe(object())
    sdk.workspace = "workspace-b"
    with pytest.raises(deployment.ModalChatDeploymentError, match="scope_mismatch"):
        scope.observe(client)


def test_installed_modal_154_api_accepts_every_direct_sdk_call_shape() -> None:
    modal = pytest.importorskip("modal")
    assert modal.__version__ == "1.5.4"
    inspect.signature(modal.Workspace.from_context).bind(client=object())
    inspect.signature(modal.Environment.from_name).bind(
        "environment-a", create_if_missing=False, client=object()
    )
    inspect.signature(modal.Volume.from_name).bind(
        "volume-a",
        environment_name="environment-a",
        create_if_missing=False,
        version=2,
        client=object(),
    )
    inspect.signature(modal.Function.from_name).bind(
        "app-a", "function-a", environment_name="environment-a", client=object()
    )
