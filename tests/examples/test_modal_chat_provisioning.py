"""Tests for explicit fresh Modal chat resource provisioning."""

from __future__ import annotations

import pytest

from examples.modal_chat.deployment import ModalChatScope
from examples.modal_chat.provisioning import (
    ModalChatProvisioner,
    ModalChatProvisioningError,
)
from examples.modal_chat.storage import ModalChatStorage
from tuner.execution.providers.modal.facade import MODAL_VOLUME_V1


class Object:
    def __init__(self, object_id, *, name=None, fail=False, after_hydrate=None):
        self.object_id, self.name = object_id, name
        self.is_hydrated = False
        self._fail = fail
        self._after_hydrate = after_hydrate

    def hydrate(self, client):
        if self._fail:
            raise RuntimeError("not exposed")
        self.is_hydrated = True
        if self._after_hydrate is not None:
            self._after_hydrate()


class SDK:
    __version__ = "1.5.4"

    def __init__(self, *, fail_hydrate=None, fail_create=None, change_after=None):
        self.calls = []
        self.fail_hydrate, self.fail_create = fail_hydrate, fail_create
        self.environment_id = "en-1"
        self.change_after = change_after
        sdk = self

        class Workspace:
            @staticmethod
            def from_context(*, client):
                return Object("ws-1", name="workspace-a")

        class Environment:
            @staticmethod
            def from_name(name, *, create_if_missing, client):
                return Object(sdk.environment_id, name=name)

        class Objects:
            def __init__(self, kind):
                self.kind = kind

            def create(self, name, *args, **kwargs):
                sdk.calls.append(("create", self.kind, name, args, kwargs))
                if sdk.fail_create == name:
                    raise RuntimeError("not exposed")

        class Volume:
            objects = Objects("volume")

            @staticmethod
            def from_name(name, **kwargs):
                sdk.calls.append(("read", "volume", name, kwargs))
                callback = None
                if sdk.change_after == name:
                    callback = lambda: setattr(sdk, "environment_id", "en-changed")
                return Object(
                    "vo-" + name,
                    fail=sdk.fail_hydrate == name,
                    after_hydrate=callback,
                )

        class Secret:
            objects = Objects("secret")

            @staticmethod
            def from_name(name, **kwargs):
                sdk.calls.append(("read", "secret", name, kwargs))
                return Object("st-" + name, fail=sdk.fail_hydrate == name)

        self.Workspace, self.Environment = Workspace, Environment
        self.Volume, self.Secret = Volume, Secret

    @staticmethod
    def current_function_call_id():
        return "unused"


def _open(tmp_path, sdk=None):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    storage = ModalChatStorage(private / "consumer.sqlite3", "consumer-a")
    sdk = sdk or SDK()
    scope = ModalChatScope(
        sdk=sdk,
        client=object(),
        environment_name="environment-a",
        client_ref="consumer-client",
    )
    sdk.calls.clear()
    return storage, sdk, scope


def _provisioner(storage, scope):
    return ModalChatProvisioner(
        scope=scope,
        storage=storage,
        training_control="training-control-a",
        artifacts="artifacts-a",
        model_cache="model-cache-a",
        runtime_secret="runtime-secret-a",
        model_key_name="MODEL_TOKEN",
        evidence_key_name="EVIDENCE_KEY",
        secret_values={"MODEL_TOKEN": "model-value", "EVIDENCE_KEY": "evidence-value"},
    )


def test_creates_three_fresh_volumes_and_one_secret_with_exact_arguments(tmp_path):
    storage, sdk, scope = _open(tmp_path)
    with storage:
        result = _provisioner(storage, scope).provision_once(attempt_ref="attempt-a")
    creates = [call for call in sdk.calls if call[0] == "create"]
    assert [call[1] for call in creates] == ["volume"] * 3 + ["secret"]
    for _, _, _, _, kwargs in creates[:3]:
        assert kwargs == {
            "version": MODAL_VOLUME_V1,
            "allow_existing": False,
            "environment_name": "environment-a",
            "client": scope.client,
        }
    secret = creates[3]
    assert secret[3] == (
        {"MODEL_TOKEN": "model-value", "EVIDENCE_KEY": "evidence-value"},
    )
    assert secret[4] == {
        "allow_existing": False,
        "environment_name": "environment-a",
        "client": scope.client,
    }
    assert set(result) == {
        "training_control",
        "artifacts",
        "model_cache",
        "runtime_secret",
    }


@pytest.mark.parametrize(
    "keys,values",
    [
        (
            ("MODAL_TOKEN_ID", "EVIDENCE_KEY"),
            {"MODAL_TOKEN_ID": "x", "EVIDENCE_KEY": "y"},
        ),
        (("MODEL_TOKEN", "EVIDENCE_KEY"), {"MODEL_TOKEN": "", "EVIDENCE_KEY": "y"}),
        (("MODEL_TOKEN", "EVIDENCE_KEY"), {"MODEL_TOKEN": "  ", "EVIDENCE_KEY": "y"}),
        (("MODEL_TOKEN", "EVIDENCE_KEY"), {"MODEL_TOKEN": "x"}),
    ],
)
def test_rejects_credentials_blank_or_incomplete_before_mutation(
    tmp_path, keys, values
):
    storage, sdk, scope = _open(tmp_path)
    with storage:
        with pytest.raises(ValueError):
            ModalChatProvisioner(
                scope=scope,
                storage=storage,
                training_control="control-a",
                artifacts="artifacts-a",
                model_cache="cache-a",
                runtime_secret="runtime-a",
                model_key_name=keys[0],
                evidence_key_name=keys[1],
                secret_values=values,
            )
    assert sdk.calls == []


def test_partial_hydration_failure_retains_creation_ack_and_forbids_retry(tmp_path):
    storage, sdk, scope = _open(tmp_path, SDK(fail_hydrate="artifacts-a"))
    provisioner = _provisioner(storage, scope)
    with storage:
        with pytest.raises(ModalChatProvisioningError):
            provisioner.provision_once(attempt_ref="attempt-a")
        with pytest.raises(ModalChatProvisioningError):
            provisioner.provision_once(attempt_ref="attempt-a")
        acks = storage.catalog(
            "provisioning-acks",
            encode=lambda value: value,
            decode=lambda value: value,
        )
        assert (
            acks.resolve(provisioner._group_ref + "-volume-artifacts-a-created")
            is not None
        )
        assert (
            acks.resolve(provisioner._group_ref + "-volume-artifacts-a-hydrated")
            is None
        )
    assert [call[2] for call in sdk.calls if call[0] == "create"] == [
        "training-control-a",
        "artifacts-a",
    ]


def test_group_claim_survives_reopen_and_blocks_renamed_attempt(tmp_path):
    storage, sdk, scope = _open(tmp_path, SDK(fail_create="training-control-a"))
    path = storage.database_path
    with storage:
        with pytest.raises(ModalChatProvisioningError):
            _provisioner(storage, scope).provision_once(attempt_ref="attempt-a")
    with ModalChatStorage(path, "consumer-a") as reopened:
        second_sdk = SDK()
        second_scope = ModalChatScope(
            sdk=second_sdk,
            client=object(),
            environment_name="environment-a",
            client_ref="consumer-client",
        )
        second_sdk.calls.clear()
        with pytest.raises(ModalChatProvisioningError):
            _provisioner(reopened, second_scope).provision_once(attempt_ref="attempt-b")
        assert second_sdk.calls == []


def test_scope_change_before_first_mutation_is_closed(tmp_path):
    storage, sdk, scope = _open(tmp_path)
    sdk.environment_id = "en-changed"
    with storage:
        with pytest.raises(ModalChatProvisioningError):
            _provisioner(storage, scope).provision_once(attempt_ref="attempt-a")
    assert [call for call in sdk.calls if call[0] == "create"] == []


def test_scope_change_between_resources_stops_before_next_mutation(tmp_path):
    storage, sdk, scope = _open(tmp_path, SDK(change_after="training-control-a"))
    with storage:
        with pytest.raises(ModalChatProvisioningError):
            _provisioner(storage, scope).provision_once(attempt_ref="attempt-a")
    assert [call[2] for call in sdk.calls if call[0] == "create"] == [
        "training-control-a"
    ]
