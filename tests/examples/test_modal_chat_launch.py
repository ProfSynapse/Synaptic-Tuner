"""Provider-free launcher ordering, confidentiality, polling and cleanup tests."""

from types import SimpleNamespace
import json
from pathlib import Path

import pytest

from examples.modal_chat import launch
from examples.modal_chat.storage import ModalChatStorage
from synaptic_tuner.api.v1.results import TrainingRunRef
from tuner.execution.providers.modal.facade import ModalFunctionCallState


def test_check_is_default_and_never_constructs_credentials_or_executes(
    monkeypatch, capsys
):
    monkeypatch.setattr(launch, "check_inputs", lambda *args, **kwargs: (1, 2, 3, 4))

    def forbidden(*args, **kwargs):
        pytest.fail("check mode entered execution")

    monkeypatch.setattr(launch, "execute", forbidden)
    assert (
        launch.main(
            [
                "--project-root",
                "/consumer",
                "--configuration",
                "configuration/smoke.json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "LOCAL_INPUTS_CHECKED"


def test_bad_arguments_never_echo_supplied_values(capsys):
    assert launch.main(["--not-a-real-flag", "private-argument-value"]) == 1
    output = capsys.readouterr()
    assert "private-argument-value" not in output.out + output.err
    assert json.loads(output.out)["retry_authorized"] is False


def test_selected_modal_profile_disables_environment_override(monkeypatch):
    pytest.importorskip("modal")
    from modal.config import config

    calls = []

    def get(name, **kwargs):
        calls.append((name, kwargs))
        return "fixture-id" if name == "token_id" else "fixture-secret"

    monkeypatch.setattr(config, "get", get)
    returned = object()

    class Client:
        @staticmethod
        def from_credentials(token_id, token_secret):
            assert (token_id, token_secret) == ("fixture-id", "fixture-secret")
            return returned

    sdk = SimpleNamespace(__version__="1.5.4", Client=Client)
    assert launch._client(sdk, "selected") is returned
    assert calls == [
        ("token_id", {"profile": "selected", "use_env": False}),
        ("token_secret", {"profile": "selected", "use_env": False}),
    ]


def test_private_output_creation_does_not_chmod_existing_shared_directory(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    with pytest.raises(launch.ModalChatLauncherError):
        launch._private_directory(shared)
    assert shared.stat().st_mode & 0o777 == 0o755
    private = launch._private_directory(tmp_path / "new" / "private")
    assert private.stat().st_mode & 0o777 == 0o700


def test_polling_uses_only_exact_known_call_pending_hint(monkeypatch):
    states = iter([ModalFunctionCallState.PENDING, ModalFunctionCallState.UNKNOWN])
    calls, sleeps, emitted = [], [], []

    class Facade:
        def observe_known_call_pending(self, reference):
            calls.append(reference)
            return next(states)

    workflow = SimpleNamespace(
        provider_run_ref=SimpleNamespace(
            reference=SimpleNamespace(provider_job_ref="fc-exact")
        )
    )
    monkeypatch.setattr(launch, "_snapshot_training", lambda *args: workflow)
    monkeypatch.setattr(launch.time, "sleep", sleeps.append)
    launch._wait_training(
        None,
        SimpleNamespace(facade=lambda: Facade()),
        None,
        TrainingRunRef("run", "project"),
        timeout_seconds=300,
        emit=emitted.append,
    )
    assert calls == ["fc-exact", "fc-exact"]
    assert sleeps == [15]
    assert emitted == ["TRAINING_PENDING"]


def _graph(*, sandboxes=(), close_error=False, submitted=True, unknown=False):
    class Ownership:
        submit_command_digest = "a" * 64

        def __init__(self, sandbox):
            self.sandbox = sandbox

        def close(self):
            if close_error:
                raise RuntimeError("private-provider-message")

    owned = [Ownership(item) for item in sandboxes]
    if unknown:
        owned.append(Ownership(None))
    return SimpleNamespace(
        runtime=SimpleNamespace(owned_lease=None),
        pending_ownership=lambda: tuple(owned),
        catalog=SimpleNamespace(
            submit_digests=lambda: ("a" * 64,) if submitted else ()
        ),
    )


class Sandbox:
    def __init__(self, object_id, returncode, fail=False):
        self.object_id, self.returncode, self.fail = object_id, returncode, fail

    def poll(self):
        if self.fail:
            raise RuntimeError("private-provider-message")
        return self.returncode


@pytest.mark.parametrize("unknown,fail", [(True, False), (False, True)])
def test_cleanup_uncertainty_is_saved_even_if_close_or_poll_raises(
    tmp_path, unknown, fail, capsys
):
    private = launch._private_directory(tmp_path / "private")
    with ModalChatStorage(private / "state.sqlite3", "consumer") as storage:
        graph = _graph(
            sandboxes=(Sandbox("sb-first", 0, fail), Sandbox("sb-second", 0)),
            close_error=True,
            unknown=unknown,
        )
        with pytest.raises(launch.ModalChatLauncherError, match="cleanup_unconfirmed"):
            launch._confirm_chat_cleanup(graph, storage, "chat")
        result = json.loads(
            storage.catalog("launch-chat-cleanup", encode=bytes, decode=bytes).resolve(
                "chat"
            )
        )
        assert result["provider_shutdown_proof"] is False
        assert result["unresolved_creation_or_cleanup"] is True
        assert result["known_instances"][1]["provider_shutdown_proof"] is True
    assert "private-provider-message" not in capsys.readouterr().out


def test_cleanup_requires_exact_stopped_readback(tmp_path):
    private = launch._private_directory(tmp_path / "private")
    with ModalChatStorage(private / "state.sqlite3", "consumer") as storage:
        launch._confirm_chat_cleanup(
            _graph(sandboxes=(Sandbox("sb-exact", 0),)), storage, "chat"
        )
        result = json.loads(
            storage.catalog("launch-chat-cleanup", encode=bytes, decode=bytes).resolve(
                "chat"
            )
        )
        assert result["provider_shutdown_proof"] is True
        assert result["known_instances"][0]["sandbox_id"] == "sb-exact"


def test_preallocation_failure_does_not_invent_cleanup_ownership(tmp_path):
    private = launch._private_directory(tmp_path / "private")
    with ModalChatStorage(private / "state.sqlite3", "consumer") as storage:
        launch._confirm_chat_cleanup(_graph(submitted=False), storage, "chat")
        result = json.loads(
            storage.catalog("launch-chat-cleanup", encode=bytes, decode=bytes).resolve(
                "chat"
            )
        )
        assert result["provider_shutdown_proof"] is False
        assert result["submit_command_retained"] is False
        assert result["unresolved_creation_or_cleanup"] is False


def test_poll_without_stopped_result_does_not_claim_cleanup(tmp_path):
    private = launch._private_directory(tmp_path / "private")
    with ModalChatStorage(private / "state.sqlite3", "consumer") as storage:
        with pytest.raises(launch.ModalChatLauncherError, match="cleanup_unconfirmed"):
            launch._confirm_chat_cleanup(
                _graph(sandboxes=(Sandbox("sb-live", None),)), storage, "chat"
            )


def test_integrated_training_and_qualification_are_importable():
    from examples.modal_chat.training import compose_modal_training_host
    from examples.modal_chat.qualification import qualify_modal_chat_run

    assert callable(compose_modal_training_host)
    assert callable(qualify_modal_chat_run)
