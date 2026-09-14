"""Focused denial-order audit for the standalone launcher."""

from types import SimpleNamespace
from pathlib import Path
import sys

import pytest

from examples.modal_chat import launch
from tuner.execution.providers.modal import coordinator_adapter


@pytest.mark.parametrize(
    ("observe", "artifact_streaming"),
    ((False, False), (False, True), (True, False)),
)
def test_public_chat_capability_denial_precedes_credentials_storage_and_cloud(
    monkeypatch, observe, artifact_streaming
):
    # Interpreter denial is covered separately; isolate the next admission gate.
    monkeypatch.setattr(launch, "_check_launcher_python", lambda: None)
    monkeypatch.setattr(
        coordinator_adapter,
        "_descriptor",
        lambda: SimpleNamespace(
            capabilities=SimpleNamespace(
                observe=observe, artifact_streaming=artifact_streaming
            )
        ),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("capability denial crossed a pre-cloud boundary")

    monkeypatch.setattr(launch, "_private_directory", forbidden)
    monkeypatch.setattr(launch, "_model_token", forbidden)
    monkeypatch.setattr(launch, "_client", forbidden)
    with pytest.raises(
        launch.ModalChatLauncherError,
        match="modal_chat_public_capabilities_unqualified",
    ):
        launch.execute(
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            mode="train-chat",
            profile="selected",
            hf_token_env_file=None,
            emit=forbidden,
        )


def test_public_chat_exact_read_capabilities_pass_the_precloud_gate(
    monkeypatch,
):
    monkeypatch.setattr(launch, "_check_launcher_python", lambda: None)
    monkeypatch.setattr(
        coordinator_adapter,
        "_descriptor",
        lambda: SimpleNamespace(
            capabilities=SimpleNamespace(observe=True, artifact_streaming=True)
        ),
    )

    class GatePassed(RuntimeError):
        pass

    def reached_local_storage(*args, **kwargs):
        raise GatePassed

    monkeypatch.setattr(launch, "_private_directory", reached_local_storage)
    monkeypatch.setitem(
        sys.modules,
        "examples.modal_chat.training",
        SimpleNamespace(
            ModalChatRunIdentity=object,
            compose_modal_training_host=lambda **kwargs: None,
        ),
    )
    with pytest.raises(GatePassed):
        launch.execute(
            SimpleNamespace(),
            SimpleNamespace(state_root=Path("/unused")),
            SimpleNamespace(attempt_ref="fixture"),
            SimpleNamespace(),
            mode="train-chat",
            profile="selected",
            hf_token_env_file=None,
            emit=lambda value: None,
        )


def test_settings_or_source_rejection_precedes_execute(monkeypatch, capsys):
    def rejected(*args, **kwargs):
        raise launch.ModalChatLauncherError("modal_chat_source_invalid")

    def forbidden(*args, **kwargs):
        pytest.fail("invalid local inputs reached execution")

    monkeypatch.setattr(launch, "check_inputs", rejected)
    monkeypatch.setattr(launch, "execute", forbidden)
    assert (
        launch.main(
            [
                "--project-root",
                "/consumer",
                "--configuration",
                "configuration/smoke.json",
                "--mode",
                "train-chat",
                "--modal-profile",
                "selected",
            ]
        )
        == 1
    )
    result = capsys.readouterr()
    assert result.err == ""
    assert "FAILED" in result.out


def test_parser_denies_unknown_execution_mode_before_input_or_cloud_work(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        launch,
        "check_inputs",
        lambda *args, **kwargs: pytest.fail("invalid mode reached input inspection"),
    )
    monkeypatch.setattr(
        launch,
        "execute",
        lambda *args, **kwargs: pytest.fail("invalid mode reached execution"),
    )
    assert (
        launch.main(
            [
                "--project-root",
                "/consumer",
                "--configuration",
                "configuration/smoke.json",
                "--mode",
                "unqualified-chat",
            ]
        )
        == 1
    )
    result = capsys.readouterr()
    assert result.err == ""
    assert "unqualified-chat" not in result.out


def test_execute_itself_denies_unknown_mode_before_storage_or_cloud(monkeypatch):

    def forbidden(*args, **kwargs):
        pytest.fail("unknown mode crossed the execution boundary")

    monkeypatch.setattr(launch, "_private_directory", forbidden)
    with pytest.raises(
        launch.ModalChatLauncherError, match="modal_chat_arguments_invalid"
    ):
        launch.execute(
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            mode="unknown",
            profile="selected",
            hf_token_env_file=None,
            emit=forbidden,
        )
