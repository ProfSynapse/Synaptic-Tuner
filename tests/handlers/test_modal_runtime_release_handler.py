from __future__ import annotations

import json
from argparse import Namespace

import pytest

from tuner.handlers.modal_runtime_release_handler import ModalRuntimeReleaseHandler


@pytest.mark.parametrize(
    "action", ("preflight", "approve", "execute", "recover", "observe", "verify",
               "qualify-preflight", "qualify-approve", "qualify-execute",
               "qualify-recover", "qualify-observe", "qualify-verify"),
)
def test_routes_exact_release_actions(action, capsys) -> None:
    args = Namespace(subcommand=action, json=True)
    calls = []
    handler = ModalRuntimeReleaseHandler(
        args=args,
        action_runner=lambda selected, **kwargs: calls.append(
            (selected, kwargs["args"], kwargs["context"])
        ) or {"status": selected.upper()},
    )
    assert handler.handle() == 0
    assert calls == [(action, args, handler.context)]
    assert json.loads(capsys.readouterr().out)["data"]["status"] == action.upper()


@pytest.mark.parametrize("action", (None, "", "retry", "deploy"))
def test_rejects_unknown_release_action_without_calling_runner(action, capsys) -> None:
    calls = []
    handler = ModalRuntimeReleaseHandler(
        args=Namespace(subcommand=action, json=True),
        action_runner=lambda *args, **kwargs: calls.append(args),
    )
    assert handler.handle() == 1
    assert calls == []
    assert json.loads(capsys.readouterr().out)["error"]["code"] \
        == "MODAL_RUNTIME_RELEASE_ERROR"


def test_sanitizes_provider_and_credential_detail(capsys) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("SYNAPTIC_MODAL_QUALIFICATION_HMAC_KEY=private")

    handler = ModalRuntimeReleaseHandler(
        args=Namespace(subcommand="execute", json=True), action_runner=fail,
    )
    assert handler.handle() == 1
    rendered = capsys.readouterr().out
    assert "private" not in rendered
    assert "HMAC_KEY" not in rendered


def test_does_not_catch_cancellation_base_exception() -> None:
    class Cancelled(BaseException):
        pass

    handler = ModalRuntimeReleaseHandler(
        args=Namespace(subcommand="observe", json=True),
        action_runner=lambda *args, **kwargs: (_ for _ in ()).throw(Cancelled()),
    )
    with pytest.raises(Cancelled):
        handler.handle()
