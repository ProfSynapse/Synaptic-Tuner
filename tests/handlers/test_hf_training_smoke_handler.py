from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from tuner.handlers.hf_training_smoke_handler import HFTrainingSmokeHandler
from tuner.project import ProjectContext


_ACTIONS = ("preflight", "approve", "execute", "recover", "observe", "verify")


def _args(action: str | None, *, json_mode: bool = True) -> Namespace:
    return Namespace(subcommand=action, json=json_mode)


@pytest.mark.parametrize("action", _ACTIONS)
def test_handler_routes_each_exact_action_and_preserves_context(
    tmp_path: Path, action: str, capsys: pytest.CaptureFixture[str]
) -> None:
    context = ProjectContext.standalone(engine_root=tmp_path)
    calls: list[tuple[str, Namespace, ProjectContext]] = []

    def run(selected: str, *, args: Namespace, context: ProjectContext):
        calls.append((selected, args, context))
        return {"status": selected.upper(), "submitted": selected == "execute"}

    args = _args(action)
    handler = HFTrainingSmokeHandler(args=args, context=context, action_runner=run)

    assert handler.name == "hf-training-smoke"
    assert handler.can_handle_direct_mode() is True
    assert handler.handle() == 0
    assert calls == [(action, args, context)]
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True
    assert payload["data"]["status"] == action.upper()


@pytest.mark.parametrize("action", [None, "", "retry", "execute-again"])
def test_handler_fails_closed_for_unknown_action_without_calling_runner(
    action: str | None, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[str] = []
    handler = HFTrainingSmokeHandler(
        args=_args(action), action_runner=lambda selected, **kwargs: calls.append(selected)
    )

    assert handler.handle() == 1
    assert calls == []
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"] == {
        "message": "Protected HF training smoke action failed.",
        "code": "HF_TRAINING_SMOKE_ERROR",
    }


def test_handler_sanitizes_ordinary_failures(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("hf_secret provider-request-value")

    handler = HFTrainingSmokeHandler(args=_args("execute"), action_runner=fail)

    assert handler.handle() == 1
    rendered = capsys.readouterr().out
    payload = json.loads(rendered)
    assert payload["error"]["code"] == "HF_TRAINING_SMOKE_ERROR"
    assert payload["error"]["message"] == "Protected HF training smoke action failed."
    assert "hf_secret" not in rendered
    assert "provider-request-value" not in rendered
    assert "RuntimeError" not in rendered


def test_handler_does_not_catch_cancellation_base_exception() -> None:
    class Cancelled(BaseException):
        pass

    def cancel(*args, **kwargs):
        raise Cancelled("stop now")

    handler = HFTrainingSmokeHandler(args=_args("observe"), action_runner=cancel)

    with pytest.raises(Cancelled, match="stop now"):
        handler.handle()


def test_handler_default_path_uses_operator_dispatcher(monkeypatch, capsys) -> None:
    from tuner.cloud import hf_training_smoke_operator as operator

    calls = []

    def run(action, *, args, context):
        calls.append((action, args, context))
        return {"status": "PASS", "submitted": False}

    monkeypatch.setattr(operator, "run_training_smoke_action", run)
    args = _args("preflight")
    handler = HFTrainingSmokeHandler(args=args)

    assert handler.handle() == 0
    assert calls == [("preflight", args, handler.context)]
    assert json.loads(capsys.readouterr().out)["data"]["status"] == "PASS"


@pytest.mark.parametrize("result", [None, [], {1: "invalid"}])
def test_handler_rejects_malformed_action_result(
    result: object, capsys: pytest.CaptureFixture[str]
) -> None:
    handler = HFTrainingSmokeHandler(
        args=_args("verify"), action_runner=lambda *args, **kwargs: result
    )

    assert handler.handle() == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "HF_TRAINING_SMOKE_ERROR"
