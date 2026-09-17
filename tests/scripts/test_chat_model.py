"""Standalone command checks and persistence without training or a real GPU."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import chat_model


@pytest.fixture
def configuration(tmp_path):
    selected = tmp_path / "chat.json"
    selected.write_text(
        json.dumps(
            {
                "model": "owner/model",
                "revision": "a" * 40,
                "adapter_path": None,
                "prompt": "hello",
                "max_tokens": 48,
                "lifetime_seconds": 120,
            }
        )
    )
    return selected


def test_check_is_credential_free_and_never_opens_a_runtime(
    configuration, monkeypatch, capsys
):
    def forbidden(*args, **kwargs):
        pytest.fail("check opened serving")

    monkeypatch.setattr(chat_model, "open_model_chat", forbidden)
    assert chat_model.main(["--configuration", str(configuration), "--check"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "status": "CHAT_INPUTS_CHECKED",
        "training_required": False,
    }


def test_one_reply_saved_no_implicit_training_or_replay(
    configuration, tmp_path, monkeypatch, capsys
):
    output = tmp_path / "private"
    output.mkdir(mode=0o700)
    calls = []

    @contextmanager
    def opened(startup, policy, **kwargs):
        calls.append((startup, policy, kwargs))
        yield SimpleNamespace(
            chat=lambda prompt: SimpleNamespace(message="saved reply")
        )

    monkeypatch.setattr(chat_model, "open_model_chat", opened)
    args = ["--configuration", str(configuration), "--output-directory", str(output)]
    assert chat_model.main(args) == 0
    result = output / "chat-result.jsonl"
    records = [json.loads(line) for line in result.read_text().splitlines()]
    assert [r["status"] for r in records] == [
        "CLAIMED",
        "REPLY_SAVED",
        "CHAT_CONTEXT_CLOSED",
    ]
    assert records[1]["response"] == "saved reply"
    assert records[-1]["provider_shutdown_proof"] is False
    assert result.stat().st_mode & 0o777 == 0o600
    assert len(calls) == 1
    assert calls[0][2]["cwd"] == Path.cwd()
    assert calls[0][2]["cwd"] != output
    before = result.read_bytes()
    assert chat_model.main(args) == 1
    assert result.read_bytes() == before
    assert len(calls) == 1
    assert "saved reply" not in capsys.readouterr().out


def test_failure_preserves_claim_without_exception_text(
    configuration, tmp_path, monkeypatch, capsys
):
    output = tmp_path / "private"
    output.mkdir(mode=0o700)

    def fail(*args, **kwargs):
        raise RuntimeError("private-credential-message")

    monkeypatch.setattr(chat_model, "open_model_chat", fail)
    assert (
        chat_model.main(
            ["--configuration", str(configuration), "--output-directory", str(output)]
        )
        == 1
    )
    assert json.loads((output / "chat-result.jsonl").read_text())["status"] == "CLAIMED"
    assert "private-credential-message" not in capsys.readouterr().out


@pytest.mark.parametrize("raw", ['{"model":"one","model":"two"}', '{"x":NaN}', "[]"])
def test_invalid_or_duplicate_configuration_rejected(configuration, raw):
    configuration.write_text(raw)
    with pytest.raises(ValueError):
        chat_model.load_configuration(configuration)


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_configuration_requires_regular_nonlink_file(configuration, tmp_path, kind):
    selected = tmp_path / "unsafe"
    if kind == "symlink":
        selected.symlink_to(configuration)
    else:
        os.mkfifo(selected)
    with pytest.raises((ValueError, OSError)):
        chat_model.load_configuration(selected)


def test_configuration_changed_during_read_is_rejected(configuration, monkeypatch):
    real_fstat = os.fstat
    calls = 0

    def changed(fd):
        nonlocal calls
        calls += 1
        info = real_fstat(fd)
        if calls == 2:
            return SimpleNamespace(
                st_dev=info.st_dev,
                st_ino=info.st_ino,
                st_size=info.st_size,
                st_mtime_ns=info.st_mtime_ns + 1,
                st_ctime_ns=info.st_ctime_ns,
            )
        return info

    monkeypatch.setattr(os, "fstat", changed)
    with pytest.raises(ValueError, match="chat_configuration_changed"):
        chat_model.load_configuration(configuration)


def test_output_directory_substitution_cannot_redirect_result(
    configuration, tmp_path, monkeypatch
):
    output = tmp_path / "private"
    output.mkdir(mode=0o700)
    retained = tmp_path / "retained"
    real_open = os.open

    def substitute(path, flags, *args, **kwargs):
        if path == "chat-result.jsonl":
            output.rename(retained)
            output.mkdir(mode=0o700)
        return real_open(path, flags, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("substituted directory opened serving")

    monkeypatch.setattr(os, "open", substitute)
    monkeypatch.setattr(chat_model, "open_model_chat", forbidden)
    assert (
        chat_model.main(
            ["--configuration", str(configuration), "--output-directory", str(output)]
        )
        == 1
    )
    assert not (output / "chat-result.jsonl").exists()
    assert (retained / "chat-result.jsonl").read_bytes() == b""


@pytest.mark.parametrize("response", [None, "x" * 16385])
def test_invalid_response_is_not_saved(configuration, tmp_path, monkeypatch, response):
    output = tmp_path / "private"
    output.mkdir(mode=0o700)

    @contextmanager
    def opened(*args, **kwargs):
        yield SimpleNamespace(chat=lambda prompt: SimpleNamespace(message=response))

    monkeypatch.setattr(chat_model, "open_model_chat", opened)
    assert (
        chat_model.main(
            ["--configuration", str(configuration), "--output-directory", str(output)]
        )
        == 1
    )
    records = (output / "chat-result.jsonl").read_text().splitlines()
    assert len(records) == 1
    assert json.loads(records[0])["status"] == "CLAIMED"
