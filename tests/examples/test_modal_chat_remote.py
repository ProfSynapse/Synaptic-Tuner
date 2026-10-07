"""Scoped subprocess boundary tests; no network calls."""

import subprocess
from types import SimpleNamespace

import pytest

from examples.modal_chat import remote

URL = "https://github.com/example/engine.git"
REF = "refs/heads/smoke"


def test_exact_ref_read_has_no_ambient_auth_or_repository_config(monkeypatch):
    calls = []
    payload = b"a" * 40 + b"\trefs/heads/smoke\n"

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(stdout=payload)

    monkeypatch.setattr(remote.subprocess, "run", run)
    reader = remote.ModalChatGitRemote(frozenset({(URL, REF)}))
    assert reader.read_ref(canonical_url=URL, exact_ref=REF) == payload
    argv, kwargs = calls[0]
    assert argv[0] == "/usr/bin/git"
    assert argv[-5:] == ["ls-remote", "--exit-code", "--refs", URL, REF]
    assert "credential.helper=" in argv
    assert "protocol.allow=never" in argv
    assert "http.followRedirects=false" in argv
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.DEVNULL
    assert kwargs["timeout"] == 30
    assert kwargs["env"] == {
        "PATH": "/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
        "GCM_INTERACTIVE": "Never",
    }


def test_undeclared_pair_rejects_before_subprocess(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected Git call")

    monkeypatch.setattr(remote.subprocess, "run", forbidden)
    reader = remote.ModalChatGitRemote(frozenset({(URL, REF)}))
    with pytest.raises(ValueError, match="not_allowed"):
        reader.read_ref(canonical_url=URL, exact_ref="refs/heads/main")


@pytest.mark.parametrize(
    "url,ref",
    [
        ("file:///tmp/repo", REF),
        ("ssh://git@github.com/example/engine.git", REF),
        ("https://user:password@example.invalid/engine.git", REF),
        (URL, "--heads"),
        (URL, "refs/tags/smoke"),
        (URL, "refs/heads/*"),
    ],
)
def test_noncanonical_credential_or_nonbranch_selectors_reject(url, ref):
    with pytest.raises(ValueError):
        remote.ModalChatGitRemote(frozenset({(url, ref)}))


@pytest.mark.parametrize("payload", [b"", b"x" * 4097, "not bytes"])
def test_bounded_output_rejects_before_evidence_issuer(monkeypatch, payload):
    monkeypatch.setattr(
        remote.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=payload),
    )
    reader = remote.ModalChatGitRemote(frozenset({(URL, REF)}))
    with pytest.raises(ValueError, match="unavailable"):
        reader.read_ref(canonical_url=URL, exact_ref=REF)


def test_provider_error_text_is_not_exposed(monkeypatch):
    def failed(*args, **kwargs):
        raise RuntimeError("untrusted provider detail")

    monkeypatch.setattr(remote.subprocess, "run", failed)
    reader = remote.ModalChatGitRemote(frozenset({(URL, REF)}))
    with pytest.raises(ValueError) as caught:
        reader.read_ref(canonical_url=URL, exact_ref=REF)
    assert str(caught.value) == "modal_chat_source_ref_unavailable"
    assert caught.value.__suppress_context__ is True
