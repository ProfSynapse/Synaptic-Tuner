from __future__ import annotations

import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tuner.core.exceptions import CloudProviderError
from tuner.handlers.hf_smoke_handler import (
    HFSmokeHandler,
    _preflight_secret_file,
    _read_claimed_hf_token,
)
from tuner.project import ProjectContext


ROOT = Path(__file__).resolve().parents[2]


def test_handler_import_is_provider_free_in_fresh_process():
    probe = (
        "import importlib,json,sys;"
        "importlib.import_module('tuner.handlers.hf_smoke_handler');"
        "print(json.dumps(sorted(name for name in sys.modules "
        "if name.split('.')[0] in {'huggingface_hub','transformers','torch'})))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], cwd=ROOT, capture_output=True, text=True, check=True
    )
    assert json.loads(completed.stdout) == []


def test_handler_dispatches_only_closed_actions(monkeypatch, capsys):
    for action in ("approve", "execute", "observe"):
        args = Namespace(subcommand=action, json=True, events=None)
        handler = HFSmokeHandler(args=args)
        monkeypatch.setattr(handler, f"_{action}", lambda action=action: {"action": action})
        assert handler.handle() == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["data"] == {"action": action}


def test_handler_rejects_missing_action_without_provider_factory_call(capsys):
    calls = []
    handler = HFSmokeHandler(
        args=Namespace(subcommand=None, json=True, events=None),
        provider_factory=lambda token: calls.append(token),
    )
    assert handler.handle() == 1
    assert calls == []
    assert "approve, execute, or observe" in capsys.readouterr().out


def _context(root: Path) -> ProjectContext:
    return ProjectContext.standalone(engine_root=root, invocation_cwd=root)


def _secret(root: Path, content: str = "HF_TOKEN=hf_file_authority\n") -> Path:
    path = root / "smoke.env"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.mark.parametrize("ambient", ["HF_TOKEN", "HF_API_KEY"])
def test_preflight_rejects_ambient_authority_without_reading_file(
    tmp_path, monkeypatch, ambient
):
    path = _secret(tmp_path)
    monkeypatch.setenv(ambient, "hf_ambient_must_not_escape")
    calls = []
    monkeypatch.setattr(Path, "read_bytes", lambda self: calls.append(self))
    with pytest.raises(CloudProviderError, match="ambient") as caught:
        _preflight_secret_file(path, context=_context(tmp_path))
    assert "hf_ambient_must_not_escape" not in str(caught.value)
    assert calls == []


def test_preflight_rejects_outside_root_before_secret_read_or_claim(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    outside = _secret(tmp_path)
    calls = []
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_API_KEY", raising=False)
    monkeypatch.setattr(Path, "read_bytes", lambda self: calls.append(self))
    with pytest.raises(CloudProviderError, match="project/config"):
        _preflight_secret_file(outside, context=_context(project))
    assert calls == []


def test_preflight_rejects_oversized_secret_without_reading_bytes(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_API_KEY", raising=False)
    path = _secret(tmp_path, "HF_TOKEN=" + ("x" * (64 * 1024)) + "\n")
    calls = []
    monkeypatch.setattr(Path, "read_bytes", lambda self: calls.append(self))
    with pytest.raises(CloudProviderError, match="bounded"):
        _preflight_secret_file(path, context=_context(tmp_path))
    assert calls == []


def test_preflight_rejects_linked_file_or_parent_chain(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_API_KEY", raising=False)
    target = _secret(tmp_path)
    link = tmp_path / "linked.env"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(CloudProviderError, match="links"):
        _preflight_secret_file(link, context=_context(tmp_path))

    real_parent = tmp_path / "real"
    real_parent.mkdir()
    nested = _secret(real_parent)
    linked_parent = tmp_path / "alias"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(CloudProviderError, match="links"):
        _preflight_secret_file(linked_parent / nested.name, context=_context(tmp_path))


@pytest.mark.parametrize(
    "content",
    [
        "HF_API_KEY=hf_alias\n",
        "OTHER=value\nHF_TOKEN=hf_value\n",
        "HF_TOKEN=hf_first\nHF_TOKEN=hf_second\n",
        "HF_TOKEN=\n",
        "export HF_TOKEN=hf_value\n",
        "HF_TOKEN=${OTHER}\n",
        "HF_TOKEN=hf_value # inline\n",
        "HF_TOKEN='unterminated\n",
        "HF_TOKEN=\"escaped\\value\"\n",
        "\x00HF_TOKEN=hf_value\n",
    ],
)
def test_post_claim_parser_rejects_every_noncanonical_secret_document(
    tmp_path, monkeypatch, content
):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_API_KEY", raising=False)
    claim = _preflight_secret_file(_secret(tmp_path, content), context=_context(tmp_path))
    with pytest.raises(CloudProviderError):
        _read_claimed_hf_token(claim)
    assert "HF_TOKEN" not in os.environ and "HF_API_KEY" not in os.environ


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("# selected authority\n\nHF_TOKEN=hf_plain\n", "hf_plain"),
        ("HF_TOKEN='hf_single'\n", "hf_single"),
        ('HF_TOKEN="hf_double"\n', "hf_double"),
    ],
)
def test_post_claim_reader_accepts_only_deterministic_token_forms_without_env_mutation(
    tmp_path, monkeypatch, content, expected
):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_API_KEY", raising=False)
    claim = _preflight_secret_file(_secret(tmp_path, content), context=_context(tmp_path))
    assert _read_claimed_hf_token(claim) == expected
    assert "HF_TOKEN" not in os.environ and "HF_API_KEY" not in os.environ


def test_metadata_selection_stores_no_identity_and_consumes_current_safe_file(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_API_KEY", raising=False)
    path = _secret(tmp_path)
    claim = _preflight_secret_file(path, context=_context(tmp_path))
    assert not hasattr(claim, "identity")
    path.write_text("HF_TOKEN=hf_changed_and_longer\n", encoding="utf-8")
    assert _read_claimed_hf_token(claim) == "hf_changed_and_longer"


def test_parent_chain_swap_after_preflight_fails_closed(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_API_KEY", raising=False)
    parent = tmp_path / "secrets"
    parent.mkdir()
    path = _secret(parent)
    claim = _preflight_secret_file(path, context=_context(tmp_path))
    moved = tmp_path / "moved-secrets"
    parent.rename(moved)
    try:
        parent.symlink_to(moved, target_is_directory=True)
    except OSError:
        moved.rename(parent)
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(CloudProviderError, match="links or reparse"):
        _read_claimed_hf_token(claim)


def test_execute_preclaim_rejection_never_reaches_submitter(tmp_path, monkeypatch):
    path = _secret(tmp_path)
    monkeypatch.setenv("HF_TOKEN", "hf_ambient")
    handler = HFSmokeHandler(
        args=Namespace(
            subcommand="execute", experiment_id="exp-1", env_file=str(path),
            json=False, events=None,
        ),
        context=_context(tmp_path),
    )
    tracking = SimpleNamespace(resolve_uri=lambda uri: path)
    experiment = SimpleNamespace(
        hf_submission_state="APPROVED",
        hf_run_approval_uri="tracking://approval.json",
    )
    monkeypatch.setattr(handler, "_state", lambda: (tracking, experiment, object()))
    monkeypatch.setattr(
        "tuner.handlers.hf_smoke_handler.load_canonical_json", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        "tuner.handlers.hf_smoke_handler.validate_hf_run_approval", lambda value: object()
    )
    calls = []
    monkeypatch.setattr(
        "tuner.handlers.hf_smoke_handler.submit_approved_bootstrap_smoke",
        lambda **kwargs: calls.append(kwargs),
    )
    with pytest.raises(CloudProviderError, match="ambient"):
        handler._execute()
    assert calls == []
