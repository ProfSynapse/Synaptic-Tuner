"""Unit tests for the M-a weights-fetch wiring in config_translation.

These assert the registry-pinned `revision` is THREADED into
huggingface_hub.snapshot_download (the review's M-a finding: the pin was
decorative because nothing wired the download). The download itself is MOCKED —
no live network, no multi-GB pull — so these run on CPU CI alongside the rest of
the P0 suite. The real download only happens on a non-dry-run GPU run (CI-deferred).

Covered:
  - revision is threaded verbatim from the registry entry into snapshot_download.
  - dry_run short-circuits with NO download (keeps --dry-run zero-network).
  - an already-populated checkpoint dir skips the download (idempotent).
  - resolve_checkpoint_dir stays a pure path namer (no network even when called).
  - a missing hf_id fails loud (ValueError naming the entry).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
_CT_PATH = REPO_ROOT / "Trainers" / "ace_step" / "src" / "config_translation.py"


def _load_config_translation():
    """Load config_translation.py under a UNIQUE module name (no sys.modules collision).

    Mirrors test_argv_contract.py's hermetic file-path loader — a bare
    ``import config_translation`` would risk shadowing a same-named module from
    another trainer dir (the documented ``data_loader``/``registry`` hazard).
    """
    spec = importlib.util.spec_from_file_location("ace_step_config_translation_fetch", _CT_PATH)
    assert spec is not None and spec.loader is not None, f"cannot load {_CT_PATH}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ct = _load_config_translation()

# The pinned aggregate-repo revision in the real model_registry.yaml — the test
# asserts THIS exact SHA threads through, so a silent un-pin (revision dropped) fails.
PINNED_2B_REVISION = "19671f406d603126926c1b7e2adc169acbcade22"


def _write_registry(repo_root: Path, models: dict) -> None:
    """Write a model_registry.yaml at the layout _load_model_registry expects."""
    cfg_dir = repo_root / "Trainers" / "ace_step" / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "model_registry.yaml").write_text(
        yaml.safe_dump({"models": models}), encoding="utf-8"
    )


@pytest.fixture
def repo_root_2b(tmp_path: Path) -> Path:
    """A temp repo_root carrying a single 2B registry entry with the pinned revision."""
    _write_registry(
        tmp_path,
        {
            "ace-step-v15-2b": {
                "hf_id": "ACE-Step/Ace-Step1.5",
                "variant": "turbo",
                "revision": PINNED_2B_REVISION,
                "components": {"dit": "acestep-v15-turbo", "vae": "vae"},
            }
        },
    )
    return tmp_path


def _config(name: str = "ace-step-v15-2b") -> dict:
    return {"model": {"registry_name": name}}


def test_fetch_threads_pinned_revision(monkeypatch, repo_root_2b: Path) -> None:
    """snapshot_download is called with revision=<the registry's pinned SHA>."""
    calls: list[dict] = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(kwargs["local_dir"])

    # Inject a stub huggingface_hub so the lazy `from huggingface_hub import
    # snapshot_download` inside fetch_checkpoint resolves to our recorder — no
    # network, no real dependency behavior.
    import sys
    import types

    stub = types.ModuleType("huggingface_hub")
    stub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", stub)

    out = ct.fetch_checkpoint(_config(), repo_root_2b, dry_run=False)

    assert len(calls) == 1, "snapshot_download should be called exactly once"
    assert calls[0]["repo_id"] == "ACE-Step/Ace-Step1.5"
    assert calls[0]["revision"] == PINNED_2B_REVISION, "the pinned revision must thread through"
    assert Path(calls[0]["local_dir"]) == out
    assert out == repo_root_2b / "Datasets" / "ace_step_models" / "ace-step-v15-2b"


def test_dry_run_does_not_download(monkeypatch, repo_root_2b: Path) -> None:
    """dry_run=True returns the dir WITHOUT importing/calling snapshot_download."""
    import sys

    # Guarantee a real import would fail loudly if attempted (so the test proves
    # dry_run is genuinely network-free, not just lucky with a cached module).
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    out = ct.fetch_checkpoint(_config(), repo_root_2b, dry_run=True)
    assert out == repo_root_2b / "Datasets" / "ace_step_models" / "ace-step-v15-2b"


def test_already_present_skips_download(monkeypatch, repo_root_2b: Path) -> None:
    """A populated DiT subfolder short-circuits the (expensive) re-download."""
    calls: list[dict] = []

    import sys
    import types

    stub = types.ModuleType("huggingface_hub")
    stub.snapshot_download = lambda **kw: calls.append(kw)
    monkeypatch.setitem(sys.modules, "huggingface_hub", stub)

    # Pre-create the dir + the DiT subfolder so the idempotency guard fires.
    ckpt = repo_root_2b / "Datasets" / "ace_step_models" / "ace-step-v15-2b"
    (ckpt / "acestep-v15-turbo").mkdir(parents=True)

    ct.fetch_checkpoint(_config(), repo_root_2b, dry_run=False)
    assert calls == [], "an already-populated checkpoint must not re-download"


def test_force_redownloads_even_when_present(monkeypatch, repo_root_2b: Path) -> None:
    """force=True overrides the idempotency skip."""
    calls: list[dict] = []

    import sys
    import types

    stub = types.ModuleType("huggingface_hub")
    stub.snapshot_download = lambda **kw: calls.append(kw)
    monkeypatch.setitem(sys.modules, "huggingface_hub", stub)

    ckpt = repo_root_2b / "Datasets" / "ace_step_models" / "ace-step-v15-2b"
    (ckpt / "acestep-v15-turbo").mkdir(parents=True)

    ct.fetch_checkpoint(_config(), repo_root_2b, dry_run=False, force=True)
    assert len(calls) == 1
    assert calls[0]["revision"] == PINNED_2B_REVISION


def test_resolve_checkpoint_dir_is_pure(repo_root_2b: Path) -> None:
    """The path namer does NO network and never creates the dir (dry-run-safe)."""
    out = ct.resolve_checkpoint_dir(_config(), repo_root_2b)
    assert out == repo_root_2b / "Datasets" / "ace_step_models" / "ace-step-v15-2b"
    assert not out.exists(), "resolve_checkpoint_dir must not materialize anything"


def test_resolve_dit_subfolder(repo_root_2b: Path) -> None:
    """F-5: the DiT component subfolder is resolved from the registry entry."""
    assert ct.resolve_dit_subfolder(_config(), repo_root_2b) == "acestep-v15-turbo"


def test_missing_hf_id_fails_loud(monkeypatch, tmp_path: Path) -> None:
    """An entry without hf_id raises ValueError naming the entry (no silent fetch)."""
    _write_registry(tmp_path, {"broken": {"variant": "turbo", "revision": "abc"}})
    with pytest.raises(ValueError, match="broken"):
        ct.fetch_checkpoint(_config("broken"), tmp_path, dry_run=False)


def test_unknown_registry_name_fails_loud(repo_root_2b: Path) -> None:
    """An unknown registry_name raises ValueError listing the known keys."""
    with pytest.raises(ValueError, match="Unknown model.registry_name"):
        ct.fetch_checkpoint(_config("does-not-exist"), repo_root_2b, dry_run=False)
