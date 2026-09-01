from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE = Path(__file__).resolve().parents[3] / "Trainers/sft/src/model_loader.py"


def _load(monkeypatch, revision: str | None, snapshot: Path):
    calls = []

    class Fast:
        @staticmethod
        def from_pretrained(**kwargs):
            calls.append(kwargs)
            model = SimpleNamespace(
                config=SimpleNamespace(_commit_hash=revision, _name_or_path=kwargs["model_name"]),
            )
            tokenizer = SimpleNamespace(
                chat_template="x", name_or_path=kwargs["model_name"],
                init_kwargs={"_commit_hash": revision}, __len__=lambda self: 10,
            )
            return model, tokenizer

    monkeypatch.setitem(sys.modules, "unsloth", SimpleNamespace(FastLanguageModel=Fast, is_bfloat16_supported=lambda: True))
    monkeypatch.setitem(
        sys.modules, "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda **kwargs: str(snapshot)),
    )
    spec = importlib.util.spec_from_file_location("protected_model_loader_test", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, calls


def test_protected_loader_resolves_exact_anonymous_snapshot(monkeypatch, tmp_path: Path) -> None:
    revision = "a" * 40
    snapshot = tmp_path / "models--owner--model" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    module, calls = _load(monkeypatch, revision, snapshot)
    module.load_model_and_tokenizer(
        "owner/model", hf_token=False, model_revision=revision,
        trust_remote_code=False, use_safetensors=True, cache_dir="/controlled",
        require_resolved_revision=True,
    )
    assert calls == [{
        "model_name": str(snapshot.resolve()), "max_seq_length": 2048, "dtype": None,
        "load_in_4bit": True, "token": False, "use_exact_model_name": True,
        "trust_remote_code": False, "use_safetensors": True, "cache_dir": "/controlled",
    }]


def test_protected_loader_rejects_resolved_revision_drift(monkeypatch, tmp_path: Path) -> None:
    snapshot = tmp_path / "models--owner--model" / "snapshots" / ("b" * 40)
    snapshot.mkdir(parents=True)
    module, _ = _load(monkeypatch, "b" * 40, snapshot)
    with pytest.raises(RuntimeError, match="does not match"):
        module.load_model_and_tokenizer("owner/model", hf_token=False, model_revision="a" * 40, require_resolved_revision=True)


def test_runtime_loader_consumes_only_the_exact_local_snapshot(
    monkeypatch, tmp_path: Path
) -> None:
    revision = "a" * 40
    cache = (tmp_path / "cache" / "model").resolve()
    snapshot = cache / "models--owner--model" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    module, calls = _load(monkeypatch, revision, snapshot)

    module.load_model_and_tokenizer(
        "owner/model",
        hf_token=False,
        model_revision=revision,
        trust_remote_code=False,
        use_safetensors=True,
        cache_dir=str(cache),
        model_snapshot=str(snapshot),
        require_local_snapshot=True,
    )

    assert calls == [
        {
            "model_name": str(snapshot),
            "max_seq_length": 2048,
            "dtype": None,
            "load_in_4bit": True,
            "token": False,
            "use_exact_model_name": True,
            "trust_remote_code": False,
            "use_safetensors": True,
            "cache_dir": str(cache),
            "local_files_only": True,
        }
    ]


def test_runtime_loader_rejects_foreign_snapshot_binding(
    monkeypatch, tmp_path: Path
) -> None:
    revision = "a" * 40
    cache = (tmp_path / "cache" / "model").resolve()
    foreign = cache / "models--other--model" / "snapshots" / revision
    foreign.mkdir(parents=True)
    (foreign / "config.json").write_text("{}", encoding="utf-8")
    module, calls = _load(monkeypatch, revision, foreign)

    with pytest.raises(RuntimeError, match="exact binding"):
        module.load_model_and_tokenizer(
            "owner/model",
            model_revision=revision,
            cache_dir=str(cache),
            model_snapshot=str(foreign),
            require_local_snapshot=True,
        )
    assert calls == []


def test_runtime_loader_rejects_redirected_snapshot_members(
    monkeypatch, tmp_path: Path
) -> None:
    revision = "a" * 40
    cache = (tmp_path / "cache" / "model").resolve()
    snapshot = cache / "models--owner--model" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"model")
    try:
        (snapshot / "model.bin").symlink_to(outside)
    except OSError:
        pytest.skip("file symlinks are unavailable")
    module, calls = _load(monkeypatch, revision, snapshot)

    with pytest.raises(RuntimeError, match="redirect"):
        module.load_model_and_tokenizer(
            "owner/model",
            model_revision=revision,
            cache_dir=str(cache),
            model_snapshot=str(snapshot),
            require_local_snapshot=True,
        )
    assert calls == []


def test_runtime_loader_never_falls_back_when_local_loading_fails(
    monkeypatch, tmp_path: Path
) -> None:
    revision = "a" * 40
    cache = (tmp_path / "cache" / "model").resolve()
    snapshot = cache / "models--owner--model" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    module, _ = _load(monkeypatch, revision, snapshot)
    calls = []

    def fail(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("local load failed")

    monkeypatch.setattr(module.FastLanguageModel, "from_pretrained", fail)
    with pytest.raises(RuntimeError, match="local load failed"):
        module.load_model_and_tokenizer(
            "owner/model",
            model_revision=revision,
            cache_dir=str(cache),
            model_snapshot=str(snapshot),
            require_local_snapshot=True,
        )
    assert len(calls) == 1
    assert calls[0]["model_name"] == str(snapshot)
    assert calls[0]["local_files_only"] is True
