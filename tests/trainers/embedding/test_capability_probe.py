"""Capability-probe totality — probe_capabilities() NEVER raises.

CONTEXT: ``Trainers/embedding/src/model_loader.py`` chooses the Unsloth fast
path vs the plain-SentenceTransformer fallback via ``probe_capabilities()``.
Per CONTRACTS §2.2 the probe is IMPORT-GUARDED and TOTAL: every failure mode
(opt-in disabled / no CUDA / unsloth ImportError / broken torch) must degrade to
a fallback capability, never propagate an exception. This is what makes the
runtime-unverified ``FastSentenceTransformer`` a non-blocking optimization
rather than a load-bearing assumption (R1).

We exercise each branch by controlling what the probe sees:
- ``allow_fast_path=False``                      -> disabled-by-config fallback
- ``torch.cuda.is_available() -> False``          -> no-CUDA fallback
- ``import unsloth`` raises ImportError           -> unsloth-not-importable fallback
- ``import torch`` raises                          -> torch-unavailable fallback
- everything available                             -> fast path
And ``load_embedding_model`` is checked for adapter-mode validation + the
"fast probed available but load failed -> degrade, don't crash" path.

The ``unsloth`` import is intercepted via a ``sys.meta_path`` finder so we can
force ImportError or success WITHOUT a real unsloth install (there is none in
CI). ``torch.cuda.is_available`` is monkeypatched on the real torch module.
"""
from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _isolated_import import load_embedding_src  # noqa: E402

model_loader = load_embedding_src("model_loader")
registry = load_embedding_src("registry")


@pytest.fixture
def fake_spec():
    """A minimal valid EmbeddingModelSpec for load_embedding_model tests."""
    return registry.EmbeddingModelSpec(
        name="fake",
        hf_id="org/fake",
        family="bert",
        pooling="mean",
        max_seq_length=128,
    )


# ---------------------------------------------------------------------------
# probe_capabilities — never raises; correct reason per branch
# ---------------------------------------------------------------------------

def test_probe_disabled_by_config():
    caps = model_loader.probe_capabilities(allow_fast_path=False)
    assert caps.fast_path_available is False
    assert caps.cuda is False
    assert caps.reason == "disabled by config"


def test_probe_no_cuda_falls_back(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    caps = model_loader.probe_capabilities(allow_fast_path=True)
    assert caps.fast_path_available is False
    assert caps.cuda is False
    assert "no CUDA" in caps.reason


def test_probe_unsloth_import_error_falls_back(monkeypatch):
    """CUDA available but `import unsloth` raises -> fallback, no exception."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "unsloth" or name.startswith("unsloth."):
            raise ImportError("no unsloth in this env")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    caps = model_loader.probe_capabilities(allow_fast_path=True)
    assert caps.fast_path_available is False
    assert caps.cuda is True  # CUDA was present; only the unsloth import failed
    assert "unsloth not importable" in caps.reason


def test_probe_torch_unavailable_falls_back(monkeypatch):
    """A broken/absent torch import degrades to fallback, never raises."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    caps = model_loader.probe_capabilities(allow_fast_path=True)
    assert caps.fast_path_available is False
    assert caps.cuda is False
    assert "torch unavailable" in caps.reason


def test_probe_fast_path_available(monkeypatch):
    """CUDA present AND unsloth.FastSentenceTransformer importable -> fast path."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    fake_unsloth = types.ModuleType("unsloth")
    fake_unsloth.FastSentenceTransformer = type("FastSentenceTransformer", (), {})
    monkeypatch.setitem(sys.modules, "unsloth", fake_unsloth)

    caps = model_loader.probe_capabilities(allow_fast_path=True)
    assert caps.fast_path_available is True
    assert caps.cuda is True
    assert caps.reason == "unsloth available"


@pytest.mark.parametrize("allow", [True, False])
def test_probe_never_raises_under_any_flag(allow, monkeypatch):
    """Property: regardless of flag/environment, probe returns (never raises)."""
    # Even with a hostile torch that raises on cuda access, the probe must cope.
    import torch

    def boom():
        raise RuntimeError("cuda subsystem exploded")

    monkeypatch.setattr(torch.cuda, "is_available", boom)
    caps = model_loader.probe_capabilities(allow_fast_path=allow)
    assert caps.fast_path_available is False  # never claims fast on failure


# ---------------------------------------------------------------------------
# load_embedding_model — adapter-mode validation + degrade-not-crash
# ---------------------------------------------------------------------------

def test_qlora_adapter_mode_is_deferred(fake_spec):
    with pytest.raises(ValueError) as exc:
        model_loader.load_embedding_model(fake_spec, adapter_mode="qlora")
    assert "qlora" in str(exc.value).lower()
    assert "defer" in str(exc.value).lower()


def test_unknown_adapter_mode_raises(fake_spec):
    with pytest.raises(ValueError) as exc:
        model_loader.load_embedding_model(fake_spec, adapter_mode="bogus")
    assert "bogus" in str(exc.value)


@pytest.mark.parametrize("mode", ["full", "lora", "frozen_head"])
def test_fallback_path_used_when_fast_unavailable(mode, fake_spec, monkeypatch):
    """With fast path disabled, the loader builds via the plain ST fallback and
    returns loader_path='fallback' — never touching unsloth."""
    sentinel_model = object()

    def fake_load_fallback(spec):
        return sentinel_model

    monkeypatch.setattr(model_loader, "_load_fallback", fake_load_fallback)

    loaded = model_loader.load_embedding_model(
        fake_spec, adapter_mode=mode, allow_fast_path=False
    )
    assert loaded.loader_path == "fallback"
    assert loaded.model is sentinel_model
    assert loaded.capabilities.fast_path_available is False


def test_fast_load_failure_degrades_to_fallback(fake_spec, monkeypatch):
    """Probe says fast is available, but _load_fast raises -> degrade to fallback
    with a warning, NOT a crash (CONTRACTS §2 robustness)."""
    fast_caps = model_loader.LoaderCapabilities(
        fast_path_available=True, cuda=True, reason="unsloth available"
    )
    monkeypatch.setattr(
        model_loader, "probe_capabilities", lambda *, allow_fast_path=True: fast_caps
    )

    def boom_fast(spec, adapter_mode):
        raise RuntimeError("fast loader exploded")

    sentinel = object()
    monkeypatch.setattr(model_loader, "_load_fast", boom_fast)
    monkeypatch.setattr(model_loader, "_load_fallback", lambda spec: sentinel)

    with pytest.warns(UserWarning, match="falling back"):
        loaded = model_loader.load_embedding_model(fake_spec, adapter_mode="lora")

    assert loaded.loader_path == "fallback"
    assert loaded.model is sentinel


def test_fast_path_used_when_available(fake_spec, monkeypatch):
    fast_caps = model_loader.LoaderCapabilities(
        fast_path_available=True, cuda=True, reason="unsloth available"
    )
    monkeypatch.setattr(
        model_loader, "probe_capabilities", lambda *, allow_fast_path=True: fast_caps
    )
    sentinel = object()
    monkeypatch.setattr(model_loader, "_load_fast", lambda spec, adapter_mode: sentinel)

    loaded = model_loader.load_embedding_model(fake_spec, adapter_mode="full")
    assert loaded.loader_path == "fast"
    assert loaded.model is sentinel


# ---------------------------------------------------------------------------
# Prompt-dict construction (the fallback honors spec prompts)
# ---------------------------------------------------------------------------

def test_build_prompts_none_when_unset(fake_spec):
    assert model_loader._build_prompts(fake_spec) is None


def test_build_prompts_includes_set_prompts():
    spec = registry.EmbeddingModelSpec(
        name="e5", hf_id="intfloat/e5", family="bert",
        query_prompt="query: ", passage_prompt="passage: ",
    )
    assert model_loader._build_prompts(spec) == {"query": "query: ", "passage": "passage: "}
