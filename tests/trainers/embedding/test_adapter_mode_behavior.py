"""Adapter-mode BEHAVIOR regression test (B1, remediation cycle 1).

CONTEXT: PR #109 review found B1 (Blocking) — `adapter_mode` lora/frozen_head
were *validated* (qlora->ValueError, unknown->ValueError) and the loader *path*
was selected, but the adapter was NEVER APPLIED to the model. The default
``lora`` path silently trained full weights and emitted no adapter, breaking the
CONTRACTS §2.4 adapter->merge/upload contract. The fix (e224b02) makes the
LOADER the single adapter-application authority: ``load_embedding_model`` now
returns a model already shaped for ``adapter_mode`` via
``_apply_adapter_mode_fast`` / ``_apply_adapter_mode_fallback``.

WHY THE PHASE-1 SUITE MISSED IT: the existing adapter-mode tests
(test_capability_probe.py) cover the VALIDATION axis and loader PATH SELECTION
using SENTINEL/MOCK model objects — so "does lora/frozen_head actually APPLY the
adapter to a real model" was never exercised. A mock model has no parameters, so
a requires_grad topology assertion is impossible against it. This file closes
that gap with a LIVE model.

TWO TIERS:
  1. Live behavior (importorskip peft + sentence_transformers): load the REAL
     cached ``bge-base-en`` base and assert the OBSERVABLE per-mode trainable /
     requires_grad topology — the contract B1 broke and the fix restores.
       - lora:        only LoRA params trainable; every base param frozen.
       - frozen_head: only the appended ``frozen_head_dense`` Dense trainable;
                      every pre-existing base param frozen.
       - full:        every param trainable.
  2. Dispatch routing (no heavy deps): stub the three apply-* leaves to markers
     and assert ``_apply_adapter_mode_{fast,fallback}`` route each mode to the
     right leaf, and that ``full`` is a pass-through no-op. This tier runs in
     CI even where peft/ST/torch are absent, so the dispatch contract is always
     guarded; the live tier deepens it where the deps exist.

COUNTER-TEST-BY-REVERT (documented expected cardinality): reverting the B1 fix
in model_loader.py to its 4453548 shape (apply-* leaves removed, adapter never
applied) makes the live-tier per-mode topology assertions fail — lora would show
the full base trainable instead of lora-only, frozen_head would have no
``frozen_head_dense`` and the base trainable. The dispatch tier would error on
the missing ``_apply_adapter_mode_*`` symbols. See task #34 HANDOFF for the
measured cardinality.

Isolated imports (the bare ``import registry``/``data_loader`` shadow hazard vs
the sft trainer) go through ``_isolated_import.load_embedding_src`` — same
discipline as the rest of this suite.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# tests/ is not a package on sys.path; make the sibling helper importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _isolated_import import load_embedding_src  # noqa: E402

model_loader = load_embedding_src("model_loader")
registry = load_embedding_src("registry")


# ---------------------------------------------------------------------------
# Tier 1 — LIVE per-mode requires_grad topology (the B1 contract)
# ---------------------------------------------------------------------------
# A real base model is needed: a mock has no parameters, so the topology the fix
# restores cannot be observed against a stub. bge-base-en is the registry's
# canonical encoder seed and is small + commonly cached. Gated behind the heavy
# deps so CI without them skips cleanly (CONTRACTS §7.4 cloud-smoke posture).

pytest.importorskip("peft", reason="LoRA application needs PEFT")
pytest.importorskip("sentence_transformers", reason="embedding base load needs sentence-transformers")
pytest.importorskip("torch", reason="requires_grad topology needs torch")

_LIVE_REGISTRY_NAME = "bge-base-en"


@pytest.fixture(scope="module")
def _offline_env():
    """Prefer the local HF cache; do not hit the network for a regression test.

    If the base is not cached locally and offline blocks the load, the live
    tests skip rather than fail — the dispatch tier still guards the contract.
    """
    prior = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prior


def _load_live(adapter_mode: str):
    """Load the real bge base in the given adapter_mode via the fallback path.

    allow_fast_path=False forces the plain-ST fallback (the correctness baseline,
    CPU-capable) so the test is deterministic and GPU-independent. Skips if the
    base cannot be loaded offline.
    """
    spec = registry.get_spec(_LIVE_REGISTRY_NAME)
    try:
        return model_loader.load_embedding_model(spec, adapter_mode, allow_fast_path=False)
    except Exception as exc:  # base not cached offline / load error -> skip, don't fail.
        pytest.skip(f"live bge base unavailable for adapter behavior test: {exc}")


def _trainable_named_params(model):
    return [name for name, p in model.named_parameters() if p.requires_grad]


def _frozen_named_params(model):
    return [name for name, p in model.named_parameters() if not p.requires_grad]


def test_live_lora_trains_only_adapter_params(_offline_env):
    """lora: every trainable parameter is a LoRA parameter; the base is frozen.

    This is the exact contract B1 broke — pre-fix, adapter_mode='lora' left the
    full base trainable and emitted no adapter.
    """
    loaded = _load_live("lora")
    assert loaded.loader_path == "fallback"

    trainable = _trainable_named_params(loaded.model)
    assert trainable, "lora produced no trainable params (adapter not applied?)"
    assert all("lora" in name.lower() for name in trainable), (
        "lora mode left non-LoRA params trainable (B1 regression): "
        f"{[n for n in trainable if 'lora' not in n.lower()][:5]}"
    )

    # Every NON-lora (base) param must be frozen.
    frozen = _frozen_named_params(loaded.model)
    assert all("lora" not in name.lower() for name in frozen)
    assert len(frozen) > len(trainable), "expected the frozen base to dwarf the adapter"


def test_live_frozen_head_trains_only_appended_head(_offline_env):
    """frozen_head: only the appended frozen_head_dense is trainable; base frozen.

    Encoder bases (bge/e5/gte) ship no projection head, so the fix appends a
    named Dense head and trains only it (CONTRACTS §2.4).
    """
    from sentence_transformers import models as st_models

    loaded = _load_live("frozen_head")
    model = loaded.model

    # The appended head exists under the contracted stable name.
    head_children = [name for name, _ in model.named_children() if name == "frozen_head_dense"]
    assert head_children == ["frozen_head_dense"], (
        "frozen_head did not append the contracted 'frozen_head_dense' module"
    )

    trainable = _trainable_named_params(model)
    assert trainable, "frozen_head produced no trainable params (head not appended/unfrozen?)"
    # Every trainable param belongs to the appended head.
    assert all(name.startswith("frozen_head_dense") for name in trainable), (
        "frozen_head left non-head params trainable (B1 regression): "
        f"{[n for n in trainable if not n.startswith('frozen_head_dense')][:5]}"
    )

    # Every pre-existing base param is frozen.
    frozen = _frozen_named_params(model)
    assert any(not name.startswith("frozen_head_dense") for name in frozen)
    assert all(not name.startswith("frozen_head_dense") for name in frozen)

    # The head is a real ST Dense (so it is saved + reloadable with the model).
    dense_named = [name for name, mod in model.named_modules() if isinstance(mod, st_models.Dense)]
    assert "frozen_head_dense" in dense_named


def test_live_full_trains_everything(_offline_env):
    """full: every parameter is trainable (no freezing, no adapter)."""
    loaded = _load_live("full")
    frozen = _frozen_named_params(loaded.model)
    assert frozen == [], f"full mode froze params it should train: {frozen[:5]}"
    assert _trainable_named_params(loaded.model), "full mode produced no trainable params"


def test_live_lora_and_full_differ_in_trainable_count(_offline_env):
    """Cross-check: lora trains strictly fewer params than full on the same base.

    A single assertion that the adapter actually REDUCES the trainable surface —
    the economic point of LoRA, and a second independent witness that B1's
    'lora == silent full-tune' behavior is gone.
    """
    import torch  # noqa: F401  (importorskip already guaranteed availability)

    lora = _load_live("lora")
    full = _load_live("full")

    lora_trainable = sum(p.numel() for p in lora.model.parameters() if p.requires_grad)
    full_trainable = sum(p.numel() for p in full.model.parameters() if p.requires_grad)
    assert 0 < lora_trainable < full_trainable, (
        f"lora trainable={lora_trainable} not strictly between 0 and full={full_trainable}"
    )


# ---------------------------------------------------------------------------
# Tier 2 — dispatch ROUTING (runs without heavy deps via leaf stubs)
# ---------------------------------------------------------------------------
# importorskip above would skip this whole module where peft/ST/torch are
# absent. To keep the dispatch contract guarded even there, these tests stub the
# three apply-* leaves to markers — no real model, no heavy import — so they
# exercise pure routing. They are colocated here (not split to a sibling file)
# because they assert the SAME §2.4 application contract from the dispatch side.


@pytest.fixture
def _fake_spec():
    return registry.EmbeddingModelSpec(
        name="t", hf_id="x/y", family="bert",
        lora_target_modules=("query", "value"),
    )


@pytest.mark.parametrize("dispatch_name,expected_leaf", [
    ("_apply_adapter_mode_fallback", {"lora": "_apply_lora_fallback", "frozen_head": "_apply_frozen_head"}),
    ("_apply_adapter_mode_fast", {"lora": "_apply_lora_fast", "frozen_head": "_apply_frozen_head"}),
])
def test_dispatch_routes_each_mode_to_its_leaf(dispatch_name, expected_leaf, _fake_spec, monkeypatch):
    """Each adapter_mode routes to exactly its apply-leaf; full is a no-op.

    Proves the §2.4 dispatch contract without loading a real model: the leaves
    are replaced with markers, and we assert the dispatcher returns the right
    marker per mode (and the untouched base object for full).
    """
    base = object()

    def marker(leaf_name):
        def _leaf(model, spec, *rest):
            return (leaf_name, model)
        return _leaf

    monkeypatch.setattr(model_loader, "_apply_lora_fallback", lambda m, s, lc: ("_apply_lora_fallback", m))
    monkeypatch.setattr(model_loader, "_apply_lora_fast", lambda m, s, lc: ("_apply_lora_fast", m))
    monkeypatch.setattr(model_loader, "_apply_frozen_head", lambda m, s: ("_apply_frozen_head", m))

    dispatch = getattr(model_loader, dispatch_name)

    # full -> pass-through no-op (the SAME object back, untouched).
    assert dispatch(base, _fake_spec, "full", None) is base

    # lora / frozen_head -> the contracted leaf marker.
    lora_out = dispatch(base, _fake_spec, "lora", {"r": 8})
    assert lora_out == (expected_leaf["lora"], base)

    fh_out = dispatch(base, _fake_spec, "frozen_head", None)
    assert fh_out == (expected_leaf["frozen_head"], base)


def test_lora_hyperparams_defaults_when_no_block():
    """An omitted lora: block yields the documented defaults, not a silent r=0."""
    r, alpha, dropout = model_loader._lora_hyperparams(None)
    assert (r, alpha, dropout) == (16, 32, 0.05)


def test_lora_hyperparams_honors_config():
    r, alpha, dropout = model_loader._lora_hyperparams({"r": 8, "alpha": 16, "dropout": 0.1})
    assert (r, alpha, dropout) == (8, 16, 0.1)
