"""Merge-seam behavior-preservation tests (the load-bearing regression risk).

CONTEXT: ``shared/model_loading/merge.py`` was parametrized in place by
``family`` (CONTRACTS §3) to add an ``embedding`` merge branch alongside the
historical causal-LM merge. The lead-confirmed constraint is that the DEFAULT
``family="causal_lm"`` path stays BEHAVIOR-IDENTICAL to the pre-seam code so
every existing SFT/KTO/GRPO/upload caller is unaffected.

WHAT WE ASSERT — the CALL CONTRACT / merged ARTIFACT, never stdout ordering.
Per the architect/auditor note, the only behavioral delta in the seam is a
COSMETIC ``del``/``empty_cache`` reorder before the "✓ saved" print, which is
behavior-preserving. So a test that asserted on print order would be a
false-positive. Instead we mock ``unsloth.FastLanguageModel`` /
``unsloth.FastSentenceTransformer`` and capture the exact ``from_pretrained``
kwargs and ``save_pretrained_merged`` args — these ARE the production contract,
because merge always runs on the GPU+Unsloth image (the backend-coder proved
causal_lm identity this way in verify_wua.py; we lock it as a regression pin).

The expected pre-seam causal_lm call shape (held identical):
    FastLanguageModel.from_pretrained(
        model_name=str(lora_path),
        max_seq_length=<max_seq_length>,
        load_in_4bit=<load_in_4bit>,
    )
    model.save_pretrained_merged(str(output_path), tokenizer,
                                 save_method="merged_16bit")
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.model_loading import merge as merge_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Fake unsloth module: records every from_pretrained + save_pretrained_merged
# call so the test can assert on the exact call contract.
# ---------------------------------------------------------------------------

class _RecordingMergedModel:
    """Stand-in for a loaded model; records save_pretrained_merged calls."""

    def __init__(self, calls: list[dict]):
        self._calls = calls

    def save_pretrained_merged(self, *args, **kwargs):
        self._calls.append({"event": "save", "args": args, "kwargs": kwargs})


class _RecordingFastLanguageModel:
    calls: list[dict] = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.calls.append({"event": "from_pretrained", "args": args, "kwargs": kwargs})
        model = _RecordingMergedModel(cls.calls)
        tokenizer = object()  # opaque tokenizer; threaded into save call
        return model, tokenizer


class _RecordingFastSentenceTransformer:
    calls: list[dict] = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.calls.append({"event": "from_pretrained", "args": args, "kwargs": kwargs})
        # ST merge path returns a single model (no tokenizer tuple).
        return _RecordingMergedModel(cls.calls)


@pytest.fixture
def fake_unsloth(monkeypatch):
    """Install a fake ``unsloth`` module exposing both Fast* classes.

    ``merge.py`` does ``from unsloth import FastLanguageModel`` /
    ``from unsloth import FastSentenceTransformer`` INSIDE the merge functions,
    so injecting a fake ``unsloth`` into ``sys.modules`` is sufficient — no real
    unsloth/GPU needed. Also stub ``torch.cuda.empty_cache`` so the cleanup
    tail does not require a real CUDA context.
    """
    _RecordingFastLanguageModel.calls = []
    _RecordingFastSentenceTransformer.calls = []

    fake = types.ModuleType("unsloth")
    fake.FastLanguageModel = _RecordingFastLanguageModel
    fake.FastSentenceTransformer = _RecordingFastSentenceTransformer
    monkeypatch.setitem(sys.modules, "unsloth", fake)

    # empty_cache is called in the cleanup tail; make it a no-op (no CUDA in CI).
    monkeypatch.setattr(merge_mod.torch.cuda, "empty_cache", lambda: None)

    return fake


# ---------------------------------------------------------------------------
# Causal-LM path: behavior-identical call contract (the regression pin)
# ---------------------------------------------------------------------------

def test_causal_lm_is_default_family():
    """The default family must be 'causal_lm' so existing callers are unchanged."""
    import inspect

    sig = inspect.signature(merge_mod.merge_lora_checkpoint)
    assert sig.parameters["family"].default == "causal_lm"
    assert sig.parameters["family"].default == merge_mod.MERGE_FAMILY_CAUSAL_LM
    # Other defaults are part of the preserved contract too.
    assert sig.parameters["max_seq_length"].default == 2048
    assert sig.parameters["load_in_4bit"].default is True


def test_causal_lm_call_contract_default(fake_unsloth, tmp_path):
    """Default invocation (no family kwarg) routes through FastLanguageModel
    with the exact pre-seam call shape."""
    lora = tmp_path / "lora_ckpt"
    out = tmp_path / "merged"
    lora.mkdir()

    result = merge_mod.merge_lora_checkpoint(lora_path=lora, output_path=out)

    assert result == out
    calls = _RecordingFastLanguageModel.calls
    # Exactly one load + one save, no embedding path touched.
    assert _RecordingFastSentenceTransformer.calls == []
    load_call = next(c for c in calls if c["event"] == "from_pretrained")
    save_call = next(c for c in calls if c["event"] == "save")

    # from_pretrained call contract: keyword-only, exact keys + values.
    assert load_call["args"] == ()
    assert load_call["kwargs"] == {
        "model_name": str(lora),
        "max_seq_length": 2048,
        "load_in_4bit": True,
    }
    # save_pretrained_merged(str(output_path), tokenizer, save_method="merged_16bit")
    assert save_call["args"][0] == str(out)
    assert len(save_call["args"]) == 2  # (output_path, tokenizer)
    assert save_call["kwargs"] == {"save_method": "merged_16bit"}


def test_causal_lm_threads_custom_kwargs(fake_unsloth, tmp_path):
    """Custom max_seq_length / load_in_4bit thread through unchanged."""
    lora = tmp_path / "lora_ckpt"
    out = tmp_path / "merged"
    lora.mkdir()

    merge_mod.merge_lora_checkpoint(
        lora_path=lora,
        output_path=out,
        max_seq_length=4096,
        load_in_4bit=False,
        family="causal_lm",
    )

    load_call = next(
        c for c in _RecordingFastLanguageModel.calls if c["event"] == "from_pretrained"
    )
    assert load_call["kwargs"] == {
        "model_name": str(lora),
        "max_seq_length": 4096,
        "load_in_4bit": False,
    }


def test_causal_lm_creates_output_dir(fake_unsloth, tmp_path):
    """The output directory is created (mkdir parents/exist_ok) before saving."""
    lora = tmp_path / "lora_ckpt"
    out = tmp_path / "nested" / "merged"
    lora.mkdir()

    assert not out.exists()
    merge_mod.merge_lora_checkpoint(lora_path=lora, output_path=out)
    assert out.exists()


# ---------------------------------------------------------------------------
# Embedding path: distinct contract — ST loader, NO load_in_4bit threaded
# ---------------------------------------------------------------------------

def test_embedding_uses_st_loader_without_4bit(fake_unsloth, tmp_path):
    """Embedding merge runs through FastSentenceTransformer and must NOT pass a
    4-bit flag (QLoRA deferred, R8; ST loader has no such kwarg)."""
    lora = tmp_path / "emb_lora"
    out = tmp_path / "emb_merged"
    lora.mkdir()

    merge_mod.merge_lora_checkpoint(
        lora_path=lora,
        output_path=out,
        load_in_4bit=True,  # supplied, but must be IGNORED for embedding
        family="embedding",
    )

    # Causal path untouched.
    assert _RecordingFastLanguageModel.calls == []
    st_calls = _RecordingFastSentenceTransformer.calls
    load_call = next(c for c in st_calls if c["event"] == "from_pretrained")
    save_call = next(c for c in st_calls if c["event"] == "save")

    # Positional lora path + max_seq_length kwarg; crucially NO load_in_4bit.
    assert load_call["args"] == (str(lora),)
    assert "load_in_4bit" not in load_call["kwargs"]
    assert load_call["kwargs"] == {"max_seq_length": 2048}

    # ST save_pretrained_merged takes (output_path) + save_method, no tokenizer.
    assert save_call["args"] == (str(out),)
    assert save_call["kwargs"] == {"save_method": "merged_16bit"}


# ---------------------------------------------------------------------------
# Dispatch table + unknown-family guard
# ---------------------------------------------------------------------------

def test_unknown_family_raises_value_error():
    with pytest.raises(ValueError) as exc:
        merge_mod._merge_loader_for_family("reranker")
    assert "reranker" in str(exc.value)


def test_merge_lora_checkpoint_unknown_family_raises(tmp_path):
    lora = tmp_path / "x"
    lora.mkdir()
    with pytest.raises(ValueError):
        merge_mod.merge_lora_checkpoint(
            lora_path=lora, output_path=tmp_path / "o", family="bogus"
        )


def test_supported_families_are_exactly_two():
    assert merge_mod.SUPPORTED_MERGE_FAMILIES == ("causal_lm", "embedding")


def test_dispatch_maps_each_family_to_its_impl():
    assert merge_mod._merge_loader_for_family("causal_lm") is merge_mod._merge_causal_lm
    assert merge_mod._merge_loader_for_family("embedding") is merge_mod._merge_embedding
