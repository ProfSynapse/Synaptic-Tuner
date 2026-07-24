"""Engine + runner integration tests on a tiny real model, CPU-only.

Location: tests/batch/test_engines_and_runner.py

Builds a tiny in-memory ``LlamaForCausalLM`` (no downloads) and drives the
hf-batched engines and the runner through it. Covers:
  - batched greedy generation == per-row greedy generation (determinism);
  - left-pad correctness across mixed-length prompts;
  - capture positions gathered correctly vs a direct single-row forward;
  - persist dtype honored + capture.jsonl index consistency;
  - resume produces the identical artifact set as an uninterrupted run;
  - OOM auto-halving (monkeypatched engine, no GPU needed).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors.torch")

from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast  # noqa: E402
from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402

from tuner.batch.engines.base import GenerateItem, CaptureItem, GenerateResult, OutOfMemoryError  # noqa: E402
from tuner.batch.engines.hf_batched import (  # noqa: E402
    HFBatchedCaptureEngine,
    HFBatchedGenerateEngine,
    _ModelBundle,
    _is_composite_text_config,
    _is_local_peft_adapter_dir,
    _load_causal_lm_with_fallback,
    _run_with_oom_halving,
)
from tuner.batch import runner as batch_runner  # noqa: E402


HIDDEN = 32
N_LAYERS = 3
VOCAB = 40


def _tiny_tokenizer() -> PreTrainedTokenizerFast:
    """A trivial whitespace word-level tokenizer mapping tokens t0..t{VOCAB-1}.

    Word-level keeps encode/decode a clean round-trip so tests can reason about
    exact token positions without a real BPE merge table.
    """
    vocab = {f"t{i}": i for i in range(VOCAB - 3)}
    vocab["[PAD]"] = VOCAB - 3
    vocab["[EOS]"] = VOCAB - 2
    vocab["[UNK]"] = VOCAB - 1
    tok_model = models.WordLevel(vocab=vocab, unk_token="[UNK]")
    backend = Tokenizer(tok_model)
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    )
    return tok


def _tiny_model() -> LlamaForCausalLM:
    cfg = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=2,
        max_position_embeddings=64,
        pad_token_id=VOCAB - 3,
        eos_token_id=VOCAB - 2,
    )
    torch.manual_seed(0)
    m = LlamaForCausalLM(cfg)
    m.eval()
    return m


@pytest.fixture(scope="module")
def model_and_tok():
    return _tiny_model(), _tiny_tokenizer()


def _gen_engine(model, tok, **kw):
    return HFBatchedGenerateEngine(
        "unused", device="cpu", model=model, tokenizer=tok, **kw
    )


def test_batched_greedy_equals_per_row_greedy(model_and_tok):
    model, tok = model_and_tok
    prompts = ["t1 t2 t3", "t4 t5", "t6 t7 t8 t9", "t2 t3"]
    items = [GenerateItem(id=str(i), prompt=p) for i, p in enumerate(prompts)]

    eng = _gen_engine(model, tok, max_new_tokens=6)
    batched = {r.id: r.completion_token_ids for r in eng.generate(items, batch_size=4)}
    per_row = {}
    for it in items:
        r = eng.generate([it], batch_size=1)[0]
        per_row[r.id] = r.completion_token_ids

    assert batched == per_row, "batched greedy must equal per-row greedy"


def test_left_pad_completions_match_individual_runs(model_and_tok):
    model, tok = model_and_tok
    # Deliberately mixed lengths in one batch; left padding must not change the
    # generated text vs running each prompt alone.
    prompts = ["t1", "t2 t3 t4 t5 t6", "t7 t8"]
    items = [GenerateItem(id=str(i), prompt=p) for i, p in enumerate(prompts)]
    eng = _gen_engine(model, tok, max_new_tokens=5)

    batched = {r.id: r.completion_text for r in eng.generate(items, batch_size=3)}
    individual = {eng.generate([it], batch_size=1)[0].id: eng.generate([it], batch_size=1)[0].completion_text for it in items}
    assert batched == individual


def test_prompt_token_len_is_real_length(model_and_tok):
    model, tok = model_and_tok
    items = [GenerateItem(id="a", prompt="t1 t2 t3"), GenerateItem(id="b", prompt="t4")]
    eng = _gen_engine(model, tok, max_new_tokens=2)
    res = {r.id: r for r in eng.generate(items, batch_size=2)}
    # "t1 t2 t3" -> 3 real tokens; "t4" -> 1 real token (no BOS in this tokenizer).
    assert res["a"].prompt_token_len == 3
    assert res["b"].prompt_token_len == 1


def test_capture_positions_match_direct_forward(model_and_tok):
    model, tok = model_and_tok
    text = "t1 t2 t3 t4"
    item = CaptureItem(id="x", text=text, positions={"first": 0, "last": "last", "mid": 2})
    eng = HFBatchedCaptureEngine("unused", device="cpu", model=model, tokenizer=tok, layers="all")
    res = eng.capture([item], batch_size=1)[0]

    # Direct single-row forward as ground truth.
    enc = tok([text], return_tensors="pt")
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    real_len = int(enc["attention_mask"][0].sum())

    assert res.n_layers == len(hs)
    assert res.hidden_dim == HIDDEN
    assert res.positions == {"first": 0, "last": real_len - 1, "mid": 2}
    for layer in range(len(hs)):
        exp = hs[layer][0, real_len - 1]
        got = res.tensors[f"last__L{layer}"]
        assert torch.allclose(got, exp.to(torch.float32), atol=1e-5)


def test_capture_batched_matches_single_row(model_and_tok):
    model, tok = model_and_tok
    items = [
        CaptureItem(id="a", text="t1 t2 t3 t4 t5", positions={"last": "last"}),
        CaptureItem(id="b", text="t6 t7", positions={"last": "last"}),
    ]
    eng = HFBatchedCaptureEngine("unused", device="cpu", model=model, tokenizer=tok, layers="1,2")
    batched = {r.id: r for r in eng.capture(items, batch_size=2)}
    singles = {eng.capture([it], batch_size=1)[0].id: eng.capture([it], batch_size=1)[0] for it in items}
    for rid in ("a", "b"):
        for key in batched[rid].tensors:
            assert torch.allclose(batched[rid].tensors[key], singles[rid].tensors[key], atol=1e-5)


def _write_prompts(tmp_path, n=5):
    p = tmp_path / "prompts.jsonl"
    with open(p, "w") as f:
        for i in range(n):
            f.write(json.dumps({"id": f"row{i}", "prompt": f"t{i % 6} t{(i + 1) % 6}", "tag": i}) + "\n")
    return p


def _patch_engine_factory(monkeypatch, model, tok):
    """Make the runner build the shared in-memory model instead of downloading."""
    def _factory(name, *, model_name, **kw):  # noqa: ARG001
        kw.pop("dtype", None)
        if name == "hf-batched":
            return HFBatchedGenerateEngine("unused", device="cpu", model=model, tokenizer=tok, **kw)
        raise ValueError(name)
    monkeypatch.setattr(batch_runner, "get_generate_engine", _factory)


def test_generate_runner_passthrough_and_index(monkeypatch, tmp_path, model_and_tok):
    model, tok = model_and_tok
    _patch_engine_factory(monkeypatch, model, tok)
    prompts = _write_prompts(tmp_path, 5)
    out = tmp_path / "gen"
    summary = batch_runner.run_batch_generate(
        prompts_path=prompts, out_dir=out, model="m", max_new_tokens=3, batch_size=2,
        log=lambda m: None,
    )
    assert summary["newly_processed"] == 5
    rows = [json.loads(l) for l in (out / "completions.jsonl").read_text().splitlines()]
    assert len(rows) == 5
    ids = {r["id"] for r in rows}
    assert ids == {f"row{i}" for i in range(5)}
    # passthrough field preserved
    assert all("tag" in r for r in rows)
    assert all("completion_token_ids" in r and "finish_reason" in r for r in rows)


def test_generate_resume_matches_uninterrupted(monkeypatch, tmp_path, model_and_tok):
    model, tok = model_and_tok
    _patch_engine_factory(monkeypatch, model, tok)
    prompts = _write_prompts(tmp_path, 6)

    # Uninterrupted reference run.
    ref = tmp_path / "ref"
    batch_runner.run_batch_generate(
        prompts_path=prompts, out_dir=ref, model="m", max_new_tokens=3, batch_size=2,
        log=lambda m: None,
    )
    ref_rows = sorted(
        (json.loads(l) for l in (ref / "completions.jsonl").read_text().splitlines()),
        key=lambda r: r["id"],
    )

    # Interrupted run: process only the first 3 rows, then resume with all 6.
    part = tmp_path / "part"
    half = tmp_path / "prompts_half.jsonl"
    lines = prompts.read_text().splitlines()
    half.write_text("\n".join(lines[:3]) + "\n")
    batch_runner.run_batch_generate(
        prompts_path=half, out_dir=part, model="m", max_new_tokens=3, batch_size=2,
        log=lambda m: None,
    )
    # Resume with the full input and --resume.
    batch_runner.run_batch_generate(
        prompts_path=prompts, out_dir=part, model="m", max_new_tokens=3, batch_size=2,
        resume=True, log=lambda m: None,
    )
    part_rows = sorted(
        (json.loads(l) for l in (part / "completions.jsonl").read_text().splitlines()),
        key=lambda r: r["id"],
    )

    # Identical artifact set: same ids, same completions, no duplicates.
    assert len(part_rows) == 6
    assert [r["id"] for r in part_rows] == [r["id"] for r in ref_rows]
    for a, b in zip(part_rows, ref_rows):
        assert a["completion_token_ids"] == b["completion_token_ids"]


def test_capture_runner_persist_dtype_and_index(monkeypatch, tmp_path, model_and_tok):
    from safetensors.torch import load_file

    model, tok = model_and_tok

    def _cap_factory(name, *, model_name, **kw):  # noqa: ARG001
        kw.pop("dtype", None)
        return HFBatchedCaptureEngine("unused", device="cpu", model=model, tokenizer=tok, **kw)

    monkeypatch.setattr(batch_runner, "get_capture_engine", _cap_factory)

    rows_path = tmp_path / "rows.jsonl"
    with open(rows_path, "w") as f:
        for i in range(4):
            f.write(json.dumps({"id": f"c{i}", "text": f"t1 t2 t{i % 5}", "positions": {"last": "last"}, "meta": i}) + "\n")

    out = tmp_path / "cap"
    summary = batch_runner.run_batch_capture(
        rows_path=rows_path, out_dir=out, model="m", layers="all",
        batch_size=2, persist_dtype="bfloat16", log=lambda m: None,
    )
    assert summary["newly_processed"] == 4
    index = [json.loads(l) for l in (out / "capture.jsonl").read_text().splitlines()]
    assert len(index) == 4
    for row in index:
        assert row["file"].startswith("tensors/")
        assert "meta" in row  # passthrough
        tensors = load_file(out / row["file"])
        # persist dtype honored
        for v in tensors.values():
            assert v.dtype == torch.bfloat16


def test_capture_resume_no_duplicates(monkeypatch, tmp_path, model_and_tok):
    model, tok = model_and_tok

    def _cap_factory(name, *, model_name, **kw):  # noqa: ARG001
        kw.pop("dtype", None)
        return HFBatchedCaptureEngine("unused", device="cpu", model=model, tokenizer=tok, **kw)

    monkeypatch.setattr(batch_runner, "get_capture_engine", _cap_factory)

    def _write(path, n):
        with open(path, "w") as f:
            for i in range(n):
                f.write(json.dumps({"id": f"c{i}", "text": f"t1 t{i % 4}", "positions": {"last": "last"}}) + "\n")

    out = tmp_path / "cap"
    half = tmp_path / "half.jsonl"
    full = tmp_path / "full.jsonl"
    _write(half, 2)
    _write(full, 5)

    batch_runner.run_batch_capture(rows_path=half, out_dir=out, model="m", batch_size=2, log=lambda m: None)
    batch_runner.run_batch_capture(rows_path=full, out_dir=out, model="m", batch_size=2, resume=True, log=lambda m: None)

    index = [json.loads(l) for l in (out / "capture.jsonl").read_text().splitlines()]
    ids = [r["id"] for r in index]
    assert sorted(ids) == [f"c{i}" for i in range(5)]
    assert len(ids) == len(set(ids)), "resume must not duplicate rows"


def test_oom_auto_halve():
    """The OOM-halving loop halves the batch on a simulated CUDA OOM and
    eventually succeeds, without needing a GPU."""

    class _FakeOOM(RuntimeError):
        pass

    # Rename so _is_cuda_oom's message check matches ("out of memory").
    err = _FakeOOM("CUDA out of memory: tried to allocate")

    calls = []
    state = {"first": True}

    def _fn(chunk, bs):
        calls.append((len(chunk), bs))
        if state["first"] and bs > 1:
            state["first"] = False
            raise err
        return [("ok", c) for c in chunk]

    warned = []
    items = list(range(6))
    results = _run_with_oom_halving(
        _fn, items, batch_size=4, on_oom=lambda o, n: warned.append((o, n)), torch=torch
    )
    assert len(results) == 6
    assert warned and warned[0] == (4, 2)  # halved 4 -> 2
    # After halving, the first (retried) chunk ran at bs=2.
    assert any(bs == 2 for _, bs in calls)


def test_oom_reraises_at_batch_size_one():
    def _always_oom(chunk, bs):
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(OutOfMemoryError):
        _run_with_oom_halving(_always_oom, [1, 2], batch_size=1, on_oom=None, torch=torch)


def test_is_local_peft_adapter_dir(tmp_path):
    plain_dir = tmp_path / "plain"
    plain_dir.mkdir()
    (plain_dir / "config.json").write_text("{}")
    assert _is_local_peft_adapter_dir(str(plain_dir)) is False
    assert _is_local_peft_adapter_dir("some/hub/repo-id") is False  # not a local dir at all

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}")
    assert _is_local_peft_adapter_dir(str(adapter_dir)) is True


def test_model_bundle_loads_local_adapter_dir_with_trainable_token_deltas_applied(tmp_path):
    """Regression guard for the silently-dropped trainable-tokens bug.

    A bare ``AutoModelForCausalLM.from_pretrained`` on a local PEFT adapter
    directory loads whatever tensors line up with the base architecture's own
    state-dict keys and drops anything that doesn't -- including a
    ``trainable_token_indices`` embedding delta (transformers reports it as
    UNEXPECTED/MISSING and moves on rather than raising). ``_ModelBundle``
    must detect the adapter directory and load it through
    ``peft.AutoPeftModelForCausalLM`` instead, so every adapter component --
    standard LoRA deltas AND the trainable-token embedding delta -- is
    actually applied.
    """
    peft = pytest.importorskip("peft")
    pytest.importorskip("safetensors.torch")
    if "trainable_token_indices" not in peft.LoraConfig.__dataclass_fields__:
        pytest.skip("installed PEFT lacks selective token rows")

    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=6,
        n_embd=8,
        n_layer=1,
        n_head=1,
        n_positions=8,
        tie_word_embeddings=False,
    )
    base_model = GPT2LMHeadModel(config)
    base_model.eval()
    base_dir = tmp_path / "base"
    base_model.save_pretrained(base_dir)
    base_embed_before = base_model.get_input_embeddings().weight.detach().clone()

    lora_config = peft.LoraConfig(
        r=2,
        lora_alpha=2,
        target_modules=["c_attn"],
        trainable_token_indices={"transformer.wte": [4, 5]},
    )
    peft_model = peft.get_peft_model(base_model, lora_config)
    embed_wrapper = next(
        module
        for name, module in peft_model.named_modules()
        if name.endswith("transformer.wte") and hasattr(module, "token_adapter")
    )
    optimizer = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad], lr=0.5
    )
    loss = peft_model(input_ids=torch.tensor([[4, 5]]), labels=torch.tensor([[5, 4]])).loss
    loss.backward()
    optimizer.step()
    trained_delta = embed_wrapper.token_adapter.trainable_tokens_delta["default"].detach()
    assert trained_delta.abs().sum().item() > 0  # the gradient step actually moved it

    adapter_dir = tmp_path / "adapter"
    peft_model.peft_config["default"].base_model_name_or_path = str(base_dir)
    peft_model.save_pretrained(adapter_dir)
    assert _is_local_peft_adapter_dir(str(adapter_dir)) is True

    bundle = _ModelBundle(str(adapter_dir), device="cpu", tokenizer=_tiny_tokenizer())
    assert type(bundle.model).__module__.startswith("peft")
    assert any("token_adapter" in name for name, _ in bundle.model.named_modules())

    loaded_embed = bundle.model.get_input_embeddings().weight.detach()
    # Trained rows must have moved away from their pristine base value...
    assert not torch.allclose(loaded_embed[4], base_embed_before[4])
    assert not torch.allclose(loaded_embed[5], base_embed_before[5])
    # ...while a row outside trainable_token_indices is untouched.
    assert torch.equal(loaded_embed[0], base_embed_before[0])


def test_model_bundle_plain_local_dir_without_adapter_config_uses_bare_loader(tmp_path):
    """Non-adapter local paths (plain saved checkpoints, and hub ids) must
    keep loading through the original AutoModelForCausalLM path -- the new
    adapter-detection branch must not misfire on an ordinary model dir."""
    pytest.importorskip("safetensors.torch")
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(vocab_size=6, n_embd=8, n_layer=1, n_head=1, n_positions=8)
    model = GPT2LMHeadModel(config)
    model_dir = tmp_path / "plain_model"
    model.save_pretrained(model_dir)
    assert _is_local_peft_adapter_dir(str(model_dir)) is False

    bundle = _ModelBundle(str(model_dir), device="cpu", tokenizer=_tiny_tokenizer())
    assert type(bundle.model).__name__ == "GPT2LMHeadModel"


# --- Composite (vision-language) config fallback -----------------------------
#
# Reproduces the real reported failure: a checkpoint whose config nests the
# text fields (vocab_size, hidden_size, ...) under `text_config` rather than
# at the top level (e.g. Qwen/Qwen3.5-4B, architecture
# Qwen3_5ForConditionalGeneration). `AutoModelForCausalLM` resolves such a
# config to a *different*, text-only architecture class
# (Qwen3_5ForCausalLM) whose flat `config.vocab_size` access satisfies
# construction, but whose state-dict key namespace (`model.*`) does not
# match the checkpoint's (`model.language_model.*`, `model.visual.*`,
# `lm_head.*`) at all -- `from_pretrained` does not raise, it just loads
# with every tensor reported MISSING/UNEXPECTED, silently producing a
# freshly-initialized (garbage) model. `AutoModelForImageTextToText`
# resolves the same config to `Qwen3_5ForConditionalGeneration`, whose
# nested structure matches the checkpoint.


def _tiny_multimodal_config(hidden=HIDDEN, n_layers=N_LAYERS, vocab=VOCAB):
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5Config,
        Qwen3_5TextConfig,
        Qwen3_5VisionConfig,
    )

    text_cfg = Qwen3_5TextConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=n_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=64,
        pad_token_id=vocab - 3,
        eos_token_id=vocab - 2,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        full_attention_interval=2,
    )
    vision_cfg = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=8,
        num_heads=1,
        intermediate_size=16,
        patch_size=4,
        spatial_merge_size=1,
        temporal_patch_size=1,
        out_hidden_size=hidden,
        num_position_embeddings=16,
    )
    return Qwen3_5Config(text_config=text_cfg, vision_config=vision_cfg, tie_word_embeddings=False)


@pytest.fixture(scope="module")
def tiny_multimodal_checkpoint(tmp_path_factory):
    """A saved, on-disk tiny Qwen3_5ForConditionalGeneration checkpoint --
    randomly initialized, no downloads -- with the same composite-config
    shape as the real Qwen3.5 checkpoints."""
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5ForConditionalGeneration,
        )
    except ImportError:
        pytest.skip("installed transformers lacks the qwen3_5 model family")

    cfg = _tiny_multimodal_config()
    torch.manual_seed(0)
    model = Qwen3_5ForConditionalGeneration(cfg)
    model.eval()
    d = tmp_path_factory.mktemp("qwen3_5_tiny")
    model.save_pretrained(d)
    return d


def test_is_composite_text_config_generic_detection():
    """`_is_composite_text_config` must use transformers' own generic
    `get_text_config()` accessor -- not a per-architecture name check -- so
    it applies to any nested-config model, not just Qwen3.5."""
    from transformers import LlamaConfig
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5Config,
        Qwen3_5TextConfig,
    )

    llama_cfg = LlamaConfig(vocab_size=10, hidden_size=8, num_hidden_layers=1, num_attention_heads=1)
    assert _is_composite_text_config(llama_cfg) is False

    flat_text_cfg = Qwen3_5TextConfig(
        vocab_size=10, hidden_size=8, num_hidden_layers=1,
        num_attention_heads=1, num_key_value_heads=1, head_dim=8,
    )
    assert _is_composite_text_config(flat_text_cfg) is False

    composite_cfg = Qwen3_5Config(text_config=flat_text_cfg)
    assert _is_composite_text_config(composite_cfg) is True


def test_model_bundle_loads_composite_multimodal_config_via_fallback(tiny_multimodal_checkpoint):
    """The real regression guard: loading the composite checkpoint through
    `_ModelBundle` (i.e. through `_load_causal_lm_with_fallback`) must
    resolve to the correct nested-aware architecture class AND actually
    load the checkpoint's real weights (not silently reinitialize them).

    Weights are checked against a reference load via
    `AutoModelForImageTextToText` directly (the known-correct loader for
    this config) rather than a hardcoded state-dict key name, since
    composite models' saved key prefixes are an internal implementation
    detail this test should not need to hardcode.
    """
    from transformers import AutoModelForImageTextToText

    tok = _tiny_tokenizer()
    bundle = _ModelBundle(str(tiny_multimodal_checkpoint), device="cpu", tokenizer=tok)
    assert type(bundle.model).__name__ == "Qwen3_5ForConditionalGeneration"

    reference = AutoModelForImageTextToText.from_pretrained(str(tiny_multimodal_checkpoint))
    reference.eval()

    loaded_embed = bundle.model.get_input_embeddings().weight.detach()
    reference_embed = reference.get_input_embeddings().weight.detach()
    assert torch.allclose(loaded_embed, reference_embed), (
        "loaded embedding weights must match a reference "
        "AutoModelForImageTextToText load -- a mismatch means the fallback "
        "resolved the wrong (flat-config) architecture class and the real "
        "weights were silently dropped"
    )

    # And a forward pass must actually agree numerically, not just at the
    # embedding table -- confirms every layer's weights loaded correctly,
    # not only the embedding.
    enc = tok(["t1 t2 t3"], return_tensors="pt")
    with torch.no_grad():
        bundle_out = bundle.model(**enc, use_cache=False).logits
        reference_out = reference(**enc, use_cache=False).logits
    assert torch.allclose(bundle_out, reference_out, atol=1e-5)


def test_load_causal_lm_with_fallback_direct(tiny_multimodal_checkpoint):
    """Unit-level check of the loader helper itself, independent of
    `_ModelBundle`, so a future refactor of the bundle can't hide a
    regression here."""
    model = _load_causal_lm_with_fallback(
        str(tiny_multimodal_checkpoint),
        revision=None,
        token=None,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    assert type(model).__name__ == "Qwen3_5ForConditionalGeneration"


def test_generate_engine_through_multimodal_wrapper(tiny_multimodal_checkpoint):
    """Generation must work end-to-end through the ConditionalGeneration
    wrapper resolved by the fallback (get_input_embeddings/generate must be
    correctly delegated through the nested model.language_model)."""
    tok = _tiny_tokenizer()
    eng = HFBatchedGenerateEngine(
        str(tiny_multimodal_checkpoint), device="cpu", tokenizer=tok, max_new_tokens=3
    )
    items = [GenerateItem(id="a", prompt="t1 t2 t3"), GenerateItem(id="b", prompt="t4 t5")]
    res = eng.generate(items, batch_size=2)
    assert len(res) == 2
    for r in res:
        assert isinstance(r.completion_token_ids, list)
        assert r.finish_reason in ("length", "eos", "stop")


def test_capture_engine_hidden_states_through_multimodal_wrapper_match_text_indexing(
    tiny_multimodal_checkpoint,
):
    """Hidden-state layer indexing through the ConditionalGeneration wrapper
    must match plain text-model semantics exactly: index 0 = embeddings,
    1..N = per-decoder-layer outputs, count == num_hidden_layers + 1, with
    numerically identical values to a direct forward call. This is what
    lets the engine's generic `len(hidden_states)` / index-based layer
    selection work unchanged for a multimodal wrapper -- no per-wrapper
    special-casing needed for hidden-state extraction."""
    tok = _tiny_tokenizer()
    eng = HFBatchedCaptureEngine(
        str(tiny_multimodal_checkpoint), device="cpu", tokenizer=tok, layers="all"
    )
    item = CaptureItem(id="x", text="t1 t2 t3 t4", positions={"last": "last"})
    res = eng.capture([item], batch_size=1)[0]

    assert res.n_layers == N_LAYERS + 1  # embeddings + N_LAYERS decoder layers
    assert res.hidden_dim == HIDDEN

    enc = tok(["t1 t2 t3 t4"], return_tensors="pt")
    with torch.no_grad():
        out = eng.bundle.model(**enc, output_hidden_states=True, use_cache=False)
    assert len(out.hidden_states) == N_LAYERS + 1

    last_idx = int(enc["attention_mask"][0].sum()) - 1
    for layer in range(len(out.hidden_states)):
        exp = out.hidden_states[layer][0, last_idx].to(torch.float32)
        got = res.tensors[f"last__L{layer}"]
        assert torch.allclose(got, exp, atol=1e-5), f"layer {layer} mismatch"
