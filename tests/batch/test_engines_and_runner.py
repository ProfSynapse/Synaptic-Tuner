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
    _run_with_oom_halving,
)
from tuner.batch import runner as batch_runner  # noqa: E402
from tuner.batch.persistence import ConfigMismatchError  # noqa: E402


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


def test_generate_resume_requires_identical_input(monkeypatch, tmp_path, model_and_tok):
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
    # Expanding or changing the input is a different config and must not mix.
    with pytest.raises(ConfigMismatchError):
        batch_runner.run_batch_generate(
            prompts_path=prompts, out_dir=part, model="m", max_new_tokens=3,
            batch_size=2, resume=True, log=lambda m: None,
        )
    # The exact same input resumes safely as a no-op.
    batch_runner.run_batch_generate(
        prompts_path=half, out_dir=part, model="m", max_new_tokens=3,
        batch_size=2, resume=True, log=lambda m: None,
    )
    part_rows = sorted(
        (json.loads(l) for l in (part / "completions.jsonl").read_text().splitlines()),
        key=lambda r: r["id"],
    )

    # Identical artifact set: same ids, same completions, no duplicates.
    assert len(part_rows) == 3
    assert [r["id"] for r in part_rows] == [r["id"] for r in ref_rows[:3]]
    for a, b in zip(part_rows, ref_rows[:3]):
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
