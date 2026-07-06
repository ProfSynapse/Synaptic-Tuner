"""CPU correctness guard for batched steer generation.

_run_batch (MechInterp/cli.py) exists so the steer cell's arm loop can decode
several rows per model.generate() call instead of one row at a time. The
speedup is worthless if it is not provably identical to the historical
one-row-at-a-time path (_run_one_pass): a left-padding or position bug would
silently corrupt every steered generation without ever raising an error.

This module builds a tiny, randomly initialized, plain-HF GPT-2 with a small
from-scratch word-level tokenizer entirely offline (no network, no hub
download), then drives the SAME rows through _run_one_pass (looped, one row
per generate() call -- batch_size=1 semantics) and through _run_batch
(chunked, several rows per generate() call), and asserts:

  (a) IDENTITY   the raw generated token ids match row-for-row, exactly, for
                 every row in the batch (greedy decoding is deterministic --
                 any mismatch here IS a padding/position/masking bug, not
                 numerical noise to tolerate).
  (b) EQUIVALENCE the hooked layer's hidden state at the edited column (the
                 prefill anchor column for "anchor" mode, the first decode
                 column for "gen_stream" mode) agrees between the batched and
                 unbatched runs within intervention/equivalence.py's
                 tolerance, confirming the edit landed on the same physical
                 token for every row despite the batch's left padding.

The row fixture deliberately mixes: different prompt lengths (so left padding
is non-trivial), nontrivial per-row gains (positive, negative, fractional),
a genuinely inactive row (never selected for this arm -- a true no-op), and
an active row whose resolved gain is exactly 0.0 (the erase_write "ablate"
case -- force_active must still write the zero setpoint for that row without
touching the neighboring true no-op row). A 5-row fixture run through
_run_batch in chunks of 4 then 1 also exercises the "remainder batch smaller
than batch_size" case the real arm loop hits when the pending count is not a
multiple of batch_size.

If any identity assertion below ever fails, that is the batching itself being
wrong -- fix the padding/position/masking in cli.py or hooks.py. Loosening the
token-id assertion or the equivalence tolerance to make this pass would defeat
the entire point of this test.
"""

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, GPT2Config, PreTrainedTokenizerFast

from MechInterp.cli import _run_batch, _run_one_pass
from MechInterp.config import GenerationContract
from MechInterp.intervention import (
    GenerationInterventionController,
    InterventionHook,
    equivalence_ok,
    get_decoder_layer,
)

_VOCAB_WORDS = 48
_HIDDEN_DIM = 32
_LAYER_IDX = 0
_MAX_NEW_TOKENS = 6

# Distinct prompt lengths so left padding is non-trivial (max length 6, so
# padding amounts vary 3/1/4/0/2 across the batch).
_PROMPTS = [
    "w1 w2 w3",
    "w4 w5 w6 w7 w8",
    "w9 w10",
    "w11 w12 w13 w14 w15 w16",
    "w17 w18 w19 w20",
]
# Nontrivial per-row gains: positive, negative, fractional, and the two rows
# that distinguish "true no-op" (inactive) from "ablate" (active at gain 0).
_STRENGTHS = [2.0, -1.5, 0.0, 3.25, 0.0]
_ACTIVES = [True, True, False, True, True]


def _build_tiny_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {f"w{i}": i for i in range(_VOCAB_WORDS)}
    vocab["<pad>"] = _VOCAB_WORDS
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="w0"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>")


def _build_tiny_model():
    torch.manual_seed(0)
    config = GPT2Config(
        n_layer=2,
        n_embd=_HIDDEN_DIM,
        n_head=2,
        vocab_size=_VOCAB_WORDS + 1,
        n_positions=64,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    return model


def _unit_direction(hidden_dim: int) -> torch.Tensor:
    torch.manual_seed(7)
    d = torch.randn(hidden_dim, dtype=torch.float32)
    return d / d.norm()


def _make_rows() -> list[dict]:
    rows = []
    for i, (prompt, strength, active) in enumerate(zip(_PROMPTS, _STRENGTHS, _ACTIVES)):
        rows.append(
            {
                "row_key": f"r{i}",
                "_prompt": prompt,
                "_strength": strength,
                "_active": active,
            }
        )
    return rows


def _render(row: dict) -> str:
    return row["_prompt"]


def _build_controller(law: str, sigma: float = 2.0) -> GenerationInterventionController:
    direction = _unit_direction(_HIDDEN_DIM)
    hook = InterventionHook(law=law, direction=direction, strength=0.0, sigma=sigma, position="anchor")
    return GenerationInterventionController(hook)


class _CaptureHook:
    """A second forward hook, registered after the controller on the same
    layer, that records the (already-edited, if the controller edited it)
    hidden state for every forward call. Used only to extract the hooked
    layer's hidden state at a known column for the equivalence check; it does
    not alter the output."""

    def __init__(self):
        self.calls: list[torch.Tensor] = []

    def __call__(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        self.calls.append(hidden.detach().clone())
        return output


class _GenerateSpy:
    """Wraps model.generate to capture the raw GenerateDecoderOnlyOutput of
    the most recent call, so the test can read exact token ids out of a
    call to the real _run_one_pass / _run_batch without those functions
    needing to expose anything beyond their normal per-row record dict."""

    def __init__(self, model):
        self._model = model
        self._original = model.generate
        self.last = None

    def __enter__(self):
        def _spy(*args, **kwargs):
            out = self._original(*args, **kwargs)
            self.last = out
            return out

        self._model.generate = _spy
        return self

    def __exit__(self, *exc):
        self._model.generate = self._original


def _continuation_ids(sequences_row: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    """The generated continuation ids for one row of a generate() output.

    Fixed-length generation (see module docstring: the tiny vocab makes the
    model's default eos_token_id unreachable, so every row runs the full
    max_new_tokens without early stopping) means the prompt/continuation
    boundary is always exactly seq_len - max_new_tokens, whether or not that
    row's prompt was left-padded.
    """
    prompt_len = sequences_row.shape[0] - max_new_tokens
    return sequences_row[prompt_len:].clone()


def _run_unbatched_all(model, tokenizer, controller, capture, rows, generation, generation_mode):
    """Loop _run_one_pass over every row (the batch_size=1 code path), and
    for each row: capture its generated continuation ids, its record dict,
    and the hooked layer's hidden state at the prefill anchor column and at
    the first decode column."""
    records, ids_by_row, prefill_hidden, decode1_hidden = [], [], [], []
    with _GenerateSpy(model) as spy:
        for row in rows:
            capture.calls.clear()
            rec = _run_one_pass(model, tokenizer, controller, row, generation, generation_mode, _render)
            records.append(rec)
            seq = spy.last.sequences[0]
            ids_by_row.append(_continuation_ids(seq, generation.max_new_tokens))
            prompt_len = seq.shape[0] - generation.max_new_tokens
            prefill_hidden.append(capture.calls[0][0, prompt_len - 1, :].clone())
            if len(capture.calls) > 1:
                decode1_hidden.append(capture.calls[1][0, 0, :].clone())
    return records, ids_by_row, prefill_hidden, decode1_hidden


def _run_batched_all(model, tokenizer, controller, capture, rows, generation, generation_mode, batch_size):
    """Chunk rows into batch_size pieces (mirroring run_steer's arm loop) and
    run each chunk through _run_batch, collecting the same per-row artifacts
    as _run_unbatched_all so the two are directly comparable."""
    records, ids_by_row, prefill_hidden, decode1_hidden = [], [], [], []
    with _GenerateSpy(model) as spy:
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            capture.calls.clear()
            recs = _run_batch(model, tokenizer, controller, chunk, generation, generation_mode, _render)
            records.extend(recs)
            seqs = spy.last.sequences
            padded_prompt_len = seqs.shape[1] - generation.max_new_tokens
            for i in range(len(chunk)):
                ids_by_row.append(_continuation_ids(seqs[i], generation.max_new_tokens))
                prefill_hidden.append(capture.calls[0][i, padded_prompt_len - 1, :].clone())
                if len(capture.calls) > 1:
                    decode1_hidden.append(capture.calls[1][i, 0, :].clone())
    return records, ids_by_row, prefill_hidden, decode1_hidden


def _assert_ids_identical(batched_ids, unbatched_ids):
    assert len(batched_ids) == len(unbatched_ids)
    for i, (b, u) in enumerate(zip(batched_ids, unbatched_ids)):
        assert torch.equal(b, u), (
            f"row {i}: batched and unbatched generated token ids diverge "
            f"(batched={b.tolist()}, unbatched={u.tolist()}); this is a "
            "left-padding or position bug in _run_batch, not numerical noise"
        )


def _assert_hidden_equivalent(batched_hidden, unbatched_hidden, label):
    if not batched_hidden:
        return
    batched_stack = torch.stack(batched_hidden, dim=0)
    unbatched_stack = torch.stack(unbatched_hidden, dim=0)
    result = equivalence_ok(batched_stack, unbatched_stack)
    assert result["passed"], f"{label} hidden-state equivalence failed: {result}"


def _run_equivalence_case(law: str, generation_mode: str, batch_size: int):
    model = _build_tiny_model()
    tokenizer = _build_tiny_tokenizer()
    rows = _make_rows()
    generation = GenerationContract(max_new_tokens=_MAX_NEW_TOKENS, do_sample=False)

    unbatched_controller = _build_controller(law)
    unbatched_capture = _CaptureHook()
    layer = get_decoder_layer(model, _LAYER_IDX)
    h1 = layer.register_forward_hook(unbatched_controller)
    h2 = layer.register_forward_hook(unbatched_capture)
    try:
        unbatched_records, unbatched_ids, unbatched_prefill, unbatched_decode1 = _run_unbatched_all(
            model, tokenizer, unbatched_controller, unbatched_capture, rows, generation, generation_mode
        )
    finally:
        h1.remove()
        h2.remove()

    batched_controller = _build_controller(law)
    batched_capture = _CaptureHook()
    original_padding_side = tokenizer.padding_side
    h1 = layer.register_forward_hook(batched_controller)
    h2 = layer.register_forward_hook(batched_capture)
    try:
        batched_records, batched_ids, batched_prefill, batched_decode1 = _run_batched_all(
            model, tokenizer, batched_controller, batched_capture, rows, generation, generation_mode, batch_size
        )
    finally:
        h1.remove()
        h2.remove()

    assert tokenizer.padding_side == original_padding_side, (
        "_run_batch must restore the tokenizer's original padding_side"
    )

    return {
        "unbatched": (unbatched_records, unbatched_ids, unbatched_prefill, unbatched_decode1),
        "batched": (batched_records, batched_ids, batched_prefill, batched_decode1),
    }


# --------------------------------------------------------------------------
# The non-negotiable correctness gate: batched == unbatched, exactly.
# --------------------------------------------------------------------------


def test_batched_matches_unbatched_anchor_erase_write():
    result = _run_equivalence_case(law="erase_write", generation_mode="anchor", batch_size=4)
    _, unbatched_ids, unbatched_prefill, _ = result["unbatched"]
    _, batched_ids, batched_prefill, _ = result["batched"]
    _assert_ids_identical(batched_ids, unbatched_ids)
    _assert_hidden_equivalent(batched_prefill, unbatched_prefill, "anchor prefill")


def test_batched_matches_unbatched_gen_stream_erase_write():
    result = _run_equivalence_case(law="erase_write", generation_mode="gen_stream", batch_size=4)
    _, unbatched_ids, _, unbatched_decode1 = result["unbatched"]
    _, batched_ids, _, batched_decode1 = result["batched"]
    _assert_ids_identical(batched_ids, unbatched_ids)
    _assert_hidden_equivalent(batched_decode1, unbatched_decode1, "gen_stream first decode step")


def test_batched_matches_unbatched_anchor_additive():
    # Regression coverage for the OTHER law: additive's shared-column path
    # never consults active_override (a zero alpha is already a no-op via
    # the multiply), so this exercises a structurally different code path
    # than erase_write's active-mask branch.
    result = _run_equivalence_case(law="additive", generation_mode="anchor", batch_size=4)
    _, unbatched_ids, unbatched_prefill, _ = result["unbatched"]
    _, batched_ids, batched_prefill, _ = result["batched"]
    _assert_ids_identical(batched_ids, unbatched_ids)
    _assert_hidden_equivalent(batched_prefill, unbatched_prefill, "anchor prefill (additive)")


def test_batched_matches_unbatched_gen_stream_additive():
    result = _run_equivalence_case(law="additive", generation_mode="gen_stream", batch_size=4)
    _, unbatched_ids, _, unbatched_decode1 = result["unbatched"]
    _, batched_ids, _, batched_decode1 = result["batched"]
    _assert_ids_identical(batched_ids, unbatched_ids)
    _assert_hidden_equivalent(batched_decode1, unbatched_decode1, "gen_stream first decode step (additive)")


def test_remainder_batch_smaller_than_batch_size_still_matches():
    # 5 rows chunked at batch_size=4 -> chunks of 4 then 1, exactly what
    # run_steer's arm loop produces when the pending count is not a multiple
    # of batch_size.
    result = _run_equivalence_case(law="erase_write", generation_mode="anchor", batch_size=4)
    batched_records = result["batched"][0]
    assert len(batched_records) == len(_PROMPTS) == 5
    _assert_ids_identical(result["batched"][1], result["unbatched"][1])


# --------------------------------------------------------------------------
# Record-shape / degeneracy checks.
# --------------------------------------------------------------------------


def test_batch_size_one_record_matches_run_one_pass_exactly():
    model = _build_tiny_model()
    tokenizer = _build_tiny_tokenizer()
    row = _make_rows()[0]
    generation = GenerationContract(max_new_tokens=_MAX_NEW_TOKENS, do_sample=False)

    controller_a = _build_controller("erase_write")
    layer = get_decoder_layer(model, _LAYER_IDX)
    handle = layer.register_forward_hook(controller_a)
    try:
        rec_unbatched = _run_one_pass(model, tokenizer, controller_a, row, generation, "anchor", _render)
    finally:
        handle.remove()

    controller_b = _build_controller("erase_write")
    handle = layer.register_forward_hook(controller_b)
    try:
        [rec_batched] = _run_batch(model, tokenizer, controller_b, [row], generation, "anchor", _render)
    finally:
        handle.remove()

    assert rec_batched == rec_unbatched


def test_mixed_active_inactive_rows_are_both_correct_within_one_batch():
    # row index 2 is a true no-op (never selected: strength 0, active False);
    # row index 4 is the erase_write ablate case (selected, resolved gain
    # exactly 0.0, active True -- force_active must still write the zero
    # setpoint). Confirms both survive being edited in the SAME batched pass.
    result = _run_equivalence_case(law="erase_write", generation_mode="anchor", batch_size=5)
    unbatched_records = result["unbatched"][0]
    batched_records = result["batched"][0]
    assert unbatched_records[2]["active"] is False
    assert unbatched_records[4]["active"] is True
    assert unbatched_records[2]["strength"] == 0.0
    assert unbatched_records[4]["strength"] == 0.0
    _assert_ids_identical(result["batched"][1], result["unbatched"][1])
    # And the two zero-strength rows' own records agree row-for-row too.
    assert batched_records[2]["answer_text"] == unbatched_records[2]["answer_text"]
    assert batched_records[4]["answer_text"] == unbatched_records[4]["answer_text"]


def test_batch_size_default_is_one():
    from MechInterp.config import ExecutionConfig

    cfg = ExecutionConfig(output_path="out.jsonl")
    assert cfg.batch_size == 1
