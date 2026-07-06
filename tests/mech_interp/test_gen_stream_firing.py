"""CPU end-to-end proof that the gen_stream decode hook fires during a real
plain-HF model.generate() call.

Background (AK doc section 8): Amendment AK Stage 2 ran a bespoke Unsloth
harness (FastLanguageModel.for_inference) and produced 100% byte-identical
generations across all seven alphas in its gen_stream arm. The diagnosis was
that Unsloth's optimized cached generate() decode path never routes through
the hooked decoder module's Python forward(), so the per-decode-step
gen_stream edit silently never fired.

The tuner's own path is different: MechInterp/cli.py loads a plain HF
AutoModelForCausalLM (no Unsloth), and register_forward_hook on a decoder
layer is expected to fire on every forward call of model.generate(), including
each single-token decode step. That expectation was previously unproven end
to end, because the existing forward-pass smoke in _run_smoke only calls
model(**enc) once and never exercises a real generate() decode loop.

This test builds a tiny, randomly initialized, plain-HF causal LM entirely
offline (no network, no from_pretrained of a hub model), registers a
GenerationInterventionController exactly as run_steer does, and runs a real
model.generate() decode loop. It asserts:

  (a) FIRING     the hook is active on more than one decode step in
                  gen_stream mode, not only at prefill.
  (b) EFFECT      the generated token ids under gen_stream at a large strength
                  differ from the ids produced with the controller "off".
  (c) CONTRAST    in anchor mode the hook fires only at the prefill step, not
                  on any decode step, so the counter-based prefill/decode
                  gating is doing real work and gen_stream genuinely differs
                  from anchor.

If (a) or (b) ever fail on this tiny plain-HF model, that is the AK Stage 2
regression showing up somewhere in the tuner's own stack, not the bespoke
Unsloth harness -- treat it as a stop-everything signal for any amendment that
depends on gen_stream firing during generate().
"""

import torch
from transformers import AutoModelForCausalLM, GPT2Config

from MechInterp.cli import gen_stream_fires
from MechInterp.intervention import (
    GenerationInterventionController,
    InterventionHook,
    get_decoder_layer,
)

# Smallest architecture get_decoder_layer already handles ("transformer.h"):
# GPT2LMHeadModel built from a from-scratch config, never downloaded.
_VOCAB_SIZE = 64
_HIDDEN_DIM = 32
_PROMPT_LEN = 5
_LAYER_IDX = 0
_LARGE_STRENGTH = 1.0e4
_MAX_NEW_TOKENS = 10


def _build_tiny_model():
    torch.manual_seed(0)
    config = GPT2Config(
        n_layer=2,
        n_embd=_HIDDEN_DIM,
        n_head=2,
        vocab_size=_VOCAB_SIZE,
        n_positions=64,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    return model


def _unit_direction(hidden_dim: int) -> torch.Tensor:
    d = torch.zeros(hidden_dim, dtype=torch.float32)
    d[0] = 1.0
    return d


class _SpyController(GenerationInterventionController):
    """A GenerationInterventionController that records per-call firing state.

    Records, for every forward call the controller receives, the decode-vs-
    prefill sequence length and whether the wrapped hook ended up active for
    that call. Used only to make the firing assertions above legible; it does
    not change the controller's gating behavior, it just observes it.
    """

    def __init__(self, hook: InterventionHook):
        super().__init__(hook)
        self.calls: list[dict] = []

    def __call__(self, module, inputs, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        seq_len = hidden.shape[1]
        result = super().__call__(module, inputs, output)
        self.calls.append(
            {"nth_call": self._nth_call, "seq_len": seq_len, "active": bool(self.hook.active), "mode": self.mode}
        )
        return result


def _fixture():
    """Build the tiny model, register the spy controller, and encode a prompt.

    Returns (model, controller, handle, enc). The caller is responsible for
    removing handle once done.
    """
    model = _build_tiny_model()
    direction = _unit_direction(_HIDDEN_DIM)
    hook = InterventionHook(law="additive", direction=direction, strength=0.0, position="anchor_onward")
    controller = _SpyController(hook)
    layer_module = get_decoder_layer(model, _LAYER_IDX)
    handle = layer_module.register_forward_hook(controller)

    torch.manual_seed(1)
    input_ids = torch.randint(0, _VOCAB_SIZE, (1, _PROMPT_LEN))
    attention_mask = torch.ones_like(input_ids)
    enc = {"input_ids": input_ids, "attention_mask": attention_mask}
    return model, controller, handle, enc


def _run(model, controller, enc, mode: str):
    """Run one real generate() decode loop under mode, mirroring _run_one_pass's
    begin_pass / generate / reset sequence. min_new_tokens pins the decode loop
    to a fixed length so the firing count is deterministic regardless of what
    the tiny random model happens to decode."""
    controller.calls.clear()
    controller.begin_pass(mode, _LARGE_STRENGTH, attention_mask=enc["attention_mask"])
    with torch.no_grad():
        gen = model.generate(
            **enc,
            max_new_tokens=_MAX_NEW_TOKENS,
            min_new_tokens=_MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=True,
        )
    controller.reset()
    return gen.sequences[0], list(controller.calls)


def _run_all_modes():
    model, controller, handle, enc = _fixture()
    try:
        return {
            mode: _run(model, controller, enc, mode)
            for mode in ("gen_stream", "off", "anchor")
        }
    finally:
        handle.remove()


def test_gen_stream_fires_on_more_than_one_decode_step():
    # (a) FIRING: the whole point of this guard. If this is 0, the decode hook
    # never fired -- the exact AK Stage 2 regression -- and the test must fail.
    results = _run_all_modes()
    _, calls_gen_stream = results["gen_stream"]
    decode_active = [c for c in calls_gen_stream if c["seq_len"] == 1 and c["active"]]
    assert len(decode_active) > 1, (
        "gen_stream hook fired on <= 1 decode step during a real generate() "
        "call on plain HF; this is the AK section 8 regression (decode hook "
        "silently not firing), not merely a weak edit"
    )


def test_gen_stream_output_differs_from_off_at_large_strength():
    # (b) EFFECT: byte-identical output under a large edit means the edit
    # never actually reached the generation loop.
    results = _run_all_modes()
    seq_gen_stream, _ = results["gen_stream"]
    seq_off, _ = results["off"]
    assert not torch.equal(seq_gen_stream, seq_off), (
        "gen_stream generation was byte-identical to off-mode generation at a "
        "strength of 1e4; the decode edit did not change the decoded tokens"
    )


def test_anchor_fires_only_at_prefill_not_at_any_decode_step():
    # (c) CONTRAST: anchor edits the prefill step only, and the edit rides the
    # KV cache into later tokens rather than re-firing on every decode step.
    # This proves the counter-based prefill/decode gating is doing real work,
    # and that gen_stream is not just always-on regardless of mode.
    results = _run_all_modes()
    _, calls_anchor = results["anchor"]
    active_calls = [c for c in calls_anchor if c["active"]]
    assert len(active_calls) == 1
    assert active_calls[0]["nth_call"] == 1
    assert active_calls[0]["seq_len"] > 1  # the prefill call, not a decode step

    decode_active = [c for c in calls_anchor if c["seq_len"] == 1 and c["active"]]
    assert decode_active == []


def test_off_mode_never_activates_the_hook():
    # Sanity companion to (b): off mode must never mark the hook active on any
    # call, prefill or decode.
    results = _run_all_modes()
    _, calls_off = results["off"]
    assert all(not c["active"] for c in calls_off)


# --------------------------------------------------------------------------
# gen_stream_fires: the run_steer fail-closed smoke guard's compare helper.
# --------------------------------------------------------------------------
#
# run_steer cannot be driven directly in a CPU unit test (it loads a real
# model by name and requires a GPU acknowledgement flag), so these tests
# exercise the factored-out compare-generate-vs-off helper on the same tiny
# plain-HF model, with a real GenerationInterventionController registered
# exactly as run_steer registers it.


def _plain_fixture():
    """Same tiny model/prompt as _fixture, but with the real (non-spy)
    GenerationInterventionController run_steer actually registers."""
    model = _build_tiny_model()
    direction = _unit_direction(_HIDDEN_DIM)
    hook = InterventionHook(law="additive", direction=direction, strength=0.0, position="anchor_onward")
    controller = GenerationInterventionController(hook)
    layer_module = get_decoder_layer(model, _LAYER_IDX)
    handle = layer_module.register_forward_hook(controller)

    torch.manual_seed(1)
    input_ids = torch.randint(0, _VOCAB_SIZE, (1, _PROMPT_LEN))
    attention_mask = torch.ones_like(input_ids)
    enc = {"input_ids": input_ids, "attention_mask": attention_mask}
    return model, controller, handle, enc


def test_gen_stream_fires_helper_passes_when_hook_is_registered():
    model, controller, handle, enc = _plain_fixture()
    try:
        fired = gen_stream_fires(model, controller, enc, strength=_LARGE_STRENGTH, max_new_tokens=8)
    finally:
        handle.remove()
    assert fired is True


def test_gen_stream_fires_helper_fails_closed_when_hook_never_fires():
    # Simulate the AK-style regression at the guard level: a controller that
    # is never actually registered as a forward hook cannot edit anything, so
    # gen_stream and off generate() calls are byte-identical and the guard
    # must report that as a firing failure (return False), not silently pass.
    model = _build_tiny_model()
    direction = _unit_direction(_HIDDEN_DIM)
    hook = InterventionHook(law="additive", direction=direction, strength=0.0, position="anchor_onward")
    controller = GenerationInterventionController(hook)  # deliberately not registered

    torch.manual_seed(1)
    input_ids = torch.randint(0, _VOCAB_SIZE, (1, _PROMPT_LEN))
    attention_mask = torch.ones_like(input_ids)
    enc = {"input_ids": input_ids, "attention_mask": attention_mask}

    fired = gen_stream_fires(model, controller, enc, strength=_LARGE_STRENGTH, max_new_tokens=8)
    assert fired is False
