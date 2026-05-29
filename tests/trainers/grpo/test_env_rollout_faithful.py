"""Tests for token-faithful (POLAR-style) multi-turn env-GRPO rollout assembly.

These cover the pure sequence-assembly helpers, the capability gate that decides
whether to emit ``env_mask``, and the rollout_func output contract. They use only
crafted token lists and stubs — no TRL, model, or environment runtime required.
"""

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Trainers" / "grpo" / "src"))

import env_rollout
from env_rollout import (
    EpisodeRolloutResult,
    _align_logprobs,
    _assemble_faithful_sequence,
    _assemble_flat_sequence,
    _resolve_faithful_mode,
    build_rollout_func,
)


# ---------------------------------------------------------------------------
# Crafted multi-turn fixture
# ---------------------------------------------------------------------------
# Turn 0: prompt [1,2,3], assistant [10,11]
# Turn 1: prompt = prompt0 + comp0 + ext1([20,21,22]), assistant [12,13]
# Turn 2: prompt = prompt1 + comp1 + ext2([30]),        assistant [14]
THREE_TURNS = [
    ([1, 2, 3], [10, 11], [-0.1, -0.2]),
    ([1, 2, 3, 10, 11, 20, 21, 22], [12, 13], [-0.3, -0.4]),
    ([1, 2, 3, 10, 11, 20, 21, 22, 12, 13, 30], [14], [-0.5]),
]
SINGLE_TURN = [([1, 2, 3], [10, 11, 12], [-0.1, -0.2, -0.3])]


# ---------------------------------------------------------------------------
# _align_logprobs
# ---------------------------------------------------------------------------

def test_align_logprobs_exact():
    assert _align_logprobs([-0.1, -0.2], 2) == [-0.1, -0.2]


def test_align_logprobs_pads_short():
    assert _align_logprobs([-0.1], 3) == [-0.1, 0.0, 0.0]


def test_align_logprobs_truncates_long():
    assert _align_logprobs([-0.1, -0.2, -0.3], 2) == [-0.1, -0.2]


# ---------------------------------------------------------------------------
# Single-turn parity: faithful and flat must agree exactly
# ---------------------------------------------------------------------------

def test_single_turn_parity():
    base_f, comp_f, mask_f, lp_f = _assemble_faithful_sequence(SINGLE_TURN)
    base_d, comp_d, lp_d = _assemble_flat_sequence(SINGLE_TURN)

    assert base_f == base_d == [1, 2, 3]
    assert comp_f == comp_d == [10, 11, 12]
    assert lp_f == lp_d == [-0.1, -0.2, -0.3]
    # env_mask is all-ones for a single assistant turn (no external context)
    assert mask_f == [1, 1, 1]


# ---------------------------------------------------------------------------
# Multi-turn faithful assembly
# ---------------------------------------------------------------------------

def test_multi_turn_faithful_sequence():
    base, completion, env_mask, logprobs = _assemble_faithful_sequence(THREE_TURNS)

    assert base == [1, 2, 3]
    # assistant0 ++ ext1 ++ assistant1 ++ ext2 ++ assistant2
    assert completion == [10, 11, 20, 21, 22, 12, 13, 30, 14]
    assert env_mask == [1, 1, 0, 0, 0, 1, 1, 0, 1]
    assert logprobs == [-0.1, -0.2, 0.0, 0.0, 0.0, -0.3, -0.4, 0.0, -0.5]


def test_multi_turn_lengths_aligned():
    _base, completion, env_mask, logprobs = _assemble_faithful_sequence(THREE_TURNS)
    assert len(completion) == len(env_mask) == len(logprobs)


def test_multi_turn_mask_covers_only_assistant_tokens():
    _base, completion, env_mask, _logprobs = _assemble_faithful_sequence(THREE_TURNS)
    trained = [tok for tok, m in zip(completion, env_mask) if m == 1]
    # exactly the sampled assistant tokens across all turns, in order
    assert trained == [10, 11, 12, 13, 14]


def test_multi_turn_flat_drops_context():
    base, completion, logprobs = _assemble_flat_sequence(THREE_TURNS)
    assert base == [1, 2, 3]
    # only assistant tokens, context [20,21,22,30] dropped entirely
    assert completion == [10, 11, 12, 13, 14]
    assert logprobs == [-0.1, -0.2, -0.3, -0.4, -0.5]


def test_assemble_handles_empty():
    assert _assemble_faithful_sequence([]) == ([], [], [], [])
    assert _assemble_flat_sequence([]) == ([], [], [])


def test_faithful_negative_ext_len_is_safe():
    # If a turn's prompt is shorter than prompt(t-1)+comp(t-1) (drift edge case),
    # ext_len <= 0 and no context tokens are inserted — never crashes, never
    # produces a negative-length slice.
    segments = [
        ([1, 2, 3], [10, 11], [-0.1, -0.2]),
        ([1, 2], [12], [-0.3]),  # impossibly short prompt
    ]
    _base, completion, env_mask, logprobs = _assemble_faithful_sequence(segments)
    assert completion == [10, 11, 12]
    assert env_mask == [1, 1, 1]
    assert len(completion) == len(env_mask) == len(logprobs)


# ---------------------------------------------------------------------------
# Capability gate
# ---------------------------------------------------------------------------

def test_resolve_faithful_requires_all_conditions():
    on = {"token_faithful": True, "context_token_policy": "mask"}
    assert _resolve_faithful_mode(on, {"has_env_mask": True}) is True


def test_resolve_faithful_falls_back_without_env_mask_support():
    on = {"token_faithful": True, "context_token_policy": "mask"}
    assert _resolve_faithful_mode(on, {"has_env_mask": False}) is False
    assert _resolve_faithful_mode(on, None) is False


def test_resolve_faithful_respects_drop_policy():
    cfg = {"token_faithful": True, "context_token_policy": "drop"}
    assert _resolve_faithful_mode(cfg, {"has_env_mask": True}) is False


def test_resolve_faithful_respects_disabled():
    cfg = {"token_faithful": False, "context_token_policy": "mask"}
    assert _resolve_faithful_mode(cfg, {"has_env_mask": True}) is False


def test_resolve_faithful_defaults_on_when_supported():
    # token_faithful defaults to True, policy defaults to "mask"
    assert _resolve_faithful_mode({}, {"has_env_mask": True}) is True


# ---------------------------------------------------------------------------
# rollout_func output contract
# ---------------------------------------------------------------------------

def _patch_openenv(monkeypatch):
    """Stub the TRL openenv import so build_rollout_func works without TRL."""
    fake = types.SimpleNamespace(generate_rollout_completions=lambda *a, **k: None)
    monkeypatch.setattr(env_rollout, "_import_openenv_helpers", lambda: fake)


def _canned_result():
    return EpisodeRolloutResult(
        prompt_ids=[1, 2, 3],
        completion_ids=[10, 11, 20, 12],
        logprobs=[-0.1, -0.2, 0.0, -0.3],
        completion_text="hi",
        env_passed=True,
        env_reward=1.0,
        stop_reason="environment_passed",
        total_turns=2,
        total_tool_calls=1,
        final_text_satisfied=True,
        env_mask=[1, 1, 0, 1],
    )


def test_rollout_func_emits_env_mask_when_faithful(monkeypatch):
    _patch_openenv(monkeypatch)
    monkeypatch.setattr(env_rollout, "_run_single_episode", lambda **kw: _canned_result())

    rollout = build_rollout_func(
        registry={"p": object()},
        env_training_cfg={"token_faithful": True, "context_token_policy": "mask"},
        runtime_support={"has_env_mask": True},
    )
    out = rollout(["p"], trainer=None)

    assert "env_mask" in out
    assert out["env_mask"] == [[1, 1, 0, 1]]
    assert out["completion_ids"] == [[10, 11, 20, 12]]
    assert len(out["env_mask"][0]) == len(out["completion_ids"][0])


def test_rollout_func_omits_env_mask_when_unsupported(monkeypatch):
    _patch_openenv(monkeypatch)
    monkeypatch.setattr(env_rollout, "_run_single_episode", lambda **kw: _canned_result())

    rollout = build_rollout_func(
        registry={"p": object()},
        env_training_cfg={"token_faithful": True, "context_token_policy": "mask"},
        runtime_support={"has_env_mask": False},  # older TRL
    )
    out = rollout(["p"], trainer=None)

    # No env_mask key -> TRL trains on all completion tokens; safe because the
    # fallback episode would use the flat (context-free) representation.
    assert "env_mask" not in out
