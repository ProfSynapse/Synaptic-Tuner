"""Tests for token-faithful (POLAR-style) multi-turn env-GRPO rollout assembly.

These cover the pure sequence-assembly helpers, the capability gate that decides
whether to emit ``env_mask``, and the rollout_func output contract. They use only
crafted token lists and stubs — no TRL, model, or environment runtime required.
"""

import logging
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Trainers" / "grpo" / "src"))

import env_rollout
from env_rollout import (
    EpisodeRolloutResult,
    EpisodeSpec,
    _align_logprobs,
    _assemble_faithful_sequence,
    _assemble_flat_sequence,
    _find_prefix_mismatches,
    _resolve_faithful_mode,
    _run_single_episode,
    build_rollout_func,
)


def _unpack(seq):
    return seq.prompt_ids, seq.completion_ids, seq.env_mask, seq.logprobs


def _assert_aligned(seq):
    assert len(seq.completion_ids) == len(seq.env_mask) == len(seq.logprobs)


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

# Turn 2's re-rendered prompt rewrote turn 0's assistant span (e.g. a template
# stripping a reasoning block from earlier assistant turns): [10, 11] -> [10].
# Its length delta is still positive, so the old length-only slicing would have
# silently taken the wrong ids as "context".
DIVERGED_AT_TURN_2 = [
    ([1, 2, 3], [10, 11], [-0.1, -0.2]),
    ([1, 2, 3, 10, 11, 20, 21, 22], [12, 13], [-0.3, -0.4]),
    ([1, 2, 3, 10, 20, 21, 22, 12, 13, 30, 31, 32], [14], [-0.5]),
]


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
    base_f, comp_f, mask_f, lp_f = _unpack(_assemble_faithful_sequence(SINGLE_TURN))
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
    base, completion, env_mask, logprobs = _unpack(_assemble_faithful_sequence(THREE_TURNS))

    assert base == [1, 2, 3]
    # assistant0 ++ ext1 ++ assistant1 ++ ext2 ++ assistant2
    assert completion == [10, 11, 20, 21, 22, 12, 13, 30, 14]
    assert env_mask == [1, 1, 0, 0, 0, 1, 1, 0, 1]
    assert logprobs == [-0.1, -0.2, 0.0, 0.0, 0.0, -0.3, -0.4, 0.0, -0.5]


def test_multi_turn_lengths_aligned():
    _assert_aligned(_assemble_faithful_sequence(THREE_TURNS))


def test_multi_turn_mask_covers_only_assistant_tokens():
    _base, completion, env_mask, _logprobs = _unpack(_assemble_faithful_sequence(THREE_TURNS))
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
    seq = _assemble_faithful_sequence([])
    assert _unpack(seq) == ([], [], [], [])
    assert seq.kept_turns == 0
    assert seq.mismatches == []
    assert _assemble_flat_sequence([]) == ([], [], [])


# ---------------------------------------------------------------------------
# Prefix verification
# ---------------------------------------------------------------------------

def test_consistent_prefix_reports_no_mismatch_and_keeps_all_turns():
    seq = _assemble_faithful_sequence(THREE_TURNS)
    assert seq.mismatches == []
    assert seq.kept_turns == 3
    assert _find_prefix_mismatches(THREE_TURNS) == []


def test_consistent_prefix_with_no_context_between_turns():
    # prompt(1) == prompt(0) + completion(0) exactly: ext_len == 0 but the
    # prefix matches, so the transition is consistent with no context ids.
    segments = [
        ([1, 2, 3], [10, 11], [-0.1, -0.2]),
        ([1, 2, 3, 10, 11], [12], [-0.3]),
    ]
    seq = _assemble_faithful_sequence(segments)
    assert seq.mismatches == []
    assert seq.completion_ids == [10, 11, 12]
    assert seq.env_mask == [1, 1, 1]
    assert seq.logprobs == [-0.1, -0.2, -0.3]


def test_diverged_prefix_truncates_at_last_consistent_turn():
    seq = _assemble_faithful_sequence(DIVERGED_AT_TURN_2)

    # Turns 0 and 1 are consistent; turn 2's prompt rewrote earlier ids.
    assert seq.kept_turns == 2
    assert seq.prompt_ids == [1, 2, 3]
    assert seq.completion_ids == [10, 11, 20, 21, 22, 12, 13]
    assert seq.env_mask == [1, 1, 0, 0, 0, 1, 1]
    assert seq.logprobs == [-0.1, -0.2, 0.0, 0.0, 0.0, -0.3, -0.4]
    _assert_aligned(seq)

    assert len(seq.mismatches) == 1
    mismatch = seq.mismatches[0]
    assert mismatch.turn_index == 2
    assert mismatch.expected_prefix_len == 10
    assert mismatch.prompt_len == 12
    assert mismatch.diverge_at == 4  # first position after [1, 2, 3, 10]


def test_diverged_prefix_never_emits_misaligned_ids():
    # The full assembled sequence must always equal some turn's real
    # prompt + completion with the base prompt stripped (i.e. a sequence the
    # model actually conditioned on / sampled), never a spliced mix.
    seq = _assemble_faithful_sequence(DIVERGED_AT_TURN_2)
    last_prompt, last_completion, _ = DIVERGED_AT_TURN_2[seq.kept_turns - 1]
    assert seq.prompt_ids + seq.completion_ids == list(last_prompt) + list(last_completion)


def test_mismatch_at_first_transition_keeps_only_first_turn():
    # Turn 0's assistant [10, 11] re-rendered as [10, 99] in turn 1's prompt
    # (boundary re-tokenization / end-of-turn marker re-added differently).
    segments = [
        ([1, 2, 3], [10, 11], [-0.1, -0.2]),
        ([1, 2, 3, 10, 99, 20, 21], [12, 13], [-0.3, -0.4]),
        ([1, 2, 3, 10, 99, 20, 21, 12, 13, 30], [14], [-0.5]),
    ]
    seq = _assemble_faithful_sequence(segments)
    assert seq.kept_turns == 1
    assert seq.completion_ids == [10, 11]
    assert seq.env_mask == [1, 1]
    assert seq.logprobs == [-0.1, -0.2]
    # Only transition 0->1 is inconsistent; 1->2 extends turn 1 correctly.
    assert [m.turn_index for m in seq.mismatches] == [1]
    assert seq.mismatches[0].diverge_at == 4


def test_faithful_negative_ext_len_is_safe():
    # A turn's prompt shorter than prompt(t-1)+comp(t-1) (ext_len < 0) cannot
    # contain the expected prefix: report it and truncate instead of appending
    # turn 1's completion with no valid context.
    segments = [
        ([1, 2, 3], [10, 11], [-0.1, -0.2]),
        ([1, 2], [12], [-0.3]),  # impossibly short prompt
    ]
    seq = _assemble_faithful_sequence(segments)
    assert seq.completion_ids == [10, 11]
    assert seq.env_mask == [1, 1]
    assert seq.logprobs == [-0.1, -0.2]
    _assert_aligned(seq)
    assert len(seq.mismatches) == 1
    assert seq.mismatches[0].turn_index == 1
    assert seq.mismatches[0].prompt_len == 2
    assert seq.mismatches[0].expected_prefix_len == 5
    assert seq.mismatches[0].diverge_at is None  # matching but too short


def test_zero_ext_len_with_diverged_prefix_is_a_mismatch():
    # Same length as prompt(0)+completion(0) (ext_len == 0) but different ids.
    segments = [
        ([1, 2, 3], [10, 11], [-0.1, -0.2]),
        ([1, 2, 3, 10, 77], [12], [-0.3]),
    ]
    seq = _assemble_faithful_sequence(segments)
    assert seq.kept_turns == 1
    assert seq.completion_ids == [10, 11]
    assert seq.mismatches[0].diverge_at == 4


def test_misaligned_segment_logprobs_are_realigned():
    # Defensive: even if a segment's logprobs are short/long, the output lists
    # stay equal length.
    segments = [
        ([1, 2, 3], [10, 11], [-0.1]),
        ([1, 2, 3, 10, 11, 20], [12], [-0.3, -0.9]),
    ]
    seq = _assemble_faithful_sequence(segments)
    _assert_aligned(seq)
    assert seq.logprobs == [-0.1, 0.0, 0.0, -0.3]


def test_lengths_aligned_for_all_fixtures():
    for segments in (SINGLE_TURN, THREE_TURNS, DIVERGED_AT_TURN_2):
        _assert_aligned(_assemble_faithful_sequence(segments))


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


def _canned_result(prefix_mismatch_count=0):
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
        prefix_mismatch_count=prefix_mismatch_count,
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
    assert out["prefix_mismatch_count"] == [0]


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


def test_rollout_func_reports_prefix_mismatch_per_episode(monkeypatch, caplog):
    _patch_openenv(monkeypatch)
    canned = iter([_canned_result(0), _canned_result(2)])
    monkeypatch.setattr(env_rollout, "_run_single_episode", lambda **kw: next(canned))

    rollout = build_rollout_func(
        registry={"a": object(), "b": object()},
        env_training_cfg={"token_faithful": True, "context_token_policy": "mask"},
        runtime_support={"has_env_mask": True},
    )
    with caplog.at_level(logging.WARNING, logger=env_rollout.logger.name):
        out = rollout(["a", "b"], trainer=None)

    assert out["prefix_mismatch_count"] == [0, 2]
    assert any("1/2 episodes" in rec.getMessage() for rec in caplog.records)


# ---------------------------------------------------------------------------
# _run_single_episode end-to-end with stubs
# ---------------------------------------------------------------------------

class _StubSession:
    def __init__(self):
        self.steps = []
        self.executed_tools = []

    def execute_response(self, _text):
        self.steps.append(object())
        return types.SimpleNamespace(
            hard_error=False, recoverable_error=False, executed_tools=[], issues=[]
        )

    def finalize(self, **_kwargs):
        return types.SimpleNamespace(passed=False, issues=[])

    def close(self):
        pass


class _StubValidator:
    def __init__(self, backend):
        self.backend = backend

    def start_session(self, **_kwargs):
        return _StubSession()


class _StubTokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        return "|".join(str(m.get("content")) for m in messages)

    def decode(self, ids, skip_special_tokens=True):
        return f"plain text reply {len(ids)}"


def _run_stub_episode(monkeypatch, segments, *, faithful=True):
    monkeypatch.setattr(env_rollout, "EnvironmentValidator", _StubValidator)
    scripted = iter(segments)

    def generate(_trainer, _prompts):
        prompt_ids, completion_ids, logprobs = next(scripted)
        return [{"prompt_ids": prompt_ids, "completion_ids": completion_ids, "logprobs": logprobs}]

    trainer = types.SimpleNamespace(processing_class=_StubTokenizer(), use_vllm=True)
    spec = EpisodeSpec(
        prompt="p",
        prompt_messages=[{"role": "user", "content": "do the thing"}],
        environment_config={},
        task_context={},
        scenario="stub",
    )
    return _run_single_episode(
        trainer=trainer,
        generate_rollout_completions=generate,
        spec=spec,
        env_training_cfg={
            "max_turns": len(segments),
            "stop_on_text_response": False,
            "require_final_text_after_pass": False,
        },
        faithful=faithful,
    )


def test_run_single_episode_consistent_has_no_mismatch(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger=env_rollout.logger.name):
        result = _run_stub_episode(monkeypatch, THREE_TURNS)
    assert result.prefix_mismatch_count == 0
    assert result.completion_ids == [10, 11, 20, 21, 22, 12, 13, 30, 14]
    assert result.env_mask == [1, 1, 0, 0, 0, 1, 1, 0, 1]
    assert not caplog.records


def test_run_single_episode_mismatch_truncates_and_warns(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger=env_rollout.logger.name):
        result = _run_stub_episode(monkeypatch, DIVERGED_AT_TURN_2)

    assert result.prefix_mismatch_count == 1
    assert result.prompt_ids == [1, 2, 3]
    assert result.completion_ids == [10, 11, 20, 21, 22, 12, 13]
    assert result.env_mask == [1, 1, 0, 0, 0, 1, 1]
    assert len(result.completion_ids) == len(result.env_mask) == len(result.logprobs)
    # Episode-level stats still describe the whole episode.
    assert result.total_turns == 3

    messages = [rec.getMessage() for rec in caplog.records]
    assert len(messages) == 1
    assert "turn 2" in messages[0]
    assert "expected prefix len 10" in messages[0]
    assert "first 2 of 3 turns" in messages[0]


def test_run_single_episode_flat_path_reports_zero(monkeypatch):
    result = _run_stub_episode(monkeypatch, DIVERGED_AT_TURN_2, faithful=False)
    assert result.prefix_mismatch_count == 0
    assert result.env_mask == []
    assert result.completion_ids == [10, 11, 12, 13, 14]
