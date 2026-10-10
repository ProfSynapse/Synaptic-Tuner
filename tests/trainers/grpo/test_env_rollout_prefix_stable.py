"""Prefix-stable multi-turn prompt construction for env-GRPO rollouts.

Each turn's prompt must be ``prompt(t-1) + completion(t-1) + suffix`` where the
suffix holds only the new context (end-of-turn marker if the completion lacks
one, tool/feedback/nudge messages, generation prompt). These tests drive real
``_run_single_episode`` episodes through a fake chat template that behaves like
Qwen3.5's (earlier assistant turns lose their ``<think>`` block once a new user
message arrives), so naive whole-conversation re-rendering would diverge on
every turn after the first.

No TRL, torch, model or network is needed. The optional real-tokenizer test at
the bottom downloads only the Qwen3.5-4B tokenizer files (``RUN_LIVE_HUB=1``).
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Trainers" / "grpo" / "src"))

import env_rollout
from env_rollout import (
    EpisodeSpec,
    _context_suffix_ids,
    _find_prefix_mismatches,
    _run_single_episode,
    build_rollout_func,
)
from shared.environments.tool_executor import format_tool_results_message


# ---------------------------------------------------------------------------
# Fake Qwen3.5-style tokenizer + chat template
# ---------------------------------------------------------------------------

class QwenLikeTokenizer:
    """Character-level tokenizer with a Qwen3.5-like chat template.

    Template behaviour that matters here (mirrors Qwen3.5's chat_template.jinja):
    * the "last query" is the last ``user`` message not wrapped in
      ``<tool_response>``; assistant turns at or before it are rendered WITHOUT
      their ``<think>`` block (reasoning dropped, content trimmed);
    * the generation prompt is ``<|im_start|>assistant\\n<think>\\n`` by default
      and ``...<think>\\n\\n</think>\\n\\n`` with ``enable_thinking=False``.
    """

    SPECIAL = {"<|im_start|>": 1, "<|im_end|>": 2, "<|endoftext|>": 3}
    ADDED = {"<think>": 4, "</think>": 5, "<tool_response>": 6, "</tool_response>": 7}
    OFFSET = 100

    def __init__(self):
        self.vocab = {**self.SPECIAL, **self.ADDED}
        self.inverse = {v: k for k, v in self.vocab.items()}
        self.eos_token_id = self.SPECIAL["<|im_end|>"]
        self.template_calls: List[Dict[str, Any]] = []

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        ids: List[int] = []
        pos = 0
        while pos < len(text):
            for token, token_id in self.vocab.items():
                if text.startswith(token, pos):
                    ids.append(token_id)
                    pos += len(token)
                    break
            else:
                ids.append(ord(text[pos]) + self.OFFSET)
                pos += 1
        return ids

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        parts = []
        for token_id in ids:
            if token_id in self.inverse:
                if skip_special_tokens and self.inverse[token_id] in self.SPECIAL:
                    continue
                parts.append(self.inverse[token_id])
            else:
                parts.append(chr(token_id - self.OFFSET))
        return "".join(parts)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        assert tokenize is False
        self.template_calls.append(dict(kwargs))
        last_query = -1
        for index, message in enumerate(messages):
            content = str(message.get("content") or "")
            if message["role"] == "user" and not (
                content.startswith("<tool_response>") and content.endswith("</tool_response>")
            ):
                last_query = index
        out = ""
        for index, message in enumerate(messages):
            role = message["role"]
            content = str(message.get("content") or "")
            if role in ("system", "user"):
                out += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            elif role == "tool":
                out += f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"
            elif role == "assistant":
                reasoning = ""
                if "</think>" in content:
                    reasoning = content.split("</think>")[0].split("<think>")[-1].strip("\n")
                    content = content.split("</think>")[-1].lstrip("\n")
                if index > last_query and (reasoning or index == len(messages) - 1):
                    out += f"<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{content}<|im_end|>\n"
                else:
                    out += f"<|im_start|>assistant\n{content.strip()}<|im_end|>\n"
        if add_generation_prompt:
            out += "<|im_start|>assistant\n"
            if kwargs.get("enable_thinking", True) is False:
                out += "<think>\n\n</think>\n\n"
            else:
                out += "<think>\n"
        return out


# ---------------------------------------------------------------------------
# Scripted environment and generator
# ---------------------------------------------------------------------------

class _ScriptedSession:
    """Env session: passes the preview after ``pass_after`` executed steps."""

    def __init__(self, pass_after: int):
        self.pass_after = pass_after
        self.steps: List[Any] = []
        self.executed_tools: List[Any] = []

    def execute_response(self, _text):
        self.steps.append(object())
        return types.SimpleNamespace(hard_error=False, recoverable_error=False, executed_tools=[], issues=[])

    def finalize(self, **_kwargs):
        return types.SimpleNamespace(passed=len(self.steps) >= self.pass_after, issues=[])

    def close(self):
        pass


def _fake_parse_response(text):
    """Tool call iff the visible content (after any think block) starts with CALL."""
    visible = text.split("</think>")[-1].strip()
    return types.SimpleNamespace(has_tool_calls=visible.startswith("CALL"), text_content=visible)


class _RecordingVLLM:
    """Stands in for TRL's generate_rollout_completions (vLLM colocate/server)."""

    def __init__(self, tokenizer, completions: List[str]):
        self.tokenizer = tokenizer
        self.completions = list(completions)
        self.prompts: List[Any] = []
        self.sampled: List[List[int]] = []
        self.logprobs: List[List[float]] = []

    def __call__(self, trainer, prompts, *, as_chat=None):
        assert as_chat is False
        assert len(prompts) == 1
        prompt = prompts[0]
        self.prompts.append(prompt)
        prompt_ids = prompt["prompt_token_ids"] if isinstance(prompt, dict) else prompt
        turn = len(self.sampled)
        completion_ids = self.tokenizer.encode(self.completions[turn], add_special_tokens=False)
        logprobs = [-(turn + 1) - 0.001 * i for i in range(len(completion_ids))]
        self.sampled.append(completion_ids)
        self.logprobs.append(logprobs)
        return [{"prompt_ids": list(prompt_ids), "completion_ids": completion_ids, "logprobs": logprobs}]

    def sent_ids(self) -> List[List[int]]:
        return [p["prompt_token_ids"] if isinstance(p, dict) else p for p in self.prompts]


SYSTEM = "You manage a vault. Use tools."
USER = "Move notes/a.md to archive/ and tell me when done."
FINAL_TEXT_PROMPT = "The task is complete. Reply briefly. Do not call tools."
FEEDBACK = format_tool_results_message(executions=[], issues=[], format_name="json")


def _run_episode(
    monkeypatch,
    tokenizer,
    completions: List[str],
    *,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
    vllm_mode: str = "colocate",
    stop_ids: Optional[List[int]] = None,
    pass_after: int = 2,
):
    monkeypatch.setattr(env_rollout, "parse_response", _fake_parse_response)
    monkeypatch.setattr(
        env_rollout,
        "EnvironmentValidator",
        lambda backend: types.SimpleNamespace(start_session=lambda **_kw: _ScriptedSession(pass_after)),
    )
    generator = _RecordingVLLM(tokenizer, completions)
    trainer = types.SimpleNamespace(
        processing_class=tokenizer,
        use_vllm=True,
        vllm_mode=vllm_mode,
        eos_token_id=tokenizer.eos_token_id,
        generation_config=types.SimpleNamespace(eos_token_id=stop_ids or [tokenizer.eos_token_id]),
    )
    spec = EpisodeSpec(
        prompt="p",
        prompt_messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}],
        environment_config={},
        task_context={},
        scenario="prefix-stable",
    )
    result = _run_single_episode(
        trainer=trainer,
        generate_rollout_completions=generator,
        spec=spec,
        env_training_cfg={
            "max_turns": 6,
            "final_text_prompt": FINAL_TEXT_PROMPT,
            "require_final_text_after_pass": True,
            "stop_on_environment_pass": True,
        },
        faithful=True,
        chat_template_kwargs=chat_template_kwargs,
    )
    return result, generator


# Default-thinking completions: the generation prompt opens <think>, so each
# completion carries reasoning, closes </think>, then content and <|im_end|>.
THINKING_COMPLETIONS = [
    "I should list the folder first.\n</think>\n\nCALL list notes/<|im_end|>",
    "Now move the file.\n</think>\n\nCALL move notes/a.md archive/a.md<|im_end|>",
    "Summarise.\n</think>\n\nMoved notes/a.md to archive/.<|im_end|>",
]


def _expected_suffix_text(new_turn_text: str, *, thinking: bool = True, closed: bool = True) -> str:
    gen = "<|im_start|>assistant\n" + ("<think>\n" if thinking else "<think>\n\n</think>\n\n")
    return ("" if closed else "<|im_end|>") + "\n" + new_turn_text + gen


# ---------------------------------------------------------------------------
# Core property: zero mismatches and exact alignment over 3 turns
# ---------------------------------------------------------------------------

def test_naive_rerender_would_diverge_with_this_template():
    """Sanity: the fake template really strips earlier <think> blocks."""
    tok = QwenLikeTokenizer()
    base = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]
    prompt1 = tok.encode(tok.apply_chat_template(base, add_generation_prompt=True))
    completion1 = tok.encode(THINKING_COMPLETIONS[0])
    history = base + [
        {"role": "assistant", "content": tok.decode(completion1, skip_special_tokens=True)},
        {"role": "user", "content": FEEDBACK},
    ]
    naive_prompt2 = tok.encode(tok.apply_chat_template(history, add_generation_prompt=True))
    expected_prefix = prompt1 + completion1
    assert naive_prompt2[: len(expected_prefix)] != expected_prefix
    assert _find_prefix_mismatches([(prompt1, completion1, []), (naive_prompt2, [], [])])


def test_prefix_stable_three_turn_episode_has_zero_mismatches(monkeypatch, caplog):
    tok = QwenLikeTokenizer()
    with caplog.at_level("WARNING", logger=env_rollout.logger.name):
        result, gen = _run_episode(monkeypatch, tok, THINKING_COMPLETIONS)

    assert result.stop_reason == "environment_passed_final_text"
    assert len(gen.prompts) == 3
    assert result.prefix_mismatch_count == 0
    assert not [rec for rec in caplog.records if "prefix" in rec.getMessage()]

    p1, p2, p3 = gen.sent_ids()
    c1, c2, c3 = gen.sampled
    # Every prompt extends the previous prompt + the exact sampled ids.
    assert p2[: len(p1) + len(c1)] == p1 + c1
    assert p3[: len(p2) + len(c2)] == p2 + c2
    s1 = p2[len(p1) + len(c1):]
    s2 = p3[len(p2) + len(c2):]
    assert tok.decode(s1) == _expected_suffix_text(f"<|im_start|>user\n{FEEDBACK}<|im_end|>\n")
    assert tok.decode(s2) == _expected_suffix_text(
        f"<|im_start|>user\n{FEEDBACK}<|im_end|>\n<|im_start|>user\n{FINAL_TEXT_PROMPT}<|im_end|>\n"
    )
    # The model keeps seeing its own earlier reasoning (not re-rendered away).
    assert "I should list the folder first." in tok.decode(p3)
    # First prompt is the plain template render of the initial messages.
    first_render = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}],
        add_generation_prompt=True,
    )
    assert p1 == tok.encode(first_render)

    # Faithful sequence: prompt + [c1 | s1 | c2 | s2 | c3] with mask/logprobs aligned.
    assert result.prompt_ids == p1
    assert result.completion_ids == c1 + s1 + c2 + s2 + c3
    assert result.env_mask == [1] * len(c1) + [0] * len(s1) + [1] * len(c2) + [0] * len(s2) + [1] * len(c3)
    assert len(result.logprobs) == len(result.completion_ids)
    lp1, lp2, lp3 = gen.logprobs
    assert result.logprobs == lp1 + [0.0] * len(s1) + lp2 + [0.0] * len(s2) + lp3
    # Masked-in ids are exactly the sampled ids, in order.
    sampled = [i for i, m in zip(result.completion_ids, result.env_mask) if m == 1]
    assert sampled == c1 + c2 + c3
    # The full sequence is exactly the last prompt plus the last completion.
    assert result.prompt_ids + result.completion_ids == p3 + c3


def test_truncated_completion_without_eos_gets_template_end_of_turn_as_context(monkeypatch):
    tok = QwenLikeTokenizer()
    truncated = [
        "I should list the folder first.\n</think>\n\nCALL list notes/",  # hit max length: no <|im_end|>
        THINKING_COMPLETIONS[1],
        THINKING_COMPLETIONS[2],
    ]
    result, gen = _run_episode(monkeypatch, tok, truncated)

    assert result.prefix_mismatch_count == 0
    p1, p2, _p3 = gen.sent_ids()
    c1 = gen.sampled[0]
    assert c1[-1] != tok.eos_token_id
    s1 = p2[len(p1) + len(c1):]
    assert s1[0] == tok.eos_token_id
    assert tok.decode(s1) == _expected_suffix_text(f"<|im_start|>user\n{FEEDBACK}<|im_end|>\n", closed=False)
    # The appended end-of-turn id is context (mask 0, logprob 0.0), never trained.
    eot_position = len(c1)
    assert result.completion_ids[eot_position] == tok.eos_token_id
    assert result.env_mask[eot_position] == 0
    assert result.logprobs[eot_position] == 0.0
    assert len(result.completion_ids) == len(result.env_mask) == len(result.logprobs)


def test_completion_closed_by_other_stop_id_does_not_get_a_second_end_of_turn():
    tok = QwenLikeTokenizer()
    endoftext = tok.SPECIAL["<|endoftext|>"]
    suffix = _context_suffix_ids(
        tok,
        [{"role": "user", "content": "next"}],
        completion_ids=[200, 201, endoftext],
        stop_token_ids=frozenset({tok.eos_token_id, endoftext}),
    )
    assert tok.decode(suffix) == "\n<|im_start|>user\nnext<|im_end|>\n<|im_start|>assistant\n<think>\n"


def test_suffix_with_no_new_messages_is_just_the_generation_prompt():
    tok = QwenLikeTokenizer()
    suffix = _context_suffix_ids(
        tok, [], completion_ids=[200, tok.eos_token_id], stop_token_ids=frozenset({tok.eos_token_id})
    )
    assert tok.decode(suffix) == "\n<|im_start|>assistant\n<think>\n"


def test_tool_role_feedback_is_supported():
    tok = QwenLikeTokenizer()
    suffix = _context_suffix_ids(
        tok,
        [{"role": "tool", "content": "ok"}],
        completion_ids=[200, tok.eos_token_id],
        stop_token_ids=frozenset({tok.eos_token_id}),
    )
    assert tok.decode(suffix) == (
        "\n<|im_start|>user\n<tool_response>\nok\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n"
    )


def test_template_that_drops_the_sentinel_is_refused():
    class DroppingTokenizer(QwenLikeTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return "<|im_start|>assistant\n"

    with pytest.raises(ValueError, match="sentinel"):
        _context_suffix_ids(
            DroppingTokenizer(), [], completion_ids=[2], stop_token_ids=frozenset({2})
        )


# ---------------------------------------------------------------------------
# Chat-template kwargs pass-through
# ---------------------------------------------------------------------------

def test_chat_template_kwargs_reach_every_render(monkeypatch):
    tok = QwenLikeTokenizer()
    completions = [
        "CALL list notes/<|im_end|>",
        "CALL move notes/a.md archive/a.md<|im_end|>",
        "Moved notes/a.md to archive/.<|im_end|>",
    ]
    result, gen = _run_episode(
        monkeypatch, tok, completions, chat_template_kwargs={"enable_thinking": False}
    )

    assert result.prefix_mismatch_count == 0
    assert len(tok.template_calls) == 3  # first prompt + two suffix renders
    assert all(call == {"enable_thinking": False} for call in tok.template_calls)
    p1, p2, p3 = gen.sent_ids()
    assert tok.decode(p1).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert tok.decode(p2).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert tok.decode(p3).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_build_rollout_func_forwards_chat_template_kwargs(monkeypatch):
    seen: Dict[str, Any] = {}

    def fake_episode(**kwargs):
        seen.update(kwargs)
        return types.SimpleNamespace(
            prompt_ids=[1], completion_ids=[2], logprobs=[0.0], completion_text="", env_passed=True,
            env_reward=1.0, stop_reason="x", total_turns=1, total_tool_calls=0, final_text_satisfied=True,
            env_mask=[1], prefix_mismatch_count=0, executed_tool_names=[], executed_tool_statuses=[],
            environment_issue_levels=[], expected_tool_names=[],
        )

    monkeypatch.setattr(env_rollout, "_run_single_episode", fake_episode)
    rollout = build_rollout_func(
        registry={"p": object()},
        env_training_cfg={},
        use_vllm=False,
        runtime_support={"has_env_mask": True},
        chat_template_kwargs={"enable_thinking": False},
    )
    rollout(["p"], trainer=None)
    assert seen["chat_template_kwargs"] == {"enable_thinking": False}
    assert seen["generate_rollout_completions"] is None


# ---------------------------------------------------------------------------
# Generator interface: token-id prompts on both vLLM modes
# ---------------------------------------------------------------------------

def test_vllm_colocate_receives_tokens_prompt_and_server_receives_id_list(monkeypatch):
    tok = QwenLikeTokenizer()
    _result, colocate = _run_episode(monkeypatch, tok, THINKING_COMPLETIONS, vllm_mode="colocate")
    assert all(isinstance(p, dict) and set(p) == {"prompt_token_ids"} for p in colocate.prompts)

    tok2 = QwenLikeTokenizer()
    result, server = _run_episode(monkeypatch, tok2, THINKING_COMPLETIONS, vllm_mode="server")
    assert all(isinstance(p, list) and all(isinstance(i, int) for i in p) for p in server.prompts)
    assert server.sent_ids() == colocate.sent_ids()
    assert result.prefix_mismatch_count == 0


def test_vllm_trainer_without_helper_is_refused():
    trainer = types.SimpleNamespace(use_vllm=True)
    with pytest.raises(RuntimeError, match="use_vllm"):
        env_rollout._generate_one_completion(
            trainer=trainer, generate_rollout_completions=None, prompt_ids=[1, 2]
        )


# ---------------------------------------------------------------------------
# TRL >= 1.9: generate_rollout_completions no longer exists
# ---------------------------------------------------------------------------

def _install_trl_without_rollout_helper(monkeypatch):
    trl = types.ModuleType("trl")
    experimental = types.ModuleType("trl.experimental")
    openenv = types.ModuleType("trl.experimental.openenv")  # helper removed in TRL 1.9
    trl.experimental = experimental
    experimental.openenv = openenv
    for name, module in (
        ("trl", trl),
        ("trl.experimental", experimental),
        ("trl.experimental.openenv", openenv),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    for name in ("trl.experimental.open_env", "trl.extras", "trl.extras.openenv"):
        monkeypatch.delitem(sys.modules, name, raising=False)


def test_rollout_module_imports_and_builds_without_rollout_helper(monkeypatch):
    _install_trl_without_rollout_helper(monkeypatch)
    monkeypatch.delitem(sys.modules, "env_rollout")
    fresh = importlib.import_module("env_rollout")
    try:
        rollout = fresh.build_rollout_func(
            registry={}, env_training_cfg={}, use_vllm=False, runtime_support={"has_env_mask": True}
        )
        assert callable(rollout)
        with pytest.raises(ImportError, match="generate_rollout_completions"):
            fresh.build_rollout_func(registry={}, env_training_cfg={}, use_vllm=True)
    finally:
        sys.modules["env_rollout"] = env_rollout


def test_transformers_path_never_imports_the_helper(monkeypatch):
    def boom():
        raise AssertionError("transformers path must not import TRL's rollout helper")

    monkeypatch.setattr(env_rollout, "_import_openenv_helpers", boom)
    assert callable(build_rollout_func(registry={}, env_training_cfg={}, use_vllm=False))


# ---------------------------------------------------------------------------
# Opt-in: the REAL Qwen3.5-4B tokenizer/template at the smoke recipe's revision
# ---------------------------------------------------------------------------

_LIVE_HUB = os.environ.get("RUN_LIVE_HUB") == "1"
_SMOKE_RECIPE = ROOT / "Trainers" / "recipes" / "qwen35_4b_modal_train_eval_smoke.yaml"


@pytest.mark.skipif(
    not _LIVE_HUB,
    reason="network-gated; set RUN_LIVE_HUB=1 to download the Qwen3.5-4B tokenizer files",
)
@pytest.mark.parametrize("enable_thinking", [None, False], ids=["thinking-default", "thinking-off"])
def test_real_qwen35_tokenizer_prefix_stable(monkeypatch, enable_thinking):
    from transformers import AutoTokenizer

    recipe = yaml.safe_load(_SMOKE_RECIPE.read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(recipe["model"]["name"], revision=recipe["model"]["revision"])
    stop_ids = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    kwargs = None if enable_thinking is None else {"enable_thinking": enable_thinking}
    if enable_thinking is False:
        completions = [
            "CALL list notes/<|im_end|>",
            "CALL move notes/a.md archive/a.md<|im_end|>",
            "Moved notes/a.md to archive/.<|im_end|>",
        ]
    else:
        completions = THINKING_COMPLETIONS

    result, gen = _run_episode(
        monkeypatch, tok, completions, chat_template_kwargs=kwargs, stop_ids=stop_ids
    )

    assert result.prefix_mismatch_count == 0
    p1, p2, p3 = gen.sent_ids()
    c1, c2, c3 = gen.sampled
    assert p2[: len(p1) + len(c1)] == p1 + c1
    assert p3[: len(p2) + len(c2)] == p2 + c2
    s1 = p2[len(p1) + len(c1):]
    suffix_text = tok.decode(s1)
    assert suffix_text.startswith("\n<|im_start|>user\n")
    assert suffix_text.endswith(
        "<|im_start|>assistant\n<think>\n\n</think>\n\n" if enable_thinking is False
        else "<|im_start|>assistant\n<think>\n"
    )
    sampled = [i for i, m in zip(result.completion_ids, result.env_mask) if m == 1]
    assert sampled == c1 + c2 + c3
    assert len(result.completion_ids) == len(result.env_mask) == len(result.logprobs)

    # The naive approach (re-render the whole conversation) diverges with the real template.
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
        {"role": "assistant", "content": tok.decode(c1, skip_special_tokens=True)},
        {"role": "user", "content": FEEDBACK},
    ]
    naive = tok.encode(
        tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **(kwargs or {})),
        add_special_tokens=False,
    )
    assert naive[: len(p1) + len(c1)] != p1 + c1
