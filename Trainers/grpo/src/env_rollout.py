"""Multi-step SynthChat environment rollout bridge for stock TRL GRPO."""

from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

from shared.environments import EnvironmentValidator
from shared.environments.tool_executor import format_tool_results_message
from shared.validation.parsing.response_parser import parse_response


@dataclass
class EpisodeSpec:
    prompt: str
    prompt_messages: List[Dict[str, Any]]
    environment_config: Dict[str, Any]
    task_context: Dict[str, Any]
    scenario: str


@dataclass
class EpisodeRolloutResult:
    prompt_ids: List[int]
    completion_ids: List[int]
    logprobs: List[float]
    completion_text: str
    env_passed: bool
    env_reward: float
    stop_reason: str
    total_turns: int
    total_tool_calls: int
    final_text_satisfied: bool
    # Token-faithful (POLAR-style) loss mask aligned 1:1 with completion_ids:
    # 1 = model-sampled token (trainable), 0 = external context token (tool
    # result / user feedback) that the model conditioned on but did not emit.
    # Empty when the legacy flattened representation is used.
    env_mask: List[int] = field(default_factory=list)
    # Number of turn transitions whose recorded prompt did not extend the
    # previous turn's prompt + sampled completion (see _assemble_faithful_sequence).
    # Non-zero means the faithful sequence was truncated at the last consistent
    # turn. Always 0 on the legacy flattened path, which does not use later
    # turns' prompts.
    prefix_mismatch_count: int = 0
    executed_tool_names: List[str] = field(default_factory=list)
    executed_tool_statuses: List[str] = field(default_factory=list)
    environment_issue_levels: List[str] = field(default_factory=list)
    expected_tool_names: List[str] = field(default_factory=list)


# A single generated turn: (prompt_ids, completion_ids, logprobs). prompt_ids is
# the exact id sequence sent to the generator for that turn (built prefix-stably,
# see _context_suffix_ids); completion_ids / logprobs are the actual sampled
# tokens and their sampling log-probabilities.
TurnSegment = Tuple[List[int], List[int], List[float]]

# Stand-in assistant content used to locate the end of an assistant turn inside
# a rendered dummy conversation (see _context_suffix_ids). Plain ASCII with no
# whitespace or markup so no chat template trims, splits or escapes it.
_ASSISTANT_BOUNDARY_SENTINEL = "EnvRolloutAssistantBoundary7f3a9c51"


def _assemble_flat_sequence(turn_segments: Sequence[TurnSegment]) -> Tuple[List[int], List[int], List[float]]:
    """Legacy representation: first turn's prompt, all assistant turns concatenated.

    Intermediate tool-result / user-feedback tokens are dropped entirely. This is
    the historical behavior and is not token-faithful for multi-turn episodes, but
    it is exactly correct (and identical to the faithful path) for single-turn
    episodes.
    """
    if not turn_segments:
        return [], [], []
    base_prompt = list(turn_segments[0][0])
    completion: List[int] = []
    logprobs: List[float] = []
    for _prompt_ids, completion_ids, turn_logprobs in turn_segments:
        completion.extend(completion_ids)
        logprobs.extend(turn_logprobs)
    return base_prompt, completion, logprobs


@dataclass(frozen=True)
class PrefixMismatch:
    """A turn transition whose rendered prompt does not extend the prior turn.

    ``turn_index`` is the 0-based index of the turn whose prompt diverged.
    ``expected_prefix_len`` is ``len(prompt(t-1)) + len(completion(t-1))``;
    ``prompt_len`` is ``len(prompt(t))``. ``diverge_at`` is the first position
    where the ids differ, or ``None`` when the prompt is simply too short to
    contain the expected prefix.
    """

    turn_index: int
    expected_prefix_len: int
    prompt_len: int
    diverge_at: Optional[int]


@dataclass
class FaithfulSequence:
    """Output of :func:`_assemble_faithful_sequence`.

    ``completion_ids``, ``env_mask`` and ``logprobs`` are always the same length.
    ``kept_turns`` is how many leading turns made it into the sequence; it is
    smaller than the number of input turns only when ``mismatches`` is non-empty.
    """

    prompt_ids: List[int]
    completion_ids: List[int]
    env_mask: List[int]
    logprobs: List[float]
    kept_turns: int
    mismatches: List[PrefixMismatch] = field(default_factory=list)


def _find_prefix_mismatches(turn_segments: Sequence[TurnSegment]) -> List[PrefixMismatch]:
    """Check every transition: does ``prompt(t)`` start with ``prompt(t-1) + completion(t-1)``?

    The rollout loop builds each prompt as ``prompt(t-1) + completion(t-1) +
    suffix`` (see :func:`_context_suffix_ids`), so in normal operation this
    finds nothing. It is kept as a safety net: if a generator ever hands back
    prompt ids that were re-rendered or re-tokenized (end-of-turn markers
    re-added differently, boundary re-tokenization, templates that strip
    reasoning blocks from earlier assistant turns), the length delta between
    consecutive prompts would no longer identify the new context ids. A
    transition where the
    prompt is not longer than the expected prefix but still matches it exactly
    (no context ids in between) is consistent and is not reported.
    """
    mismatches: List[PrefixMismatch] = []
    for idx in range(1, len(turn_segments)):
        prev_prompt, prev_completion, _prev_logprobs = turn_segments[idx - 1]
        cur_prompt = turn_segments[idx][0]
        expected = list(prev_prompt) + list(prev_completion)
        if len(cur_prompt) >= len(expected) and list(cur_prompt[: len(expected)]) == expected:
            continue
        diverge_at: Optional[int] = None
        for pos, (got, want) in enumerate(zip(cur_prompt, expected)):
            if got != want:
                diverge_at = pos
                break
        mismatches.append(
            PrefixMismatch(
                turn_index=idx,
                expected_prefix_len=len(expected),
                prompt_len=len(cur_prompt),
                diverge_at=diverge_at,
            )
        )
    return mismatches


def _assemble_faithful_sequence(turn_segments: Sequence[TurnSegment]) -> FaithfulSequence:
    """Token-faithful representation: full interleaved sequence + per-token mask.

    ``completion_ids`` is everything after the initial prompt — assistant turns
    interleaved with the exact tool-result / user-feedback context ids the model
    saw between turns. ``env_mask`` is 1 on assistant-sampled ids and 0 on
    external context ids; ``logprobs`` carries the sampling log-prob on
    assistant ids and 0.0 on context ids. All three lists are the same length,
    matching TRL's ``env_mask`` contract (mask is multiplied into the completion
    loss mask, so context ids contribute nothing to the loss while still being
    attended to).

    Context ids for the transition into turn ``t`` are the tail of turn ``t``'s
    prompt past ``prompt(t-1) + completion(t-1)``. The rollout loop builds
    prompts that way by construction; every transition is still verified first
    as a safety net (idea borrowed from agent-lightning's ``ids_startswith``
    check). On the first transition that fails, the sequence
    is truncated after the last consistent turn: every id kept is then exactly
    what the model conditioned on or sampled, with correct logprob alignment.
    A single TRL rollout row per episode cannot hold a second segment, so the
    later turns are dropped rather than spliced in with a wrong context.
    Assistant spans use the raw sampled ids (not re-templated text).
    """
    if not turn_segments:
        return FaithfulSequence([], [], [], [], kept_turns=0)

    mismatches = _find_prefix_mismatches(turn_segments)
    kept_turns = mismatches[0].turn_index if mismatches else len(turn_segments)

    base_prompt = list(turn_segments[0][0])
    completion: List[int] = []
    env_mask: List[int] = []
    logprobs: List[float] = []

    for idx in range(kept_turns):
        cur_prompt, cur_completion, cur_logprobs = turn_segments[idx]
        if idx > 0:
            prev_prompt, prev_completion, _prev_logprobs = turn_segments[idx - 1]
            prefix_len = len(prev_prompt) + len(prev_completion)
            ext_ids = list(cur_prompt[prefix_len:])
            completion.extend(ext_ids)
            env_mask.extend([0] * len(ext_ids))
            logprobs.extend([0.0] * len(ext_ids))

        completion.extend(cur_completion)
        env_mask.extend([1] * len(cur_completion))
        logprobs.extend(_align_logprobs(list(cur_logprobs), len(cur_completion)))

    return FaithfulSequence(
        prompt_ids=base_prompt,
        completion_ids=completion,
        env_mask=env_mask,
        logprobs=logprobs,
        kept_turns=kept_turns,
        mismatches=mismatches,
    )


def build_prompt_registry(dataset) -> Dict[str, EpisodeSpec]:
    registry: Dict[str, EpisodeSpec] = {}
    for row in dataset:
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Env-GRPO row missing string prompt")
        if prompt in registry:
            raise ValueError("Duplicate prompt detected in env-GRPO dataset; prompt lookup must be unique")
        metadata = row.get("metadata") or {}
        registry[prompt] = EpisodeSpec(
            prompt=prompt,
            prompt_messages=list(row.get("prompt_messages") or []),
            environment_config=dict(row.get("resolved_environment_config") or {}),
            task_context=dict(row.get("task_context") or {}),
            scenario=str(metadata.get("scenario") or row.get("scenario") or "unknown"),
        )
    return registry


def _resolve_faithful_mode(
    env_training_cfg: Mapping[str, Any],
    runtime_support: Optional[Mapping[str, Any]],
) -> bool:
    """Decide whether to emit the token-faithful representation + env_mask.

    Faithful mode requires (a) the config opting in (default on), (b) the mask
    policy, and (c) the installed TRL actually honoring ``env_mask`` as a loss
    mask. If faithful is requested but the runtime cannot honor the mask, we fall
    back to the safe legacy flattened path rather than silently stuffing
    untrainable context tokens into ``completion_ids`` (which an older TRL would
    train on). This is the Phase 0 capability gate.
    """
    token_faithful = bool(env_training_cfg.get("token_faithful", True))
    context_policy = str(env_training_cfg.get("context_token_policy", "mask"))
    if not token_faithful or context_policy != "mask":
        return False

    supports_env_mask = bool((runtime_support or {}).get("has_env_mask"))
    if not supports_env_mask:
        logger.warning(
            "token_faithful requested but the installed TRL does not honor "
            "env_mask (requires trl>=0.28.0). Falling back to the legacy "
            "flattened rollout representation to avoid training on context tokens."
        )
        return False
    return True


def build_rollout_func(
    *,
    registry: Dict[str, EpisodeSpec],
    env_training_cfg: Dict[str, Any],
    use_vllm: bool,
    runtime_support: Optional[Mapping[str, Any]] = None,
    chat_template_kwargs: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Build the TRL ``rollout_func`` for multi-turn env episodes.

    ``use_vllm`` must match ``GRPOConfig.use_vllm``. TRL's
    ``generate_rollout_completions`` helper is imported only on the vLLM path:
    TRL 1.9 removed it, and the transformers path never needs it.
    ``chat_template_kwargs`` (e.g. ``{"enable_thinking": False}``) is passed to
    every chat-template render the rollout performs.
    """
    generate_rollout_completions = None
    if use_vllm:
        openenv_module = _import_openenv_helpers()
        generate_rollout_completions = getattr(openenv_module, "generate_rollout_completions")
    faithful = _resolve_faithful_mode(env_training_cfg, runtime_support)
    template_kwargs = dict(chat_template_kwargs or {})

    def rollout_func(prompts: List[str], trainer) -> Dict[str, List[Any]]:
        results: List[EpisodeRolloutResult] = []
        for prompt in prompts:
            spec = registry.get(prompt)
            if spec is None:
                raise KeyError("Prompt not found in env-GRPO registry")
            results.append(
                _run_single_episode(
                    trainer=trainer,
                    generate_rollout_completions=generate_rollout_completions,
                    spec=spec,
                    env_training_cfg=env_training_cfg,
                    faithful=faithful,
                    chat_template_kwargs=template_kwargs,
                )
            )

        output: Dict[str, List[Any]] = {
            "prompt_ids": [item.prompt_ids for item in results],
            "completion_ids": [item.completion_ids for item in results],
            "logprobs": [item.logprobs for item in results],
            "env_reward": [item.env_reward for item in results],
            "env_passed": [item.env_passed for item in results],
            "stop_reason": [item.stop_reason for item in results],
            "total_turns": [item.total_turns for item in results],
            "total_tool_calls": [item.total_tool_calls for item in results],
            "final_text_satisfied": [item.final_text_satisfied for item in results],
            "executed_tool_names": [item.executed_tool_names for item in results],
            "executed_tool_statuses": [item.executed_tool_statuses for item in results],
            "environment_issue_levels": [item.environment_issue_levels for item in results],
            "expected_tool_names": [item.expected_tool_names for item in results],
            "completion_text": [item.completion_text for item in results],
            "prefix_mismatch_count": [item.prefix_mismatch_count for item in results],
        }
        mismatched = sum(1 for item in results if item.prefix_mismatch_count)
        if mismatched:
            logger.warning(
                "env-GRPO rollout batch: %d/%d episodes had a prompt prefix "
                "mismatch and were truncated to their last consistent turn.",
                mismatched,
                len(results),
            )
        if faithful:
            # TRL pops env_mask from the rollout output and multiplies it into the
            # completion loss mask (model tokens=1, external context tokens=0).
            output["env_mask"] = [item.env_mask for item in results]
        return output

    return rollout_func


def _run_single_episode(
    *,
    trainer,
    generate_rollout_completions,
    spec: EpisodeSpec,
    env_training_cfg: Dict[str, Any],
    faithful: bool = False,
    chat_template_kwargs: Optional[Mapping[str, Any]] = None,
) -> EpisodeRolloutResult:
    tokenizer = trainer.processing_class
    template_kwargs = dict(chat_template_kwargs or {})
    stop_token_ids = _stop_token_ids(trainer, tokenizer)
    env_backend = str(env_training_cfg.get("env_backend") or "local")
    validator = EnvironmentValidator(backend=env_backend)
    messages = [dict(msg) for msg in spec.prompt_messages]
    system_prompt = _first_system_prompt(messages)
    session = validator.start_session(
        system_prompt=system_prompt,
        environment_config=spec.environment_config,
    )

    loop_cfg = spec.environment_config.get("loop")
    if not isinstance(loop_cfg, Mapping):
        loop_cfg = {}
    max_turns = int(env_training_cfg.get("max_turns", loop_cfg.get("max_turns", 6)))
    max_tool_steps = int(env_training_cfg.get("max_tool_steps", loop_cfg.get("max_tool_steps", 0)))
    stop_on_text_response = bool(env_training_cfg.get("stop_on_text_response", loop_cfg.get("stop_on_text_response", True)))
    stop_on_environment_pass = bool(env_training_cfg.get("stop_on_environment_pass", loop_cfg.get("stop_on_environment_pass", True)))
    require_final_text_after_pass = bool(
        env_training_cfg.get("require_final_text_after_pass", loop_cfg.get("require_final_text_after_pass", True))
    )
    expected_tools = spec.environment_config.get("expected_tools")
    if not isinstance(expected_tools, list):
        expected_tools = None
    require_expected_tools = bool(spec.environment_config.get("require_expected_tools"))
    final_text_prompt = str(
        env_training_cfg.get("final_text_prompt")
        or loop_cfg.get("final_text_prompt")
        or "The task is complete. Reply to the user with a brief final text-only response. Do not call any more tools."
    )

    turn_segments: List[TurnSegment] = []
    completion_text_parts: List[str] = []
    stop_reason = "max_turns_reached"
    awaiting_final_text = False
    final_text_satisfied = False

    # Index in ``messages`` of the latest assistant turn; everything after it is
    # new context (tool results, feedback, nudges) for the next generation.
    last_assistant_index = -1

    try:
        for turn_index in range(1, max_turns + 1):
            if turn_segments:
                # Prefix-stable: earlier turns are never re-rendered. The next
                # prompt is exactly what the model already saw and sampled, plus
                # only the new context ids.
                prev_prompt_ids, prev_completion_ids, _prev_logprobs = turn_segments[-1]
                suffix_ids = _context_suffix_ids(
                    tokenizer,
                    messages[last_assistant_index + 1:],
                    completion_ids=prev_completion_ids,
                    stop_token_ids=stop_token_ids,
                    chat_template_kwargs=template_kwargs,
                    leading_messages=_leading_system_messages(messages),
                )
                prompt_ids = list(prev_prompt_ids) + list(prev_completion_ids) + suffix_ids
            else:
                prompt_ids = _encode_text(
                    tokenizer,
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        **template_kwargs,
                    ),
                )
            outputs = _generate_one_completion(
                trainer=trainer,
                generate_rollout_completions=generate_rollout_completions,
                prompt_ids=prompt_ids,
                stop_token_ids=stop_token_ids,
            )
            # Record the ids the generator reports it conditioned on (normally
            # identical to what was sent). If a backend ever altered them (e.g.
            # added a BOS), the prefix check below catches it instead of the
            # episode silently training on ids the model never saw.
            prompt_ids = list(outputs.get("prompt_ids") or prompt_ids)
            completion_ids = list(outputs.get("completion_ids") or [])
            logprobs = [float(value) for value in (outputs.get("logprobs") or [])]
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)

            # Keep logprobs aligned 1:1 with completion_ids. TRL pads/uses the
            # sampling logprobs positionally, so a per-turn mismatch would
            # corrupt downstream alignment. Pad short, truncate long.
            logprobs = _align_logprobs(logprobs, len(completion_ids))

            turn_segments.append((prompt_ids, completion_ids, logprobs))
            completion_text_parts.append(completion_text)

            parsed = parse_response(completion_text)
            has_tool_calls = parsed.has_tool_calls
            text_content = parsed.text_content.strip()

            # The decoded text drives parsing and the environment. The ids the
            # model sees next come from turn_segments, not from re-rendering
            # this message.
            messages.append({"role": "assistant", "content": completion_text})
            last_assistant_index = len(messages) - 1

            if awaiting_final_text:
                if has_tool_calls:
                    stop_reason = "final_text_tool_calls_emitted"
                    break
                if not text_content:
                    stop_reason = "final_text_missing"
                    break
                final_text_satisfied = True
                stop_reason = "environment_passed_final_text"
                break

            step = session.execute_response(completion_text)
            if step.hard_error:
                stop_reason = "environment_execution_failed"
                break

            environment_preview = session.finalize(
                expected_tools=expected_tools if require_expected_tools else None,
                total_turns=turn_index,
                stop_reason="preview",
            )

            feedback = None
            if has_tool_calls or (step.recoverable_error and bool(env_training_cfg.get("continue_on_execution_error", False))):
                feedback = format_tool_results_message(
                    executions=step.executed_tools,
                    issues=step.issues,
                    format_name=str(env_training_cfg.get("tool_result_format") or "json"),
                )
                messages.append({"role": "user", "content": feedback})

            if stop_on_environment_pass and environment_preview.passed:
                if require_final_text_after_pass:
                    awaiting_final_text = True
                    messages.append({"role": "user", "content": final_text_prompt})
                    continue
                stop_reason = "environment_passed"
                break

            if max_tool_steps and len(session.executed_tools) > max_tool_steps:
                stop_reason = "max_tool_steps_exceeded"
                break

            if not has_tool_calls:
                if require_final_text_after_pass and not environment_preview.passed:
                    stop_reason = "text_response_before_completion"
                    break
                if stop_on_text_response:
                    stop_reason = "text_response"
                    break

        environment_result = session.finalize(
            expected_tools=expected_tools if require_expected_tools else None,
            total_turns=len(session.steps),
            stop_reason=stop_reason,
        )
    finally:
        session.close()

    env_passed = bool(environment_result.passed)
    env_reward = 1.0 if env_passed else 0.0
    completion_text = "\n".join(part for part in completion_text_parts if part.strip())
    executed_tool_names = [
        str(getattr(item, "name", "")).strip()
        for item in session.executed_tools
        if str(getattr(item, "name", "")).strip()
    ]
    executed_tool_statuses = [
        str(getattr(item, "status", "")).strip()
        for item in session.executed_tools
        if str(getattr(item, "status", "")).strip()
    ]
    environment_issue_levels = [
        str(getattr(item, "level", "")).strip()
        for item in getattr(environment_result, "issues", [])
        if str(getattr(item, "level", "")).strip()
    ]
    expected_tool_names = [
        str(item).strip()
        for item in (expected_tools or [])
        if str(item).strip()
    ]

    prefix_mismatch_count = 0
    if faithful:
        assembled = _assemble_faithful_sequence(turn_segments)
        prompt_ids = assembled.prompt_ids
        completion_ids = assembled.completion_ids
        env_mask = assembled.env_mask
        logprobs = assembled.logprobs
        prefix_mismatch_count = len(assembled.mismatches)
        if assembled.mismatches:
            first = assembled.mismatches[0]
            logger.warning(
                "env-GRPO faithful rollout: turn %d prompt does not extend the previous "
                "turn (expected prefix len %d, prompt len %d, first divergence at %s); "
                "%d transition(s) inconsistent. Truncating episode sequence to the first "
                "%d of %d turns. scenario=%s",
                first.turn_index,
                first.expected_prefix_len,
                first.prompt_len,
                "n/a (prompt shorter than prefix)" if first.diverge_at is None else first.diverge_at,
                prefix_mismatch_count,
                assembled.kept_turns,
                len(turn_segments),
                spec.scenario,
            )
    else:
        prompt_ids, completion_ids, logprobs = _assemble_flat_sequence(turn_segments)
        env_mask = []

    _write_debug_rollout(
        env_training_cfg=env_training_cfg,
        spec=spec,
        completion_text=completion_text,
        env_passed=env_passed,
        stop_reason=stop_reason,
        total_turns=len(session.steps),
        total_tool_calls=len(session.executed_tools),
        final_text_satisfied=final_text_satisfied,
        expected_tool_names=expected_tool_names,
        environment_result=environment_result,
        executed_tools=session.executed_tools,
        prefix_mismatch_count=prefix_mismatch_count,
    )

    return EpisodeRolloutResult(
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        logprobs=logprobs,
        completion_text=completion_text,
        env_passed=env_passed,
        env_reward=env_reward,
        stop_reason=stop_reason,
        total_turns=len(session.steps),
        total_tool_calls=len(session.executed_tools),
        final_text_satisfied=final_text_satisfied,
        env_mask=env_mask,
        prefix_mismatch_count=prefix_mismatch_count,
        executed_tool_names=executed_tool_names,
        executed_tool_statuses=executed_tool_statuses,
        environment_issue_levels=environment_issue_levels,
        expected_tool_names=expected_tool_names,
    )


def _align_logprobs(logprobs: List[float], target_len: int) -> List[float]:
    """Force ``logprobs`` to length ``target_len`` (pad with 0.0, truncate excess)."""
    if len(logprobs) == target_len:
        return logprobs
    if len(logprobs) < target_len:
        return logprobs + [0.0] * (target_len - len(logprobs))
    return logprobs[:target_len]


def _write_debug_rollout(
    *,
    env_training_cfg: Dict[str, Any],
    spec: EpisodeSpec,
    completion_text: str,
    env_passed: bool,
    stop_reason: str,
    total_turns: int,
    total_tool_calls: int,
    final_text_satisfied: bool,
    expected_tool_names: List[str],
    environment_result: Any,
    executed_tools: Any,
    prefix_mismatch_count: int,
) -> None:
    debug_path = env_training_cfg.get("debug_rollouts_path")
    if not debug_path:
        return

    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "scenario": spec.scenario,
        "env_passed": env_passed,
        "stop_reason": stop_reason,
        "total_turns": total_turns,
        "total_tool_calls": total_tool_calls,
        "final_text_satisfied": final_text_satisfied,
        "prefix_mismatch_count": prefix_mismatch_count,
        "expected_tool_names": expected_tool_names,
        "environment_issues": getattr(environment_result, "issues", []),
        "executed_tools": [
            getattr(item, "__dict__", item)
            for item in (executed_tools or [])
        ],
        "task_context": spec.task_context,
        "completion_text": completion_text,
    }
    try:
        path = Path(str(debug_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:
        print(f"[env-grpo] failed to write debug rollout to {debug_path}: {exc}", flush=True)
        return


def _generate_one_completion(
    *,
    trainer,
    generate_rollout_completions,
    prompt_ids: Sequence[int],
    stop_token_ids: frozenset = frozenset(),
) -> Dict[str, Any]:
    """Sample one completion conditioned on exactly ``prompt_ids`` (no re-tokenization)."""
    if not bool(getattr(trainer, "use_vllm", False)):
        return _generate_one_completion_transformers(
            trainer=trainer,
            prompt_ids=prompt_ids,
            stop_token_ids=stop_token_ids,
        )

    if generate_rollout_completions is None:
        raise RuntimeError(
            "trainer.use_vllm is true but the rollout was built with use_vllm=False; "
            "build_rollout_func(use_vllm=...) must match GRPOConfig.use_vllm"
        )
    # Token-id prompts so vLLM conditions on exactly these ids. Colocate mode
    # hands prompts straight to LLM.generate (a vLLM TokensPrompt). Server mode
    # sends a token-id list, which needs a TRL VLLMClient.generate that accepts
    # list[list[int]] (TRL 1.x; 0.28's client takes text only).
    if str(getattr(trainer, "vllm_mode", "colocate")) == "server":
        prompt: Any = list(prompt_ids)
    else:
        prompt = {"prompt_token_ids": list(prompt_ids)}
    outputs = generate_rollout_completions(trainer, [prompt], as_chat=False)

    if not outputs:
        raise RuntimeError("generate_rollout_completions returned no outputs")
    first = outputs[0]
    if not isinstance(first, Mapping):
        raise RuntimeError("generate_rollout_completions returned invalid output shape")
    return dict(first)


def _generate_one_completion_transformers(
    *,
    trainer,
    prompt_ids: Sequence[int],
    stop_token_ids: frozenset = frozenset(),
) -> Dict[str, Any]:
    import torch

    device = trainer.accelerator.device
    input_ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    prompt_length = input_ids.shape[1]

    generation_kwargs = {
        "generation_config": trainer.generation_config,
    }
    if "disable_compile" in inspect.signature(trainer.model.generate).parameters:
        generation_kwargs["disable_compile"] = True

    was_training = bool(trainer.model.training)
    trainer.model.eval()
    try:
        with torch.no_grad():
            generated = trainer.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )
    finally:
        if was_training:
            trainer.model.train()

    completion_ids = generated[0, prompt_length:].tolist()
    # generate() pads finished sequences: cut after the first stop (or pad) id
    # and keep that id, as TRL does for its own completions.
    cut_ids = set(stop_token_ids)
    pad_token_id = getattr(trainer, "pad_token_id", None)
    if pad_token_id is not None:
        cut_ids.add(pad_token_id)
    for index, token_id in enumerate(completion_ids):
        if token_id in cut_ids:
            completion_ids = completion_ids[: index + 1]
            break

    return {
        "prompt_ids": list(prompt_ids),
        "completion_ids": completion_ids,
        "logprobs": None,
    }


def _encode_text(tokenizer, text: str) -> List[int]:
    """Tokenize already-templated text without adding BOS/EOS a second time."""
    return list(tokenizer.encode(text, add_special_tokens=False))


def _stop_token_ids(trainer, tokenizer) -> frozenset:
    """Ids that end an assistant turn when sampled (tokenizer / generation-config EOS)."""
    ids = set()
    candidates: List[Any] = [
        getattr(tokenizer, "eos_token_id", None),
        getattr(trainer, "eos_token_id", None),
        getattr(getattr(trainer, "generation_config", None), "eos_token_id", None),
    ]
    for value in candidates:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set, frozenset)):
            ids.update(int(item) for item in value if item is not None)
        else:
            ids.add(int(value))
    return frozenset(ids)


def _leading_system_messages(messages: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    leading: List[Dict[str, Any]] = []
    for message in messages:
        if str(message.get("role", "")).strip() != "system":
            break
        leading.append(dict(message))
    return leading


def _context_suffix_ids(
    tokenizer,
    new_messages: Sequence[Mapping[str, Any]],
    *,
    completion_ids: Sequence[int],
    stop_token_ids: frozenset,
    chat_template_kwargs: Optional[Mapping[str, Any]] = None,
    leading_messages: Sequence[Mapping[str, Any]] = (),
) -> List[int]:
    """Ids the model sees between its last sampled completion and its next turn.

    Adapted from TRL's ``GRPOTrainer._get_tool_suffix_ids`` (its multi-turn tool
    loop): render a fixed minimal conversation that ends in an assistant turn,
    append ``new_messages`` and the generation prompt, and keep only the ids
    after that assistant turn, aligned at the end-of-turn (EOS) id. The caller
    builds ``prompt + completion + suffix``, so earlier turns are never
    re-rendered.

    TRL finds the boundary by rendering the dummy conversation a second time
    without the new messages. That needs a prefix-preserving template, and TRL
    swaps in its own training template when the model's is not. Templates such
    as Qwen3.5 render the dummy assistant turn differently depending on what
    follows it, so here the boundary is located inside the one full render: the
    dummy assistant content is a sentinel, and the suffix is the tokenized text
    after it. That text starts with the template's assistant end-of-turn
    marker. If the sampled completion already ended with a stop id, the marker
    up to and including the first stop id is dropped (the model emitted its
    own). If it did not (truncated at max length, or no EOS sampled), the
    template's whole end-of-turn marker is kept as context so the next turn
    still opens cleanly. Nothing here is specific to one chat template.
    """
    dummy = [dict(message) for message in leading_messages]
    dummy.append({"role": "user", "content": "dummy"})
    dummy.append({"role": "assistant", "content": _ASSISTANT_BOUNDARY_SENTINEL})
    rendered = tokenizer.apply_chat_template(
        dummy + [dict(message) for message in new_messages],
        tokenize=False,
        add_generation_prompt=True,
        **dict(chat_template_kwargs or {}),
    )
    if rendered.count(_ASSISTANT_BOUNDARY_SENTINEL) != 1:
        raise ValueError(
            "chat template did not render the assistant boundary sentinel exactly once; "
            "cannot compute prefix-stable context ids"
        )
    tail_ids = _encode_text(tokenizer, rendered.split(_ASSISTANT_BOUNDARY_SENTINEL, 1)[1])

    end_of_turn = next(
        (index + 1 for index, token_id in enumerate(tail_ids) if token_id in stop_token_ids),
        0,
    )
    completion_closed = bool(completion_ids) and completion_ids[-1] in stop_token_ids
    if end_of_turn and completion_closed:
        return tail_ids[end_of_turn:]
    return tail_ids


def _first_system_prompt(messages: Sequence[Mapping[str, Any]]) -> str:
    for message in messages:
        if str(message.get("role", "")).strip() == "system":
            content = message.get("content")
            return content if isinstance(content, str) else ""
    return ""


def _import_openenv_helpers():
    for module_name in (
        "trl.experimental.openenv",
        "trl.experimental.open_env",
        "trl.extras.openenv",
    ):
        try:
            module = __import__(module_name, fromlist=["generate_rollout_completions"])
        except Exception:
            continue
        if hasattr(module, "generate_rollout_completions"):
            return module
    raise ImportError("Could not import TRL OpenEnv helpers with generate_rollout_completions")
