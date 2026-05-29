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
    executed_tool_names: List[str] = field(default_factory=list)
    executed_tool_statuses: List[str] = field(default_factory=list)
    environment_issue_levels: List[str] = field(default_factory=list)
    expected_tool_names: List[str] = field(default_factory=list)


# A single generated turn: (prompt_ids, completion_ids, logprobs). prompt_ids is
# the re-tokenized conversation prefix rendered for that turn's generation;
# completion_ids / logprobs are the actual sampled tokens and their sampling
# log-probabilities.
TurnSegment = Tuple[List[int], List[int], List[float]]


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


def _assemble_faithful_sequence(
    turn_segments: Sequence[TurnSegment],
) -> Tuple[List[int], List[int], List[int], List[float]]:
    """Token-faithful representation: full interleaved sequence + per-token mask.

    Returns ``(base_prompt_ids, completion_ids, env_mask, logprobs)`` where
    ``completion_ids`` is everything after the initial prompt — assistant turns
    interleaved with the exact tool-result / user-feedback context tokens the
    model saw between turns. ``env_mask`` is 1 on assistant-sampled tokens and 0
    on external context tokens; ``logprobs`` carries the sampling log-prob on
    assistant tokens and 0.0 on context tokens. All three lists are the same
    length, matching TRL's ``env_mask`` contract (mask is multiplied into the
    completion loss mask, so context tokens contribute nothing to the loss while
    still being attended to).

    Context tokens for the transition into turn ``t`` are recovered as the tail of
    turn ``t``'s rendered prompt that extends past ``prompt(t-1) + completion(t-1)``
    — the same length-delta slicing TRL's own internal tool loop uses. Assistant
    spans use the raw sampled token ids (not re-templated text) so the trained
    tokens stay byte-faithful to what was sampled.
    """
    if not turn_segments:
        return [], [], [], []

    base_prompt = list(turn_segments[0][0])
    completion: List[int] = []
    env_mask: List[int] = []
    logprobs: List[float] = []

    _first_prompt, first_completion, first_logprobs = turn_segments[0]
    completion.extend(first_completion)
    env_mask.extend([1] * len(first_completion))
    logprobs.extend(first_logprobs)

    for idx in range(1, len(turn_segments)):
        prev_prompt, prev_completion, _prev_logprobs = turn_segments[idx - 1]
        cur_prompt, cur_completion, cur_logprobs = turn_segments[idx]

        ext_len = len(cur_prompt) - len(prev_prompt) - len(prev_completion)
        if ext_len > 0:
            ext_ids = list(cur_prompt[-ext_len:])
            completion.extend(ext_ids)
            env_mask.extend([0] * ext_len)
            logprobs.extend([0.0] * ext_len)

        completion.extend(cur_completion)
        env_mask.extend([1] * len(cur_completion))
        logprobs.extend(cur_logprobs)

    return base_prompt, completion, env_mask, logprobs


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
    runtime_support: Optional[Mapping[str, Any]] = None,
) -> Any:
    openenv_module = _import_openenv_helpers()
    generate_rollout_completions = getattr(openenv_module, "generate_rollout_completions")
    faithful = _resolve_faithful_mode(env_training_cfg, runtime_support)

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
        }
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
) -> EpisodeRolloutResult:
    tokenizer = trainer.processing_class
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

    try:
        for turn_index in range(1, max_turns + 1):
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            outputs = _generate_one_completion(
                trainer=trainer,
                generate_rollout_completions=generate_rollout_completions,
                prompt_text=prompt_text,
            )
            prompt_ids = list(outputs.get("prompt_ids") or [])
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

            messages.append({"role": "assistant", "content": completion_text})

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
    )

    if faithful:
        prompt_ids, completion_ids, env_mask, logprobs = _assemble_faithful_sequence(turn_segments)
    else:
        prompt_ids, completion_ids, logprobs = _assemble_flat_sequence(turn_segments)
        env_mask = []

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


def _generate_one_completion(*, trainer, generate_rollout_completions, prompt_text: str) -> Dict[str, Any]:
    if not bool(getattr(trainer, "use_vllm", False)):
        return _generate_one_completion_transformers(trainer=trainer, prompt_text=prompt_text)

    signature = inspect.signature(generate_rollout_completions)
    if len(signature.parameters) == 2:
        outputs = generate_rollout_completions(trainer, [prompt_text])
    else:
        outputs = generate_rollout_completions(
            prompts=[prompt_text],
            args=trainer.args,
            processing_class=trainer.processing_class,
            model=trainer.model,
        )

    if not outputs:
        raise RuntimeError("generate_rollout_completions returned no outputs")
    first = outputs[0]
    if not isinstance(first, Mapping):
        raise RuntimeError("generate_rollout_completions returned invalid output shape")
    return dict(first)


def _generate_one_completion_transformers(*, trainer, prompt_text: str) -> Dict[str, Any]:
    import torch

    tokenizer = trainer.processing_class
    inputs = tokenizer(prompt_text, return_tensors="pt")
    device = trainer.accelerator.device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    prompt_ids_tensor = inputs["input_ids"]
    prompt_length = prompt_ids_tensor.shape[1]

    generation_kwargs = {
        "generation_config": trainer.generation_config,
    }
    if "disable_compile" in inspect.signature(trainer.model.generate).parameters:
        generation_kwargs["disable_compile"] = True

    was_training = bool(trainer.model.training)
    trainer.model.eval()
    try:
        with torch.no_grad():
            generated = trainer.model.generate(**inputs, **generation_kwargs)
    finally:
        if was_training:
            trainer.model.train()

    completion_tensor = generated[0, prompt_length:]
    eos_token_id = getattr(trainer, "eos_token_id", None)
    pad_token_id = getattr(trainer, "pad_token_id", None)
    completion_ids = completion_tensor.tolist()
    for index, token_id in enumerate(completion_ids):
        if token_id in {eos_token_id, pad_token_id}:
            completion_ids = completion_ids[: index + 1]
            break

    return {
        "prompt_ids": prompt_ids_tensor[0].tolist(),
        "completion_ids": completion_ids,
        "logprobs": None,
    }


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
