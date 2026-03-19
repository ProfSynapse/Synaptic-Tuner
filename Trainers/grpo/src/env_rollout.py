"""Multi-step SynthChat environment rollout bridge for stock TRL GRPO."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

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


def build_rollout_func(
    *,
    registry: Dict[str, EpisodeSpec],
    env_training_cfg: Dict[str, Any],
) -> Any:
    openenv_module = _import_openenv_helpers()
    generate_rollout_completions = getattr(openenv_module, "generate_rollout_completions")

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
                )
            )

        return {
            "prompt_ids": [item.prompt_ids for item in results],
            "completion_ids": [item.completion_ids for item in results],
            "logprobs": [item.logprobs for item in results],
            "env_reward": [item.env_reward for item in results],
            "env_passed": [item.env_passed for item in results],
            "stop_reason": [item.stop_reason for item in results],
            "total_turns": [item.total_turns for item in results],
            "total_tool_calls": [item.total_tool_calls for item in results],
            "final_text_satisfied": [item.final_text_satisfied for item in results],
            "completion_text": [item.completion_text for item in results],
        }

    return rollout_func


def _run_single_episode(
    *,
    trainer,
    generate_rollout_completions,
    spec: EpisodeSpec,
    env_training_cfg: Dict[str, Any],
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

    max_turns = int(env_training_cfg.get("max_turns", 6))
    max_tool_steps = int(env_training_cfg.get("max_tool_steps", 0))
    stop_on_text_response = bool(env_training_cfg.get("stop_on_text_response", True))
    stop_on_environment_pass = bool(env_training_cfg.get("stop_on_environment_pass", True))
    require_final_text_after_pass = bool(env_training_cfg.get("require_final_text_after_pass", True))
    final_text_prompt = str(
        env_training_cfg.get("final_text_prompt")
        or "The task is complete. Reply to the user with a brief final text-only response. Do not call any more tools."
    )

    all_completion_ids: List[int] = []
    all_logprobs: List[float] = []
    first_prompt_ids: Optional[List[int]] = None
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

            if first_prompt_ids is None:
                first_prompt_ids = prompt_ids
            all_completion_ids.extend(completion_ids)
            all_logprobs.extend(logprobs)
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
                expected_tools=None,
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
            expected_tools=None,
            total_turns=len(session.steps),
            stop_reason=stop_reason,
        )
    finally:
        session.close()

    env_passed = bool(environment_result.passed)
    env_reward = 1.0 if env_passed else 0.0
    completion_text = "\n".join(part for part in completion_text_parts if part.strip())

    return EpisodeRolloutResult(
        prompt_ids=first_prompt_ids or [],
        completion_ids=all_completion_ids,
        logprobs=all_logprobs,
        completion_text=completion_text,
        env_passed=env_passed,
        env_reward=env_reward,
        stop_reason=stop_reason,
        total_turns=len(session.steps),
        total_tool_calls=len(session.executed_tools),
        final_text_satisfied=final_text_satisfied,
    )


def _generate_one_completion(*, trainer, generate_rollout_completions, prompt_text: str) -> Dict[str, Any]:
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

