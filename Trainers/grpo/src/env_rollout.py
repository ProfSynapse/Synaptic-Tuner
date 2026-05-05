"""Multi-step SynthChat environment rollout bridge for stock TRL GRPO."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
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
        metadata = _as_mapping(row.get("metadata"))
        registry[prompt] = EpisodeSpec(
            prompt=prompt,
            prompt_messages=_as_list(row.get("prompt_messages")),
            environment_config=_as_mapping(row.get("resolved_environment_config")),
            task_context=_as_mapping(row.get("task_context")),
            scenario=str(metadata.get("scenario") or row.get("scenario") or "unknown"),
        )
    return registry


def build_rollout_func(
    *,
    registry: Dict[str, EpisodeSpec],
    env_training_cfg: Dict[str, Any],
) -> Any:
    generate_completion = _build_completion_generator(env_training_cfg)

    def rollout_func(prompts: List[str], trainer) -> Dict[str, List[Any]]:
        results: List[EpisodeRolloutResult] = []
        for prompt in prompts:
            spec = registry.get(prompt)
            if spec is None:
                raise KeyError("Prompt not found in env-GRPO registry")
            results.append(
                _run_single_episode(
                    trainer=trainer,
                    generate_completion=generate_completion,
                    spec=spec,
                    env_training_cfg=env_training_cfg,
                )
            )

        return {
            "prompt_ids": [item.prompt_ids for item in results],
            "completion_ids": [item.completion_ids for item in results],
            "logprobs": [item.logprobs for item in results],
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
    generate_completion,
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
    expected_tools = _resolve_expected_tools(
        spec=spec,
        env_training_cfg=env_training_cfg,
        tool_schema=validator.tool_schema,
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
    turn_records: List[Dict[str, Any]] = []
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
                generate_completion=generate_completion,
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
                turn_records.append(
                    _build_turn_record(
                        turn_index=turn_index,
                        completion_text=completion_text,
                        has_tool_calls=has_tool_calls,
                        text_content=text_content,
                        step=None,
                        environment_preview=None,
                        tool_feedback=None,
                        env_training_cfg=env_training_cfg,
                    )
                )
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
                turn_records.append(
                    _build_turn_record(
                        turn_index=turn_index,
                        completion_text=completion_text,
                        has_tool_calls=has_tool_calls,
                        text_content=text_content,
                        step=step,
                        environment_preview=None,
                        tool_feedback=None,
                        env_training_cfg=env_training_cfg,
                    )
                )
                stop_reason = "environment_execution_failed"
                break

            environment_preview = session.finalize(
                expected_tools=expected_tools,
                total_turns=turn_index,
                stop_reason="preview",
            )

            feedback = None
            if has_tool_calls or (step.recoverable_error and bool(env_training_cfg.get("continue_on_execution_error", False))):
                feedback = format_tool_results_message(
                    executions=step.executed_tools,
                    issues=step.issues,
                    format_name=str(env_training_cfg.get("tool_result_format") or "json"),
                    tool_schema=validator.tool_schema,
                    tool_name_format=str(env_training_cfg.get("tool_result_name_format") or "executor"),
                )
                messages.append({"role": "user", "content": feedback})

            turn_records.append(
                _build_turn_record(
                    turn_index=turn_index,
                    completion_text=completion_text,
                    has_tool_calls=has_tool_calls,
                    text_content=text_content,
                    step=step,
                    environment_preview=environment_preview,
                    tool_feedback=feedback,
                    env_training_cfg=env_training_cfg,
                )
            )

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
            expected_tools=expected_tools,
            total_turns=len(session.steps),
            stop_reason=stop_reason,
        )
    finally:
        session.close()

    env_passed = bool(environment_result.passed)
    completion_text = "\n".join(part for part in completion_text_parts if part.strip())

    result = EpisodeRolloutResult(
        prompt_ids=first_prompt_ids or [],
        completion_ids=all_completion_ids,
        logprobs=all_logprobs,
        completion_text=completion_text,
        env_passed=env_passed,
        stop_reason=stop_reason,
        total_turns=len(session.steps),
        total_tool_calls=len(session.executed_tools),
        final_text_satisfied=final_text_satisfied,
    )
    _write_trajectory_log(
        env_training_cfg=env_training_cfg,
        spec=spec,
        result=result,
        turn_records=turn_records,
        environment_result=environment_result,
    )
    return result


def _build_turn_record(
    *,
    turn_index: int,
    completion_text: str,
    has_tool_calls: bool,
    text_content: str,
    step,
    environment_preview,
    tool_feedback: Optional[str],
    env_training_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    max_chars = _optional_int(env_training_cfg.get("trajectory_log_string_chars"), 8000)
    record: Dict[str, Any] = {
        "turn_index": turn_index,
        "completion_text": _truncate_text(completion_text, max_chars),
        "has_tool_calls": has_tool_calls,
        "text_content": _truncate_text(text_content, max_chars),
    }
    if step is not None:
        record["step"] = _truncate_payload_strings(_to_plain_data(step), max_chars)
    if environment_preview is not None:
        record["environment_preview"] = {
            "passed": bool(getattr(environment_preview, "passed", False)),
            "issues": _truncate_payload_strings(
                [_to_plain_data(issue) for issue in getattr(environment_preview, "issues", [])],
                max_chars,
            ),
            "assertions_run": int(getattr(environment_preview, "assertions_run", 0) or 0),
        }
    if tool_feedback is not None:
        record["tool_feedback"] = _truncate_text(tool_feedback, max_chars)
    return record


def _write_trajectory_log(
    *,
    env_training_cfg: Dict[str, Any],
    spec: EpisodeSpec,
    result: EpisodeRolloutResult,
    turn_records: List[Dict[str, Any]],
    environment_result,
) -> None:
    if not bool(env_training_cfg.get("log_trajectories", False)):
        return
    log_path = str(env_training_cfg.get("trajectory_log_path") or "").strip()
    if not log_path:
        return

    max_chars = _optional_int(env_training_cfg.get("trajectory_log_string_chars"), 8000)
    payload: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scenario": spec.scenario,
        "env_passed": result.env_passed,
        "stop_reason": result.stop_reason,
        "total_turns": result.total_turns,
        "total_tool_calls": result.total_tool_calls,
        "final_text_satisfied": result.final_text_satisfied,
        "completion_text": _truncate_text(result.completion_text, max_chars),
        "turns": turn_records,
        "environment_result": _truncate_payload_strings(_to_plain_data(environment_result), max_chars),
    }
    if bool(env_training_cfg.get("trajectory_log_include_prompt", False)):
        payload["prompt"] = _truncate_text(spec.prompt, max_chars)
    if bool(env_training_cfg.get("trajectory_log_include_task_context", True)):
        payload["task_context"] = _truncate_payload_strings(spec.task_context, max_chars)

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _to_plain_data(value: Any) -> Any:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if isinstance(value, Mapping):
        return {str(key): _to_plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain_data(item) for item in value]
    return value


def _truncate_payload_strings(value: Any, max_chars: Optional[int]) -> Any:
    if isinstance(value, str):
        return _truncate_text(value, max_chars)
    if isinstance(value, Mapping):
        return {key: _truncate_payload_strings(item, max_chars) for key, item in value.items()}
    if isinstance(value, list):
        return [_truncate_payload_strings(item, max_chars) for item in value]
    return value


def _truncate_text(value: str, max_chars: Optional[int]) -> str:
    if max_chars is None or max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[:max_chars] + f"...[truncated {len(value) - max_chars} chars]"


def _optional_int(value: Any, default: Optional[int]) -> Optional[int]:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _build_completion_generator(env_training_cfg: Dict[str, Any]):
    backend = str(env_training_cfg.get("backend") or "trl_rollout_func").strip().lower()
    if backend in {"trl_rollout_func", "trl_openenv", "openenv"}:
        openenv_module = _import_openenv_helpers()
        generate_rollout_completions = getattr(openenv_module, "generate_rollout_completions")

        def generate_with_trl(*, trainer, prompt_text: str) -> Dict[str, Any]:
            return _generate_one_completion_trl(
                trainer=trainer,
                generate_rollout_completions=generate_rollout_completions,
                prompt_text=prompt_text,
            )

        return generate_with_trl

    if backend in {"hf_generate", "transformers_generate", "model_generate"}:
        return _generate_one_completion_hf

    raise ValueError(f"Unsupported env_training.backend: {backend}")


def _resolve_expected_tools(
    *,
    spec: EpisodeSpec,
    env_training_cfg: Dict[str, Any],
    tool_schema: Optional[Dict[str, Any]],
) -> Optional[List[str]]:
    expected_cfg = env_training_cfg.get("expected_tools")
    expected_tools: Any = None

    if isinstance(expected_cfg, list):
        expected_tools = expected_cfg
        transform = str(env_training_cfg.get("expected_tools_transform") or "identity")
    elif isinstance(expected_cfg, Mapping):
        field_path = str(expected_cfg.get("field") or expected_cfg.get("path") or "").strip()
        transform = str(expected_cfg.get("transform") or "identity")
        expected_tools = _resolve_dot_path(
            {
                "task_context": spec.task_context,
                "environment_config": spec.environment_config,
            },
            field_path,
        )
    else:
        transform = "identity"
        expected_tools = spec.environment_config.get("expected_tools")

    if not isinstance(expected_tools, list):
        return None

    normalized = [str(item).strip() for item in expected_tools if str(item).strip()]
    if not normalized:
        return None
    if transform in {"schema_command", "tool_schema_command", "command_prefix"}:
        return _expected_commands_to_schema_commands(normalized, tool_schema)
    return normalized


def _expected_commands_to_schema_commands(
    expected_tools: List[str],
    tool_schema: Optional[Dict[str, Any]],
) -> List[str]:
    if not isinstance(tool_schema, dict):
        return expected_tools

    commands: List[str] = []
    for tools in (tool_schema.get("tools") or {}).values():
        if not isinstance(tools, list):
            continue
        for tool in tools:
            if not isinstance(tool, Mapping):
                continue
            command = str(tool.get("command") or "").strip()
            if command:
                commands.append(command)

    commands = sorted(set(commands), key=len, reverse=True)
    if not commands:
        return expected_tools

    resolved: List[str] = []
    for expected in expected_tools:
        match = next(
            (
                command
                for command in commands
                if expected == command or expected.startswith(command + " ")
            ),
            None,
        )
        resolved.append(match or expected)
    return resolved


def _resolve_dot_path(payload: Mapping[str, Any], path: str) -> Any:
    if not path:
        return None
    current: Any = payload
    for part in path.split("."):
        if isinstance(current, Mapping):
            current = current.get(part)
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes)) and part.isdigit():
            index = int(part)
            current = current[index] if index < len(current) else None
        else:
            return None
        if current is None:
            return None
    return current


def _generate_one_completion(*, trainer, generate_completion, prompt_text: str) -> Dict[str, Any]:
    outputs = generate_completion(trainer=trainer, prompt_text=prompt_text)
    if not isinstance(outputs, Mapping):
        raise RuntimeError("Configured completion generator returned invalid output shape")
    return dict(outputs)


def _generate_one_completion_trl(*, trainer, generate_rollout_completions, prompt_text: str) -> Dict[str, Any]:
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


def _generate_one_completion_hf(*, trainer, prompt_text: str) -> Dict[str, Any]:
    import torch

    tokenizer = trainer.processing_class
    model = trainer.model
    args = trainer.args
    encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_ids = encoded["input_ids"][0].tolist()

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    max_new_tokens = int(getattr(args, "max_completion_length", 256) or 256)
    temperature = float(getattr(args, "temperature", 0.0) or 0.0)
    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": temperature > 0.0,
    }
    if temperature > 0.0:
        generation_kwargs["temperature"] = temperature
    for attr in ("top_p", "top_k", "min_p"):
        value = getattr(args, attr, None)
        if value is not None:
            generation_kwargs[attr] = value

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            generated = model.generate(**encoded, **generation_kwargs)
            sequence = generated[:1]
            prompt_len = len(prompt_ids)
            completion_ids = sequence[0, prompt_len:].tolist()
            logprobs = _completion_logprobs(
                model=model,
                input_ids=sequence.to(device),
                prompt_len=prompt_len,
            )
    finally:
        if was_training:
            model.train()

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
    }


def _completion_logprobs(*, model, input_ids, prompt_len: int) -> List[float]:
    import torch

    if input_ids.shape[1] <= prompt_len:
        return []
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        log_probs = outputs.logits.log_softmax(dim=-1)
        target_ids = input_ids[:, 1:]
        token_logprobs = torch.gather(log_probs[:, :-1, :], 2, target_ids.unsqueeze(-1)).squeeze(-1)
    completion_logprobs = token_logprobs[0, prompt_len - 1 :].detach().cpu().tolist()
    return [float(value) for value in completion_logprobs]


def _as_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return []
        return decoded if isinstance(decoded, list) else []
    return []


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
