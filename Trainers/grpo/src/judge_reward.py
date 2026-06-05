"""GRPO/GSPO reward backed by the shared LLM-as-judge service.

Wires shared/judge into the GRPO reward pipeline. Each completion is sent to a
configured judge LLM with a list of user-supplied rubrics, and the per-rubric
scores are aggregated into a single scalar reward via a configurable strategy.

Activated from `configs/rewards/judge_rubric.yaml` (type: judge_rubric) and
dispatched in `Trainers/grpo/src/rewards.py::build_combined_reward_function`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List

from shared.judge import (
    JudgeConfig,
    JudgeService,
    RubricDef,
    RubricLoader,
)
from shared.llm import LLMConfig, create_client
from shared.verifiers.builtins.llm_judge import (
    AGGREGATIONS as _AGGREGATIONS,
    aggregate,
    render_combined_prompt,
)

logger = logging.getLogger(__name__)


def _coerce_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict) and "content" in first:
            return str(first.get("content") or "")
    if isinstance(completion, dict) and "content" in completion:
        return str(completion.get("content") or "")
    return str(completion)


def _coerce_prompt_text(prompt: Any) -> str:
    if prompt is None:
        return ""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list) and prompt:
        last = prompt[-1]
        if isinstance(last, dict) and "content" in last:
            return str(last.get("content") or "")
    return str(prompt)


def _resolve_dir(value: str, base_dir: Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base_dir / p).resolve()


def _build_judge_service(judge_cfg: Dict[str, Any]) -> JudgeService:
    backend = judge_cfg.get("backend")
    model = judge_cfg.get("model")
    if not backend or not model:
        raise ValueError(
            "judge_rubric reward requires 'judge.backend' and 'judge.model' in YAML"
        )

    llm_config = LLMConfig.from_env(env_prefix="IMPROVEMENT")
    llm_config.provider = str(backend).lower()
    llm_config.model = str(model)

    for key in (
        "lmstudio_host",
        "lmstudio_port",
        "ollama_host",
        "ollama_port",
        "provider_routing",
        "openrouter_timeout_seconds",
        "unsloth_max_seq_length",
        "unsloth_load_in_4bit",
        "unsloth_top_p",
    ):
        if key in judge_cfg:
            setattr(llm_config, key, judge_cfg[key])

    llm_client = create_client(config=llm_config)

    judge_config = JudgeConfig(
        temperature=float(judge_cfg.get("temperature", 0.3)),
        max_tokens=int(judge_cfg.get("max_tokens", 2048)),
    )
    return JudgeService(llm_client, judge_config)


def build_judge_rubric_reward(
    rubric_config: Dict[str, Any],
    base_dir: Path,
) -> Callable:
    """Build a TRL-compatible reward function that scores via shared/judge.

    Args:
        rubric_config: Loaded YAML config for the judge_rubric reward. Expected:
            - rubrics_dir (str): Directory containing rubric YAMLs (compatible
              with shared.judge.RubricLoader).
            - rubric_keys (list[str]): Filename stems (without .yaml) to load.
            - judge (dict): Judge LLM config; requires 'backend' and 'model'.
              Optional: 'temperature', 'max_tokens', and any matching
              shared.llm.LLMConfig field (e.g. 'lmstudio_host', 'ollama_port',
              'provider_routing').
            - aggregation (str): One of mean_score | mean_passed | min_score |
              all_pass. Default: mean_passed.
        base_dir: Base directory for resolving relative paths.

    Returns:
        Reward fn with signature (completions, prompts=None, **kwargs) -> List[float]
    """
    rubrics_dir_raw = rubric_config.get("rubrics_dir")
    rubric_keys = rubric_config.get("rubric_keys") or []
    judge_cfg = rubric_config.get("judge") or {}
    aggregation = rubric_config.get("aggregation", "mean_passed")

    if not rubrics_dir_raw:
        raise ValueError("judge_rubric reward requires 'rubrics_dir' in YAML")
    if not rubric_keys:
        raise ValueError(
            "judge_rubric reward requires non-empty 'rubric_keys' in YAML"
        )
    if aggregation not in _AGGREGATIONS:
        raise ValueError(
            f"judge_rubric 'aggregation' must be one of {_AGGREGATIONS}, "
            f"got '{aggregation}'"
        )

    rubrics_dir = _resolve_dir(rubrics_dir_raw, base_dir)
    loader = RubricLoader(rubrics_dir)
    rubrics: List[RubricDef] = loader.load_many(list(rubric_keys))

    judge_service = _build_judge_service(judge_cfg)

    logger.info(
        "judge_rubric reward initialized: %d rubric(s) from %s, backend=%s, "
        "aggregation=%s",
        len(rubrics),
        rubrics_dir,
        judge_cfg.get("backend"),
        aggregation,
    )

    def _reward(completions, prompts=None, **kwargs):
        rewards: List[float] = []
        n = len(completions)
        if isinstance(prompts, list):
            prompts_list = prompts
        else:
            prompts_list = [prompts] * n

        for i, completion in enumerate(completions):
            response_text = _coerce_completion_text(completion)
            prompt_value = prompts_list[i] if i < len(prompts_list) else prompts_list[-1]
            prompt_text = _coerce_prompt_text(prompt_value)

            template_vars = {
                "prompt": prompt_text,
                "user_prompt": prompt_text,
                "response": response_text,
            }
            rendered = render_combined_prompt(rubrics, template_vars)

            result = judge_service.judge(prompt=rendered, rubrics=rubrics)
            if result.error:
                logger.warning("judge_rubric judge call failed: %s", result.error)
                rewards.append(0.0)
                continue

            rewards.append(aggregate(result, aggregation))

        return rewards

    return _reward
