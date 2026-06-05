"""LLM-judge verifier: score completions via the shared judge service.

Registered under the ``llm_judge`` type. This module is the CANONICAL home for
two pieces of logic that currently also live (as a copy to be deleted in a
later phase) in ``Trainers/grpo/src/judge_reward.py``:

- :func:`aggregate` — collapse a :class:`shared.judge.JudgeResult` into a single
  scalar via one of ``mean_score`` / ``mean_passed`` / ``min_score`` /
  ``all_pass``.
- :func:`render_combined_prompt` — concatenate per-rubric judge prompts after
  template-variable substitution.

The verifier loads rubrics via :class:`shared.judge.RubricLoader`, renders the
combined prompt, runs a :class:`shared.judge.JudgeService`, and returns the
aggregated score. The ``JudgeService`` (or any object exposing a compatible
``judge(prompt, rubrics, ...)`` method) may be INJECTED via params so tests do
not hit the network; no LLM client is constructed at import time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping

from shared.judge import JudgeResult, RubricDef, RubricLoader

from ..contract import VerifierInput, VerifierOutput
from ..registry import register

AGGREGATIONS = ("mean_score", "mean_passed", "min_score", "all_pass")


def aggregate(result: JudgeResult, strategy: str) -> float:
    """Collapse a :class:`JudgeResult` into a single scalar.

    This is the single source of truth for judge aggregation. It reproduces,
    exactly, the four strategies from ``judge_reward._aggregate``.

    Args:
        result: The judge result whose ``scores`` are aggregated.
        strategy: One of ``mean_score``, ``mean_passed``, ``min_score``,
            ``all_pass``.

    Returns:
        Aggregated score in ``[0.0, 1.0]``. Returns ``0.0`` when there are no
        scores.

    Raises:
        ValueError: If ``strategy`` is unknown.
    """
    if not result.scores:
        return 0.0
    if strategy == "mean_score":
        return sum(s.score for s in result.scores) / len(result.scores)
    if strategy == "mean_passed":
        return sum(1.0 if s.passed else 0.0 for s in result.scores) / len(result.scores)
    if strategy == "min_score":
        return min(s.score for s in result.scores)
    if strategy == "all_pass":
        return 1.0 if all(s.passed for s in result.scores) else 0.0
    raise ValueError(
        f"Unknown aggregation '{strategy}'. Must be one of: {', '.join(AGGREGATIONS)}"
    )


def render_combined_prompt(
    rubrics: List[RubricDef],
    template_vars: Dict[str, str],
) -> str:
    """Concatenate per-rubric ``judge_prompt`` templates after substituting vars.

    Mirrors ``judge_reward._render_combined_prompt``: a single rubric renders
    bare; multiple rubrics are separated with ``=== <name> ===`` headers.
    """
    parts: List[str] = []
    for rubric in rubrics:
        rendered = rubric.judge_prompt
        for k, v in template_vars.items():
            rendered = rendered.replace(f"{{{k}}}", str(v))
        if len(rubrics) == 1:
            parts.append(rendered)
        else:
            parts.append(f"=== {rubric.name} ===\n{rendered}")
    return "\n\n".join(parts)


def _resolve_dir(value: str, base_dir: Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base_dir / p).resolve()


class LLMJudgeVerifier:
    """Verifier that scores completions via an LLM judge.

    Args:
        name: Verifier name.
        rubrics: Loaded rubric definitions.
        judge_service: An object exposing ``judge(prompt, rubrics)`` returning a
            :class:`JudgeResult`. Injected so tests avoid network calls.
        aggregation: Aggregation strategy (see :func:`aggregate`).
        pass_threshold: Threshold for the ``passed`` flag.
    """

    def __init__(
        self,
        name: str,
        rubrics: List[RubricDef],
        judge_service: Any,
        aggregation: str = "mean_passed",
        pass_threshold: float = 0.5,
    ):
        if aggregation not in AGGREGATIONS:
            raise ValueError(
                f"llm_judge 'aggregation' must be one of {AGGREGATIONS}, "
                f"got '{aggregation}'"
            )
        self.name = name
        self.rubrics = rubrics
        self.judge_service = judge_service
        self.aggregation = aggregation
        self.pass_threshold = pass_threshold

    def verify(self, sample: VerifierInput) -> VerifierOutput:
        template_vars = {
            "prompt": sample.prompt_text,
            "user_prompt": sample.prompt_text,
            "response": sample.completion_text,
        }
        rendered = render_combined_prompt(self.rubrics, template_vars)

        result = self.judge_service.judge(prompt=rendered, rubrics=self.rubrics)

        if getattr(result, "error", None):
            return VerifierOutput(
                score=0.0,
                passed=False,
                detail={"error": result.error, "aggregation": self.aggregation},
            )

        score = aggregate(result, self.aggregation)
        return VerifierOutput(
            score=score,
            passed=score >= self.pass_threshold,
            detail={
                "aggregation": self.aggregation,
                "scores": [
                    {
                        "rubric_key": s.rubric_key,
                        "score": s.score,
                        "passed": s.passed,
                    }
                    for s in result.scores
                ],
            },
        )


@register("llm_judge")
def _build_llm_judge(spec: Mapping) -> LLMJudgeVerifier:
    params = spec.get("params", spec)

    judge_service = params.get("judge_service")
    if judge_service is None:
        raise ValueError(
            "llm_judge verifier requires an injected 'judge_service'. "
            "Constructing an LLM client at build time is not supported here; "
            "build the JudgeService externally and inject it."
        )

    rubrics = params.get("rubrics")
    if rubrics is None:
        rubrics_dir_raw = params.get("rubrics_dir")
        rubric_keys = params.get("rubric_keys") or []
        if not rubrics_dir_raw:
            raise ValueError(
                "llm_judge verifier requires either inline 'rubrics' or "
                "'rubrics_dir' + 'rubric_keys'"
            )
        if not rubric_keys:
            raise ValueError("llm_judge verifier requires non-empty 'rubric_keys'")
        base_dir = Path(params.get("base_dir", "."))
        rubrics_dir = _resolve_dir(str(rubrics_dir_raw), base_dir)
        loader = RubricLoader(rubrics_dir)
        rubrics = loader.load_many(list(rubric_keys))

    return LLMJudgeVerifier(
        name=spec.get("name", "llm_judge"),
        rubrics=list(rubrics),
        judge_service=judge_service,
        aggregation=params.get("aggregation", "mean_passed"),
        pass_threshold=params.get("pass_threshold", 0.5),
    )
