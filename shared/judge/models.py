"""Data model definitions for the shared judge module.

Location: shared/judge/models.py
Summary: Typed dataclasses for rubric definitions, judge scores, aggregate
         results, and judge execution configuration. These models are generic
         and consumed by both the Evaluator and SynthChat pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RubricDef:
    """A loaded rubric definition from YAML.

    Attributes:
        key: Filename stem used as identifier (e.g., "tool_call_quality").
        name: Human-readable display name.
        description: What this rubric evaluates.
        scope: Which part of the response to judge (e.g., "response", "tool_call").
        pass_threshold: Score >= this means pass (0.0-1.0).
        judge_prompt: Prompt template with {variables} for the judge LLM.
        output_schema: JSON Schema dict for structured LLM output.
        improver_prompt: Prompt template for improvement (SynthChat only, None for Evaluator).
    """

    key: str
    name: str
    description: str
    scope: str
    pass_threshold: float
    judge_prompt: str
    output_schema: Dict[str, Any]
    improver_prompt: Optional[str] = None


@dataclass
class JudgeScore:
    """Score from a single rubric evaluation.

    Attributes:
        rubric_key: Identifier matching the RubricDef key.
        rubric_name: Human-readable rubric name.
        score: Numeric score from 0.0 to 1.0.
        passed: Whether score >= pass_threshold.
        pass_threshold: The threshold used for this rubric.
        feedback: LLM's explanation text (from the feedback field).
    """

    rubric_key: str
    rubric_name: str
    score: float
    passed: bool
    pass_threshold: float
    feedback: Optional[str] = None


@dataclass
class JudgeResult:
    """Aggregate result from judging across all rubrics.

    Attributes:
        passed: Overall pass (all rubrics passed).
        scores: Per-rubric JudgeScore instances.
        raw_output: Raw structured output dict from the LLM.
        error: Error message if the judge call failed.
        latency_s: Judge call latency in seconds.
    """

    passed: bool
    scores: List[JudgeScore] = field(default_factory=list)
    raw_output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    latency_s: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict for JSON output.

        Follows the same pattern as BehaviorValidationResult and
        ValidationResult for consistent serialization across the pipeline.
        """
        return {
            "passed": self.passed,
            "scores": [
                {
                    "rubric_key": s.rubric_key,
                    "rubric_name": s.rubric_name,
                    "score": s.score,
                    "passed": s.passed,
                    "pass_threshold": s.pass_threshold,
                    "feedback": s.feedback,
                }
                for s in self.scores
            ],
            "raw_output": self.raw_output,
            "error": self.error,
            "latency_s": self.latency_s,
        }


@dataclass
class JudgeConfig:
    """Configuration for judge execution behavior.

    This configures how the judge interprets LLM output, not LLM connectivity
    (which comes from shared/llm/LLMConfig).

    Attributes:
        feedback_field: Name of the feedback field in the combined schema.
        score_field_suffix: Suffix appended to rubric keys to form score field names.
        temperature: Sampling temperature for judge calls (low for consistency).
        max_tokens: Maximum tokens for judge LLM response.
    """

    feedback_field: str = "overall_feedback"
    score_field_suffix: str = "_score"
    temperature: float = 0.3
    max_tokens: int = 2048
