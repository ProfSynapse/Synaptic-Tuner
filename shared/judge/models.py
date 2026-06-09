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
        dimensions: Optional machine-readable per-dimension weights. When present,
            the judge emits a bare per-dimension {reasoning, score} for each entry
            and OUR code (JudgeService._parse_scores) computes the weighted composite
            score = sum(weight_i * score_i); the judge LLM does NOT compute any
            composite. Each entry is a dict with keys: ``key`` (matches the
            output_schema dimension property name), ``name`` (human-readable), and
            ``weight`` (float). Weights are expected to sum to 1.0. None for legacy
            single-composite rubrics, which fall back to the ``<key>_score`` path.
        weights_ratified: When True, the dimension weights are a ratified, stable
            contract and a sum != 1.0 (beyond tolerance) is a hard error. When
            False/None (placeholder rubrics pending client sign-off), a non-unit
            weight sum is normalized-and-warned instead of failing the run.
    """

    key: str
    name: str
    description: str
    scope: str
    pass_threshold: float
    judge_prompt: str
    output_schema: Dict[str, Any]
    improver_prompt: Optional[str] = None
    dimensions: Optional[List[Dict[str, Any]]] = None
    weights_ratified: bool = False


@dataclass
class JudgeScore:
    """Score from a single rubric evaluation.

    Attributes:
        rubric_key: Identifier matching the RubricDef key.
        rubric_name: Human-readable rubric name.
        score: Numeric score from 0.0 to 1.0. For a dimensioned rubric this is the
            weighted composite our code computes from the per-dimension scores
            (the SELECTION/scoring view); for a legacy single-composite rubric it
            is the bare ``<key>_score`` the judge emitted.
        passed: Whether score >= pass_threshold.
        pass_threshold: The threshold used for this rubric.
        feedback: LLM's explanation text (from the feedback field).
        per_dimension: For a dimensioned rubric, the per-dimension AUDIT view that
            must survive in the record for editorial review and judge-human
            calibration: an ordered list of dicts, one per dimension, each carrying
            ``key``, ``name``, ``weight``, ``reasoning`` (the judge's reason-first
            justification), and ``score`` (the bare per-dimension score). None for
            legacy single-composite rubrics.
    """

    rubric_key: str
    rubric_name: str
    score: float
    passed: bool
    pass_threshold: float
    feedback: Optional[str] = None
    per_dimension: Optional[List[Dict[str, Any]]] = None


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
                    "per_dimension": s.per_dimension,
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
        temperature: Sampling temperature for judge calls (low for consistency);
            None omits it from the request (required for gpt-5-family reasoning
            models, which reject the temperature parameter entirely).
        max_tokens: Maximum tokens for judge LLM response.
        reasoning_effort: gpt-5-family reasoning effort (minimal|low|medium|high)
            for judge calls; "minimal" by default. None omits the reasoning field.
    """

    feedback_field: str = "overall_feedback"
    score_field_suffix: str = "_score"
    temperature: Optional[float] = None
    max_tokens: int = 2048
    reasoning_effort: Optional[str] = "minimal"

    def __post_init__(self) -> None:
        valid = ("minimal", "low", "medium", "high")
        if self.reasoning_effort is not None and self.reasoning_effort not in valid:
            raise ValueError(
                f"reasoning_effort must be one of {valid} or None, "
                f"got {self.reasoning_effort!r}"
            )
