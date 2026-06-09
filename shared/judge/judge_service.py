"""Judge service -- execute LLM judge calls and parse results.

Location: shared/judge/judge_service.py
Summary: Executes a single LLM judge call via BaseLLMClient.structured_output(),
         parses the raw structured output into typed JudgeResult with per-rubric
         JudgeScore instances. On LLM errors, returns a failed JudgeResult with
         the error message instead of raising. Used by both Evaluator (via
         JudgeValidator) and SynthChat consumers.
"""

import logging
import time
from typing import Dict, List, Optional

from shared.llm import BaseLLMClient

from .models import JudgeConfig, JudgeResult, JudgeScore, RubricDef
from .schema_builder import SchemaBuilder

logger = logging.getLogger(__name__)

# Allowed drift when checking that a dimensioned rubric's weights sum to 1.0.
# Tolerates floating-point accumulation (e.g. 0.35+0.25+0.25+0.15) without
# masking a genuinely mis-summed weight set.
WEIGHT_SUM_TOLERANCE = 1e-6


class JudgeService:
    """Execute judge LLM calls and parse results into JudgeResult.

    Owns a SchemaBuilder instance (composition) to build the combined schema
    from the rubrics passed to each judge() call.

    Args:
        llm_client: A BaseLLMClient implementation for LLM API calls.
        judge_config: Configuration for judge execution behavior.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        judge_config: JudgeConfig,
    ):
        self.llm_client = llm_client
        self.schema_builder = SchemaBuilder(judge_config)
        self.judge_config = judge_config

    def judge(
        self,
        prompt: str,
        rubrics: List[RubricDef],
        system_prompt: Optional[str] = None,
    ) -> JudgeResult:
        """Execute judge call and return structured result.

        Args:
            prompt: Rendered judge prompt (template variables already filled).
            rubrics: Rubrics to judge against (defines schema and thresholds).
            system_prompt: Optional system-level instruction for the judge LLM.

        Returns:
            JudgeResult with scores for each rubric. On LLM error, returns
            JudgeResult(passed=False, error=str(exc)) instead of raising.
        """
        try:
            # Build combined schema from all rubrics
            schema = self.schema_builder.build(rubrics)

            # Assemble messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Call LLM with timing
            start = time.perf_counter()
            # reasoning_effort is only meaningful for the openai_responses
            # provider (gpt-5-family judge); other providers ignore it.
            raw_output = self.llm_client.structured_output(
                messages=messages,
                schema=schema,
                temperature=self.judge_config.temperature,
                max_tokens=self.judge_config.max_tokens,
                reasoning_effort=self.judge_config.reasoning_effort,
            )
            latency = time.perf_counter() - start

            # Parse raw output into per-rubric scores
            scores = self._parse_scores(raw_output, rubrics)
            overall_passed = all(s.passed for s in scores)

            return JudgeResult(
                passed=overall_passed,
                scores=scores,
                raw_output=raw_output,
                error=None,
                latency_s=round(latency, 3),
            )

        except Exception as exc:
            logger.error("Judge call failed: %s", exc)
            return JudgeResult(
                passed=False,
                error=str(exc),
            )

    def _parse_scores(
        self,
        raw_output: Dict,
        rubrics: List[RubricDef],
    ) -> List[JudgeScore]:
        """Extract per-rubric JudgeScore instances from raw LLM output.

        Two contracts are supported per rubric:

        * Dimensioned (rubric.dimensions set): the judge emits a bare
          per-dimension {reasoning, score}; OUR code computes the weighted
          composite score = sum(weight_i * score_i). The composite is the
          JudgeScore.score (SELECTION view) and the per-dimension
          {key,name,weight,reasoning,score} breakdown is preserved on
          JudgeScore.per_dimension (AUDIT view). See _score_dimensioned_rubric.
        * Legacy single-composite (rubric.dimensions None): looks for a score
          field by the <rubric_key><score_suffix> convention (with the
          normalized / substring fallbacks in _find_score_field).

        Args:
            raw_output: Parsed JSON dict from the LLM structured output.
            rubrics: The rubrics that were judged.

        Returns:
            List of JudgeScore instances, one per rubric.
        """
        scores = []
        feedback = raw_output.get(self.judge_config.feedback_field)
        suffix = self.judge_config.score_field_suffix

        for rubric in rubrics:
            if rubric.dimensions:
                scores.append(
                    self._score_dimensioned_rubric(raw_output, rubric, feedback)
                )
                continue

            # Legacy path: single <rubric_key>_score composite emitted by the LLM.
            score_field = self._find_score_field(raw_output, rubric.key, suffix)
            score_value = raw_output.get(score_field, 0.0) if score_field else 0.0

            # Clamp to 0.0-1.0
            score_value = max(0.0, min(1.0, float(score_value)))

            scores.append(
                JudgeScore(
                    rubric_key=rubric.key,
                    rubric_name=rubric.name,
                    score=score_value,
                    passed=score_value >= rubric.pass_threshold,
                    pass_threshold=rubric.pass_threshold,
                    feedback=feedback,
                )
            )

        return scores

    def _score_dimensioned_rubric(
        self,
        raw_output: Dict,
        rubric: RubricDef,
        feedback: Optional[str],
    ) -> JudgeScore:
        """Compute the weighted composite for a dimensioned rubric in OUR code.

        Reads each dimension's bare ``score`` from
        ``raw_output[dimension_key]['score']`` (clamped to 0.0-1.0), computes the
        weighted composite ``sum(weight_i * score_i)`` (also clamped), and
        preserves the per-dimension {key,name,weight,reasoning,score} breakdown on
        the returned JudgeScore.per_dimension for editorial/calibration audit.

        Weight handling per rubric authoring contract: weights MUST be present,
        numeric, and non-negative (a structural error raises ValueError, surfaced
        by judge() as a failed JudgeResult). The weight SUM is expected to be 1.0;
        a rubric that opts into ``weights_ratified: true`` raises on any drift
        beyond WEIGHT_SUM_TOLERANCE (the ratified rubrics are a stable contract),
        while a non-ratified (placeholder) rubric normalizes-and-warns so client
        sign-off can iterate on weights without breaking the run.

        Args:
            raw_output: Parsed JSON dict from the LLM structured output.
            rubric: The dimensioned rubric being scored.
            feedback: The shared overall_feedback string (or None).

        Returns:
            A single JudgeScore carrying the composite (score) plus the
            per_dimension audit breakdown.
        """
        weights = self._validate_dimension_weights(rubric)

        per_dimension = []
        composite = 0.0
        for dimension in rubric.dimensions:
            dim_key = dimension["key"]
            weight = weights[dim_key]
            dim_obj = raw_output.get(dim_key) or {}
            raw_score = dim_obj.get("score", 0.0) if isinstance(dim_obj, dict) else 0.0
            dim_score = max(0.0, min(1.0, float(raw_score)))
            reasoning = dim_obj.get("reasoning") if isinstance(dim_obj, dict) else None

            composite += weight * dim_score
            per_dimension.append(
                {
                    "key": dim_key,
                    "name": dimension.get("name", dim_key),
                    "weight": weight,
                    "reasoning": reasoning,
                    "score": dim_score,
                }
            )

        composite = max(0.0, min(1.0, composite))

        return JudgeScore(
            rubric_key=rubric.key,
            rubric_name=rubric.name,
            score=composite,
            passed=composite >= rubric.pass_threshold,
            pass_threshold=rubric.pass_threshold,
            feedback=feedback,
            per_dimension=per_dimension,
        )

    def _validate_dimension_weights(self, rubric: RubricDef) -> Dict[str, float]:
        """Validate dimension weights and return a {key: weight} map.

        Raises ValueError on structural errors (missing/duplicate keys, missing,
        non-numeric, or negative weights) for ANY rubric. For the weight SUM:
        a ``weights_ratified: true`` rubric raises on drift beyond
        WEIGHT_SUM_TOLERANCE; otherwise weights are normalized to sum to 1.0 with
        a warning (placeholder rubrics pending client sign-off).
        """
        weights: Dict[str, float] = {}
        for dimension in rubric.dimensions:
            key = dimension.get("key")
            if not key:
                raise ValueError(
                    f"Rubric '{rubric.key}' dimension missing 'key': {dimension!r}"
                )
            if key in weights:
                raise ValueError(
                    f"Rubric '{rubric.key}' has duplicate dimension key '{key}'"
                )
            if "weight" not in dimension:
                raise ValueError(
                    f"Rubric '{rubric.key}' dimension '{key}' missing 'weight'"
                )
            try:
                weight = float(dimension["weight"])
            except (TypeError, ValueError):
                raise ValueError(
                    f"Rubric '{rubric.key}' dimension '{key}' weight is not numeric: "
                    f"{dimension['weight']!r}"
                )
            if weight < 0:
                raise ValueError(
                    f"Rubric '{rubric.key}' dimension '{key}' weight is negative: "
                    f"{weight}"
                )
            weights[key] = weight

        if not weights:
            raise ValueError(f"Rubric '{rubric.key}' has no dimension weights")

        total = sum(weights.values())
        if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
            if rubric.weights_ratified:
                raise ValueError(
                    f"Rubric '{rubric.key}' ratified dimension weights sum to "
                    f"{total}, expected 1.0 (within {WEIGHT_SUM_TOLERANCE})"
                )
            if total <= 0:
                raise ValueError(
                    f"Rubric '{rubric.key}' dimension weights sum to {total}; "
                    f"cannot normalize"
                )
            logger.warning(
                "Rubric '%s' dimension weights sum to %s (expected 1.0); "
                "normalizing (placeholder rubric pending ratification).",
                rubric.key,
                total,
            )
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def _find_score_field(
        self,
        raw_output: Dict,
        rubric_key: str,
        suffix: str,
    ) -> Optional[str]:
        """Find the score field name in the raw output for a given rubric.

        Tries these strategies in order:
        1. Exact match: rubric_key + suffix (e.g., "tool_call_quality_score")
        2. Normalized match: rubric_key without underscores + suffix
        3. Any field ending with suffix that contains the rubric key substring

        Args:
            raw_output: The raw LLM output dict.
            rubric_key: The rubric identifier.
            suffix: The score field suffix (e.g., "_score").

        Returns:
            The matching field name, or None if not found.
        """
        # Strategy 1: Exact convention
        exact_key = f"{rubric_key}{suffix}"
        if exact_key in raw_output:
            return exact_key

        # Strategy 2: Normalized (no underscores)
        normalized_key = f"{rubric_key.replace('_', '')}{suffix}"
        if normalized_key in raw_output:
            return normalized_key

        # Strategy 3: Substring match on score fields
        for field_name in raw_output:
            if field_name.endswith(suffix) and field_name != self.judge_config.feedback_field:
                # Check if rubric key parts appear in the field name
                key_parts = rubric_key.replace("_", "")
                field_normalized = field_name.replace("_", "").replace(suffix, "")
                if key_parts == field_normalized:
                    return field_name

        logger.warning(
            "Could not find score field for rubric '%s' in output keys: %s",
            rubric_key,
            list(raw_output.keys()),
        )
        return None
