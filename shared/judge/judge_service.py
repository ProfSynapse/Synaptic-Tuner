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
            raw_output = self.llm_client.structured_output(
                messages=messages,
                schema=schema,
                temperature=self.judge_config.temperature,
                max_tokens=self.judge_config.max_tokens,
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

        Looks for score fields using the convention: <rubric_key><score_suffix>.
        Falls back to searching for any numeric field containing the rubric key
        if the exact convention doesn't match.

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
            # Try exact convention: rubrickey_score (with underscores normalized)
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
