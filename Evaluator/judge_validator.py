"""Evaluator-specific wrapper around shared/judge.

Location: Evaluator/judge_validator.py
Summary: Bridges the generic shared/judge module with the Evaluator's specific
         context. Renders judge prompt templates with evaluator-specific variables
         (response, system_prompt, user_prompt, tool_calls, expected_tools, pass_fail,
         thinking_content), coordinates judge calls via JudgeService, and logs
         interactions for KTO training. This is the only Evaluator-specific judge
         class -- everything else lives in shared/judge/.

Used by: Evaluator/runner.py (_evaluate_single_case) when --judge is enabled.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from shared.judge import (
    InteractionLogger,
    JudgeConfig,
    JudgeResult,
    JudgeService,
    RubricDef,
)
from shared.llm import BaseLLMClient
from shared.validation.parsing import ParsedResponse

logger = logging.getLogger(__name__)


@dataclass
class JudgeValidationResult:
    """Result of judge validation for a single evaluation case.

    Attributes:
        judge_result: Scores and pass/fail from shared/judge.
        judge_mode: Composition mode ("and", "or", "judge_only") that was used.
    """

    judge_result: JudgeResult
    judge_mode: str

    @property
    def passed(self) -> bool:
        """Whether the judge considered this case as passed."""
        return self.judge_result.passed

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict for JSON output."""
        return {
            "judge_mode": self.judge_mode,
            **self.judge_result.to_dict(),
        }


class JudgeValidator:
    """Evaluator-specific wrapper around shared/judge.

    Responsibilities:
    - Render judge prompt templates with evaluator-specific variables
    - Coordinate the judge call via JudgeService
    - Log interactions for KTO training

    Args:
        llm_client: LLM client for judge calls (may differ from eval backend).
        rubrics: Loaded rubric definitions to judge against.
        judge_config: Configuration for judge execution behavior.
        interaction_logger: Optional logger for KTO interaction logging.
        default_judge_mode: Global judge mode from CLI (overridable per-scenario).
        eval_model: Name of the model being evaluated (for interaction logging).
        judge_model: Name of the model used as judge (for interaction logging).
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        rubrics: List[RubricDef],
        judge_config: JudgeConfig,
        interaction_logger: Optional[InteractionLogger] = None,
        default_judge_mode: str = "and",
        eval_model: Optional[str] = None,
        judge_model: Optional[str] = None,
    ):
        self.judge_service = JudgeService(llm_client, judge_config)
        self.rubrics = rubrics
        self.judge_config = judge_config
        self.interaction_logger = interaction_logger
        self.default_judge_mode = default_judge_mode
        self.eval_model = eval_model
        self.judge_model = judge_model

    def validate(
        self,
        parsed_response: ParsedResponse,
        case_metadata: Dict[str, Any],
        judge_mode: Optional[str] = None,
    ) -> JudgeValidationResult:
        """Run judge validation on a parsed response.

        Args:
            parsed_response: Parsed model response (from shared/validation/parsing).
            case_metadata: Scenario metadata containing system prompt, user question,
                          expected tools, pattern match result, etc.
            judge_mode: Per-case override for composition mode. Falls back to
                        default_judge_mode from CLI if None.

        Returns:
            JudgeValidationResult with scores and pass/fail.
        """
        effective_mode = judge_mode or self.default_judge_mode

        # Render the combined judge prompt from all rubrics
        rendered_prompt = self._render_combined_prompt(
            parsed_response, case_metadata
        )

        # Call the judge service
        judge_result = self.judge_service.judge(
            prompt=rendered_prompt,
            rubrics=self.rubrics,
        )

        # Log interaction for KTO training
        if self.interaction_logger is not None and judge_result.raw_output is not None:
            rubric_names = ", ".join(r.name for r in self.rubrics)
            scores_dict = {s.rubric_key: s.score for s in judge_result.scores}
            self.interaction_logger.log_judge_interaction(
                judge_prompt=rendered_prompt,
                judge_response_raw=json.dumps(judge_result.raw_output, ensure_ascii=False),
                rubric_name=rubric_names,
                scores=scores_dict,
                passed=judge_result.passed,
                case_id=case_metadata.get("case_id"),
                judge_mode=effective_mode,
                eval_model=self.eval_model,
                judge_model=self.judge_model,
            )

        return JudgeValidationResult(
            judge_result=judge_result,
            judge_mode=effective_mode,
        )

    def _render_combined_prompt(
        self,
        parsed_response: ParsedResponse,
        case_metadata: Dict[str, Any],
    ) -> str:
        """Render a combined prompt from all rubrics with template variables filled.

        When multiple rubrics are active, their prompts are concatenated with
        clear separators so the judge LLM evaluates all dimensions in one call.

        Args:
            parsed_response: Parsed model response.
            case_metadata: Scenario metadata for template variable values.

        Returns:
            Fully rendered prompt string with all {variables} replaced.
        """
        template_vars = self._build_template_vars(parsed_response, case_metadata)

        if len(self.rubrics) == 1:
            return self._render_single_prompt(self.rubrics[0], template_vars)

        # Multiple rubrics: concatenate with separators
        parts = []
        for rubric in self.rubrics:
            rendered = self._render_single_prompt(rubric, template_vars)
            parts.append(f"=== {rubric.name} ===\n{rendered}")
        return "\n\n".join(parts)

    @staticmethod
    def _render_single_prompt(
        rubric: RubricDef,
        template_vars: Dict[str, str],
    ) -> str:
        """Fill template variables in a single rubric's judge_prompt.

        Uses simple string replacement (str.replace) for each known variable,
        avoiding str.format_map which could allow Python attribute access via
        format strings. Missing variables are left as-is.

        Args:
            rubric: The rubric whose prompt to render.
            template_vars: Dict of variable name -> value.

        Returns:
            Rendered prompt string.
        """
        prompt = rubric.judge_prompt
        for key, val in template_vars.items():
            prompt = prompt.replace(f"{{{key}}}", str(val))
        return prompt

    def _build_template_vars(
        self,
        parsed_response: ParsedResponse,
        case_metadata: Dict[str, Any],
    ) -> Dict[str, str]:
        """Build the template variable dict from evaluator context.

        Template variables available:
        - {response}: Full model response text
        - {system_prompt}: System prompt from the scenario
        - {user_prompt}: User's question/instruction
        - {tool_calls}: Formatted tool calls (name + arguments)
        - {expected_tools}: Comma-separated expected tool names
        - {pass_fail}: Pattern-match result ("PASS" or "FAIL")
        - {thinking_content}: Thinking block content (if present)

        Args:
            parsed_response: Parsed model response.
            case_metadata: Scenario metadata.

        Returns:
            Dict mapping variable names to string values.
        """
        # Format tool calls as readable strings
        tool_calls_str = self._format_tool_calls(parsed_response)

        # Get expected tools from metadata
        expected_tools = case_metadata.get("expected_tools", [])
        if isinstance(expected_tools, list):
            expected_tools_str = ", ".join(expected_tools)
        else:
            expected_tools_str = str(expected_tools)

        # Determine pass/fail from pattern match result
        pass_fail = "PASS" if case_metadata.get("pattern_passed", True) else "FAIL"

        return {
            "response": parsed_response.text_content or "",
            "system_prompt": case_metadata.get("system", ""),
            "user_prompt": case_metadata.get("user_prompt", ""),
            "tool_calls": tool_calls_str,
            "expected_tools": expected_tools_str,
            "pass_fail": pass_fail,
            "thinking_content": parsed_response.thinking or "",
        }

    @staticmethod
    def _format_tool_calls(parsed_response: ParsedResponse) -> str:
        """Format tool calls into a human-readable string.

        Each tool call is formatted as: tool_name(arg1=val1, arg2=val2)

        Args:
            parsed_response: Parsed response containing tool calls.

        Returns:
            Formatted string of tool calls, one per line. Returns
            "(none)" if no tool calls were made.
        """
        if not parsed_response.tool_calls:
            return "(none)"

        lines = []
        for tc in parsed_response.tool_calls:
            # Format arguments as key=value pairs
            if tc.arguments:
                if isinstance(tc.arguments, dict):
                    arg_parts = []
                    for k, v in tc.arguments.items():
                        # Truncate long values for readability
                        v_str = json.dumps(v, ensure_ascii=False)
                        if len(v_str) > 100:
                            v_str = v_str[:100] + "..."
                        arg_parts.append(f"{k}={v_str}")
                    args_str = ", ".join(arg_parts)
                else:
                    args_str = str(tc.arguments)
            else:
                args_str = ""
            lines.append(f"{tc.name}({args_str})")

        return "\n".join(lines)
