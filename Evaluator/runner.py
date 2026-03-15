"""Evaluation orchestration logic.

This module provides the core evaluation loop that runs prompt cases
against a backend client and collects validation results.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

from .behavior_validator import BehaviorIssue, BehaviorValidationResult, validate_behavior
from .prompt_sets import PromptCase
from .protocols import BackendClient
from .schema_validator import ValidationResult, ValidatorIssue, validate_assistant_response

try:
    from shared.environments import EnvironmentValidationResult, EnvironmentValidator
except ImportError:
    EnvironmentValidationResult = None
    EnvironmentValidator = None

try:
    from .judge_validator import JudgeValidationResult, JudgeValidator
except ImportError:
    JudgeValidationResult = None
    JudgeValidator = None


@dataclass
class EvaluationRecord:
    """Result of evaluating a single prompt case.

    Attributes:
        case: The prompt case that was evaluated
        response_text: Model's response content (string or dict)
        validator: Schema validation result
        latency_s: Response time in seconds
        raw_response: Complete API response
        error: Error message if request/validation failed
        behavior: Behavior validation result (if expectations defined)
        environment: Environment validation result (if enabled)
        judge: Judge validation result (if --judge enabled)
    """

    case: PromptCase
    response_text: Optional[Any]
    validator: Optional[ValidationResult]
    latency_s: Optional[float]
    raw_response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    behavior: Optional[BehaviorValidationResult] = None
    environment: Optional["EnvironmentValidationResult"] = None
    judge: Optional["JudgeValidationResult"] = None
    scoring: Optional["PathScoringResult"] = None

    @property
    def status(self) -> str:
        """Get evaluation status: 'pass', 'warn', or 'fail'.

        - pass: Tool correct + behavior expectations met (or no behavior expectations)
        - warn: Tool correct + behavior expectations NOT met
        - fail: Tool incorrect OR error

        When judge is enabled, the judge_mode controls how pattern-match and
        judge results combine:
        - "and": Both pattern match AND judge must pass
        - "or": Either pattern match OR judge can pass
        - "judge_only": Only judge result matters for pass/fail
        """
        if self.error is not None:
            return "fail"

        # Determine pattern match result (schema + expected tools)
        pattern_passed = self.validator is not None and self.validator.passed

        # Environment validation is independent -- always fails if it fails
        if self.environment is not None and not self.environment.passed:
            return "fail"

        # Combine pattern match with judge result
        if self.judge is not None:
            judge_passed = self.judge.passed
            mode = self.judge.judge_mode

            if mode == "and":
                if not pattern_passed or not judge_passed:
                    return "fail"
            elif mode == "or":
                if not pattern_passed and not judge_passed:
                    return "fail"
            elif mode == "judge_only":
                if not judge_passed:
                    return "fail"
            else:
                logger.warning("Unknown judge_mode '%s', defaulting to 'and'", mode)
                if not pattern_passed or not judge_passed:
                    return "fail"
        else:
            # No judge -- original behavior
            if not pattern_passed:
                return "fail"

        # Behavior check (advisory: warn, not fail)
        if self.behavior is not None and not self.behavior.passed:
            return "warn"
        return "pass"

    @property
    def passed(self) -> bool:
        """Check if evaluation passed all validations (pass or warn)."""
        return self.status in ("pass", "warn")

    @property
    def failed(self) -> bool:
        """Check if evaluation hard-failed (tool incorrect or error)."""
        return self.status == "fail"

    @property
    def warned(self) -> bool:
        """Check if evaluation warned (tool correct but behavior failed)."""
        return self.status == "warn"

    @property
    def schema_passed(self) -> bool:
        """Check if schema validation passed (ignoring behavior)."""
        return self.error is None and self.validator is not None and self.validator.passed

    @property
    def behavior_passed(self) -> bool:
        """Check if behavior validation passed (or not applicable)."""
        return self.behavior is None or self.behavior.passed

    @property
    def judge_passed(self) -> bool:
        """Check if judge validation passed (or not applicable)."""
        return self.judge is None or self.judge.passed

    @property
    def score(self) -> Optional[float]:
        """Normalized config-driven score, if present."""
        return self.scoring.normalized_score if self.scoring is not None else None


@dataclass
class PathScoreMatch:
    """Match result for one configured scoring path."""

    name: str
    tier: str
    score: float
    matched: bool
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tier": self.tier,
            "score": self.score,
            "matched": self.matched,
            "reasons": list(self.reasons),
        }


@dataclass
class PathScoringResult:
    """Aggregate score for a case with preferred/acceptable paths."""

    max_score: float
    awarded_score: float
    matched_path: Optional[str] = None
    matched_tier: Optional[str] = None
    matches: List[PathScoreMatch] = field(default_factory=list)

    @property
    def normalized_score(self) -> float:
        if self.max_score <= 0:
            return 0.0
        return self.awarded_score / self.max_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_score": self.max_score,
            "awarded_score": self.awarded_score,
            "normalized_score": self.normalized_score,
            "matched_path": self.matched_path,
            "matched_tier": self.matched_tier,
            "matches": [match.to_dict() for match in self.matches],
        }


def evaluate_cases(
    cases: Sequence[PromptCase],
    client: BackendClient,
    dry_run: bool = False,
    on_record: Callable[[EvaluationRecord], None] | None = None,
    validate_context: bool = False,
    environment_validator: "EnvironmentValidator" | None = None,
    judge_validator: "JudgeValidator" | None = None,
) -> List[EvaluationRecord]:
    """Run evaluation for the provided prompts.

    Args:
        cases: Prompt cases to evaluate
        client: Backend client implementing BackendClient protocol
        dry_run: Skip backend calls (for testing)
        on_record: Optional callback for each completed record
        validate_context: If True, validate that model uses IDs from
                         expected_context in prompt metadata
        environment_validator: Optional environment validator for
                              runtime-backed tool execution checks
        judge_validator: Optional LLM-as-judge validator for semantic evaluation

    Returns:
        List of evaluation records
    """
    records: List[EvaluationRecord] = []

    for case in cases:
        record = _evaluate_single_case(
            case,
            client,
            dry_run,
            validate_context,
            environment_validator=environment_validator,
            judge_validator=judge_validator,
        )
        records.append(record)
        if on_record:
            on_record(record)

    return records


def _evaluate_single_case(
    case: PromptCase,
    client: BackendClient,
    dry_run: bool,
    validate_context: bool = False,
    environment_validator: "EnvironmentValidator" | None = None,
    judge_validator: "JudgeValidator" | None = None,
) -> EvaluationRecord:
    """Evaluate a single prompt case.

    Args:
        case: The prompt case to evaluate
        client: Backend client
        dry_run: Skip backend calls
        validate_context: If True, validate IDs against expected_context in metadata
        judge_validator: Optional LLM-as-judge validator

    Returns:
        EvaluationRecord with results
    """
    # Handle dry run
    if dry_run:
        return EvaluationRecord(
            case=case,
            response_text=None,
            validator=None,
            latency_s=None,
            raw_response=None,
            error=None,
        )

    # Get expected_context from prompt metadata (if validation enabled)
    # The system prompt is already in case.metadata["system"] and will be
    # included via case.chat_messages()
    eval_context = None
    if validate_context:
        eval_context = case.metadata.get("expected_context")

    # Make request to backend
    try:
        response = client.chat(case.chat_messages())
    except Exception as exc:
        return EvaluationRecord(
            case=case,
            response_text=None,
            validator=None,
            latency_s=None,
            raw_response=None,
            error=str(exc),
        )

    # Run schema validation (with optional context validation)
    try:
        validator_result = validate_assistant_response(response.message, eval_context)
    except Exception as exc:
        return EvaluationRecord(
            case=case,
            response_text=response.message,
            validator=None,
            latency_s=response.latency_s,
            raw_response=response.raw,
            error=f"Validation error: {exc}",
        )

    # Check expected tools
    _check_expected_tools(case, validator_result)

    # Run behavior validation - pass extracted tool names from schema validation
    extracted_tool_names = [tc.name for tc in validator_result.tool_calls]
    behavior_result = _run_behavior_validation(case, response.message, extracted_tool_names)

    environment_result = None
    if environment_validator is not None:
        try:
            system_prompt = case.metadata.get("system", "")
            environment_config = case.metadata.get("environment") or {}
            expected_for_env = case.expected_tools if environment_config.get("require_expected_tools") else None
            environment_result = environment_validator.validate_response(
                system_prompt=system_prompt,
                response=response.message,
                environment_config=environment_config,
                expected_tools=expected_for_env,
            )
        except Exception as exc:
            # Environment validation should fail this case, but we keep schema/behavior data.
            if EnvironmentValidationResult is not None:
                from shared.environments import EnvironmentIssue

                environment_result = EnvironmentValidationResult(
                    passed=False,
                    issues=[EnvironmentIssue(level="error", message=f"Environment validation error: {exc}")],
                )
            else:
                return EvaluationRecord(
                    case=case,
                    response_text=response.message,
                    validator=validator_result,
                    latency_s=response.latency_s,
                    raw_response=response.raw,
                    error=f"Environment validation error: {exc}",
                    behavior=behavior_result,
                )

    # Run judge validation if enabled
    judge_result = None
    if judge_validator is not None:
        # Determine judge_mode: per-test override > global CLI default
        judge_meta = case.metadata.get("judge", {})
        per_case_mode = judge_meta.get("mode") if judge_meta else None
        effective_mode = per_case_mode or judge_validator.default_judge_mode

        # AND optimization: skip judge if pattern match already fails in "and" mode
        pattern_passed = validator_result is not None and validator_result.passed
        skip_judge = (effective_mode == "and" and not pattern_passed)

        if not skip_judge:
            try:
                from shared.validation.parsing import parse_response

                parsed = parse_response(response.message)

                # Build case metadata for template rendering
                case_meta = {
                    "system": case.metadata.get("system", ""),
                    "user_prompt": case.question,
                    "expected_tools": case.expected_tools or case.acceptable_tools or [],
                    "pattern_passed": pattern_passed,
                    "case_id": case.case_id,
                }

                judge_result = judge_validator.validate(
                    parsed_response=parsed,
                    case_metadata=case_meta,
                    judge_mode=per_case_mode,
                )
            except Exception as exc:
                # Judge failure should not crash the evaluation
                logger.error("Judge validation error for %s: %s", case.case_id, exc)

    scoring_result = _run_path_scoring(
        case=case,
        validator_result=validator_result,
        behavior_result=behavior_result,
        environment_result=environment_result,
        judge_result=judge_result,
    )

    return EvaluationRecord(
        case=case,
        response_text=response.message,
        validator=validator_result,
        latency_s=response.latency_s,
        raw_response=response.raw,
        error=None,
        behavior=behavior_result,
        environment=environment_result,
        judge=judge_result,
        scoring=scoring_result,
    )


def _check_expected_tools(case: PromptCase, validator_result: ValidationResult) -> None:
    """Check if expected tools were called, updating validator_result in place.

    Supports two modes:
    - expected_tools: AND logic - ALL listed tools must be called (primary expectation)
    - acceptable_tools: OR logic - ANY listed tool is valid as an alternative (includes TEXT_ONLY)

    When both are defined, expected_tools are ALSO acceptable (they're what we want!).
    acceptable_tools provides additional alternatives beyond the expected ones.
    """
    called_tool_names = {tc.name for tc in validator_result.tool_calls}
    has_tool_calls = len(called_tool_names) > 0

    # Build the full set of acceptable tools
    # expected_tools are always acceptable (they're what we want)
    # acceptable_tools provides additional alternatives
    all_acceptable = set()
    if case.expected_tools:
        all_acceptable.update(case.expected_tools)
    if case.acceptable_tools:
        all_acceptable.update(case.acceptable_tools)

    # If we have acceptable alternatives (OR logic)
    if all_acceptable:
        # TEXT_ONLY is a pseudo-tool meaning no tool call is acceptable
        text_only_acceptable = "TEXT_ONLY" in all_acceptable
        tool_options = [t for t in all_acceptable if t != "TEXT_ONLY"]

        # Check if any acceptable tool was called OR text-only is valid
        any_acceptable_called = bool(called_tool_names & set(tool_options))
        text_only_response = not has_tool_calls and text_only_acceptable

        if not any_acceptable_called and not text_only_response:
            validator_result.passed = False
            options_str = ", ".join(sorted(all_acceptable))
            validator_result.issues.append(
                ValidatorIssue(
                    level="error",
                    message=f"No acceptable tool called. Valid options: {options_str}"
                )
            )


def _run_behavior_validation(
    case: PromptCase,
    response: Any,
    extracted_tool_names: List[str] = None,
) -> Optional[BehaviorValidationResult]:
    """Run behavior validation if expectations are defined.

    Args:
        case: The prompt case with potential behavior expectations
        response: Model's response
        extracted_tool_names: Tool names already extracted by schema validation
                             (with useTools expansion applied)

    Returns:
        BehaviorValidationResult or None if no expectations defined
    """
    behavior_expectations = case.metadata.get("behavior_expectations")
    expected_response_type = case.metadata.get("expected_response_type")
    anti_patterns = case.metadata.get("anti_patterns_to_avoid")

    # Skip if no behavior expectations defined
    if not (behavior_expectations or expected_response_type or anti_patterns):
        return None

    try:
        return validate_behavior(
            response=response,
            behavior_expectations=behavior_expectations,
            expected_response_type=expected_response_type,
            anti_patterns=anti_patterns,
            extracted_tool_names=extracted_tool_names,
        )
    except Exception as exc:
        # Return error result instead of failing completely
        return BehaviorValidationResult(
            passed=False,
            issues=[BehaviorIssue(
                check="validation_error",
                expected="successful validation",
                actual=str(exc),
                passed=False,
                message=f"Behavior validation error: {exc}"
            )]
        )


def _run_path_scoring(
    *,
    case: PromptCase,
    validator_result: Optional[ValidationResult],
    behavior_result: Optional[BehaviorValidationResult],
    environment_result: Optional["EnvironmentValidationResult"],
    judge_result: Optional["JudgeValidationResult"],
) -> Optional[PathScoringResult]:
    scoring_cfg = case.metadata.get("scoring")
    if not isinstance(scoring_cfg, dict):
        return None

    paths = scoring_cfg.get("paths")
    if not isinstance(paths, list) or not paths:
        return None

    tool_names = [tc.name for tc in (validator_result.tool_calls if validator_result else [])]
    matches: List[PathScoreMatch] = []
    max_score = 0.0
    best_score = 0.0
    best_name = None
    best_tier = None

    for idx, path_cfg in enumerate(paths, start=1):
        if not isinstance(path_cfg, dict):
            continue

        score_value = float(path_cfg.get("score", 0.0) or 0.0)
        max_score = max(max_score, score_value)
        matched, reasons = _matches_scoring_path(
            path_cfg=path_cfg,
            tool_names=tool_names,
            validator_result=validator_result,
            behavior_result=behavior_result,
            environment_result=environment_result,
            judge_result=judge_result,
        )
        name = str(path_cfg.get("name") or f"path_{idx}")
        tier = str(path_cfg.get("tier") or "acceptable")
        matches.append(
            PathScoreMatch(
                name=name,
                tier=tier,
                score=score_value,
                matched=matched,
                reasons=reasons,
            )
        )

        if matched and score_value >= best_score:
            best_score = score_value
            best_name = name
            best_tier = tier

    return PathScoringResult(
        max_score=max_score,
        awarded_score=best_score,
        matched_path=best_name,
        matched_tier=best_tier,
        matches=matches,
    )


def _matches_scoring_path(
    *,
    path_cfg: Dict[str, Any],
    tool_names: List[str],
    validator_result: Optional[ValidationResult],
    behavior_result: Optional[BehaviorValidationResult],
    environment_result: Optional["EnvironmentValidationResult"],
    judge_result: Optional["JudgeValidationResult"],
) -> tuple[bool, List[str]]:
    reasons: List[str] = []

    all_tools = _string_list(path_cfg.get("all_tools"))
    if all_tools:
        missing = [tool for tool in all_tools if tool not in tool_names]
        if missing:
            reasons.append(f"missing tools: {', '.join(missing)}")

    any_tools = _string_list(path_cfg.get("any_tools"))
    if any_tools and not any(tool in tool_names for tool in any_tools):
        reasons.append(f"needs any of: {', '.join(any_tools)}")

    ordered_tools = _string_list(path_cfg.get("ordered_tools"))
    if ordered_tools and not _contains_subsequence(tool_names, ordered_tools):
        reasons.append(f"ordered tools not matched: {', '.join(ordered_tools)}")

    first_tool = str(path_cfg.get("first_tool", "")).strip()
    if first_tool and (not tool_names or tool_names[0] != first_tool):
        reasons.append(f"first tool should be {first_tool}")

    first_tool_any_of = _string_list(path_cfg.get("first_tool_any_of"))
    if first_tool_any_of and (not tool_names or tool_names[0] not in first_tool_any_of):
        reasons.append(f"first tool should be one of: {', '.join(first_tool_any_of)}")

    max_tool_calls = path_cfg.get("max_tool_calls")
    if max_tool_calls is not None and len(tool_names) > int(max_tool_calls):
        reasons.append(f"too many tool calls: {len(tool_names)} > {int(max_tool_calls)}")

    min_tool_calls = path_cfg.get("min_tool_calls")
    if min_tool_calls is not None and len(tool_names) < int(min_tool_calls):
        reasons.append(f"too few tool calls: {len(tool_names)} < {int(min_tool_calls)}")

    if path_cfg.get("require_schema_pass") and not (validator_result and validator_result.passed):
        reasons.append("schema validation did not pass")

    if path_cfg.get("require_behavior_pass") and not (behavior_result is None or behavior_result.passed):
        reasons.append("behavior validation did not pass")

    if path_cfg.get("require_environment_pass") and not (environment_result and environment_result.passed):
        reasons.append("environment validation did not pass")

    if path_cfg.get("require_judge_pass") and not (judge_result and judge_result.passed):
        reasons.append("judge validation did not pass")

    return len(reasons) == 0, reasons


def _contains_subsequence(items: List[str], subsequence: List[str]) -> bool:
    if not subsequence:
        return True
    pos = 0
    for item in items:
        if item == subsequence[pos]:
            pos += 1
            if pos == len(subsequence):
                return True
    return False


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        clean = value.strip()
        return [clean] if clean else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []
