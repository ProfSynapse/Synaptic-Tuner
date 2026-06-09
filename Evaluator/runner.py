"""Evaluation orchestration logic.

This module provides the core evaluation loop that runs prompt cases
against a backend client and collects validation results.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

from shared.verifiers.builtins.assertion_verifier import (
    CorrectnessResult,
    evaluate_correctness,
    has_correctness_config,
)
from shared.verifiers.builtins.tool_sequence import evaluate_tool_sequence
from .prompt_sets import PromptCase
from .protocols import BackendClient
from .response_view import build_response_view
from .schema_validator import ToolCall, ValidationResult, ValidatorIssue, validate_assistant_response

try:
    from shared.environments import EnvironmentValidationResult, EnvironmentValidator
except ImportError:
    EnvironmentValidationResult = None
    EnvironmentValidator = None

try:
    from shared.agentic_loop import run_environment_episode
except ImportError:
    run_environment_episode = None

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
        environment: Environment validation result (if enabled)
        judge: Judge validation result (if --judge enabled)
    """

    case: PromptCase
    response_text: Optional[Any]
    validator: Optional[ValidationResult]
    latency_s: Optional[float]
    raw_response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    behavior: Optional[Any] = None
    environment: Optional["EnvironmentValidationResult"] = None
    judge: Optional["JudgeValidationResult"] = None
    scoring: Optional["PathScoringResult"] = None
    correctness: Optional[CorrectnessResult] = None
    conversation_trace: Optional[List[Dict[str, Any]]] = None

    @property
    def status(self) -> str:
        """Get evaluation status: 'pass', 'warn', or 'fail'.

        - pass: configured correctness assertions passed
        - fail: configured correctness assertions, judge, environment, or transport failed

        When judge is enabled, the judge_mode controls how pattern-match and
        judge results combine:
        - "and": Both pattern match AND judge must pass
        - "or": Either pattern match OR judge can pass
        - "judge_only": Only judge result matters for pass/fail
        """
        if self.error is not None:
            return "fail"

        if self.correctness is not None:
            if not self.correctness.passed:
                return "fail"
            if self.environment is not None and not self.environment.passed:
                return "fail"
            if self.judge is not None and not self.judge.passed:
                return "fail"
            return "pass"

        if self.environment is not None and not self.environment.passed:
            return "fail"

        if self.judge is not None:
            return "pass" if self.judge.passed else "fail"

        return "fail"

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
        """Check if evaluation warned."""
        return self.status == "warn"

    @property
    def schema_passed(self) -> bool:
        """Check if schema validation passed (ignoring behavior)."""
        return self.error is None and self.validator is not None and self.validator.passed

    @property
    def behavior_passed(self) -> bool:
        """Behavior validation is not part of the assertion-driven runtime."""
        return True

    @property
    def judge_passed(self) -> bool:
        """Check if judge validation passed (or not applicable)."""
        return self.judge is None or self.judge.passed

    @property
    def correctness_passed(self) -> bool:
        """Check if configured correctness assertions passed."""
        return self.correctness is None or self.correctness.passed

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
    parallel: bool = False,
    max_workers: int = 4,
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

    if not parallel or len(cases) <= 1:
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

    bounded_workers = max(1, min(int(max_workers or 1), len(cases)))
    pending_by_index: Dict[int, EvaluationRecord] = {}
    next_emit_index = 0

    with ThreadPoolExecutor(max_workers=bounded_workers) as executor:
        future_to_index: Dict[Future[EvaluationRecord], int] = {
            executor.submit(
                _evaluate_single_case,
                case,
                client,
                dry_run,
                validate_context,
                environment_validator,
                judge_validator,
            ): index
            for index, case in enumerate(cases)
        }

        while future_to_index:
            done, _ = wait(tuple(future_to_index.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                index = future_to_index.pop(future)
                pending_by_index[index] = future.result()

            while next_emit_index in pending_by_index:
                record = pending_by_index.pop(next_emit_index)
                records.append(record)
                if on_record:
                    on_record(record)
                next_emit_index += 1

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

    environment_config = case.metadata.get("environment") or {}
    loop_cfg = environment_config.get("loop") if isinstance(environment_config.get("loop"), dict) else {}
    loop_enabled = bool(environment_validator is not None and loop_cfg.get("enabled"))

    if loop_enabled and environment_validator is not None:
        return _evaluate_case_with_environment_loop(
            case=case,
            client=client,
            eval_context=eval_context,
            environment_validator=environment_validator,
            judge_validator=judge_validator,
            environment_config=environment_config,
        )

    # Make request to backend.
    # Baseline-fidelity spec §4.3: when the scenario carries a response_schema
    # AND the client supports structured output, route the TARGET call through
    # the provider's json_schema strict path (mirrors production). Absent a
    # schema, behavior is byte-identical to the legacy chat() path — zero
    # regression for every non-structured scenario.
    try:
        response_schema = case.metadata.get("response_schema")
        response_schema_name = case.metadata.get("response_schema_name")
        if response_schema and hasattr(client, "structured_chat"):
            response = client.structured_chat(
                case.chat_messages(), response_schema, response_schema_name
            )
        else:
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
        validator_result = _validate_case_response(case, response.message, eval_context)
    except Exception as exc:
        return EvaluationRecord(
            case=case,
            response_text=response.message,
            validator=None,
            latency_s=response.latency_s,
            raw_response=response.raw,
            error=f"Validation error: {exc}",
        )

    response_view = build_response_view(response.message, response.raw)
    correctness_result = _run_correctness_validation(case, response_view)
    behavior_result = None

    environment_result = None
    if environment_validator is not None:
        try:
            system_prompt = case.metadata.get("system", "")
            environment_result = environment_validator.validate_response(
                system_prompt=system_prompt,
                response=response.message,
                environment_config=environment_config,
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
        correctness=correctness_result,
        conversation_trace=_build_single_turn_trace(case, response.message),
    )


def _evaluate_case_with_environment_loop(
    *,
    case: PromptCase,
    client: BackendClient,
    eval_context: Optional[Dict[str, Any]],
    environment_validator: "EnvironmentValidator",
    judge_validator: "JudgeValidator" | None,
    environment_config: Dict[str, Any],
) -> EvaluationRecord:
    """Evaluate a case with a persistent environment and multi-turn tool loop."""
    loop_cfg = environment_config.get("loop") if isinstance(environment_config.get("loop"), dict) else {}
    max_turns = int(loop_cfg.get("max_turns", 6) or 6)
    max_tool_steps = int(loop_cfg.get("max_tool_steps", environment_config.get("max_steps", 0)) or 0)
    stop_on_text_response = bool(loop_cfg.get("stop_on_text_response", True))
    stop_on_environment_pass = bool(loop_cfg.get("stop_on_environment_pass", False))
    require_final_text = bool(loop_cfg.get("require_final_text", False))
    require_final_text_after_pass = bool(
        loop_cfg.get("require_final_text_after_pass", require_final_text)
    )
    final_text_prompt = loop_cfg.get("final_text_prompt")
    tool_result_format = str(loop_cfg.get("tool_result_format", "json") or "json")
    continue_on_execution_error = bool(
        loop_cfg.get("continue_on_execution_error", str(loop_cfg.get("mode", "strict")).strip().lower() == "agentic")
    )
    stuck_repeat_limit = int(loop_cfg.get("stuck_repeat_limit", 2) or 2)
    no_progress_window = int(loop_cfg.get("no_progress_window", 3) or 3)

    messages = case.chat_messages()
    session = environment_validator.start_session(
        system_prompt=case.metadata.get("system", ""),
        environment_config=environment_config,
    )

    try:
        episode = run_environment_episode(
            initial_messages=messages,
            session=session,
            respond=lambda episode_messages, turn_index: client.chat(episode_messages),
            validate=lambda message: _validate_response_structure(message, eval_context),
            max_turns=max_turns,
            max_tool_steps=max_tool_steps,
            stop_on_text_response=stop_on_text_response,
            stop_on_environment_pass=stop_on_environment_pass,
            continue_on_execution_error=continue_on_execution_error,
            stuck_repeat_limit=stuck_repeat_limit,
            no_progress_window=no_progress_window,
            tool_result_format=tool_result_format,
            stringify_response=_stringify_assistant_response,
            require_final_text_after_pass=require_final_text_after_pass,
            final_text_prompt=final_text_prompt,
        )
    finally:
        session.close()

    turn_validators = [turn.validation for turn in episode.turns]
    final_response = episode.final_response
    final_raw = episode.final_raw
    final_latency = episode.total_latency_s
    environment_result = episode.environment_result
    conversation_trace = episode.conversation_trace

    combined_validator = _combine_validation_results(turn_validators)
    combined_behavior = None
    if require_final_text and combined_validator is not None:
        final_text = _extract_text_content(final_response)
        if not final_text.strip():
            combined_validator.passed = False
            combined_validator.issues.append(
                ValidatorIssue(
                    level="error",
                    message="Loop expected a final text response but none was produced.",
                )
            )

    judge_result = None
    if judge_validator is not None and final_response is not None and combined_validator is not None:
        judge_meta = case.metadata.get("judge", {})
        per_case_mode = judge_meta.get("mode") if judge_meta else None
        effective_mode = per_case_mode or judge_validator.default_judge_mode
        pattern_passed = combined_validator is not None and combined_validator.passed
        skip_judge = effective_mode == "and" and not pattern_passed
        if not skip_judge:
            try:
                from shared.validation.parsing import parse_response

                parsed = parse_response(final_response)
                case_meta = {
                    "system": case.metadata.get("system", ""),
                    "user_prompt": case.question,
                    "pattern_passed": pattern_passed,
                    "case_id": case.case_id,
                }
                judge_result = judge_validator.validate(
                    parsed_response=parsed,
                    case_metadata=case_meta,
                    judge_mode=per_case_mode,
                )
            except Exception as exc:
                logger.error("Judge validation error for %s: %s", case.case_id, exc)

    scoring_result = _run_path_scoring(
        case=case,
        validator_result=combined_validator,
        behavior_result=combined_behavior,
        environment_result=environment_result,
        judge_result=judge_result,
    )

    return EvaluationRecord(
        case=case,
        response_text=final_response,
        validator=combined_validator,
        latency_s=final_latency or None,
        raw_response=final_raw,
        error=None,
        behavior=combined_behavior,
        environment=environment_result,
        judge=judge_result,
        scoring=scoring_result,
        correctness=_run_correctness_validation(case, build_response_view(final_response, final_raw)) if final_response is not None else None,
        conversation_trace=conversation_trace,
    )


def _validate_case_response(
    case: PromptCase,
    response: Any,
    eval_context: Optional[Dict[str, Any]],
) -> ValidationResult:
    return _validate_response_structure(response, eval_context)


def _validate_response_structure(
    response: Any,
    eval_context: Optional[Dict[str, Any]],
) -> ValidationResult:
    return validate_assistant_response(response, eval_context)


def _has_correctness_expectation(case: PromptCase) -> bool:
    return has_correctness_config(case.metadata.get("correct"))


def _run_correctness_validation(
    case: PromptCase,
    response_view: Mapping[str, Any],
) -> Optional[CorrectnessResult]:
    correct = case.metadata.get("correct")
    if not has_correctness_config(correct):
        return None
    return evaluate_correctness(correct, response_view)


def _combine_validation_results(results: List[ValidationResult]) -> Optional[ValidationResult]:
    if not results:
        return None

    combined_tool_calls: List[ToolCall] = []
    combined_issues: List[ValidatorIssue] = []
    combined_context: Dict[str, Any] = {"all_match": True}

    for result in results:
        combined_tool_calls.extend(result.tool_calls)
        combined_issues.extend(result.issues)
        if result.context_validation:
            for key, value in result.context_validation.items():
                if key == "all_match":
                    combined_context["all_match"] = combined_context.get("all_match", True) and bool(value)
                    continue
                combined_context.setdefault(key, [])
                if isinstance(value, list):
                    combined_context[key].extend(value)

    return ValidationResult(
        passed=all(result.passed for result in results),
        issues=combined_issues,
        tool_calls=combined_tool_calls,
        context_validation=combined_context if len(combined_context) > 1 else None,
    )


def _stringify_assistant_response(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        content = response.get("content")
        tool_calls = response.get("tool_calls") or []
        parts: List[str] = []
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
        if tool_calls:
            parts.append(f"Tool calls: {tool_calls}")
        return "\n\n".join(parts).strip() or str(response)
    return str(response)


def _extract_text_content(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        content = response.get("content")
        if isinstance(content, str):
            return content
    return ""


def _build_single_turn_trace(case: PromptCase, response: Any) -> List[Dict[str, Any]]:
    trace = []
    for index, message in enumerate(case.chat_messages(), start=1):
        trace.append(
            {
                "index": index,
                "role": str(message.get("role", "")),
                "kind": "prompt_message",
                "content": message.get("content"),
            }
        )
    trace.append(
        {
            "index": len(trace) + 1,
            "role": "assistant",
            "kind": "assistant_response",
            "content": _stringify_assistant_response(response),
            "raw": response,
        }
    )
    return trace


def _run_path_scoring(
    *,
    case: PromptCase,
    validator_result: Optional[ValidationResult],
    behavior_result: Optional[Any],
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
    behavior_result: Optional[Any],
    environment_result: Optional["EnvironmentValidationResult"],
    judge_result: Optional["JudgeValidationResult"],
) -> tuple[bool, List[str]]:
    _matched, reasons = evaluate_tool_sequence(tool_names, path_cfg)

    if path_cfg.get("require_schema_pass") and not (validator_result and validator_result.passed):
        reasons.append("schema validation did not pass")

    if path_cfg.get("require_behavior_pass") and not (behavior_result is None or behavior_result.passed):
        reasons.append("behavior validation did not pass")

    if path_cfg.get("require_environment_pass") and not (environment_result and environment_result.passed):
        reasons.append("environment validation did not pass")

    if path_cfg.get("require_judge_pass") and not (judge_result and judge_result.passed):
        reasons.append("judge validation did not pass")

    return len(reasons) == 0, reasons
