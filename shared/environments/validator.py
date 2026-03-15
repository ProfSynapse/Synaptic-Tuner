"""High-level environment validator for tool-call responses."""

from __future__ import annotations

import re
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .base import EnvironmentRuntime
from .e2b_runtime import E2BEnvironmentRuntime
from .fixture_parser import merge_environment_fixture, parse_environment_fixture
from .local_runtime import LocalEnvironmentRuntime
from .tool_executor import execute_response_tool_calls
from .types import (
    EnvironmentEpisodeTrace,
    EnvironmentIssue,
    EnvironmentStepResult,
    EnvironmentValidationResult,
    ExecutedToolCall,
)

try:
    import yaml
except ImportError:
    yaml = None


DEFAULT_TOOL_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "Evaluator" / "config" / "tool_schema.yaml"
DEFAULT_EXECUTION_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "Evaluator" / "config" / "environment_execution.yaml"
)


class EnvironmentValidator:
    """Execute tool calls in an isolated runtime and validate state assertions."""

    def __init__(
        self,
        backend: str = "local",
        e2b_template: Optional[str] = None,
        e2b_api_key: Optional[str] = None,
        timeout_seconds: float = 120.0,
        tool_schema_path: Optional[str] = None,
        execution_config_path: Optional[str] = None,
    ):
        backend_normalized = (backend or "local").strip().lower()
        if backend_normalized not in {"local", "e2b"}:
            raise ValueError(f"Unsupported environment backend: {backend}")

        self.backend = backend_normalized
        self.e2b_template = e2b_template
        self.e2b_api_key = e2b_api_key
        self.timeout_seconds = timeout_seconds
        self.tool_schema_path = Path(tool_schema_path) if tool_schema_path else DEFAULT_TOOL_SCHEMA_PATH
        self.execution_config_path = (
            Path(execution_config_path) if execution_config_path else DEFAULT_EXECUTION_CONFIG_PATH
        )

        self.tool_schema = _load_yaml_file(self.tool_schema_path)
        self.execution_config = _load_execution_config(self.execution_config_path)

    def validate_response(
        self,
        system_prompt: str,
        response: Any,
        environment_config: Optional[Dict[str, Any]] = None,
        expected_tools: Optional[Iterable[str]] = None,
    ) -> EnvironmentValidationResult:
        """Validate a response by executing its tool calls against runtime state."""
        session = self.start_session(system_prompt=system_prompt, environment_config=environment_config)
        try:
            session.execute_response(response)
            return session.finalize(expected_tools=expected_tools, total_turns=1, stop_reason="single_response")
        finally:
            session.close()

    def start_session(
        self,
        system_prompt: str,
        environment_config: Optional[Dict[str, Any]] = None,
    ) -> "EnvironmentSession":
        """Create a persistent environment session for multi-turn episodes."""
        return EnvironmentSession(
            validator=self,
            system_prompt=system_prompt,
            environment_config=environment_config or {},
        )

    def _create_runtime(self) -> EnvironmentRuntime:
        if self.backend == "local":
            return LocalEnvironmentRuntime()
        return E2BEnvironmentRuntime(
            template=self.e2b_template,
            api_key=self.e2b_api_key,
            timeout_seconds=self.timeout_seconds,
        )


@dataclass
class EnvironmentSession:
    """Persistent runtime session for one environment-backed episode."""

    validator: EnvironmentValidator
    system_prompt: str
    environment_config: Dict[str, Any] = field(default_factory=dict)
    runtime: EnvironmentRuntime = field(init=False)
    assertions: List[Dict[str, Any]] = field(init=False, default_factory=list)
    allowed_tools: Optional[Iterable[str]] = field(init=False, default=None)
    max_steps: int = field(init=False, default=0)
    action_hints: Dict[str, str] = field(init=False, default_factory=dict)
    key_hints: Dict[str, List[str]] = field(init=False, default_factory=dict)
    verb_rules: Dict[str, List[str]] = field(init=False, default_factory=dict)
    strict_schema: bool = field(init=False, default=False)
    default_action: str = field(init=False, default="simulate")
    loop_mode: str = field(init=False, default="strict")
    continue_on_execution_error: bool = field(init=False, default=False)
    issues: List[EnvironmentIssue] = field(init=False, default_factory=list)
    executed_tools: List[ExecutedToolCall] = field(init=False, default_factory=list)
    steps: List[EnvironmentStepResult] = field(init=False, default_factory=list)
    closed: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        config = self.environment_config or {}
        self.assertions = config.get("assertions", []) if isinstance(config.get("assertions"), list) else []
        self.allowed_tools = config.get("allowed_tools")
        self.max_steps = int(config.get("max_steps", 0) or 0)
        execution_overrides = config.get("execution") if isinstance(config.get("execution"), dict) else {}
        self.strict_schema = bool(
            execution_overrides.get(
                "strict_schema",
                self.validator.execution_config.get("strict_schema", False),
            )
        )
        self.default_action = str(
            execution_overrides.get(
                "default_action",
                self.validator.execution_config.get("default_action", "simulate"),
            )
        ).strip().lower() or "simulate"

        global_action_hints = self.validator.execution_config.get("tool_action_hints", {})
        override_action_hints = (
            execution_overrides.get("tool_action_hints")
            if isinstance(execution_overrides.get("tool_action_hints"), dict)
            else {}
        )
        inline_action_hints = config.get("action_hints") if isinstance(config.get("action_hints"), dict) else {}
        self.action_hints = _merge_dicts(
            _merge_dicts(global_action_hints, override_action_hints),
            inline_action_hints,
        )

        override_key_hints = execution_overrides.get("key_hints")
        if not isinstance(override_key_hints, dict):
            override_key_hints = {}
        self.key_hints = _merge_rule_lists(
            self.validator.execution_config.get("key_hints", {}),
            override_key_hints,
        )

        override_verb_rules = execution_overrides.get("verb_rules")
        if not isinstance(override_verb_rules, dict):
            override_verb_rules = {}
        self.verb_rules = _merge_rule_lists(
            self.validator.execution_config.get("verb_rules", {}),
            override_verb_rules,
        )

        self.runtime = self.validator._create_runtime()
        loop_cfg = config.get("loop") if isinstance(config.get("loop"), dict) else {}
        self.loop_mode = str(loop_cfg.get("mode", "strict") or "strict").strip().lower()
        self.continue_on_execution_error = bool(
            loop_cfg.get(
                "continue_on_execution_error",
                self.loop_mode == "agentic",
            )
        )

        fixture = merge_environment_fixture(
            parse_environment_fixture(self.system_prompt),
            config.get("fixture"),
        )
        self.runtime.setup(fixture)

    def execute_response(self, response: Any) -> EnvironmentStepResult:
        """Execute one assistant response against the persistent runtime."""
        if self.closed:
            raise RuntimeError("Environment session is closed")

        step_issues: List[EnvironmentIssue] = []
        before_signature = _snapshot_signature(self.runtime)
        try:
            executions, exec_issues = execute_response_tool_calls(
                runtime=self.runtime,
                response=response,
                allowed_tools=self.allowed_tools,
                tool_schema=self.validator.tool_schema,
                action_hints=self.action_hints,
                strict_schema=self.strict_schema,
                key_hints=self.key_hints,
                verb_rules=self.verb_rules,
                default_action=self.default_action,
            )
        except Exception as exc:
            executions = []
            exec_issues = [EnvironmentIssue("error", f"Environment validation failed: {exc}")]

        step_issues.extend(exec_issues)
        self.issues.extend(step_issues)
        self.executed_tools.extend(executions)
        after_signature = _snapshot_signature(self.runtime)
        state_changed = before_signature != after_signature
        has_errors = any(issue.level.lower() == "error" for issue in step_issues)
        recoverable_error = has_errors and all(issue.recoverable is not False for issue in step_issues)
        hard_error = has_errors and not (self.continue_on_execution_error and recoverable_error)

        step = EnvironmentStepResult(
            turn_index=len(self.steps) + 1,
            executed_tools=executions,
            issues=step_issues,
            hard_error=hard_error,
            recoverable_error=recoverable_error,
            state_changed=state_changed,
            action_signature=_build_action_signature(executions),
            issue_signature=_build_issue_signature(step_issues),
        )
        self.steps.append(step)
        return step

    def finalize(
        self,
        expected_tools: Optional[Iterable[str]] = None,
        total_turns: Optional[int] = None,
        stop_reason: Optional[str] = None,
    ) -> EnvironmentValidationResult:
        """Run final checks and return the aggregate environment result."""
        assertions_run = 0
        snapshot: Dict[str, Any] = {}
        final_issues = [_normalize_final_issue(issue) for issue in self.issues]

        if self.max_steps and len(self.executed_tools) > self.max_steps:
            final_issues.append(
                EnvironmentIssue(
                    "error",
                    f"Response executed {len(self.executed_tools)} tool calls, exceeding max_steps={self.max_steps}",
                )
            )

        if expected_tools:
            expected = set(expected_tools)
            called = {tool.name for tool in self.executed_tools}
            missing = sorted(expected - called)
            if missing:
                final_issues.append(
                    EnvironmentIssue(
                        "error",
                        f"Expected tool(s) not executed in environment simulation: {', '.join(missing)}",
                    )
                )

        if isinstance(self.assertions, list):
            assertion_issues = _run_assertions(self.runtime, self.assertions)
            assertions_run = len(self.assertions)
            final_issues.extend(assertion_issues)

        try:
            snapshot = self.runtime.snapshot()
        except Exception as exc:
            final_issues.append(EnvironmentIssue("warning", f"Environment snapshot warning: {exc}"))

        passed = all(issue.level.lower() != "error" for issue in final_issues)
        trace = EnvironmentEpisodeTrace(
            steps=list(self.steps),
            total_turns=total_turns if total_turns is not None else len(self.steps),
            total_tool_calls=len(self.executed_tools),
            stop_reason=stop_reason,
            hard_failure=any(step.hard_error for step in self.steps),
            recovered_after_error=_did_recover_after_error(self.steps, passed),
        )
        return EnvironmentValidationResult(
            passed=passed,
            issues=final_issues,
            executed_tools=list(self.executed_tools),
            assertions_run=assertions_run,
            snapshot=snapshot,
            episode_trace=trace,
        )

    def close(self) -> None:
        """Tear down the underlying runtime."""
        if self.closed:
            return
        self.closed = True
        try:
            self.runtime.teardown()
        except Exception as exc:
            self.issues.append(EnvironmentIssue("warning", f"Environment teardown warning: {exc}"))


def _run_assertions(runtime: EnvironmentRuntime, assertions: List[Dict[str, Any]]) -> List[EnvironmentIssue]:
    issues: List[EnvironmentIssue] = []
    for assertion in assertions:
        if not isinstance(assertion, dict):
            issues.append(EnvironmentIssue("warning", "Skipping malformed assertion (expected object)"))
            continue

        atype = str(assertion.get("type", "")).strip()
        path = str(assertion.get("path", "")).strip()

        if atype == "path_exists":
            if not runtime.exists(path):
                issues.append(EnvironmentIssue("error", f"Assertion failed: expected path to exist: {path}"))
            continue

        if atype == "path_not_exists":
            if runtime.exists(path):
                issues.append(EnvironmentIssue("error", f"Assertion failed: expected path to be absent: {path}"))
            continue

        if atype == "file_contains":
            needle = str(assertion.get("text", ""))
            content, error = _read_file(runtime, path)
            if error:
                issues.append(EnvironmentIssue("error", error, code="assertion_read_failed", recoverable=False))
                continue
            if needle not in content:
                issues.append(
                    EnvironmentIssue(
                        "error",
                        f"Assertion failed: '{path}' does not contain expected text",
                        code="assertion_file_contains",
                        recoverable=False,
                    )
                )
            continue

        if atype == "file_not_contains":
            needle = str(assertion.get("text", ""))
            content, error = _read_file(runtime, path)
            if error:
                issues.append(EnvironmentIssue("error", error, code="assertion_read_failed", recoverable=False))
                continue
            if needle in content:
                issues.append(
                    EnvironmentIssue(
                        "error",
                        f"Assertion failed: '{path}' contains forbidden text",
                        code="assertion_file_not_contains",
                        recoverable=False,
                    )
                )
            continue

        if atype == "file_matches_regex":
            pattern = str(assertion.get("pattern", ""))
            if not pattern:
                issues.append(EnvironmentIssue("warning", "Assertion file_matches_regex missing 'pattern'"))
                continue
            content, error = _read_file(runtime, path)
            if error:
                issues.append(EnvironmentIssue("error", error, code="assertion_read_failed", recoverable=False))
                continue
            if re.search(pattern, content, re.MULTILINE) is None:
                issues.append(
                    EnvironmentIssue(
                        "error",
                        f"Assertion failed: '{path}' does not match expected regex",
                        code="assertion_file_matches_regex",
                        recoverable=False,
                    )
                )
            continue

        if atype in {"file_line_contains", "file_line_not_contains"}:
            expected_text = str(assertion.get("text", ""))
            line_number = int(assertion.get("line", 0) or 0)
            if line_number <= 0:
                issues.append(EnvironmentIssue("warning", f"Assertion {atype} missing valid 'line'"))
                continue
            content, error = _read_file(runtime, path)
            if error:
                issues.append(EnvironmentIssue("error", error, code="assertion_read_failed", recoverable=False))
                continue
            lines = content.splitlines()
            if line_number > len(lines):
                issues.append(
                    EnvironmentIssue(
                        "error",
                        f"Assertion failed: '{path}' has no line {line_number}",
                        code="assertion_line_missing",
                        recoverable=False,
                    )
                )
                continue
            line = lines[line_number - 1]
            contains = expected_text in line
            if atype == "file_line_contains" and not contains:
                issues.append(
                    EnvironmentIssue(
                        "error",
                        f"Assertion failed: line {line_number} of '{path}' does not contain expected text",
                        code="assertion_file_line_contains",
                        recoverable=False,
                    )
                )
            if atype == "file_line_not_contains" and contains:
                issues.append(
                    EnvironmentIssue(
                        "error",
                        f"Assertion failed: line {line_number} of '{path}' contains forbidden text",
                        code="assertion_file_line_not_contains",
                        recoverable=False,
                    )
                )
            continue

        if atype == "dir_contains":
            item_name = str(assertion.get("item", "")).strip()
            if not item_name:
                issues.append(EnvironmentIssue("warning", "Assertion dir_contains missing 'item'"))
                continue
            items = runtime.list_dir(path or ".")
            if item_name not in items:
                issues.append(
                    EnvironmentIssue(
                        "error",
                        f"Assertion failed: directory '{path or '.'}' does not contain '{item_name}'",
                    )
                )
            continue

        if atype == "frontmatter_has_key":
            field = str(assertion.get("field", "")).strip()
            if not field:
                issues.append(EnvironmentIssue("warning", "Assertion frontmatter_has_key missing 'field'"))
                continue
            frontmatter, error = _read_frontmatter(runtime, path)
            if error:
                issues.append(EnvironmentIssue("error", error))
                continue
            if field not in frontmatter:
                issues.append(
                    EnvironmentIssue(
                        "error",
                        f"Assertion failed: '{path}' front matter is missing key '{field}'",
                    )
                )
            continue

        if atype == "frontmatter_field_equals":
            field = str(assertion.get("field", "")).strip()
            expected = assertion.get("value")
            if not field:
                issues.append(EnvironmentIssue("warning", "Assertion frontmatter_field_equals missing 'field'"))
                continue
            frontmatter, error = _read_frontmatter(runtime, path)
            if error:
                issues.append(EnvironmentIssue("error", error))
                continue
            actual = frontmatter.get(field)
            if actual != expected:
                issues.append(
                    EnvironmentIssue(
                        "error",
                        f"Assertion failed: '{path}' front matter field '{field}' expected {expected!r}, got {actual!r}",
                    )
                )
            continue

        if atype == "frontmatter_field_contains":
            field = str(assertion.get("field", "")).strip()
            expected = assertion.get("value")
            if not field:
                issues.append(EnvironmentIssue("warning", "Assertion frontmatter_field_contains missing 'field'"))
                continue
            frontmatter, error = _read_frontmatter(runtime, path)
            if error:
                issues.append(EnvironmentIssue("error", error))
                continue
            actual = frontmatter.get(field)
            if not _value_contains(actual, expected):
                issues.append(
                    EnvironmentIssue(
                        "error",
                        f"Assertion failed: '{path}' front matter field '{field}' does not contain {expected!r}",
                    )
                )
            continue

        issues.append(EnvironmentIssue("warning", f"Unknown assertion type: {atype}"))

    return issues


def _snapshot_signature(runtime: EnvironmentRuntime) -> str:
    """Return a normalized state signature for no-progress detection."""
    try:
        snapshot = runtime.snapshot()
    except Exception:
        return "snapshot_unavailable"

    directories = sorted(str(item) for item in snapshot.get("directories", []) if isinstance(item, str))
    files = snapshot.get("files", [])
    normalized_files = []
    if isinstance(files, list):
        for item in files:
            if not isinstance(item, dict):
                continue
            normalized_files.append(
                {
                    "path": str(item.get("path", "")),
                    "size": int(item.get("size", 0) or 0),
                }
            )
    normalized_files.sort(key=lambda item: item["path"])
    payload = {"directories": directories, "files": normalized_files}
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _build_action_signature(executions: List[ExecutedToolCall]) -> str:
    payload = []
    for tool in executions:
        payload.append(
            {
                "name": tool.name,
                "arguments": tool.arguments,
                "status": tool.status,
                "error": tool.error,
            }
        )
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _build_issue_signature(issues: List[EnvironmentIssue]) -> str:
    payload = []
    for issue in issues:
        payload.append(
            {
                "level": issue.level,
                "message": issue.message,
                "code": issue.code,
            }
        )
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    if yaml is None or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _load_execution_config(path: Path) -> Dict[str, Any]:
    defaults = {
        "strict_schema": False,
        "default_action": "simulate",
        "verb_rules": {},
        "key_hints": {},
        "tool_action_hints": {},
    }
    data = _load_yaml_file(path)
    section = data.get("environment_execution") if isinstance(data.get("environment_execution"), dict) else data
    if not isinstance(section, dict):
        return defaults

    merged = dict(defaults)
    merged.update({k: v for k, v in section.items() if k in {"strict_schema", "default_action"}})
    merged["verb_rules"] = section.get("verb_rules", {}) if isinstance(section.get("verb_rules"), dict) else {}
    merged["key_hints"] = section.get("key_hints", {}) if isinstance(section.get("key_hints"), dict) else {}
    merged["tool_action_hints"] = (
        section.get("tool_action_hints", {}) if isinstance(section.get("tool_action_hints"), dict) else {}
    )
    return merged


def _merge_rule_lists(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for key, values in (base or {}).items():
        if isinstance(values, list):
            out[key] = [str(v) for v in values]
    for key, values in (override or {}).items():
        if isinstance(values, list):
            out[key] = [str(v) for v in values]
    return out


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base or {})
    merged.update(override or {})
    return merged


def _read_frontmatter(runtime: EnvironmentRuntime, path: str) -> tuple[Dict[str, Any], Optional[str]]:
    if yaml is None:
        return {}, "Front matter assertions require PyYAML to be installed"
    try:
        content = runtime.read_text(path)
    except Exception as exc:
        return {}, f"Assertion failed reading {path}: {exc}"

    frontmatter_text = _extract_frontmatter_block(content)
    if frontmatter_text is None:
        return {}, f"Assertion failed: '{path}' does not contain YAML front matter"

    try:
        parsed = yaml.safe_load(frontmatter_text) or {}
    except Exception as exc:
        return {}, f"Assertion failed parsing front matter for {path}: {exc}"

    if not isinstance(parsed, dict):
        return {}, f"Assertion failed: '{path}' front matter is not a mapping"
    return parsed, None


def _extract_frontmatter_block(content: str) -> Optional[str]:
    if not isinstance(content, str):
        return None
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            return "\n".join(lines[1:idx])
    return None


def _value_contains(actual: Any, expected: Any) -> bool:
    if isinstance(actual, list):
        return expected in actual
    if isinstance(actual, str):
        return str(expected) in actual
    if isinstance(actual, dict):
        return expected in actual
    return actual == expected


def _read_file(runtime: EnvironmentRuntime, path: str) -> tuple[str, Optional[str]]:
    try:
        return runtime.read_text(path), None
    except Exception as exc:
        return "", f"Assertion failed reading {path}: {exc}"


def _did_recover_after_error(steps: List[EnvironmentStepResult], passed: bool) -> bool:
    if not passed:
        return False
    return any(step.recoverable_error for step in steps)


def _normalize_final_issue(issue: EnvironmentIssue) -> EnvironmentIssue:
    if issue.level.lower() == "error" and issue.recoverable:
        return EnvironmentIssue(
            level="warning",
            message=issue.message,
            code=issue.code,
            recoverable=issue.recoverable,
        )
    return issue
