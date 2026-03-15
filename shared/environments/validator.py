"""High-level environment validator for tool-call responses."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .base import EnvironmentRuntime
from .e2b_runtime import E2BEnvironmentRuntime
from .fixture_parser import merge_environment_fixture, parse_environment_fixture
from .local_runtime import LocalEnvironmentRuntime
from .tool_executor import execute_response_tool_calls
from .types import EnvironmentIssue, EnvironmentValidationResult

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
        config = environment_config or {}
        assertions = config.get("assertions", [])
        allowed_tools = config.get("allowed_tools")
        max_steps = int(config.get("max_steps", 0) or 0)
        execution_overrides = config.get("execution") if isinstance(config.get("execution"), dict) else {}
        strict_schema = bool(
            execution_overrides.get(
                "strict_schema",
                self.execution_config.get("strict_schema", False),
            )
        )
        default_action = str(
            execution_overrides.get(
                "default_action",
                self.execution_config.get("default_action", "simulate"),
            )
        ).strip().lower() or "simulate"

        global_action_hints = self.execution_config.get("tool_action_hints", {})
        override_action_hints = (
            execution_overrides.get("tool_action_hints")
            if isinstance(execution_overrides.get("tool_action_hints"), dict)
            else {}
        )
        inline_action_hints = config.get("action_hints") if isinstance(config.get("action_hints"), dict) else {}
        action_hints = _merge_dicts(
            _merge_dicts(global_action_hints, override_action_hints),
            inline_action_hints,
        )

        override_key_hints = execution_overrides.get("key_hints")
        if not isinstance(override_key_hints, dict):
            override_key_hints = {}
        key_hints = _merge_rule_lists(
            self.execution_config.get("key_hints", {}),
            override_key_hints,
        )

        override_verb_rules = execution_overrides.get("verb_rules")
        if not isinstance(override_verb_rules, dict):
            override_verb_rules = {}
        verb_rules = _merge_rule_lists(
            self.execution_config.get("verb_rules", {}),
            override_verb_rules,
        )

        runtime = self._create_runtime()
        issues: List[EnvironmentIssue] = []
        assertions_run = 0
        snapshot: Dict[str, Any] = {}
        executions = []

        try:
            fixture = merge_environment_fixture(
                parse_environment_fixture(system_prompt),
                config.get("fixture"),
            )
            runtime.setup(fixture)

            executions, exec_issues = execute_response_tool_calls(
                runtime=runtime,
                response=response,
                allowed_tools=allowed_tools,
                tool_schema=self.tool_schema,
                action_hints=action_hints,
                strict_schema=strict_schema,
                key_hints=key_hints,
                verb_rules=verb_rules,
                default_action=default_action,
            )
            issues.extend(exec_issues)

            if max_steps and len(executions) > max_steps:
                issues.append(
                    EnvironmentIssue(
                        "error",
                        f"Response executed {len(executions)} tool calls, exceeding max_steps={max_steps}",
                    )
                )

            if expected_tools:
                expected = set(expected_tools)
                called = {tool.name for tool in executions}
                missing = sorted(expected - called)
                if missing:
                    issues.append(
                        EnvironmentIssue(
                            "error",
                            f"Expected tool(s) not executed in environment simulation: {', '.join(missing)}",
                        )
                    )

            if isinstance(assertions, list):
                assertion_issues = _run_assertions(runtime, assertions)
                assertions_run = len(assertions)
                issues.extend(assertion_issues)

            snapshot = runtime.snapshot()
        except Exception as exc:
            issues.append(EnvironmentIssue("error", f"Environment validation failed: {exc}"))
        finally:
            try:
                runtime.teardown()
            except Exception as exc:
                issues.append(EnvironmentIssue("warning", f"Environment teardown warning: {exc}"))

        passed = all(issue.level.lower() != "error" for issue in issues)
        return EnvironmentValidationResult(
            passed=passed,
            issues=issues,
            executed_tools=executions,
            assertions_run=assertions_run,
            snapshot=snapshot,
        )

    def _create_runtime(self) -> EnvironmentRuntime:
        if self.backend == "local":
            return LocalEnvironmentRuntime()
        return E2BEnvironmentRuntime(
            template=self.e2b_template,
            api_key=self.e2b_api_key,
            timeout_seconds=self.timeout_seconds,
        )


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
            try:
                content = runtime.read_text(path)
            except Exception as exc:
                issues.append(EnvironmentIssue("error", f"Assertion failed reading {path}: {exc}"))
                continue
            if needle not in content:
                issues.append(EnvironmentIssue("error", f"Assertion failed: '{path}' does not contain expected text"))
            continue

        if atype == "file_not_contains":
            needle = str(assertion.get("text", ""))
            try:
                content = runtime.read_text(path)
            except Exception as exc:
                issues.append(EnvironmentIssue("error", f"Assertion failed reading {path}: {exc}"))
                continue
            if needle in content:
                issues.append(EnvironmentIssue("error", f"Assertion failed: '{path}' contains forbidden text"))
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
