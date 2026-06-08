"""Generic config-driven assertion engine, wired as an ``assertions`` verifier.

This module is the canonical home for the assertion engine (moved verbatim from
``Evaluator/assertions.py``): the result dataclasses, ``evaluate_correctness``,
``select_path`` and the small JSONPath-subset helpers. The Evaluator CONSUMES
these symbols; it no longer owns them.

The :class:`AssertionsVerifier` (registered under ``assertions``) wires the
engine into the shared verifier contract. It is fully config-driven: the
``correct`` config block and the field paths it references come from the spec /
sample, nothing is hardcoded.

``shared.verifiers`` MUST NOT import ``Evaluator/`` or ``Trainers/`` — the
engine here is stdlib-only so consumers can depend on it (and not the reverse).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ..contract import VerifierInput, VerifierOutput
from ..registry import register


_MISSING = object()


@dataclass
class AssertionResult:
    """Result for one configured assertion."""

    name: str
    type: str
    passed: bool
    expected: Any = None
    actual: Any = None
    message: str = ""
    path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "passed": self.passed,
            "expected": self.expected,
            "actual": None if self.actual is _MISSING else self.actual,
            "message": self.message,
            "path": self.path,
        }


@dataclass
class AssertionPathResult:
    """Result for one allowed correctness path."""

    name: str
    passed: bool
    assertions: List[AssertionResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "assertions": [assertion.to_dict() for assertion in self.assertions],
        }


@dataclass
class CorrectnessResult:
    """Result for configured response correctness."""

    passed: bool
    matched_path: Optional[str] = None
    paths: List[AssertionPathResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "matched_path": self.matched_path,
            "paths": [path.to_dict() for path in self.paths],
        }


def has_correctness_config(config: Any) -> bool:
    return isinstance(config, Mapping) and bool(config)


def evaluate_correctness(correct: Mapping[str, Any], response_view: Mapping[str, Any]) -> CorrectnessResult:
    """Evaluate a `correct` config object against a response view."""
    paths = _configured_paths(correct)
    if not paths:
        return CorrectnessResult(passed=True, matched_path=None, paths=[])

    path_results: List[AssertionPathResult] = []
    matched_path: Optional[str] = None
    for index, path_config in enumerate(paths, start=1):
        path_result = _evaluate_path(path_config, response_view, index)
        path_results.append(path_result)
        if path_result.passed and matched_path is None:
            matched_path = path_result.name

    return CorrectnessResult(
        passed=matched_path is not None,
        matched_path=matched_path,
        paths=path_results,
    )


def _configured_paths(correct: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    any_paths = correct.get("any")
    if isinstance(any_paths, list):
        return [path for path in any_paths if isinstance(path, Mapping)]

    all_paths = correct.get("all")
    if isinstance(all_paths, list):
        return [
            {
                "name": "all",
                "assertions": [
                    {"type": "all", "assertions": all_paths},
                ],
            }
        ]

    assertions = correct.get("assertions")
    if isinstance(assertions, list):
        return [{"name": str(correct.get("name") or "default"), "assertions": assertions}]

    return []


def _evaluate_path(
    path_config: Mapping[str, Any],
    response_view: Mapping[str, Any],
    index: int,
) -> AssertionPathResult:
    assertions = path_config.get("assertions")
    path_name = str(path_config.get("name") or f"path_{index}")
    if not isinstance(assertions, list):
        result = AssertionResult(
            name=f"{path_name}.assertions",
            type="config",
            passed=False,
            expected="list",
            actual=type(assertions).__name__,
            message="Correctness path is missing an assertions list.",
        )
        return AssertionPathResult(name=path_name, passed=False, assertions=[result])

    results = [
        evaluate_assertion(assertion, response_view, f"{path_name}.{idx}")
        for idx, assertion in enumerate(assertions, start=1)
        if isinstance(assertion, Mapping)
    ]
    return AssertionPathResult(
        name=path_name,
        passed=bool(results) and all(result.passed for result in results),
        assertions=results,
    )


def evaluate_assertion(
    assertion: Mapping[str, Any],
    response_view: Mapping[str, Any],
    fallback_name: str = "assertion",
) -> AssertionResult:
    atype = str(assertion.get("type", "")).strip()
    name = str(assertion.get("name") or fallback_name)

    if atype in {"all", "any"}:
        nested = assertion.get("assertions")
        if not isinstance(nested, list):
            return AssertionResult(
                name=name,
                type=atype,
                passed=False,
                expected="assertions list",
                actual=type(nested).__name__,
                message=f"{atype} assertion missing nested assertions list.",
            )
        nested_results = [
            evaluate_assertion(item, response_view, f"{name}.{idx}")
            for idx, item in enumerate(nested, start=1)
            if isinstance(item, Mapping)
        ]
        passed = all(result.passed for result in nested_results) if atype == "all" else any(
            result.passed for result in nested_results
        )
        return AssertionResult(
            name=name,
            type=atype,
            passed=passed,
            expected=atype,
            actual=[result.to_dict() for result in nested_results],
            message=f"{atype} nested assertions {'passed' if passed else 'failed'}.",
        )

    if atype == "not":
        nested = assertion.get("assertion")
        if not isinstance(nested, Mapping):
            return AssertionResult(
                name=name,
                type=atype,
                passed=False,
                expected="assertion object",
                actual=type(nested).__name__,
                message="not assertion missing nested assertion object.",
            )
        nested_result = evaluate_assertion(nested, response_view, f"{name}.not")
        passed = not nested_result.passed
        return AssertionResult(
            name=name,
            type=atype,
            passed=passed,
            expected="nested assertion to fail",
            actual=nested_result.to_dict(),
            message="Nested assertion failed as expected." if passed else "Nested assertion passed unexpectedly.",
        )

    value, path = _selected_value(assertion, response_view, atype)
    expected = assertion.get("value")

    if atype in {"exists", "jsonpath_exists"}:
        passed = value is not _MISSING and value is not None
        return _result(name, atype, passed, "value exists", value, path)

    if atype in {"absent", "jsonpath_absent"}:
        passed = value is _MISSING or value is None
        return _result(name, atype, passed, "value absent", value, path)

    if atype in {"equals", "jsonpath_equals"}:
        passed = value == expected
        return _result(name, atype, passed, expected, value, path)

    if atype in {"not_equals", "jsonpath_not_equals"}:
        passed = value != expected
        return _result(name, atype, passed, f"not {expected!r}", value, path)

    if atype in {"contains", "jsonpath_contains", "text_contains"}:
        needle = assertion.get("value", assertion.get("text"))
        passed = _contains(value, needle)
        return _result(name, atype, passed, needle, value, path)

    if atype in {"not_contains", "jsonpath_not_contains", "text_not_contains"}:
        needle = assertion.get("value", assertion.get("text"))
        passed = not _contains(value, needle)
        return _result(name, atype, passed, f"not contains {needle!r}", value, path)

    if atype in {"regex", "jsonpath_regex", "text_regex"}:
        pattern = str(assertion.get("pattern", ""))
        passed = _matches_regex(value, pattern)
        return _result(name, atype, passed, pattern, value, path)

    if atype in {"not_regex", "jsonpath_not_regex", "text_not_regex"}:
        pattern = str(assertion.get("pattern", ""))
        passed = not _matches_regex(value, pattern)
        return _result(name, atype, passed, f"not regex {pattern}", value, path)

    if atype in {"length_equals", "jsonpath_length_equals"}:
        expected_len = int(assertion.get("value", assertion.get("length", 0)) or 0)
        actual_len = _length(value)
        return _result(name, atype, actual_len == expected_len, expected_len, actual_len, path)

    if atype in {"length_min", "jsonpath_length_min"}:
        expected_len = int(assertion.get("value", assertion.get("min", 0)) or 0)
        actual_len = _length(value)
        return _result(name, atype, actual_len >= expected_len, f">= {expected_len}", actual_len, path)

    if atype in {"length_max", "jsonpath_length_max"}:
        expected_len = int(assertion.get("value", assertion.get("max", 0)) or 0)
        actual_len = _length(value)
        return _result(name, atype, actual_len <= expected_len, f"<= {expected_len}", actual_len, path)

    if atype == "json_subset":
        passed = _is_subset(expected, value)
        return _result(name, atype, passed, expected, value, path)

    return AssertionResult(
        name=name,
        type=atype or "unknown",
        passed=False,
        expected="known assertion type",
        actual=atype,
        message=f"Unknown assertion type: {atype}",
        path=path,
    )


def _selected_value(
    assertion: Mapping[str, Any],
    response_view: Mapping[str, Any],
    atype: str,
) -> Tuple[Any, Optional[str]]:
    path = assertion.get("path") or assertion.get("selector")
    if atype.startswith("text_") and path is None:
        path = "$.content"
    if isinstance(path, str) and path.strip():
        return select_path(response_view, path), path
    return response_view, "$"


def select_path(data: Any, path: str) -> Any:
    """Select from nested JSON-like data using a small JSONPath subset."""
    path = str(path or "").strip()
    if path in {"", "$"}:
        return data
    if not path.startswith("$."):
        return _MISSING

    tokens = _parse_path(path[2:])
    current = data
    for token in tokens:
        current = _resolve_token(current, token)
        if current is _MISSING:
            return _MISSING
    return current


def _parse_path(path: str) -> List[Any]:
    tokens: List[Any] = []
    buf = ""
    i = 0
    while i < len(path):
        char = path[i]
        if char == ".":
            if buf:
                tokens.append(buf)
                buf = ""
            i += 1
            continue
        if char == "[":
            if buf:
                tokens.append(buf)
                buf = ""
            end = path.find("]", i)
            if end == -1:
                tokens.append(path[i:])
                break
            raw = path[i + 1:end].strip()
            if raw == "*":
                tokens.append("*")
            elif (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
                tokens.append(raw[1:-1])
            else:
                try:
                    tokens.append(int(raw))
                except ValueError:
                    tokens.append(raw)
            i = end + 1
            continue
        buf += char
        i += 1
    if buf:
        tokens.append(buf)
    return tokens


def _resolve_token(current: Any, token: Any) -> Any:
    if token == "*":
        if isinstance(current, list):
            return list(current)
        if isinstance(current, Mapping):
            return list(current.values())
        return _MISSING

    if isinstance(current, list):
        if isinstance(token, int):
            try:
                return current[token]
            except IndexError:
                return _MISSING
        resolved = [_resolve_token(item, token) for item in current]
        return [item for item in resolved if item is not _MISSING]

    if isinstance(current, Mapping):
        return current.get(token, _MISSING)

    return _MISSING


def _result(
    name: str,
    atype: str,
    passed: bool,
    expected: Any,
    actual: Any,
    path: Optional[str],
) -> AssertionResult:
    actual_display = None if actual is _MISSING else actual
    return AssertionResult(
        name=name,
        type=atype,
        passed=passed,
        expected=expected,
        actual=actual_display,
        message="passed" if passed else f"expected {expected!r}, got {actual_display!r}",
        path=path,
    )


def _matches_regex(value: Any, pattern: str) -> bool:
    if value is _MISSING:
        return False
    if isinstance(value, list):
        return any(_matches_regex(item, pattern) for item in value)
    return re.search(pattern, str(value), flags=re.MULTILINE | re.DOTALL) is not None


def _contains(value: Any, needle: Any) -> bool:
    if value is _MISSING:
        return False
    if isinstance(value, str):
        return str(needle) in value
    if isinstance(value, list):
        return needle in value or any(_contains(item, needle) for item in value)
    if isinstance(value, Mapping):
        return needle in value or needle in value.values()
    return value == needle


def _length(value: Any) -> int:
    if value is _MISSING or value is None:
        return 0
    if isinstance(value, (str, list, tuple, dict)):
        return len(value)
    return len(str(value))


def _is_subset(expected: Any, actual: Any) -> bool:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return False
        for key, expected_value in expected.items():
            if key not in actual or not _is_subset(expected_value, actual[key]):
                return False
        return True
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        return all(any(_is_subset(expected_item, actual_item) for actual_item in actual) for expected_item in expected)
    return expected == actual


# ---------------------------------------------------------------------------
# Verifier wiring
# ---------------------------------------------------------------------------

def _minimal_response_view(sample: VerifierInput) -> Mapping[str, Any]:
    """Build a minimal response view from ``parsed`` / ``completion_text``.

    Used only when no ``response_view`` signal is supplied. The Evaluator's own
    ``build_response_view`` produces a richer view; this fallback keeps the
    shared package free of any Evaluator import while still exposing the common
    ``content`` / ``tool_calls`` fields config assertions reference.
    """
    parsed = sample.parsed
    content = getattr(parsed, "text_content", None)
    if content is None:
        content = sample.completion_text
    tool_calls = getattr(parsed, "tool_calls", None)
    return {
        "content": content if isinstance(content, str) else str(content or ""),
        "tool_calls": list(tool_calls) if isinstance(tool_calls, list) else [],
    }


class AssertionsVerifier:
    """Verifier that evaluates a config-driven ``correct`` block.

    The ``correct`` config (the same structure consumed by
    :func:`evaluate_correctness`) is read from the spec ``params`` under the
    ``correct`` key. The response view is read from
    ``sample.signals['response_view']`` when present, falling back to a minimal
    view built from ``sample.parsed`` / ``sample.completion_text``.

    Scoring follows :class:`CorrectnessResult`: ``passed`` maps to ``1.0`` /
    ``0.0`` (``CorrectnessResult`` carries no numeric score of its own).
    """

    def __init__(self, name: str = "assertions", correct: Mapping[str, Any] | None = None):
        self.name = name
        self.correct = dict(correct) if correct else {}

    def verify(self, sample: VerifierInput) -> VerifierOutput:
        response_view = sample.signals.get("response_view")
        if not isinstance(response_view, Mapping):
            response_view = _minimal_response_view(sample)

        result = evaluate_correctness(self.correct, response_view)
        score = 1.0 if result.passed else 0.0
        return VerifierOutput(
            score=score,
            passed=result.passed,
            detail=result.to_dict(),
        )


@register("assertions")
def _build_assertions(spec: Mapping) -> AssertionsVerifier:
    params = spec.get("params", spec)
    return AssertionsVerifier(
        name=spec.get("name", "assertions"),
        correct=params.get("correct"),
    )

