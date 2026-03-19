"""Config-driven deterministic stage gates."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


@dataclass
class StageGateResult:
    gate_type: str
    passed: bool
    message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_type": self.gate_type,
            "passed": self.passed,
            "message": self.message,
            "metadata": dict(self.metadata),
        }


def run_stage_gates(gates: Sequence[Mapping[str, Any]], payload: Mapping[str, Any]) -> List[StageGateResult]:
    results: List[StageGateResult] = []
    for gate in gates or []:
        if not isinstance(gate, Mapping):
            continue
        gate_type = str(gate.get("type") or "").strip()
        if not gate_type:
            continue
        handler = _GATE_HANDLERS.get(gate_type)
        if handler is None:
            results.append(
                StageGateResult(
                    gate_type=gate_type,
                    passed=False,
                    message=f"Unknown stage gate type: {gate_type}",
                )
            )
            continue
        results.append(handler(payload, gate))
    return results


def _gate_non_empty_text(payload: Mapping[str, Any], gate: Mapping[str, Any]) -> StageGateResult:
    field = str(gate.get("field") or "text")
    value = _resolve_dotted(payload, field)
    text = str(value or "").strip()
    return StageGateResult(
        gate_type="non_empty_text",
        passed=bool(text),
        message=None if text else f"Field '{field}' is empty.",
    )


def _gate_plain_text(payload: Mapping[str, Any], gate: Mapping[str, Any]) -> StageGateResult:
    field = str(gate.get("field") or "text")
    text = str(_resolve_dotted(payload, field) or "").strip()
    looks_json = False
    if text.startswith("{") or text.startswith("["):
        try:
            json.loads(text)
            looks_json = True
        except Exception:
            looks_json = False
    has_code_fence = text.startswith("```") or text.endswith("```")
    passed = bool(text) and not looks_json and not has_code_fence
    reason = None
    if not text:
        reason = f"Field '{field}' is empty."
    elif looks_json:
        reason = f"Field '{field}' looks like JSON instead of plain text."
    elif has_code_fence:
        reason = f"Field '{field}' contains markdown fences."
    return StageGateResult(gate_type="plain_text", passed=passed, message=reason)


def _gate_no_tool_names(payload: Mapping[str, Any], gate: Mapping[str, Any]) -> StageGateResult:
    field = str(gate.get("field") or "text")
    text = str(_resolve_dotted(payload, field) or "")
    tool_names = gate.get("tool_names")
    if not isinstance(tool_names, list):
        tool_names = payload.get("allowed_tools") or []
    leaks = [name for name in tool_names if isinstance(name, str) and name and name in text]
    return StageGateResult(
        gate_type="no_tool_names",
        passed=not leaks,
        message=None if not leaks else f"Found tool names in text: {', '.join(leaks)}",
        metadata={"leaked_tool_names": leaks},
    )


def _gate_no_exact_paths_from_context(payload: Mapping[str, Any], gate: Mapping[str, Any]) -> StageGateResult:
    field = str(gate.get("field") or "text")
    text = str(_resolve_dotted(payload, field) or "")
    sources = gate.get("sources")
    if not isinstance(sources, list) or not sources:
        sources = ["task_context"]
    leaked_paths: List[str] = []
    for source in sources:
        values = _collect_path_strings(_resolve_dotted(payload, str(source)))
        for value in values:
            candidate = str(value).strip()
            if not _looks_like_path(candidate):
                continue
            if candidate and candidate in text:
                leaked_paths.append(candidate)
    leaked_paths = sorted(set(leaked_paths))
    return StageGateResult(
        gate_type="no_exact_paths_from_context",
        passed=not leaked_paths,
        message=None if not leaked_paths else f"Found exact path leakage: {', '.join(leaked_paths)}",
        metadata={"leaked_paths": leaked_paths},
    )


def _gate_environment_payload_shape(payload: Mapping[str, Any], gate: Mapping[str, Any]) -> StageGateResult:
    field = str(gate.get("field") or "value")
    value = _resolve_dotted(payload, field)
    environment = value.get("environment") if isinstance(value, Mapping) else None
    fixture = environment.get("fixture") if isinstance(environment, Mapping) else None
    assertions = environment.get("assertions") if isinstance(environment, Mapping) else None
    passed = isinstance(value, Mapping) and isinstance(environment, Mapping) and isinstance(fixture, Mapping) and isinstance(assertions, list)
    return StageGateResult(
        gate_type="environment_payload_shape",
        passed=passed,
        message=None if passed else "Generated environment payload is missing environment.fixture or environment.assertions.",
    )


def _gate_field_equals(payload: Mapping[str, Any], gate: Mapping[str, Any]) -> StageGateResult:
    field = str(gate.get("field") or "").strip()
    expected = gate.get("value")
    actual = _resolve_dotted(payload, field) if field else None
    passed = actual == expected
    return StageGateResult(
        gate_type="field_equals",
        passed=passed,
        message=None if passed else f"Field '{field}' expected {expected!r} but got {actual!r}.",
        metadata={"field": field, "expected": expected, "actual": actual},
    )


def _resolve_dotted(value: Any, dotted: str) -> Any:
    current = value
    for part in [piece for piece in str(dotted or "").split(".") if piece]:
        if isinstance(current, Mapping):
            current = current.get(part)
        else:
            return None
    return current


def _collect_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _collect_strings(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _collect_strings(item)


def _collect_path_strings(value: Any, key_hint: Optional[str] = None) -> Iterable[str]:
    if isinstance(value, str):
        if _is_path_field_name(key_hint) and "://" not in value:
            yield value
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _collect_path_strings(item, str(key))
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _collect_path_strings(item, key_hint)


def _is_path_field_name(key: Optional[str]) -> bool:
    key_text = str(key or "").strip().lower()
    if not key_text:
        return False
    return any(token in key_text for token in ("path", "paths", "folder", "scope", "directory"))


def _looks_like_path(value: str) -> bool:
    if not value or len(value) < 4:
        return False
    if "/" in value:
        return True
    return bool(re.search(r"\.(md|markdown|txt|yaml|yml|json)$", value, re.IGNORECASE))


_GATE_HANDLERS = {
    "non_empty_text": _gate_non_empty_text,
    "plain_text": _gate_plain_text,
    "no_tool_names": _gate_no_tool_names,
    "no_exact_paths_from_context": _gate_no_exact_paths_from_context,
    "environment_payload_shape": _gate_environment_payload_shape,
    "field_equals": _gate_field_equals,
}
