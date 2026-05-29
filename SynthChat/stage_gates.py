"""Config-driven deterministic stage gates."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import jsonschema


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


def _gate_json_schema(payload: Mapping[str, Any], gate: Mapping[str, Any]) -> StageGateResult:
    field = str(gate.get("field") or "value")
    value = _resolve_dotted(payload, field)
    schema_config = gate.get("schema")
    if isinstance(schema_config, str):
        if schema_config != "canonical_environment":
            return StageGateResult(
                gate_type="json_schema",
                passed=False,
                message=f"Unknown JSON schema reference: {schema_config}",
                metadata={"field": field},
            )
        from SynthChat.schemas.environment_schema import _build_canonical_environment_schema

        schema = _build_canonical_environment_schema()
    elif isinstance(schema_config, Mapping):
        schema = dict(schema_config)
    else:
        return StageGateResult(
            gate_type="json_schema",
            passed=False,
            message="JSON schema gate requires a mapping schema or known schema reference.",
            metadata={"field": field},
        )
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(value)
    except jsonschema.ValidationError as exc:
        path = ".".join(str(part) for part in exc.absolute_path)
        location = f" at {path}" if path else ""
        return StageGateResult(
            gate_type="json_schema",
            passed=False,
            message=f"Field '{field}' failed JSON schema{location}: {exc.message}",
            metadata={"field": field, "path": path},
        )
    except jsonschema.SchemaError as exc:
        return StageGateResult(
            gate_type="json_schema",
            passed=False,
            message=f"Configured JSON schema is invalid: {exc.message}",
            metadata={"field": field},
        )
    return StageGateResult(gate_type="json_schema", passed=True, metadata={"field": field})


def _gate_no_placeholder_strings(payload: Mapping[str, Any], gate: Mapping[str, Any]) -> StageGateResult:
    field = str(gate.get("field") or "value")
    value = _resolve_dotted(payload, field)
    patterns = [
        r"\{\{[^}]+\}\}",
        r"<[^>\n]+>",
        r"\bTODO\b",
        r"\bTBD\b",
        r"\.\.\.",
        r"```",
    ]
    configured_patterns = gate.get("patterns")
    if isinstance(configured_patterns, list):
        patterns.extend(str(pattern) for pattern in configured_patterns if str(pattern or "").strip())
    compiled = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    matches: List[str] = []
    for text in _collect_strings(value):
        for pattern in compiled:
            if pattern.search(text):
                matches.append(text[:120])
                break
    matches = matches[:5]
    return StageGateResult(
        gate_type="no_placeholder_strings",
        passed=not matches,
        message=None if not matches else f"Field '{field}' contains placeholder or banned strings.",
        metadata={"field": field, "matches": matches},
    )


def _gate_min_fixture_items(payload: Mapping[str, Any], gate: Mapping[str, Any]) -> StageGateResult:
    field = str(gate.get("field") or "value.environment.fixture")
    fixture = _resolve_dotted(payload, field)
    if not isinstance(fixture, Mapping):
        return StageGateResult(
            gate_type="min_fixture_items",
            passed=False,
            message=f"Field '{field}' is not a fixture mapping.",
            metadata={"field": field},
        )
    directory_count = _count_fixture_collection(fixture.get("directories"))
    file_count = _count_fixture_collection(fixture.get("files"))
    note_count = _count_fixture_collection(fixture.get("notes"))
    total = directory_count + file_count + note_count
    requirements = {
        "min_directories": int(gate.get("min_directories") or 0),
        "min_files": int(gate.get("min_files") or 0),
        "min_notes": int(gate.get("min_notes") or 0),
        "min_total": int(gate.get("min_total") or 0),
    }
    failures = []
    if directory_count < requirements["min_directories"]:
        failures.append(f"directories {directory_count} < {requirements['min_directories']}")
    if file_count < requirements["min_files"]:
        failures.append(f"files {file_count} < {requirements['min_files']}")
    if note_count < requirements["min_notes"]:
        failures.append(f"notes {note_count} < {requirements['min_notes']}")
    if total < requirements["min_total"]:
        failures.append(f"total {total} < {requirements['min_total']}")
    return StageGateResult(
        gate_type="min_fixture_items",
        passed=not failures,
        message=None if not failures else "Fixture item minimums not met: " + "; ".join(failures),
        metadata={
            "field": field,
            "directories": directory_count,
            "files": file_count,
            "notes": note_count,
            "total": total,
            **requirements,
        },
    )


def _gate_required_mapping_keys(payload: Mapping[str, Any], gate: Mapping[str, Any]) -> StageGateResult:
    field = str(gate.get("field") or "value")
    value = _resolve_dotted(payload, field)
    keys = gate.get("keys")
    if not isinstance(keys, list):
        keys = []
    if not isinstance(value, Mapping):
        return StageGateResult(
            gate_type="required_mapping_keys",
            passed=False,
            message=f"Field '{field}' is not a mapping.",
            metadata={"field": field, "missing": list(keys)},
        )
    missing = [str(key) for key in keys if str(key) not in value]
    return StageGateResult(
        gate_type="required_mapping_keys",
        passed=not missing,
        message=None if not missing else f"Field '{field}' is missing required keys: {', '.join(missing)}",
        metadata={"field": field, "missing": missing},
    )


def _count_fixture_collection(value: Any) -> int:
    if isinstance(value, Mapping):
        return len(value)
    if isinstance(value, (list, tuple)):
        return len(value)
    return 0


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
    "json_schema": _gate_json_schema,
    "no_placeholder_strings": _gate_no_placeholder_strings,
    "min_fixture_items": _gate_min_fixture_items,
    "required_mapping_keys": _gate_required_mapping_keys,
}
