"""Config-driven deterministic stage gates."""

from __future__ import annotations

from collections import Counter
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


def _gate_json_schema(payload: Mapping[str, Any], gate: Mapping[str, Any]) -> StageGateResult:
    field = str(gate.get("field") or "value")
    value = _resolve_dotted(payload, field)
    schema_spec = gate.get("schema")
    schema = _resolve_json_schema(schema_spec)
    if not isinstance(schema, Mapping):
        return StageGateResult(
            gate_type="json_schema",
            passed=False,
            message=f"Unknown or missing JSON schema for gate: {schema_spec!r}.",
            metadata={"field": field, "schema": schema_spec},
        )
    try:
        import jsonschema

        jsonschema.validate(instance=value, schema=schema)
    except Exception as exc:
        return StageGateResult(
            gate_type="json_schema",
            passed=False,
            message=f"Field '{field}' failed JSON schema validation: {exc}",
            metadata={"field": field, "schema": schema_spec},
        )
    return StageGateResult(
        gate_type="json_schema",
        passed=True,
        metadata={"field": field, "schema": schema_spec},
    )


def _gate_no_placeholder_strings(payload: Mapping[str, Any], gate: Mapping[str, Any]) -> StageGateResult:
    field = str(gate.get("field") or "value")
    value = _resolve_dotted(payload, field)
    raw_patterns = gate.get("patterns")
    if not isinstance(raw_patterns, list) or not raw_patterns:
        raw_patterns = [r"\.\.\.", r"\{\{[^}]+\}\}", r"<[^>]+>", r"\bTODO\b", r"\bTBD\b"]
    compiled = []
    for pattern in raw_patterns:
        try:
            compiled.append(re.compile(str(pattern), re.IGNORECASE))
        except re.error:
            return StageGateResult(
                gate_type="no_placeholder_strings",
                passed=False,
                message=f"Invalid placeholder pattern: {pattern!r}.",
                metadata={"field": field},
            )

    matches: List[Dict[str, str]] = []
    for path, text in _collect_strings_with_paths(value):
        for pattern in compiled:
            match = pattern.search(text)
            if match:
                matches.append(
                    {
                        "path": path,
                        "pattern": pattern.pattern,
                        "match": match.group(0),
                    }
                )
                break
    return StageGateResult(
        gate_type="no_placeholder_strings",
        passed=not matches,
        message=None if not matches else f"Found {len(matches)} placeholder-like string(s).",
        metadata={"field": field, "matches": matches[:20], "match_count": len(matches)},
    )


def _gate_required_mapping_keys(payload: Mapping[str, Any], gate: Mapping[str, Any]) -> StageGateResult:
    field = str(gate.get("field") or "value")
    value = _resolve_dotted(payload, field)
    keys = gate.get("keys")
    if not isinstance(keys, list):
        keys = []
    expected = [str(key) for key in keys if str(key).strip()]
    if not isinstance(value, Mapping):
        return StageGateResult(
            gate_type="required_mapping_keys",
            passed=False,
            message=f"Field '{field}' expected a mapping but got {type(value).__name__}.",
            metadata={"field": field, "missing_keys": expected},
        )
    missing = [key for key in expected if key not in value]
    return StageGateResult(
        gate_type="required_mapping_keys",
        passed=not missing,
        message=None if not missing else f"Field '{field}' is missing required key(s): {', '.join(missing)}.",
        metadata={"field": field, "missing_keys": missing},
    )


def _gate_min_fixture_items(payload: Mapping[str, Any], gate: Mapping[str, Any]) -> StageGateResult:
    field = str(gate.get("field") or "value.environment.fixture")
    fixture = _resolve_dotted(payload, field)
    directories: List[Any] = []
    files: List[Any] = []
    if isinstance(fixture, Mapping):
        raw_directories = fixture.get("directories")
        if isinstance(raw_directories, list):
            directories = raw_directories
        raw_files = fixture.get("files")
        if isinstance(raw_files, list):
            files = raw_files
        elif isinstance(raw_files, Mapping):
            files = list(raw_files.keys())
    min_directories = int(gate.get("min_directories", 0) or 0)
    min_files = int(gate.get("min_files", 0) or 0)
    min_total = int(gate.get("min_total", 0) or 0)
    directory_count = len(directories)
    file_count = len(files)
    total = directory_count + file_count
    failures = []
    if directory_count < min_directories:
        failures.append(f"directories {directory_count} < {min_directories}")
    if file_count < min_files:
        failures.append(f"files {file_count} < {min_files}")
    if total < min_total:
        failures.append(f"total {total} < {min_total}")
    return StageGateResult(
        gate_type="min_fixture_items",
        passed=not failures,
        message=None if not failures else f"Fixture item thresholds failed: {', '.join(failures)}.",
        metadata={
            "field": field,
            "directory_count": directory_count,
            "file_count": file_count,
            "total": total,
            "min_directories": min_directories,
            "min_files": min_files,
            "min_total": min_total,
        },
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


def _gate_list_empty(payload: Mapping[str, Any], gate: Mapping[str, Any]) -> StageGateResult:
    field = str(gate.get("field") or "").strip()
    actual = _resolve_dotted(payload, field) if field else None
    is_list = isinstance(actual, list)
    passed = is_list and len(actual) == 0
    reason = None
    if not is_list:
        reason = f"Field '{field}' expected an empty list but got {type(actual).__name__}."
    elif actual:
        reason = f"Field '{field}' expected an empty list but had {len(actual)} item(s)."
    return StageGateResult(
        gate_type="list_empty",
        passed=passed,
        message=reason,
        metadata={"field": field, "actual_length": len(actual) if is_list else None},
    )


def _gate_all_items_field_equals(payload: Mapping[str, Any], gate: Mapping[str, Any]) -> StageGateResult:
    field = str(gate.get("field") or "").strip()
    item_field = str(gate.get("item_field") or "").strip()
    expected = gate.get("value")
    allow_empty = bool(gate.get("allow_empty", False))
    actual = _resolve_dotted(payload, field) if field else None
    if not isinstance(actual, list):
        return StageGateResult(
            gate_type="all_items_field_equals",
            passed=False,
            message=f"Field '{field}' expected a list but got {type(actual).__name__}.",
            metadata={"field": field, "item_field": item_field, "expected": expected},
        )
    if not actual and not allow_empty:
        return StageGateResult(
            gate_type="all_items_field_equals",
            passed=False,
            message=f"Field '{field}' expected at least one item.",
            metadata={"field": field, "item_field": item_field, "expected": expected},
        )
    failures: List[Dict[str, Any]] = []
    for index, item in enumerate(actual):
        item_value = _resolve_dotted(item, item_field) if item_field else item
        if item_value != expected:
            failures.append({"index": index, "actual": item_value})
    return StageGateResult(
        gate_type="all_items_field_equals",
        passed=not failures,
        message=None if not failures else (
            f"Field '{field}' had {len(failures)} item(s) where '{item_field}' did not equal {expected!r}."
        ),
        metadata={
            "field": field,
            "item_field": item_field,
            "expected": expected,
            "failures": failures,
        },
    )


def _gate_expected_cli_commands_executed(payload: Mapping[str, Any], gate: Mapping[str, Any]) -> StageGateResult:
    expected_field = str(gate.get("expected_field") or "task_context.expected_command_sequence").strip()
    executed_field = str(gate.get("executed_field") or "environment_result.executed_tools").strip()
    expected_value = _resolve_dotted(payload, expected_field)
    executed_value = _resolve_dotted(payload, executed_field)
    if not isinstance(expected_value, list):
        return StageGateResult(
            gate_type="expected_cli_commands_executed",
            passed=False,
            message=f"Field '{expected_field}' expected a list.",
            metadata={"expected_field": expected_field, "executed_field": executed_field},
        )
    if not isinstance(executed_value, list):
        return StageGateResult(
            gate_type="expected_cli_commands_executed",
            passed=False,
            message=f"Field '{executed_field}' expected a list.",
            metadata={"expected_field": expected_field, "executed_field": executed_field},
        )

    expected_commands = [_normalize_cli_command(item) for item in expected_value if isinstance(item, str) and item.strip()]
    status_value = gate.get("status", "ok")
    require_status = None if status_value is None else str(status_value)
    renderers = gate.get("renderers") if isinstance(gate.get("renderers"), Mapping) else {}

    executed_commands: List[str] = []
    for item in executed_value:
        if not isinstance(item, Mapping):
            continue
        if require_status is not None and str(item.get("status")) != require_status:
            continue
        raw_command = item.get("command") or item.get("tool")
        if isinstance(raw_command, str) and raw_command.strip():
            executed_commands.append(_normalize_cli_command(raw_command))
            continue
        name = str(item.get("name") or "").strip()
        template = renderers.get(name) if name else None
        if isinstance(template, str) and template.strip():
            rendered = _render_template_from_mapping(template, item)
            if rendered:
                executed_commands.append(_normalize_cli_command(rendered))

    expected_counts = Counter(expected_commands)
    executed_counts = Counter(executed_commands)
    missing: List[str] = []
    for command, count in expected_counts.items():
        missing_count = count - executed_counts.get(command, 0)
        missing.extend([command] * max(0, missing_count))

    order_failures: List[str] = []
    if bool(gate.get("require_order", False)) and not missing:
        search_start = 0
        for command in expected_commands:
            try:
                found_at = executed_commands.index(command, search_start)
            except ValueError:
                order_failures.append(command)
                break
            search_start = found_at + 1

    passed = not missing and not order_failures
    message = None
    if missing:
        message = "Missing expected CLI command(s): " + ", ".join(missing)
    elif order_failures:
        message = "Expected CLI command order was not preserved at: " + ", ".join(order_failures)
    return StageGateResult(
        gate_type="expected_cli_commands_executed",
        passed=passed,
        message=message,
        metadata={
            "expected_field": expected_field,
            "executed_field": executed_field,
            "expected_commands": expected_commands,
            "executed_commands": executed_commands,
            "missing_commands": missing,
            "order_failures": order_failures,
        },
    )


def _resolve_dotted(value: Any, dotted: str) -> Any:
    current = value
    for part in [piece for piece in str(dotted or "").split(".") if piece]:
        if isinstance(current, Mapping):
            current = current.get(part)
        else:
            return None
    return current


def _render_template_from_mapping(template: str, value: Mapping[str, Any]) -> Optional[str]:
    def replace(match: re.Match[str]) -> str:
        field = match.group(1).strip()
        resolved = _resolve_dotted(value, field)
        if resolved is None:
            raise KeyError(field)
        return str(resolved)

    try:
        return re.sub(r"\{([^{}]+)\}", replace, template)
    except KeyError:
        return None


def _normalize_cli_command(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


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


def _collect_strings_with_paths(value: Any, path: str = "$") -> Iterable[tuple[str, str]]:
    if isinstance(value, str):
        yield path, value
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _collect_strings_with_paths(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _collect_strings_with_paths(item, f"{path}[{index}]")


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


def _resolve_json_schema(schema_spec: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(schema_spec, Mapping):
        return schema_spec
    if str(schema_spec or "").strip() == "canonical_environment":
        from .schemas.environment_schema import _build_canonical_environment_schema

        return _build_canonical_environment_schema()
    return None


_GATE_HANDLERS = {
    "non_empty_text": _gate_non_empty_text,
    "plain_text": _gate_plain_text,
    "no_tool_names": _gate_no_tool_names,
    "no_exact_paths_from_context": _gate_no_exact_paths_from_context,
    "environment_payload_shape": _gate_environment_payload_shape,
    "json_schema": _gate_json_schema,
    "no_placeholder_strings": _gate_no_placeholder_strings,
    "required_mapping_keys": _gate_required_mapping_keys,
    "min_fixture_items": _gate_min_fixture_items,
    "field_equals": _gate_field_equals,
    "list_empty": _gate_list_empty,
    "all_items_field_equals": _gate_all_items_field_equals,
    "expected_cli_commands_executed": _gate_expected_cli_commands_executed,
}
