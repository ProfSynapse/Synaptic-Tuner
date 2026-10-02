"""Helpers for config-driven tool-call format detection and recovery."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional, Sequence

from SynthChat.config.format_resolver import load_tool_call_formats


def _decode_lenient_cli_string(value: str) -> str:
    return (
        value
        .replace("\\n", "\n")
        .replace("\\r", "\r")
        .replace("\\t", "\t")
        .replace('\\"', '"')
        .replace("\\\\", "\\")
    )


def _command_escapes(wrapper_name: str, raw: Any) -> Dict[str, str]:
    """Validate a format's ``command_escapes``: escape character -> decoded text."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"tool-call format {wrapper_name!r}: command_escapes must be a mapping")
    escapes: Dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or len(key) != 1 or key in {"\\", '"'}:
            raise ValueError(
                f"tool-call format {wrapper_name!r}: command_escapes key {key!r} must be one character "
                "other than a backslash or double quote"
            )
        if not isinstance(value, str):
            raise ValueError(f"tool-call format {wrapper_name!r}: command_escapes[{key!r}] must be a string")
        escapes[key] = value
    return escapes


def build_wrapper_specs(formats: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Derive wrapper specs from a tool-call format registry.

    ``formats`` has the shape returned by ``load_tool_call_formats()``: format
    name -> format config. Formats without a ``wrapper_name`` call tools
    directly and contribute no spec.

    Optional format keys carried into the spec:
    - ``command_field``: the argument field holding a free-form command
      string, which lenient recovery reads up to the next declared field and
      CLI expansion parses into catalog commands.
    - ``command_escapes``: backslash escapes decoded inside double-quoted
      arguments of that command string.
    - ``prompt_bound_fields``: field -> list of ``{pattern, in_tag}`` sources
      naming where the system prompt states the field's allowed values.
    """
    specs: List[Dict[str, Any]] = []

    for _, fmt in (formats or {}).items():
        if not isinstance(fmt, dict):
            continue
        wrapper_name = str(fmt.get("wrapper_name") or "").strip()
        if not wrapper_name:
            continue

        argument_fields = fmt.get("argument_fields") or {}
        properties = dict(argument_fields.get("properties") or {})
        properties.update(fmt.get("extra_argument_fields") or {})
        required_fields = list(fmt.get("argument_required") or argument_fields.get("required") or [])
        field_names = list(properties.keys())
        command_field = str(fmt.get("command_field") or "").strip() or None
        if command_field is not None and command_field not in properties:
            raise ValueError(
                f"tool-call format {wrapper_name!r}: command_field {command_field!r} is not a declared argument field"
            )
        string_fields = [
            name
            for name, spec in properties.items()
            if isinstance(spec, dict) and str(spec.get("type", "string")).strip().lower() == "string"
        ]

        specs.append(
            {
                "wrapper_name": wrapper_name,
                "required_fields": required_fields,
                "field_names": field_names,
                "string_fields": string_fields,
                "command_escapes": _command_escapes(wrapper_name, fmt.get("command_escapes")),
                "properties": properties,
                "command_field": command_field,
                "prompt_bound_fields": dict(fmt.get("prompt_bound_fields") or {}),
            }
        )

    return specs


@lru_cache(maxsize=1)
def get_configured_wrapper_specs() -> List[Dict[str, Any]]:
    """Wrapper specs for the engine's configured tool-call formats."""
    return build_wrapper_specs(load_tool_call_formats())


def match_configured_wrapper(
    args: Any,
    function_name: Optional[str] = None,
    specs: Optional[Sequence[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Return the wrapper spec ``args`` belongs to, or None for a direct tool call.

    ``specs`` defaults to the configured registry; pass ``build_wrapper_specs``
    output to match against a different tool-call format registry.
    """
    if not isinstance(args, dict):
        return None

    for spec in get_configured_wrapper_specs() if specs is None else specs:
        if function_name and function_name == spec["wrapper_name"]:
            return spec
        required = spec["required_fields"]
        if required and all(isinstance(args.get(field), str) and str(args.get(field)).strip() for field in required):
            return spec

    return None


def sanitize_wrapper_string_fields(
    args: Any,
    function_name: Optional[str] = None,
    specs: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Trim fields leaked into a wrapper's configured ``command_field``."""
    if not isinstance(args, dict):
        return {}

    spec = match_configured_wrapper(args, function_name=function_name, specs=specs)
    if spec is None:
        return args

    command_field = spec.get("command_field")
    cleaned = dict(args)
    if command_field and isinstance(cleaned.get(command_field), str):
        cleaned[command_field] = sanitize_command_value(cleaned[command_field], spec)
    return cleaned


def sanitize_command_value(value: str, spec: Dict[str, Any]) -> str:
    """Cut a command string at the first leaked ``,"<declared field>":`` marker."""
    for field_name in spec.get("field_names", []):
        if field_name == spec.get("command_field"):
            continue
        marker = f',"{field_name}":'
        if marker in value:
            value = value.split(marker, 1)[0]
    while value.endswith('"') and value.count('"') % 2 == 1:
        value = value[:-1]
    return value.strip()


def extract_lenient_wrapper_arguments(
    args_raw: Any,
    specs: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Recover a configured wrapper's fields from malformed arguments JSON."""
    if not isinstance(args_raw, str):
        return {}

    for spec in get_configured_wrapper_specs() if specs is None else specs:
        extracted: Dict[str, Any] = {}
        command_field = spec.get("command_field")

        for field_name in spec.get("string_fields", []):
            if field_name == command_field:
                continue
            match = re.search(rf'"{re.escape(field_name)}"\s*:\s*"((?:\\.|[^"\\])*)"', args_raw, re.S)
            if match:
                try:
                    extracted[field_name] = json.loads(f'"{match.group(1)}"')
                except json.JSONDecodeError:
                    extracted[field_name] = _decode_lenient_cli_string(match.group(1))

        if command_field:
            field = re.escape(command_field)
            other_fields = [name for name in spec.get("field_names", []) if name != command_field]
            if other_fields:
                alternation = "|".join(re.escape(name) for name in other_fields)
                pattern = rf'"{field}"\s*:\s*"(?P<command>.*?)(?<!\\)"\s*(?:,\s*"(?P<next>{alternation})"\s*:|\s*}})'
            else:
                pattern = rf'"{field}"\s*:\s*"(?P<command>.*?)(?<!\\)"\s*}}'
            match = re.search(pattern, args_raw, re.S)
            if match:
                extracted[command_field] = sanitize_command_value(
                    _decode_lenient_cli_string(match.group("command").strip()),
                    spec,
                )

        required = spec.get("required_fields") or []
        if required and all(isinstance(extracted.get(field), str) and str(extracted.get(field)).strip() for field in required):
            return extracted

    return {}
