"""Helpers for config-driven tool-call format detection and recovery."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

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


@lru_cache(maxsize=1)
def get_configured_wrapper_specs() -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    formats = load_tool_call_formats()

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
            }
        )

    return specs


def match_configured_wrapper(args: Any, function_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if not isinstance(args, dict):
        return None

    for spec in get_configured_wrapper_specs():
        if function_name and function_name == spec["wrapper_name"]:
            return spec
        required = spec["required_fields"]
        if required and all(isinstance(args.get(field), str) and str(args.get(field)).strip() for field in required):
            return spec

    return None


def sanitize_wrapper_string_fields(args: Any, function_name: Optional[str] = None) -> Dict[str, Any]:
    if not isinstance(args, dict):
        return {}

    spec = match_configured_wrapper(args, function_name=function_name)
    if spec is None:
        return args

    cleaned = dict(args)
    if "tool" in cleaned and isinstance(cleaned["tool"], str):
        cleaned["tool"] = sanitize_extracted_tool_value(cleaned["tool"], spec)
    return cleaned


def sanitize_extracted_tool_value(value: str, spec: Dict[str, Any]) -> str:
    for field_name in spec.get("field_names", []):
        if field_name == "tool":
            continue
        marker = f',"{field_name}":'
        if marker in value:
            value = value.split(marker, 1)[0]
    while value.endswith('"') and value.count('"') % 2 == 1:
        value = value[:-1]
    return value.strip()


def extract_lenient_wrapper_arguments(args_raw: Any) -> Dict[str, Any]:
    if not isinstance(args_raw, str):
        return {}

    for spec in get_configured_wrapper_specs():
        extracted: Dict[str, Any] = {}

        for field_name in spec.get("string_fields", []):
            if field_name == "tool":
                continue
            match = re.search(rf'"{re.escape(field_name)}"\s*:\s*"((?:\\.|[^"\\])*)"', args_raw, re.S)
            if match:
                try:
                    extracted[field_name] = json.loads(f'"{match.group(1)}"')
                except json.JSONDecodeError:
                    extracted[field_name] = _decode_lenient_cli_string(match.group(1))

        if "tool" in spec.get("field_names", []):
            other_fields = [name for name in spec.get("field_names", []) if name != "tool"]
            if other_fields:
                alternation = "|".join(re.escape(name) for name in other_fields)
                pattern = rf'"tool"\s*:\s*"(?P<tool>.*?)(?<!\\)"\s*(?:,\s*"(?P<next>{alternation})"\s*:|\s*}})'
            else:
                pattern = r'"tool"\s*:\s*"(?P<tool>.*?)(?<!\\)"\s*}'
            match = re.search(pattern, args_raw, re.S)
            if match:
                extracted["tool"] = sanitize_extracted_tool_value(
                    _decode_lenient_cli_string(match.group("tool").strip()),
                    spec,
                )

        required = spec.get("required_fields") or []
        if required and all(isinstance(extracted.get(field), str) and str(extracted.get(field)).strip() for field in required):
            return extracted

    return {}
