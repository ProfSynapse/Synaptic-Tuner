"""Generic, lossless response views for assertion-based evaluation."""
from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional

try:
    from shared.validation.parsing import parse_qwen_tool_calls
    from shared.validation.parsing.tool_call_parser import parse_gemma_tool_calls, is_gemma_tool_call
except Exception:  # pragma: no cover - optional parsing helpers
    parse_qwen_tool_calls = None
    parse_gemma_tool_calls = None
    is_gemma_tool_call = None


def build_response_view(response: Any, raw_response: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Build a generic view used by config assertions.

    The view preserves the raw response and adds normalized convenience fields.
    Normalization is syntactic only: it parses JSON argument strings, but never
    maps emitted commands or wrappers into semantic tool names.
    """
    view: Dict[str, Any] = {
        "raw_api_message": deepcopy(raw_response) if raw_response is not None else None,
        "raw": deepcopy(response),
        "content": "",
        "content_json": None,
        "tool_calls": [],
        "raw_tool_calls": [],
    }

    if isinstance(response, Mapping):
        content = response.get("content")
        if content is not None:
            view["content"] = content if isinstance(content, str) else str(content)
            view["content_json"] = _parse_json_maybe(view["content"])

        raw_tool_calls = response.get("tool_calls")
        if isinstance(raw_tool_calls, list):
            view["raw_tool_calls"] = deepcopy(raw_tool_calls)
            view["tool_calls"] = [_normalize_tool_call(call) for call in raw_tool_calls]
        elif isinstance(view["content"], str):
            extracted = _extract_tool_calls_from_text(view["content"])
            if extracted:
                view["raw_tool_calls"] = deepcopy(extracted)
                view["tool_calls"] = [_normalize_tool_call(call) for call in extracted]
        return view

    if isinstance(response, str):
        view["content"] = response
        view["content_json"] = _parse_json_maybe(response)
        extracted = _extract_tool_calls_from_text(response)
        if extracted:
            view["raw_tool_calls"] = deepcopy(extracted)
            view["tool_calls"] = [_normalize_tool_call(call) for call in extracted]
        return view

    if response is not None:
        view["content"] = str(response)
    return view


def _normalize_tool_call(call: Any) -> Dict[str, Any]:
    if not isinstance(call, Mapping):
        return {"raw": deepcopy(call)}

    normalized = deepcopy(dict(call))
    function = normalized.get("function")
    if isinstance(function, Mapping):
        function = dict(function)
        function["arguments"] = _parse_arguments(function.get("arguments"))
        normalized["function"] = function
        return normalized

    arguments = normalized.get("arguments")
    if arguments is not None:
        normalized["arguments"] = _parse_arguments(arguments)
    return normalized


def _parse_arguments(arguments: Any) -> Any:
    if isinstance(arguments, str):
        parsed = _parse_json_maybe(arguments)
        return parsed if parsed is not None else arguments
    return deepcopy(arguments)


def _parse_json_maybe(value: Any) -> Any:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def _extract_tool_calls_from_text(text: str) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    if not isinstance(text, str) or not text.strip():
        return calls

    calls.extend(_extract_plain_tool_call_blocks(text))

    if is_gemma_tool_call is not None and parse_gemma_tool_calls is not None:
        try:
            if is_gemma_tool_call(text):
                parsed = parse_gemma_tool_calls(text)
                if isinstance(parsed, list):
                    calls.extend(call for call in parsed if isinstance(call, dict))
        except Exception:
            pass

    if not calls and parse_qwen_tool_calls is not None and "<tool_call>" in text:
        try:
            parsed = parse_qwen_tool_calls(text)
            if isinstance(parsed, list):
                calls.extend(call for call in parsed if isinstance(call, dict))
        except Exception:
            pass

    return calls


def _extract_plain_tool_call_blocks(text: str) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    search_from = 0
    marker = "tool_call:"

    while True:
        start = text.find(marker, search_from)
        if start == -1:
            return calls

        name_start = start + len(marker)
        line_end = text.find("\n", name_start)
        if line_end == -1:
            return calls
        name = text[name_start:line_end].strip()
        if not name:
            search_from = line_end + 1
            continue

        arguments_marker = "arguments:"
        args_label = text.find(arguments_marker, line_end)
        if args_label == -1:
            search_from = line_end + 1
            continue

        json_start = text.find("{", args_label + len(arguments_marker))
        if json_start == -1:
            search_from = args_label + len(arguments_marker)
            continue

        json_end = _find_matching_json_object_end(text, json_start)
        if json_end is None:
            search_from = json_start + 1
            continue

        raw_args = text[json_start:json_end]
        parsed_args = _parse_json_maybe(raw_args)
        if isinstance(parsed_args, dict):
            calls.append({"name": name, "arguments": parsed_args})

        search_from = json_end


def _find_matching_json_object_end(text: str, start: int) -> Optional[int]:
    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return None
