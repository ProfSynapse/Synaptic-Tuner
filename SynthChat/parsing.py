"""SynthChat Parsing - Response parsing and environment normalization.

Location: SynthChat/parsing.py
Purpose: Parse LLM-generated assistant responses (tool calls, thinking blocks,
         direct JSON) and normalize generated environment payloads for
         downstream validation.
Usage: Called by SynthChatGenerator methods in generator.py during the
       assistant response and environment generation stages.
"""

import json
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional

from shared.validation.parsing.configured_formats import (
    extract_lenient_wrapper_arguments,
    match_configured_wrapper,
    sanitize_wrapper_string_fields,
)


def _repair_truncated_json(json_str: str) -> str:
    """Best-effort repair for under-closed JSON emitted by the model."""
    result = []
    stack = []
    in_string = False
    escape_next = False

    for char in json_str:
        result.append(char)

        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in "}]":
            if stack and stack[-1] == char:
                stack.pop()

    if in_string:
        result.append('"')

    while stack:
        result.append(stack.pop())

    return "".join(result)


def _normalize_tool_arguments(args_raw: Any, function_name: Optional[str] = None) -> Any:
    """Unwrap common nested argument wrapper shapes into the actual args object."""
    args = args_raw

    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            lenient = extract_lenient_wrapper_arguments(args)
            if lenient:
                return lenient
            try:
                args = json.loads(_repair_truncated_json(args))
            except json.JSONDecodeError:
                return args_raw

    # Some models nest the real args under {"arguments": {...}} or
    # {"function": "...", "arguments": {...}} despite direct-tool instructions.
    for _ in range(3):
        if isinstance(args, dict) and isinstance(args.get("function"), dict):
            args = args["function"].get("arguments", args)
            continue
        if isinstance(args, dict) and isinstance(args.get("tool_calls"), list) and args["tool_calls"]:
            first = args["tool_calls"][0]
            if isinstance(first, dict):
                fn = first.get("function")
                if isinstance(fn, dict):
                    args = fn.get("arguments", args)
                    continue
        if isinstance(args, dict) and "arguments" in args:
            inner = args.get("arguments")
            if isinstance(inner, str):
                try:
                    args = json.loads(inner)
                    continue
                except json.JSONDecodeError:
                    try:
                        args = json.loads(_repair_truncated_json(inner))
                        continue
                    except json.JSONDecodeError:
                        break
            if isinstance(inner, dict):
                args = inner
                continue
        break

    if isinstance(args, dict):
        args = sanitize_wrapper_string_fields(args, function_name=function_name)

    return args


def stringify_assistant_message(response: Any) -> str:
    """Convert an assistant response dict to a plain text summary."""
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
        return "\n\n".join(parts).strip() or json.dumps(response)
    return str(response)


def parse_assistant_response(content: str, scenario: Dict) -> Dict:
    """Parse assistant response for tool calls and thinking.

    Handles four cases:
    1. Direct JSON: response is already {"content":..., "tool_calls":[...]}
    2. Text-only: content is text, no tool_calls
    3. Tool-only: content is null or empty, tool_calls present
    4. Thinking+tool: content contains <thinking>...</thinking>, tool_calls present

    Returns:
        Assistant message dict with role, content, and optional tool_calls
    """
    if content is None:
        content = ""

    content_stripped = content.strip()

    # Strip markdown code fences if present
    if content_stripped.startswith('```'):
        first_newline = content_stripped.find('\n')
        if first_newline > 0:
            content_stripped = content_stripped[first_newline + 1:]
        if content_stripped.rstrip().endswith('```'):
            content_stripped = content_stripped.rstrip()[:-3].strip()

    if content_stripped.startswith('{') and '"tool_calls"' in content_stripped:
        try:
            parsed = json.loads(content_stripped)
        except json.JSONDecodeError:
            fixed = content_stripped
            fixed = re.sub(r'\n\s*', ' ', fixed)
            try:
                parsed = json.loads(fixed)
            except json.JSONDecodeError:
                parsed = None

        if parsed and "tool_calls" in parsed and isinstance(parsed["tool_calls"], list):
                message = {"role": "assistant"}
                message["content"] = parsed.get("content")

                tool_calls = []
                for tc in parsed["tool_calls"]:
                    normalized_tc = {
                        "id": tc.get("id", f"call_{len(tool_calls)+1:04d}"),
                        "type": tc.get("type", "function"),
                        "function": {}
                    }
                    fn = tc.get("function", {})
                    normalized_tc["function"]["name"] = fn.get("name", "")

                    args = fn.get("arguments", "{}")
                    normalized_args = _normalize_tool_arguments(args, function_name=fn.get("name", ""))
                    normalized_name = fn.get("name", "")
                    wrapper_spec = match_configured_wrapper(normalized_args, function_name=normalized_name)
                    if wrapper_spec is not None:
                        normalized_name = wrapper_spec["wrapper_name"]

                    normalized_tc["function"]["name"] = normalized_name
                    if isinstance(normalized_args, dict):
                        normalized_tc["function"]["arguments"] = json.dumps(normalized_args)
                    else:
                        normalized_tc["function"]["arguments"] = args

                    tool_calls.append(normalized_tc)

                message["tool_calls"] = tool_calls
                return message
        if parsed and parsed.get("tool_calls") is None and isinstance(parsed.get("content"), str):
            return {
                "role": "assistant",
                "content": parsed.get("content"),
            }

    parsed_object = parse_json_object(content_stripped)
    wrapper_spec = match_configured_wrapper(parsed_object)
    if wrapper_spec is not None:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_0001",
                    "type": "function",
                    "function": {
                        "name": wrapper_spec["wrapper_name"],
                        "arguments": json.dumps(parsed_object),
                    },
                }
            ],
        }

    # Extract thinking block if present
    thinking_match = re.search(r'<thinking>(.*?)</thinking>', content, re.DOTALL)
    thinking_block = None
    if thinking_match:
        thinking_block = f"<thinking>{thinking_match.group(1)}</thinking>"
        content_without_thinking = content.replace(thinking_match.group(0), '').strip()
    else:
        content_without_thinking = content

    # Detect tool calls - look for patterns like:
    # "tool_name(args)" or "Use: tool_name" or JSON-like tool definitions
    tool_calls = []
    tool_pattern = r'(\w+Manager_\w+)\s*\((.*?)\)'
    tool_matches = re.finditer(tool_pattern, content_without_thinking)

    call_id_counter = 1
    for match in tool_matches:
        tool_name = match.group(1)
        tool_args = match.group(2).strip()

        try:
            if not tool_args.startswith('{'):
                tool_args = '{' + tool_args + '}'
            arguments_str = tool_args
        except:
            arguments_str = "{}"

        tool_calls.append({
            "id": f"call_{call_id_counter:04d}",
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": arguments_str
            }
        })
        call_id_counter += 1

    # If scenario specifies a tool but we didn't detect it, create from scenario
    if not tool_calls and scenario.get("type") == "tool":
        tool_name = scenario.get("tool", "")
        if tool_name:
            json_match = re.search(r'\{[^}]+\}', content_without_thinking)
            arguments_str = json_match.group(0) if json_match else "{}"
            parsed_args = parse_json_object(arguments_str) if arguments_str else None
            wrapper_spec = match_configured_wrapper(parsed_args, function_name=tool_name)
            if wrapper_spec is not None:
                tool_name = wrapper_spec["wrapper_name"]
                arguments_str = json.dumps(parsed_args)

            tool_calls.append({
                "id": "call_0001",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": arguments_str
                }
            })

    # Build final message
    message = {"role": "assistant"}

    if tool_calls:
        message["tool_calls"] = tool_calls
        message["content"] = thinking_block if thinking_block else None
    else:
        message["content"] = content

    return message


def parse_json_object(content: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON object from model output, tolerating code fences."""
    candidate = (content or "").strip()
    if not candidate:
        return None

    if candidate.startswith("```"):
        first_newline = candidate.find("\n")
        if first_newline != -1:
            candidate = candidate[first_newline + 1:]
        candidate = candidate.rstrip()
        if candidate.endswith("```"):
            candidate = candidate[:-3].rstrip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            parsed = json.loads(candidate[start:end + 1])
        except json.JSONDecodeError:
            return None

    return parsed if isinstance(parsed, dict) else None


def _normalize_generated_assertion(assertion: Any) -> Any:
    """Normalize field name aliases in a single assertion dict."""
    if not isinstance(assertion, dict):
        return assertion

    normalized = deepcopy(assertion)
    assertion_type = str(normalized.get("type") or "").strip()

    # Generated environments often use more natural field names than the
    # validator contract. Normalize those aliases here so environment seeds
    # stay strict downstream without having to tutor the model in the prompt.
    if assertion_type in {"file_contains", "file_not_contains"}:
        if "text" not in normalized and "content" in normalized:
            normalized["text"] = normalized.pop("content")
    elif assertion_type == "frontmatter_has_key":
        if "field" not in normalized and "key" in normalized:
            normalized["field"] = normalized.pop("key")
    elif assertion_type in {"frontmatter_field_equals", "frontmatter_field_contains"}:
        if "field" not in normalized and "key" in normalized:
            normalized["field"] = normalized.pop("key")
        if assertion_type == "frontmatter_field_equals" and "value" not in normalized and "content" in normalized:
            normalized["value"] = normalized.pop("content")
        if assertion_type == "frontmatter_field_contains" and "text" not in normalized and "content" in normalized:
            normalized["text"] = normalized.pop("content")

    return normalized


def normalize_generated_environment(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Accept a few config-driven shapes for generated environment payloads."""
    normalized = deepcopy(payload)
    environment = normalized.get("environment")

    if not isinstance(environment, dict):
        if any(key in normalized for key in ("fixture", "assertions", "allowed_tools", "max_steps", "execution")):
            environment = {
                key: normalized.pop(key)
                for key in ("fixture", "assertions", "allowed_tools", "max_steps", "execution")
                if key in normalized
            }
        elif any(key in normalized for key in ("directories", "files", "notes", "folders")):
            environment = {
                "fixture": {
                    key: normalized.pop(key)
                    for key in ("directories", "files", "notes", "folders")
                    if key in normalized
                }
            }
        else:
            environment = {}

    if environment and not environment.get("fixture"):
        fixture = {
            key: normalized.pop(key)
            for key in ("directories", "files", "notes", "folders")
            if key in normalized
        }
        if fixture:
            environment["fixture"] = fixture

    normalized["environment"] = environment
    assertions = environment.get("assertions")
    if isinstance(assertions, list):
        environment["assertions"] = [_normalize_generated_assertion(assertion) for assertion in assertions]
    normalized.setdefault("system_context", {})
    normalized.setdefault("task_context", {})
    return normalized
