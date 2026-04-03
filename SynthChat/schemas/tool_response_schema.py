"""SynthChat Tool Response Schema - JSON schema and prompt builders for tool-call responses.

Location: SynthChat/schemas/tool_response_schema.py
Purpose: Build the JSON schema and generation prompt used when the LLM generates
         tool-calling assistant responses. Schema structure is driven by a format_config
         dict loaded from tool_call_formats.yaml rather than hardcoded.
Usage: Called by generator.py during assistant response generation stage.
       Imported by tests for schema validation.
"""

from typing import Any, Dict, List, Optional, Tuple

from ..config.format_resolver import get_default_tool_call_format


# ---------------------------------------------------------------------------
# Schema builder
# ---------------------------------------------------------------------------

def build_tool_response_schema(
    format_config: Dict[str, Any],
    allowed_tools: Optional[List[str]] = None,
    context_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build JSON Schema for a tool-call response from a format config.

    Args:
        format_config: Resolved format dict from tool_call_formats.yaml.
        allowed_tools: List of allowed tool names (e.g. "fileManager_read").
        context_overrides: Optional dict of context field overrides (e.g. session_id const).
    """
    allowed_tools = [t for t in (allowed_tools or []) if isinstance(t, str) and "_" in t]
    context_overrides = context_overrides or {}

    wrapper_name = format_config.get("wrapper_name")

    if wrapper_name is None:
        # Native mode: no wrapper, each tool gets its own tool_calls entry
        return _build_native_schema(allowed_tools)

    return _build_wrapper_schema(format_config, wrapper_name, allowed_tools, context_overrides)


def _build_wrapper_schema(
    format_config: Dict[str, Any],
    wrapper_name: str,
    allowed_tools: List[str],
    context_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Build schema for wrapper-style tool calls (single function wrapping multiple calls)."""
    agent_enum = sorted({t.split("_", 1)[0] for t in allowed_tools}) if allowed_tools else []
    tool_enum = sorted({t.split("_", 1)[1] for t in allowed_tools}) if allowed_tools else []

    # Build context properties from config
    ctx_cfg = format_config.get("context_fields") or {}
    ctx_properties_cfg = ctx_cfg.get("properties") or {}
    context_properties: Dict[str, Any] = {}
    for field_name, field_schema in ctx_properties_cfg.items():
        if isinstance(field_schema, dict):
            context_properties[field_name] = dict(field_schema)
        else:
            context_properties[field_name] = {"type": "string", "minLength": 1}

    # Apply context overrides (e.g. sessionId/workspaceId const values)
    for field_name, const_value in context_overrides.items():
        if const_value and field_name in context_properties:
            context_properties[field_name] = {"const": const_value}

    context_required = list(ctx_cfg.get("required") or [])

    # Build call_item properties from config
    call_cfg = format_config.get("call_item") or {}
    call_properties_cfg = (call_cfg.get("properties") or {}).copy()
    call_required = list(call_cfg.get("required") or [])

    call_properties: Dict[str, Any] = {}
    for field_name, field_schema in call_properties_cfg.items():
        if isinstance(field_schema, dict):
            call_properties[field_name] = dict(field_schema)
        else:
            call_properties[field_name] = {"type": "string", "minLength": 1}

    # Apply agent/tool enum constraints if available
    if agent_enum and "agent" in call_properties:
        call_properties["agent"] = {"enum": agent_enum}
    if tool_enum and "tool" in call_properties:
        call_properties["tool"] = {"enum": tool_enum}

    # Build arguments properties
    arguments_properties: Dict[str, Any] = {}

    if context_properties:
        arguments_properties["context"] = {
            "type": "object",
            "additionalProperties": True,
            "properties": context_properties,
            "required": context_required,
        }

    if call_properties:
        arguments_properties["calls"] = {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": call_properties,
                "required": call_required,
            },
        }

    # Add extra argument fields from config (e.g. strategy)
    extra_fields = format_config.get("extra_argument_fields") or {}
    for field_name, field_schema in extra_fields.items():
        if isinstance(field_schema, dict):
            arguments_properties[field_name] = dict(field_schema)

    argument_required = list(format_config.get("argument_required") or [])

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "content": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "null"},
                ]
            },
            "tool_calls": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "array",
                        "maxItems": 0,
                    },
                    {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "id": {"type": "string", "minLength": 1},
                                "type": {"const": "function"},
                                "function": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "name": {"const": wrapper_name},
                                        "arguments": {
                                            "type": "object",
                                            "additionalProperties": False,
                                            "properties": arguments_properties,
                                            "required": argument_required,
                                        },
                                    },
                                    "required": ["name", "arguments"],
                                },
                            },
                            "required": ["id", "type", "function"],
                        },
                    },
                ]
            },
        },
        "required": ["content", "tool_calls"],
    }


def _build_native_schema(allowed_tools: List[str]) -> Dict[str, Any]:
    """Build schema for native (no-wrapper) tool calls."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "content": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "null"},
                ]
            },
            "tool_calls": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "array",
                        "maxItems": 0,
                    },
                    {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "id": {"type": "string", "minLength": 1},
                                "type": {"const": "function"},
                                "function": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "name": {"type": "string", "minLength": 1},
                                        "arguments": {
                                            "type": "object",
                                            "additionalProperties": True,
                                        },
                                    },
                                    "required": ["name", "arguments"],
                                },
                            },
                            "required": ["id", "type", "function"],
                        },
                    },
                ]
            },
        },
        "required": ["content", "tool_calls"],
    }


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_tool_generation_prompt(
    format_config: Dict[str, Any],
    base_prompt: str,
    allowed_tools: List[str],
) -> str:
    """Build the generation prompt from format config instructions.

    Performs {placeholder} substitution for {wrapper_name} and {allowed_tools_csv}.
    """
    wrapper_name = format_config.get("wrapper_name") or ""
    instructions = format_config.get("generation_instructions") or []

    lines = []
    for instruction in instructions:
        line = str(instruction)
        line = line.replace("{wrapper_name}", wrapper_name)
        if allowed_tools:
            line = line.replace("{allowed_tools_csv}", ", ".join(allowed_tools))
        lines.append(line)

    if allowed_tools:
        formatted = ", ".join(allowed_tools)
        lines.append(f"Allowed concrete tools for this task: {formatted}.")

    lines.append("")
    lines.append(base_prompt)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Wrapper name resolver
# ---------------------------------------------------------------------------

def resolve_wrapper_name(
    format_config: Dict[str, Any],
    tool_schema: Optional[Dict[str, Any]] = None,
) -> str:
    """Resolve wrapper function name.

    Priority: format_config["wrapper_name"] > tool_schema.tool_format.wrapper > ""
    Returns empty string when no wrapper is configured (native tool-call mode).
    """
    wrapper = format_config.get("wrapper_name")
    if wrapper:
        return str(wrapper).strip()

    if isinstance(tool_schema, dict):
        wrapper_cfg = tool_schema.get("tool_format") or {}
        wrapper = str(wrapper_cfg.get("wrapper") or "").strip()
        if wrapper:
            return wrapper

    return ""


# ---------------------------------------------------------------------------
# Unchanged helpers (no config dependency)
# ---------------------------------------------------------------------------

def _resolve_allowed_tool_names(
    *,
    scenario: Dict[str, Any],
    tool_schema: Optional[Dict[str, Any]],
) -> List[str]:
    configured = []
    for key in ("expected_tools", "acceptable_tools"):
        values = scenario.get(key) or []
        if isinstance(values, list):
            configured.extend(
                str(value).strip() for value in values
                if isinstance(value, str) and str(value).strip() and str(value).strip() != "TEXT_ONLY"
            )
    tool_name = str(scenario.get("tool") or "").strip()
    if tool_name:
        configured.append(tool_name)

    configured = sorted(dict.fromkeys(configured))
    if configured:
        return configured

    if not isinstance(tool_schema, dict):
        return []

    names: List[str] = []
    for agent, tools in (tool_schema.get("tools") or {}).items():
        if not isinstance(tools, list):
            continue
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            tool_name = str(tool.get("name", "")).strip()
            if tool_name:
                names.append(f"{agent}_{tool_name}")
    return sorted(dict.fromkeys(names))


def _resolve_context_defaults(
    *,
    system_context: Optional[Dict[str, Any]],
) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(system_context, dict):
        return None, None

    session_id = system_context.get("session_id")
    workspace_id = system_context.get("workspace_id")
    if not workspace_id and isinstance(system_context.get("selected_workspace"), dict):
        workspace_id = system_context["selected_workspace"].get("id")

    session_id = str(session_id).strip() if isinstance(session_id, str) and str(session_id).strip() else None
    workspace_id = str(workspace_id).strip() if isinstance(workspace_id, str) and str(workspace_id).strip() else None
    return session_id, workspace_id
