"""SynthChat Tool Response Schema - JSON schema and prompt builders for use_tools responses.

Location: SynthChat/schemas/tool_response_schema.py
Purpose: Build the JSON schema and generation prompt used when the LLM generates
         tool-calling assistant responses in the useTools wrapper format.
Usage: Called by generator.py during assistant response generation stage.
       Imported by tests for schema validation.
"""

from typing import Any, Dict, List, Optional, Tuple


def _build_use_tools_response_schema(
    wrapper_name: str = "useTools",
    allowed_tools: Optional[List[str]] = None,
    session_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
) -> Dict[str, Any]:
    allowed_tools = [tool for tool in (allowed_tools or []) if isinstance(tool, str) and "_" in tool]
    agent_enum = sorted({tool.split("_", 1)[0] for tool in allowed_tools})
    tool_enum = sorted({tool.split("_", 1)[1] for tool in allowed_tools})

    context_properties: Dict[str, Any] = {
        "sessionId": {"type": "string", "minLength": 1},
        "workspaceId": {"type": "string", "minLength": 1},
        "memory": {"type": "string", "minLength": 1},
        "goal": {"type": "string", "minLength": 1},
    }
    if session_id:
        context_properties["sessionId"] = {"const": session_id}
    if workspace_id:
        context_properties["workspaceId"] = {"const": workspace_id}

    call_properties: Dict[str, Any] = {
        "agent": {"type": "string", "minLength": 1},
        "tool": {"type": "string", "minLength": 1},
        "params": {
            "type": "object",
            "additionalProperties": True,
        },
    }
    if agent_enum:
        call_properties["agent"] = {"enum": agent_enum}
    if tool_enum:
        call_properties["tool"] = {"enum": tool_enum}

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
                                            "properties": {
                                                "context": {
                                                    "type": "object",
                                                    "additionalProperties": True,
                                                    "properties": context_properties,
                                                    "required": ["sessionId", "workspaceId", "memory", "goal"],
                                                },
                                                "calls": {
                                                    "type": "array",
                                                    "minItems": 1,
                                                    "items": {
                                                        "type": "object",
                                                        "additionalProperties": False,
                                                        "properties": call_properties,
                                                        "required": ["agent", "tool", "params"],
                                                    },
                                                },
                                                "strategy": {
                                                    "type": "string",
                                                    "enum": ["serial", "parallel"],
                                                },
                                            },
                                            "required": ["context", "calls"],
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


def _build_use_tools_generation_prompt(
    *,
    base_prompt: str,
    wrapper_name: str,
    allowed_tools: List[str],
) -> str:
    lines = [
        "Return a single JSON object only.",
        "Your job is to either call tools or respond via text.",
        f"If tools are needed, use exactly one tool_calls entry whose function.name is '{wrapper_name}'.",
        "If no tool call is needed, respond with normal text in content and set tool_calls to null or [].",
        "Inside function.arguments.calls, each item must use this exact shape:",
        '{"agent": "AgentName", "tool": "toolName", "params": {...}}',
        "Do not use dotted names like 'contentManager.read' for either agent or tool.",
        "Do not use nested wrappers like params.tool, params.parameters, or assistant as the agent name.",
        "Put the real tool arguments directly inside params.",
        "Use content as null when the response is tool-only.",
        "When the task is already complete, when clarification is needed, or when you are asked for a final confirmation, respond with text instead of calling tools.",
    ]
    if allowed_tools:
        formatted = ", ".join(allowed_tools)
        lines.append(f"Allowed concrete tools for this task: {formatted}.")
    lines.append("")
    lines.append(base_prompt)
    return "\n".join(lines)
