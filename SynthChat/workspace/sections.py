"""SynthChat Workspace Section Builders - Render individual sections of the workspace prompt.

Location: SynthChat/workspace/sections.py
Purpose: Build formatted text sections (workspaces, prompts, tools, notes, etc.)
         that are composed into the full mocked workspace system prompt.
Usage: Called by workspace/renderer.py to construct each prompt section.
"""

from typing import Any, Dict, List, Optional


def _render_available_workspaces(workspaces: List[Dict[str, Any]]) -> str:
    if not isinstance(workspaces, list) or not workspaces:
        return ""
    lines: List[str] = []
    for workspace in workspaces:
        if not isinstance(workspace, dict):
            continue
        lines.append(f'- {workspace.get("name", "Workspace")} (id: "{workspace.get("id", "")}")')
        description = workspace.get("description")
        if description:
            lines.append(f"  Description: {description}")
        root_folder = workspace.get("root_folder")
        if root_folder is not None:
            lines.append(f"  Root folder: {root_folder}")
        lines.append("")
    if lines:
        lines.append("Use memoryManager with loadWorkspace mode to get full workspace context.")
    return "\n".join(lines).strip()


def _render_available_prompts(prompts: List[Dict[str, Any]]) -> str:
    if not isinstance(prompts, list) or not prompts:
        return ""
    lines: List[str] = []
    for prompt in prompts:
        if not isinstance(prompt, dict):
            continue
        prompt_id = prompt.get("id")
        name = prompt.get("name", "Prompt")
        lines.append(f"- {prompt_id} - {name}" if prompt_id else f"- {name}")
        purpose = prompt.get("purpose") or prompt.get("description")
        if purpose:
            lines.append(f"  Purpose: {purpose}")
    return "\n".join(lines).strip()


def _tool_wrapper_name(tool_schema: Optional[Dict[str, Any]]) -> str:
    if not isinstance(tool_schema, dict):
        return "useTools"
    wrapper_cfg = tool_schema.get("tool_format") or {}
    return str(wrapper_cfg.get("wrapper") or "useTools").strip() or "useTools"


def _render_available_tools(tool_schema: Optional[Dict[str, Any]]) -> str:
    if not isinstance(tool_schema, dict):
        return ""

    wrapper_name = _tool_wrapper_name(tool_schema)
    lines: List[str] = [
        f"Use the `{wrapper_name}` wrapper for tool calls.",
        "Required wrapper context fields: sessionId, workspaceId, memory, goal.",
        "",
    ]

    tools = tool_schema.get("tools") or {}
    for agent in sorted(tools.keys()):
        agent_tools = tools.get(agent)
        if not isinstance(agent_tools, list) or not agent_tools:
            continue
        lines.append(f"{agent}:")
        for tool in agent_tools:
            if not isinstance(tool, dict):
                continue
            tool_name = str(tool.get("name") or "").strip()
            params = tool.get("params") or {}
            required = ", ".join(str(item) for item in params.get("required") or []) or "-"
            optional = ", ".join(str(item) for item in params.get("optional") or []) or "-"
            if tool_name:
                lines.append(f"- {tool_name}: required [{required}] optional [{optional}]")
        lines.append("")

    return "\n".join(lines).strip()


def _build_selected_workspace_section(name: str, workspace_id: str, payload: str) -> str:
    return "\n".join(
        [
            f'<selected_workspace name="{name}" id="{workspace_id}">',
            "This workspace is currently selected.",
            "",
            payload,
            "</selected_workspace>",
        ]
    )


def _build_session_context_section(session_id: str, workspace_id: str) -> str:
    return "\n".join(
        [
            "<session_context>",
            "IMPORTANT: When using tools, include these values in your tool call parameters:",
            "",
            f'- sessionId: "{session_id}"',
            f'- workspaceId: "{workspace_id}" (current workspace)',
            "",
            'Include these in the "context" parameter of your tool calls.',
            "</session_context>",
        ]
    )


def _build_wrapped_section(tag: str, content: str) -> str:
    clean = str(content or "").strip()
    if not clean:
        return ""
    return f"<{tag}>\n{clean}\n</{tag}>"


def _render_extra_sections(extra_sections: List[Dict[str, Any]]) -> str:
    rendered: List[str] = []
    for section in extra_sections:
        if not isinstance(section, dict):
            continue
        tag = str(section.get("tag", "")).strip()
        content = str(section.get("content", "")).strip()
        if tag and content:
            rendered.append(_build_wrapped_section(tag, content))
    return "\n\n".join(rendered)


def _render_note_contents(note_entries: Any) -> str:
    if not note_entries:
        return ""
    lines: List[str] = []
    for entry in note_entries:
        if not isinstance(entry, dict):
            continue
        path = str(entry.get("path", "")).strip()
        content = str(entry.get("content", "")).rstrip()
        if not path:
            continue
        lines.append(f"- {path}")
        if content:
            for line in content.splitlines():
                lines.append(f"  {line}")
        else:
            lines.append("  ")
        lines.append("")
    return "\n".join(lines).strip()
