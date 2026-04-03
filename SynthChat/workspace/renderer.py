"""SynthChat Workspace Renderer - Compose mocked workspace system prompts.

Location: SynthChat/workspace/renderer.py
Purpose: Main entry point for rendering production-style mocked workspace system prompts.
         Uses config-driven rendering via workspace_formats.yaml for flexible section composition.
Usage: from SynthChat.workspace.renderer import render_workspace_prompt
"""

import json
from typing import Any, Dict, List, Optional

from .fixture_helpers import (
    _merged_fixture_from_config,
    _workspace_structure_from_fixture,
    _vault_structure_text_from_fixture,
    _note_entries_from_fixture,
)
from .sections import (
    _render_available_workspaces,
    _render_available_prompts,
    _render_available_tools,
    _build_selected_workspace_section,
    _build_wrapped_section,
    _render_extra_sections,
    _render_note_contents,
)


# ---------------------------------------------------------------------------
# Config-driven workspace prompt renderer
# ---------------------------------------------------------------------------

def render_workspace_prompt(
    system_context: Dict[str, Any],
    environment_config: Dict[str, Any],
    tool_schema: Optional[Dict[str, Any]],
    format_config: Dict[str, Any],
    tool_call_format: Optional[Dict[str, Any]] = None,
) -> str:
    """Render workspace system prompt from format config.

    Iterates over format_config["sections"], resolving each section's content
    from its source, applying templates, and wrapping in XML tags.

    Args:
        system_context: Runtime context (session_id, workspace_id, etc.)
        environment_config: Environment fixture config.
        tool_schema: Tool schema dict (agents, tools, params).
        format_config: Resolved workspace format from workspace_formats.yaml.
        tool_call_format: Resolved tool-call format for available_tools rendering.
    """
    defaults = format_config.get("defaults") or {}
    session_id = str(system_context.get("session_id") or defaults.get("session_id", "session_eval_001"))
    workspace_id = str(system_context.get("workspace_id") or "default")
    fixture = _merged_fixture_from_config(environment_config)

    # Precompute data sources used by section renderers
    available_workspaces = system_context.get("available_workspaces") or []
    available_prompts = system_context.get("available_prompts") or []
    selected_workspace = dict(system_context.get("selected_workspace") or {})
    note_contents = system_context.get("note_contents")
    note_paths = system_context.get("note_content_paths") or []
    extra_sections = system_context.get("extra_sections") or []

    default_ws_name = defaults.get("workspace_name", "Current Workspace")
    selected_workspace.setdefault("id", workspace_id)
    selected_workspace.setdefault("name", default_ws_name)

    # Build selected_workspace JSON payload from config fields
    selected_workspace_json = _build_selected_workspace_json(
        selected_workspace, available_workspaces, workspace_id, fixture, format_config
    )

    if not note_contents:
        note_contents = _note_entries_from_fixture(fixture, note_paths=note_paths)

    # Template variables for {placeholder} substitution
    template_vars = dict(system_context)
    template_vars["session_id"] = session_id
    template_vars["workspace_id"] = workspace_id

    # Built-in source renderers
    source_renderers = {
        "vault_structure": lambda: _vault_structure_text_from_fixture(fixture),
        "available_workspaces": lambda: _render_available_workspaces(available_workspaces),
        "available_prompts": lambda: _render_available_prompts(available_prompts),
        "available_tools": lambda: _render_available_tools(tool_schema, tool_call_format),
        "selected_workspace": lambda: selected_workspace_json,
        "note_contents": lambda: _render_note_contents(note_contents),
        "extra_sections": lambda: _render_extra_sections(extra_sections),
        "assistant_instructions": lambda: str(system_context.get("assistant_instructions", "")).strip(),
    }

    # Iterate over format_config sections
    sections_config = format_config.get("sections") or []
    rendered_sections: List[str] = []

    for section_def in sections_config:
        if not isinstance(section_def, dict):
            continue

        tag = str(section_def.get("tag", "")).strip() or None
        source = str(section_def.get("source", "")).strip() or None
        template = section_def.get("template")
        is_optional = bool(section_def.get("optional", False))
        is_raw = bool(section_def.get("raw", False))

        # Resolve content
        content = ""
        if template is not None:
            # Template: perform {placeholder} substitution
            content = _substitute_template(str(template), template_vars)
        elif source in source_renderers:
            content = source_renderers[source]()
        elif source and source.startswith("system_context."):
            key = source.split(".", 1)[1]
            content = str(system_context.get(key, "")).strip()

        content = str(content or "").strip()

        if is_optional and not content:
            continue

        # Special handling for selected_workspace (name/id attributes)
        if source == "selected_workspace" and tag == "selected_workspace":
            ws_name = selected_workspace.get("name") or default_ws_name
            ws_id = selected_workspace.get("id") or workspace_id
            rendered_sections.append(
                _build_selected_workspace_section(ws_name, ws_id, content)
            )
            continue

        # Special handling for extra_sections (already rendered with tags)
        if source == "extra_sections":
            if content:
                rendered_sections.append(content)
            continue

        # Wrap in XML tag or emit raw
        if is_raw or not tag:
            if content:
                rendered_sections.append(content)
        else:
            wrapped = _build_wrapped_section(tag, content)
            if wrapped:
                rendered_sections.append(wrapped)

    return "\n\n".join(section for section in rendered_sections if section)


def _substitute_template(template: str, variables: Dict[str, Any]) -> str:
    """Simple {placeholder} substitution using a variables dict.

    Only replaces known placeholders; leaves unknown ones untouched.
    """
    result = template
    for key, value in variables.items():
        if not isinstance(key, str):
            continue
        placeholder = "{" + key + "}"
        if placeholder in result:
            result = result.replace(placeholder, str(value) if value is not None else "")
    return result


def _build_selected_workspace_json(
    selected_workspace: Dict[str, Any],
    available_workspaces: List[Dict[str, Any]],
    workspace_id: str,
    fixture: Dict[str, Any],
    format_config: Dict[str, Any],
) -> str:
    """Build the JSON payload for selected_workspace using config-defined fields."""
    from .fixture_helpers import _workspace_structure_from_fixture

    matched_workspace = None
    for ws in available_workspaces:
        if isinstance(ws, dict) and ws.get("id") == selected_workspace.get("id"):
            matched_workspace = ws
            break

    context_payload = dict(selected_workspace.get("context") or {})
    context_payload.setdefault("id", selected_workspace.get("id"))
    context_payload.setdefault("name", selected_workspace.get("name"))
    if matched_workspace:
        context_payload.setdefault("description", matched_workspace.get("description"))
        context_payload.setdefault("rootFolder", matched_workspace.get("root_folder", ""))
    context_payload.setdefault("rootFolder", selected_workspace.get("root_folder", ""))

    # Build payload from config-defined field list
    fields = format_config.get("selected_workspace_fields") or [
        "context", "workspaceStructure", "recentFiles", "keyFiles",
        "workflows", "preferences", "sessions",
    ]

    # Map field names to data sources
    workspace_structure = selected_workspace.get("workspace_structure") or _workspace_structure_from_fixture(fixture)
    field_data = {
        "context": context_payload,
        "workspaceStructure": workspace_structure,
        "recentFiles": selected_workspace.get("recent_files") or [],
        "keyFiles": selected_workspace.get("key_files") or [],
        "workflows": selected_workspace.get("workflows") or [],
        "preferences": selected_workspace.get("preferences", ""),
        "sessions": selected_workspace.get("sessions") or [],
    }

    payload = {}
    for field in fields:
        field_str = str(field).strip()
        if field_str in field_data:
            payload[field_str] = field_data[field_str]
        elif field_str in selected_workspace:
            payload[field_str] = selected_workspace[field_str]
        else:
            payload[field_str] = None

    return json.dumps(payload, indent=2)


