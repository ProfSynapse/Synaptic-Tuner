"""SynthChat Workspace Renderer - Compose mocked workspace system prompts.

Location: SynthChat/workspace/renderer.py
Purpose: Main entry point for rendering production-style mocked workspace system prompts
         from structured config. Composes fixture helpers and section builders.
Usage: from SynthChat.workspace.renderer import render_mocked_workspace_system_prompt
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
    _build_session_context_section,
    _build_wrapped_section,
    _render_extra_sections,
    _render_note_contents,
)


def render_mocked_workspace_system_prompt(
    system_context: Dict[str, Any],
    environment_config: Dict[str, Any],
    tool_schema: Optional[Dict[str, Any]] = None,
) -> str:
    """Render production-style mocked workspace prompt from structured config."""
    session_id = str(system_context.get("session_id") or "session_eval_001")
    workspace_id = str(system_context.get("workspace_id") or "default")
    fixture = _merged_fixture_from_config(environment_config)

    available_workspaces = system_context.get("available_workspaces") or []
    available_prompts = system_context.get("available_prompts") or []
    selected_workspace = dict(system_context.get("selected_workspace") or {})
    note_contents = system_context.get("note_contents")
    note_paths = system_context.get("note_content_paths") or []
    extra_sections = system_context.get("extra_sections") or []

    selected_workspace.setdefault("id", workspace_id)
    selected_workspace.setdefault("name", "Current Workspace")

    matched_workspace = None
    for workspace in available_workspaces:
        if isinstance(workspace, dict) and workspace.get("id") == selected_workspace.get("id"):
            matched_workspace = workspace
            break

    context_payload = dict(selected_workspace.get("context") or {})
    context_payload.setdefault("id", selected_workspace.get("id"))
    context_payload.setdefault("name", selected_workspace.get("name"))
    if matched_workspace:
        context_payload.setdefault("description", matched_workspace.get("description"))
        context_payload.setdefault("rootFolder", matched_workspace.get("root_folder", ""))
    context_payload.setdefault("rootFolder", selected_workspace.get("root_folder", ""))

    workspace_structure = selected_workspace.get("workspace_structure") or _workspace_structure_from_fixture(fixture)
    recent_files = selected_workspace.get("recent_files") or []
    key_files = selected_workspace.get("key_files") or []
    workflows = selected_workspace.get("workflows") or []
    preferences = selected_workspace.get("preferences", "")
    sessions = selected_workspace.get("sessions") or []

    selected_workspace_json = json.dumps(
        {
            "context": context_payload,
            "workspaceStructure": workspace_structure,
            "recentFiles": recent_files,
            "keyFiles": key_files,
            "workflows": workflows,
            "preferences": preferences,
            "sessions": sessions,
        },
        indent=2,
    )

    if not note_contents:
        note_contents = _note_entries_from_fixture(fixture, note_paths=note_paths)

    sections = [
        _build_session_context_section(session_id, workspace_id),
        _build_wrapped_section("vault_structure", _vault_structure_text_from_fixture(fixture)),
        _build_wrapped_section("available_workspaces", _render_available_workspaces(available_workspaces)),
        _build_wrapped_section("available_prompts", _render_available_prompts(available_prompts)),
        _build_wrapped_section("available_tools", _render_available_tools(tool_schema)),
        _build_selected_workspace_section(
            selected_workspace.get("name") or context_payload.get("name") or "Current Workspace",
            selected_workspace.get("id") or context_payload.get("id") or workspace_id,
            selected_workspace_json,
        ),
        _build_wrapped_section("note_contents", _render_note_contents(note_contents)),
        _render_extra_sections(extra_sections),
        str(system_context.get("assistant_instructions", "")).strip(),
    ]
    return "\n\n".join(section for section in sections if section)
