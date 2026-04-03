"""Tests for SynthChat.workspace — renderer, sections, and fixture helpers."""
from __future__ import annotations

import json
import pytest
from SynthChat.workspace.sections import (
    _build_session_context_section,
    _build_selected_workspace_section,
    _build_wrapped_section,
    _render_available_prompts,
    _render_available_tools,
    _render_available_workspaces,
    _render_extra_sections,
    _render_note_contents,
    _tool_wrapper_name,
)
from SynthChat.workspace.renderer import render_mocked_workspace_system_prompt


# ---- _tool_wrapper_name ----

class TestToolWrapperName:
    def test_default_name(self):
        assert _tool_wrapper_name(None) == "useTools"

    def test_custom_wrapper(self):
        schema = {"tool_format": {"wrapper": "customWrapper"}}
        assert _tool_wrapper_name(schema) == "customWrapper"

    def test_empty_wrapper_falls_back(self):
        schema = {"tool_format": {"wrapper": ""}}
        assert _tool_wrapper_name(schema) == "useTools"

    def test_no_tool_format(self):
        assert _tool_wrapper_name({}) == "useTools"


# ---- _build_wrapped_section ----

class TestBuildWrappedSection:
    def test_wraps_content(self):
        result = _build_wrapped_section("my_tag", "hello")
        assert result == "<my_tag>\nhello\n</my_tag>"

    def test_empty_content_returns_empty(self):
        assert _build_wrapped_section("tag", "") == ""
        assert _build_wrapped_section("tag", None) == ""


# ---- _build_session_context_section ----

class TestBuildSessionContextSection:
    def test_includes_session_and_workspace_ids(self):
        result = _build_session_context_section("sess_1", "ws_1")
        assert "sess_1" in result
        assert "ws_1" in result
        assert "<session_context>" in result
        assert "</session_context>" in result


# ---- _build_selected_workspace_section ----

class TestBuildSelectedWorkspaceSection:
    def test_includes_name_and_id(self):
        result = _build_selected_workspace_section("My WS", "ws_1", '{"data": true}')
        assert 'name="My WS"' in result
        assert 'id="ws_1"' in result
        assert '{"data": true}' in result
        assert "<selected_workspace" in result


# ---- _render_available_workspaces ----

class TestRenderAvailableWorkspaces:
    def test_empty_list(self):
        assert _render_available_workspaces([]) == ""

    def test_none_input(self):
        assert _render_available_workspaces(None) == ""

    def test_renders_workspace_entries(self):
        workspaces = [
            {"name": "Project A", "id": "a", "description": "Main project"},
            {"name": "Project B", "id": "b"},
        ]
        result = _render_available_workspaces(workspaces)
        assert "Project A" in result
        assert '"a"' in result
        assert "Main project" in result
        assert "Project B" in result
        assert "memoryManager" in result


# ---- _render_available_prompts ----

class TestRenderAvailablePrompts:
    def test_empty_list(self):
        assert _render_available_prompts([]) == ""

    def test_renders_prompts(self):
        prompts = [{"id": "p1", "name": "Summarize", "purpose": "Summarize notes"}]
        result = _render_available_prompts(prompts)
        assert "p1" in result
        assert "Summarize" in result


# ---- _render_available_tools ----

class TestRenderAvailableTools:
    def test_none_schema(self):
        assert _render_available_tools(None) == ""

    def test_renders_tool_agents(self):
        schema = {
            "tools": {
                "fileManager": [
                    {"name": "read", "params": {"required": ["path"]}},
                    {"name": "write", "params": {"required": ["path", "content"]}},
                ],
            }
        }
        result = _render_available_tools(schema)
        assert "fileManager:" in result
        assert "read" in result
        assert "write" in result
        assert "useTools" in result  # default wrapper

    def test_custom_wrapper_in_output(self):
        schema = {
            "tool_format": {"wrapper": "myWrapper"},
            "tools": {"agent": [{"name": "doStuff", "params": {}}]},
        }
        result = _render_available_tools(schema)
        assert "myWrapper" in result


# ---- _render_note_contents ----

class TestRenderNoteContents:
    def test_empty_input(self):
        assert _render_note_contents([]) == ""
        assert _render_note_contents(None) == ""

    def test_renders_note_entries(self):
        entries = [
            {"path": "notes/test.md", "content": "Hello world"},
            {"path": "notes/empty.md", "content": ""},
        ]
        result = _render_note_contents(entries)
        assert "notes/test.md" in result
        assert "Hello world" in result
        assert "notes/empty.md" in result


# ---- _render_extra_sections ----

class TestRenderExtraSections:
    def test_empty_list(self):
        assert _render_extra_sections([]) == ""

    def test_renders_sections(self):
        sections = [
            {"tag": "custom_a", "content": "Alpha"},
            {"tag": "custom_b", "content": "Beta"},
        ]
        result = _render_extra_sections(sections)
        assert "<custom_a>" in result
        assert "Alpha" in result
        assert "<custom_b>" in result

    def test_skips_empty_content(self):
        sections = [{"tag": "empty", "content": ""}]
        assert _render_extra_sections(sections) == ""


# ---- render_mocked_workspace_system_prompt (integration) ----

class TestRenderMockedWorkspaceSystemPrompt:
    def test_basic_rendering(self):
        system_context = {
            "session_id": "sess_1",
            "workspace_id": "ws_1",
            "available_workspaces": [{"name": "Main", "id": "ws_1"}],
            "available_prompts": [],
            "selected_workspace": {"id": "ws_1", "name": "Main"},
        }
        environment_config = {
            "fixture": {
                "directories": ["notes/"],
                "files": {"notes/test.md": "content"},
            }
        }
        result = render_mocked_workspace_system_prompt(system_context, environment_config)
        assert "sess_1" in result
        assert "ws_1" in result
        assert "Main" in result

    def test_includes_note_contents(self):
        system_context = {
            "note_contents": [{"path": "notes/a.md", "content": "Hello"}],
        }
        environment_config = {"fixture": {}}
        result = render_mocked_workspace_system_prompt(system_context, environment_config)
        assert "notes/a.md" in result
        assert "Hello" in result

    def test_includes_tool_schema(self):
        system_context = {}
        environment_config = {"fixture": {}}
        tool_schema = {
            "tools": {"fileManager": [{"name": "read", "params": {"required": ["path"]}}]}
        }
        result = render_mocked_workspace_system_prompt(
            system_context, environment_config, tool_schema=tool_schema
        )
        assert "fileManager" in result
        assert "useTools" in result

    def test_extra_sections_rendered(self):
        system_context = {
            "extra_sections": [{"tag": "rules", "content": "Be helpful."}],
        }
        environment_config = {"fixture": {}}
        result = render_mocked_workspace_system_prompt(system_context, environment_config)
        assert "<rules>" in result
        assert "Be helpful." in result
