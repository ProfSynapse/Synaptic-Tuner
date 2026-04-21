"""SynthChat Config Format Resolver - Resolution logic for per-scenario format lookup.

Location: SynthChat/config/format_resolver.py
Purpose: Resolve which tool-call format, workspace format, and label mappings
         apply to a given scenario by following a layered priority chain.
Usage: Called by generator.py at generation time to resolve per-scenario configs.
"""

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.yaml_loader import load_yaml

_CONFIG_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Built-in defaults (used when YAML files are missing or as merge base)
# ---------------------------------------------------------------------------

_DEFAULT_TOOL_CALL_FORMAT: Dict[str, Any] = {
    "wrapper_name": "useTools",
    "argument_fields": {
        "required": ["sessionId", "workspaceId", "memory", "goal", "tool"],
        "properties": {
            "sessionId": {"type": "string", "minLength": 1},
            "workspaceId": {"type": "string", "minLength": 1},
            "memory": {"type": "string", "minLength": 1},
            "goal": {"type": "string", "minLength": 1},
            "constraints": {"type": "string", "minLength": 1},
            "tool": {"type": "string", "minLength": 1},
        },
    },
    "extra_argument_fields": {
        "strategy": {
            "type": "string",
            "enum": ["serial", "parallel"],
        },
    },
    "argument_required": ["sessionId", "workspaceId", "memory", "goal", "tool"],
    "generation_instructions": [
        "Return a single JSON object only.",
        "Your job is to either call tools or respond via text.",
        "If tools are needed, use exactly one tool_calls entry whose function.name is '{wrapper_name}'.",
        "If no tool call is needed, respond with normal text in content and set tool_calls to null or [].",
        "The function.arguments value must be a single JSON object with top-level keys: workspaceId, sessionId, memory, goal, tool, and optional constraints/strategy.",
        "The tool field must be a CLI command string, not a nested object.",
        "Use CLI command names like 'storage move', 'content read', 'prompt execute-prompts', 'memory load-workspace', and 'search search-content'.",
        "When multiple commands are needed, join them in the tool field as a comma-separated sequence.",
        "Do not use context/calls wrappers, params objects, agent/tool arrays, or direct per-tool function names.",
        "Good: {\"workspaceId\":\"default\",\"sessionId\":\"session_123\",\"memory\":\"Need to inspect notes.\",\"goal\":\"Move a note and read it back.\",\"constraints\":\"Do not touch unrelated files.\",\"tool\":\"storage move \\\"notes/today.md\\\" \\\"archive/today.md\\\", content read \\\"archive/today.md\\\"\",\"strategy\":\"serial\"}",
        "Bad: {\"context\": {...}, \"calls\": [...]}, {\"tool_calls\": [...]}, or function.name='contentManager_read'",
        "Use content as null when the response is tool-only.",
        "When the task is already complete, when clarification is needed, or when you are asked for a final confirmation, respond with text instead of calling tools.",
    ],
    "available_tools_instruction": "Required wrapper context fields: {context_required_csv}.",
}

_DEFAULT_LABEL_MAPPINGS: Dict[str, Any] = {
    "issue_classifiers": [
        {"match": "expected tool(s) not executed", "label": "missing_expected_tool"},
        {"match": "no acceptable tool called", "label": "wrong_tool_called"},
        {"match": "front matter", "label": "frontmatter_missing"},
        {"match": "yaml front matter", "label": "frontmatter_missing"},
        {"match": "expected path to exist", "label": "path_state_mismatch"},
        {"match": "expected path to be absent", "label": "path_state_mismatch"},
        {"match": "does not contain expected text", "label": "content_mismatch"},
        {"match": "contains forbidden text", "label": "content_mismatch"},
        {"match": "failed reading", "label": "read_failure"},
        {"match": "is a directory", "label": "path_type_error"},
        {"match": "file exists", "label": "path_type_error"},
        {"match": "strict schema", "label": "schema_error"},
        {"match": "missing required args", "label": "schema_error"},
        {"match": "searchmanager_searchcontent", "label": "retrieval_missing"},
        {"match": "searchmanager_searchdirectory", "label": "retrieval_missing"},
        {"match": "clarification", "label": "clarification_expected"},
        {"match": "tool '", "match_also": "failed:", "label": "tool_runtime_error"},
    ],
    "behavior_rollups": {
        "behavior:retrieval_failure": ["missing_expected_tool", "retrieval_missing"],
        "behavior:structure_failure": ["frontmatter_missing"],
        "behavior:tool_execution_failure": ["wrong_tool_called", "tool_runtime_error"],
        "behavior:clarification_failure": ["clarification_expected"],
    },
    "failure_type_rollups": {
        "failure_type:environment": ["environment"],
        "failure_type:behavior": ["response", "thinking"],
        "failure_type:generation": [
            "system_prompt", "user", "system_generation", "user_generation",
            "assistant_generation", "environment_generation", "final",
        ],
    },
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_tool_call_formats(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load named tool-call format definitions.

    Returns dict mapping format names to their config dicts.
    Falls back to built-in defaults if file not found.
    """
    path = Path(config_path) if config_path else _CONFIG_DIR / "tool_call_formats.yaml"
    if path.is_file():
        data = load_yaml(path)
        if isinstance(data, dict) and "formats" in data:
            return data["formats"]
    return {"default": deepcopy(_DEFAULT_TOOL_CALL_FORMAT)}


def load_workspace_formats(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load named workspace prompt format definitions.

    Returns dict mapping format names to their config dicts.
    Falls back to built-in defaults if file not found.
    """
    path = Path(config_path) if config_path else _CONFIG_DIR / "workspace_formats.yaml"
    if path.is_file():
        data = load_yaml(path)
        if isinstance(data, dict) and "formats" in data:
            return data["formats"]
    return {"default": _default_workspace_format()}


def load_label_mappings(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load label mapping config.

    Returns dict with keys: issue_classifiers, behavior_rollups, failure_type_rollups.
    Falls back to built-in defaults if file not found.
    """
    path = Path(config_path) if config_path else _CONFIG_DIR / "label_mappings.yaml"
    if path.is_file():
        data = load_yaml(path)
        if isinstance(data, dict):
            return {
                "issue_classifiers": data.get("issue_classifiers", _DEFAULT_LABEL_MAPPINGS["issue_classifiers"]),
                "behavior_rollups": data.get("behavior_rollups", _DEFAULT_LABEL_MAPPINGS["behavior_rollups"]),
                "failure_type_rollups": data.get("failure_type_rollups", _DEFAULT_LABEL_MAPPINGS["failure_type_rollups"]),
            }
    return deepcopy(_DEFAULT_LABEL_MAPPINGS)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge override into a copy of base. Override wins on conflicts."""
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def resolve_tool_call_format(
    scenario: Dict[str, Any],
    formats_registry: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolve the tool call format for a scenario.

    Priority:
    1. scenario["tool_call_format"] as inline dict -> deep-merge with default
    2. scenario["tool_call_format"] as string -> lookup in registry
    3. scenario["tool_format"]["wrapper"] -> merge into default
    4. "default" from registry
    """
    default_format = formats_registry.get("default", _DEFAULT_TOOL_CALL_FORMAT)
    tcf = scenario.get("tool_call_format")

    if isinstance(tcf, dict):
        return _deep_merge(default_format, tcf)

    if isinstance(tcf, str):
        named = formats_registry.get(tcf)
        if named:
            return deepcopy(named)
        return deepcopy(default_format)

    # Check legacy tool_format.wrapper field
    tool_format = scenario.get("tool_format")
    if isinstance(tool_format, dict):
        wrapper = str(tool_format.get("wrapper") or "").strip()
        context_required = tool_format.get("context", {}).get("required") if isinstance(tool_format.get("context"), dict) else None
        if wrapper or context_required:
            merged = deepcopy(default_format)
            if wrapper:
                merged["wrapper_name"] = wrapper
            if isinstance(context_required, list) and context_required:
                merged["context_fields"]["required"] = context_required
            return merged

    return deepcopy(default_format)


def resolve_workspace_format(
    scenario: Dict[str, Any],
    formats_registry: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Resolve the workspace format for a scenario.

    Priority:
    1. scenario["workspace_format"] as inline dict
    2. scenario["workspace_format"] as string -> lookup in registry
    3. system_template == "mocked_workspace_vault" -> "default"
    4. None (no workspace rendering)
    """
    default_format = formats_registry.get("default")

    wsf = scenario.get("workspace_format")
    if isinstance(wsf, dict):
        if default_format:
            return _deep_merge(default_format, wsf)
        return deepcopy(wsf)

    if isinstance(wsf, str):
        named = formats_registry.get(wsf)
        if named:
            return deepcopy(named)
        return deepcopy(default_format) if default_format else None

    system_template = str(scenario.get("system_template") or "").strip()
    if system_template == "mocked_workspace_vault":
        return deepcopy(default_format) if default_format else None

    return None


def get_default_tool_call_format() -> Dict[str, Any]:
    """Return a copy of the built-in default tool call format."""
    return deepcopy(_DEFAULT_TOOL_CALL_FORMAT)


def get_default_label_mappings() -> Dict[str, Any]:
    """Return a copy of the built-in default label mappings."""
    return deepcopy(_DEFAULT_LABEL_MAPPINGS)


# ---------------------------------------------------------------------------
# Workspace format default (kept here to avoid circular imports)
# ---------------------------------------------------------------------------

def _default_workspace_format() -> Dict[str, Any]:
    """Return the built-in default workspace format config."""
    return {
        "sections": [
            {
                "tag": "session_context",
                "source": "session_context",
                "template": (
                    "IMPORTANT: When using tools, include these values in your tool call parameters:\n"
                    "\n"
                    '- sessionId: "{session_id}"\n'
                    '- workspaceId: "{workspace_id}" (current workspace)\n'
                    "\n"
                    'Include these as top-level fields in the useTools arguments payload.\n'
                ),
            },
            {"tag": "vault_structure", "source": "vault_structure", "optional": True},
            {"tag": "available_workspaces", "source": "available_workspaces", "optional": True},
            {"tag": "available_prompts", "source": "available_prompts", "optional": True},
            {"tag": "available_tools", "source": "available_tools", "optional": True},
            {"tag": "selected_workspace", "source": "selected_workspace"},
            {"tag": "note_contents", "source": "note_contents", "optional": True},
            {"source": "extra_sections"},
            {"source": "assistant_instructions", "raw": True},
        ],
        "selected_workspace_fields": [
            "context",
            "workspaceStructure",
            "recentFiles",
            "keyFiles",
            "workflows",
            "preferences",
            "sessions",
        ],
        "defaults": {
            "session_id": "session_eval_001",
            "workspace_name": "Current Workspace",
        },
    }
