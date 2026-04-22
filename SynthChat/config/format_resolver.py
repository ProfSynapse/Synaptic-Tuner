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
    "wrapper_name": None,
    "argument_fields": {
        "required": [],
        "properties": {},
    },
    "extra_argument_fields": {},
    "argument_required": [],
    "generation_instructions": [
        "Return a single JSON object only.",
        "Use the configured tool-call format for this scenario.",
        "If no tool call is needed, respond with normal text in content and set tool_calls to null or [].",
    ],
    "available_tools_instruction": "Use the configured tool-call format for this scenario.",
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
    3. "default" from registry
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
                    "Include these in the tool-call context fields required by the active format.\n"
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
