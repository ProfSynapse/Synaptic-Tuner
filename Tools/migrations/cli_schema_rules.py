"""Classification rules for the CLI-schema migration pipeline."""

from __future__ import annotations

from typing import Any, Dict, List


IN_SCOPE_NONTHINKING_AGENTS = (
    "contentManager",
    "memoryManager",
    "promptManager",
    "searchManager",
    "storageManager",
)


AUTO_TOOLS = {
    "contentManager": {"insert", "read", "replace", "setProperty", "write"},
    "memoryManager": {
        "archiveWorkspace",
        "createState",
        "listStates",
        "listWorkspaces",
        "loadState",
        "loadWorkspace",
    },
    "promptManager": {
        "archivePrompt",
        "createPrompt",
        "generateImage",
        "getPrompt",
        "listModels",
        "listPrompts",
        "updatePrompt",
    },
    "searchManager": {"searchContent", "searchDirectory", "searchMemory"},
    "storageManager": {"archive", "copy", "createFolder", "list", "move", "open"},
}


HEURISTIC_TOOLS = {
    "contentManager": {"update"},
    "memoryManager": {"createWorkspace"},
}


REGENERATE_TOOLS = {
    "promptManager": {"executePrompts"},
    "memoryManager": {"updateWorkspace"},
}


def classify_call(agent: str, tool: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Classify a normalized call for migration readiness."""
    reasons: List[str] = []

    if agent not in IN_SCOPE_NONTHINKING_AGENTS:
        return {"bucket": "out_of_scope", "reasons": [f"agent:{agent}"]}

    if tool in REGENERATE_TOOLS.get(agent, set()):
        return {"bucket": "regenerate", "reasons": ["tool_marked_for_regeneration"]}

    if agent == "storageManager" and tool == "open":
        mode = params.get("mode")
        if mode in {"create", "splitview"}:
            return {
                "bucket": "regenerate",
                "reasons": [f"unsupported_open_mode:{mode}"],
            }

    if agent == "contentManager" and tool == "update":
        reasons.append("content_update_requires_operation_mapping")
        if "endLine" in params:
            reasons.append("replace_without_old_content")
        if "startLine" in params:
            reasons.append("line_based_edit")
        if "oldText" in params or "findText" in params:
            reasons.append("text_replacement_inference")
        return {"bucket": "heuristic", "reasons": reasons}

    if agent == "memoryManager" and tool in {"createWorkspace", "updateWorkspace"}:
        if "description" in params and "purpose" not in params:
            reasons.append("infer_purpose_from_description")
        else:
            reasons.append("workspace_metadata_review")
        return {"bucket": "heuristic", "reasons": reasons}

    if tool in AUTO_TOOLS.get(agent, set()):
        if agent == "storageManager":
            if "sourcePath" in params:
                reasons.append("rename_sourcePath_to_path")
            if "targetPath" in params:
                reasons.append("rename_targetPath_to_newPath")
        if agent == "searchManager" and tool == "searchMemory" and "workspaceId" in params:
            reasons.append("remove_workspaceId_from_searchMemory")
        if agent == "memoryManager" and tool == "loadWorkspace" and "workspaceId" in params:
            reasons.append("rename_workspaceId_to_id")
        if agent == "promptManager" and tool == "updatePrompt" and "enabled" in params:
            reasons.append("rename_enabled_to_isEnabled")
        return {"bucket": "auto", "reasons": reasons}

    return {"bucket": "regenerate", "reasons": ["unsupported_or_unknown_tool_shape"]}
