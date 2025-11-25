#!/usr/bin/env python3
"""Add system prompts and expected_context to evaluator prompt sets.

This script adds fixed context (sessionId, workspaceId, agents) to each prompt
so the model can be tested on whether it uses the provided IDs correctly.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Fixed IDs for evaluation (not randomly generated)
EVAL_SESSION_ID = "session_1732300800000_eval01234"
EVAL_WORKSPACE_ID = "ws_1732300800000_atlasroll"
EVAL_WORKSPACE_NAME = "Atlas Rollout"
EVAL_WORKSPACE_DESC = "Project tracking for Atlas customer rollout"
EVAL_WORKSPACE_ROOT = "Projects/"

# Secondary workspace for multi-workspace scenarios
EVAL_WORKSPACE_2_ID = "ws_1732300800001_phoenixmg"
EVAL_WORKSPACE_2_NAME = "Phoenix Migration"
EVAL_WORKSPACE_2_DESC = "Phoenix system migration tracking"
EVAL_WORKSPACE_2_ROOT = "Projects/Phoenix/"

# Fixed agents for evaluation
EVAL_AGENTS = [
    {
        "id": "agent_1732300800000_workspace_auditor",
        "name": "Workspace Auditor",
        "desc": "Audits workspace contents and reports issues"
    },
    {
        "id": "agent_1732300800001_release_briefing",
        "name": "Release Briefing",
        "desc": "Generates release briefing summaries"
    },
    {
        "id": "agent_1732300800002_incident_scribe",
        "name": "Incident Scribe",
        "desc": "Documents incidents and post-mortems"
    },
    {
        "id": "agent_1732300800003_legacy_intake",
        "name": "Legacy Intake",
        "desc": "Processes legacy system data intake"
    },
    {
        "id": "agent_1732300800004_qa_prototype",
        "name": "QA Prototype",
        "desc": "Prototype agent for QA testing"
    },
]


def build_session_context(session_id: str, workspace_id: str, is_default: bool = False) -> str:
    """Build <session_context> section."""
    prompt = "<session_context>\n"
    prompt += "IMPORTANT: When using tools, include these values in your tool call parameters:\n\n"
    prompt += f'- sessionId: "{session_id}"\n'

    if is_default:
        prompt += '- workspaceId: "default" (no specific workspace selected)\n'
        prompt += "\nInclude these in the \"context\" parameter of your tool calls.\n"
        prompt += "NOTE: Use \"default\" as the workspaceId when no specific workspace context is needed.\n"
    else:
        prompt += f'- workspaceId: "{workspace_id}" (current workspace)\n'
        prompt += "\nInclude these in the \"context\" parameter of your tool calls.\n"

    prompt += "</session_context>"
    return prompt


def build_available_workspaces(workspaces: List[Dict[str, str]]) -> str:
    """Build <available_workspaces> section."""
    prompt = "<available_workspaces>\n"
    prompt += "The following workspaces are available in this vault:\n\n"

    for ws in workspaces:
        prompt += f'- {ws["name"]} (id: "{ws["id"]}")\n'
        prompt += f'  Description: {ws["desc"]}\n'
        prompt += f'  Root folder: {ws["root"]}\n\n'

    prompt += "Use memoryManager with loadWorkspace mode to get full workspace context.\n"
    prompt += "</available_workspaces>"
    return prompt


def build_available_agents(agents: List[Dict[str, str]]) -> str:
    """Build <available_agents> section."""
    prompt = "<available_agents>\n"
    prompt += "The following custom agents are available:\n\n"

    for agent in agents:
        prompt += f'- {agent["name"]} (id: "{agent["id"]}")\n'
        prompt += f'  {agent["desc"]}\n\n'

    prompt += "</available_agents>"
    return prompt


def build_system_prompt(
    session_id: str,
    workspace_id: str,
    workspaces: Optional[List[Dict[str, str]]] = None,
    agents: Optional[List[Dict[str, str]]] = None,
    is_default_workspace: bool = False,
) -> str:
    """Build full system prompt."""
    sections = [build_session_context(session_id, workspace_id, is_default_workspace)]

    if workspaces:
        sections.append(build_available_workspaces(workspaces))

    if agents:
        sections.append(build_available_agents(agents))

    return "\n".join(sections)


def build_expected_context(
    session_id: str,
    workspace_id: str,
    workspace_ids: List[str],
    agent_ids: List[str],
) -> Dict[str, Any]:
    """Build expected_context for validation."""
    return {
        "session_id": session_id,
        "workspace_id": workspace_id,
        "workspace_ids": workspace_ids,
        "agent_ids": agent_ids,
    }


def add_context_to_prompt(prompt: Dict[str, Any]) -> Dict[str, Any]:
    """Add system prompt and expected_context to a single prompt."""
    prompt_id = prompt.get("id", "")
    tags = prompt.get("expected_tools", [])

    # Determine if this is an agent-related prompt
    is_agent_prompt = any("agentManager" in t for t in tags)

    # Determine if this needs multiple workspaces
    needs_multi_workspace = any(kw in prompt_id.lower() for kw in ["load", "list", "batch"])

    # Build workspaces list
    workspaces = [
        {"id": EVAL_WORKSPACE_ID, "name": EVAL_WORKSPACE_NAME, "desc": EVAL_WORKSPACE_DESC, "root": EVAL_WORKSPACE_ROOT}
    ]
    if needs_multi_workspace:
        workspaces.append({
            "id": EVAL_WORKSPACE_2_ID, "name": EVAL_WORKSPACE_2_NAME,
            "desc": EVAL_WORKSPACE_2_DESC, "root": EVAL_WORKSPACE_2_ROOT
        })

    # Build agents list (only for agent prompts)
    agents = EVAL_AGENTS if is_agent_prompt else None

    # Build system prompt
    system = build_system_prompt(
        session_id=EVAL_SESSION_ID,
        workspace_id=EVAL_WORKSPACE_ID,
        workspaces=workspaces,
        agents=agents,
        is_default_workspace=False,
    )

    # Build expected_context
    workspace_ids = [ws["id"] for ws in workspaces]
    agent_ids = [a["id"] for a in EVAL_AGENTS] if is_agent_prompt else []

    expected_context = build_expected_context(
        session_id=EVAL_SESSION_ID,
        workspace_id=EVAL_WORKSPACE_ID,
        workspace_ids=workspace_ids,
        agent_ids=agent_ids,
    )

    # Create updated prompt
    updated = dict(prompt)
    updated["system"] = system
    updated["expected_context"] = expected_context

    return updated


def process_prompt_file(input_path: Path, output_path: Optional[Path] = None) -> int:
    """Process a prompt file and add context to all prompts."""
    if output_path is None:
        output_path = input_path

    with open(input_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    updated_prompts = [add_context_to_prompt(p) for p in prompts]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(updated_prompts, f, indent=2)

    return len(updated_prompts)


def main():
    """Process all prompt files."""
    prompts_dir = Path(__file__).parent / "prompts"

    files_to_process = [
        "full_coverage.json",
        "baseline.json",
        "tool_combos.json",
    ]

    for filename in files_to_process:
        filepath = prompts_dir / filename
        if filepath.exists():
            count = process_prompt_file(filepath)
            print(f"Updated {filename}: {count} prompts")
        else:
            print(f"Skipping {filename}: not found")


if __name__ == "__main__":
    main()
