"""Context injection utilities for evaluation prompts.

This module provides dataclasses and utilities for building system prompts
that match the production SystemPromptBuilder format.

NOTE: For evaluation, system prompts and expected_context should be defined
directly in the prompt set JSON files, not generated at runtime.
Use add_context_to_prompts.py to add context to prompt sets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class WorkspaceInfo:
    """Information about an available workspace."""
    id: str
    name: str
    description: str
    root_folder: str


@dataclass
class AgentInfo:
    """Information about an available agent."""
    id: str
    name: str
    description: str


@dataclass
class EvaluationContext:
    """Context for evaluation prompts.

    This dataclass can be used to programmatically build system prompts
    when creating new prompt sets.
    """
    session_id: str
    workspace_id: str  # Current workspace (can be "default")
    workspaces: List[WorkspaceInfo] = field(default_factory=list)
    agents: List[AgentInfo] = field(default_factory=list)

    def to_system_prompt(self) -> str:
        """Generate a system prompt matching production SystemPromptBuilder format."""
        sections = []

        # Session context section
        sections.append(self._build_session_context())

        # Available workspaces section (if any non-default workspaces)
        if self.workspaces:
            sections.append(self._build_available_workspaces())

        # Available agents section (if any)
        if self.agents:
            sections.append(self._build_available_agents())

        return "\n".join(sections)

    def _build_session_context(self) -> str:
        prompt = "<session_context>\n"
        prompt += "IMPORTANT: When using tools, include these values in your tool call parameters:\n\n"
        prompt += f'- sessionId: "{self.session_id}"\n'

        if self.workspace_id == "default":
            prompt += '- workspaceId: "default" (no specific workspace selected)\n'
            prompt += "\nInclude these in the \"context\" parameter of your tool calls.\n"
            prompt += "NOTE: Use \"default\" as the workspaceId when no specific workspace context is needed.\n"
        else:
            prompt += f'- workspaceId: "{self.workspace_id}" (current workspace)\n'
            prompt += "\nInclude these in the \"context\" parameter of your tool calls.\n"

        prompt += "</session_context>"
        return prompt

    def _build_available_workspaces(self) -> str:
        prompt = "<available_workspaces>\n"
        prompt += "The following workspaces are available in this vault:\n\n"

        for ws in self.workspaces:
            prompt += f'- {ws.name} (id: "{ws.id}")\n'
            prompt += f"  Description: {ws.description}\n"
            prompt += f"  Root folder: {ws.root_folder}\n\n"

        prompt += "Use memoryManager with loadWorkspace mode to get full workspace context.\n"
        prompt += "</available_workspaces>"
        return prompt

    def _build_available_agents(self) -> str:
        prompt = "<available_agents>\n"
        prompt += "The following custom agents are available:\n\n"

        for agent in self.agents:
            prompt += f'- {agent.name} (id: "{agent.id}")\n'
            prompt += f"  {agent.description}\n\n"

        prompt += "</available_agents>"
        return prompt

    def to_expected_context(self) -> Dict[str, Any]:
        """Convert to expected_context dict for validation."""
        return {
            "session_id": self.session_id,
            "workspace_id": self.workspace_id,
            "workspace_ids": [ws.id for ws in self.workspaces],
            "agent_ids": [a.id for a in self.agents],
        }
