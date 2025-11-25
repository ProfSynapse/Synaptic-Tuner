"""Context injection for evaluation prompts.

This module generates system prompts that mirror production SystemPromptBuilder output,
providing session context, available workspaces, and available agents to the model.

The model should use the IDs from this context rather than hallucinating new ones.
"""
from __future__ import annotations

import random
import string
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
    """Context to inject into evaluation prompts."""
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


def generate_session_id() -> str:
    """Generate a realistic session ID."""
    timestamp = int(time.time() * 1000)
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=9))
    return f"session_{timestamp}_{suffix}"


def generate_workspace_id() -> str:
    """Generate a realistic workspace ID."""
    timestamp = int(time.time() * 1000)
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=9))
    return f"ws_{timestamp}_{suffix}"


def generate_agent_id(name: str) -> str:
    """Generate a realistic agent ID based on name."""
    timestamp = int(time.time() * 1000)
    # Convert name to lowercase with underscores
    name_part = name.lower().replace(" ", "_").replace("-", "_")[:20]
    return f"agent_{timestamp}_{name_part}"


# Predefined workspace templates for realistic evaluation
WORKSPACE_TEMPLATES = [
    ("Budget Tracker", "Monthly budget and expense tracking", "Finance/"),
    ("Research Hub", "Academic research and paper management", "Research/"),
    ("Project Management", "Project tracking and task management", "Projects/"),
    ("Content Hub", "Blog posts and content creation", "Content/"),
    ("Meeting Notes", "Meeting recordings and action items", "Meetings/"),
    ("Recipe Collection", "Personal recipes and meal planning", "Recipes/"),
    ("Fitness Tracker", "Workout logs and health metrics", "Fitness/"),
    ("Learning Center", "Course materials and study notes", "Courses/"),
    ("Client Work", "Client projects and deliverables", "Clients/"),
    ("Development", "Code documentation and dev notes", "Dev/"),
]

# Predefined agent templates for realistic evaluation
AGENT_TEMPLATES = [
    ("Research Assistant", "Helps with deep research and analysis tasks"),
    ("Code Reviewer", "Reviews code for quality and best practices"),
    ("Writing Coach", "Assists with writing and editing"),
    ("Data Analyst", "Analyzes data and creates reports"),
    ("Task Manager", "Helps organize and prioritize tasks"),
]


def create_evaluation_context(
    use_default_workspace: bool = False,
    num_workspaces: int = 2,
    num_agents: int = 1,
    include_specific_workspace: Optional[str] = None,
) -> EvaluationContext:
    """Create an evaluation context with realistic IDs.

    Args:
        use_default_workspace: If True, use "default" as current workspace
        num_workspaces: Number of available workspaces to include
        num_agents: Number of available agents to include
        include_specific_workspace: If set, include a workspace with this name

    Returns:
        EvaluationContext with generated IDs
    """
    session_id = generate_session_id()

    # Generate workspaces
    workspaces = []
    workspace_templates = random.sample(WORKSPACE_TEMPLATES, min(num_workspaces, len(WORKSPACE_TEMPLATES)))

    for name, desc, folder in workspace_templates:
        ws_id = generate_workspace_id()
        workspaces.append(WorkspaceInfo(
            id=ws_id,
            name=name,
            description=desc,
            root_folder=folder,
        ))

    # Add specific workspace if requested
    if include_specific_workspace:
        ws_id = generate_workspace_id()
        workspaces.append(WorkspaceInfo(
            id=ws_id,
            name=include_specific_workspace,
            description=f"Workspace for {include_specific_workspace.lower()}",
            root_folder=f"{include_specific_workspace.replace(' ', '')}/"
        ))

    # Set current workspace
    if use_default_workspace or not workspaces:
        workspace_id = "default"
    else:
        workspace_id = workspaces[0].id

    # Generate agents
    agents = []
    agent_templates = random.sample(AGENT_TEMPLATES, min(num_agents, len(AGENT_TEMPLATES)))

    for name, desc in agent_templates:
        agent_id = generate_agent_id(name)
        agents.append(AgentInfo(
            id=agent_id,
            name=name,
            description=desc,
        ))

    return EvaluationContext(
        session_id=session_id,
        workspace_id=workspace_id,
        workspaces=workspaces,
        agents=agents,
    )


def inject_context_into_prompt(
    prompt_metadata: Dict[str, Any],
    context: Optional[EvaluationContext] = None,
) -> Dict[str, Any]:
    """Inject evaluation context into prompt metadata.

    Args:
        prompt_metadata: Original prompt metadata dict
        context: EvaluationContext to inject (creates new one if None)

    Returns:
        Updated metadata dict with system prompt
    """
    if context is None:
        context = create_evaluation_context()

    # Create a copy to avoid modifying original
    updated = dict(prompt_metadata)

    # Set or append to system prompt
    existing_system = updated.get("system", "")
    context_prompt = context.to_system_prompt()

    if existing_system:
        updated["system"] = f"{context_prompt}\n\n{existing_system}"
    else:
        updated["system"] = context_prompt

    # Store context IDs for validation
    updated["_eval_context"] = {
        "session_id": context.session_id,
        "workspace_id": context.workspace_id,
        "workspace_ids": [ws.id for ws in context.workspaces],
        "agent_ids": [a.id for a in context.agents],
    }

    return updated
