"""SynthChat Workspace - Mocked workspace rendering for synthetic prompt generation.

Location: SynthChat/workspace/__init__.py
Purpose: Subpackage for workspace system prompt rendering, including fixture helpers,
         section builders, and the main renderer that composes them.
Usage: from SynthChat.workspace import render_mocked_workspace_system_prompt
"""

from .renderer import render_mocked_workspace_system_prompt

__all__ = ["render_mocked_workspace_system_prompt"]
