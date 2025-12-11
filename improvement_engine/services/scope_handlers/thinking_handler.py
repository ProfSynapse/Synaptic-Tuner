"""Thinking handler - handles thinking block scope.

Single Responsibility: ALL thinking scope logic.
"""

import json
import re
from typing import Dict
from .base import ScopeHandler


class ThinkingHandler(ScopeHandler):
    """
    Handler for thinking scope.

    Applies improvements to thinking blocks in assistant messages.
    """

    def apply_improvement(self, example: Dict, improved_content: str) -> Dict:
        """
        Apply improved thinking block.

        Logic moved from ImprovementApplicator._apply_thinking()
        """
        improved = json.loads(json.dumps(example))  # Deep copy
        conversations = improved.get("conversations", [])

        # Get scope definition for markers
        scope_def = self.scope_config.get_scope("thinking")
        if not scope_def:
            return improved

        markers = scope_def.markers

        for conv in conversations:
            if conv.get("role") == "assistant":
                content = conv.get("content", "")

                # Extract thinking block from new content using markers
                pattern = f"{re.escape(markers.start)}(.*?){re.escape(markers.end)}"
                match = re.search(pattern, improved_content, re.DOTALL)

                if match:
                    new_thinking = f"{markers.start}\n{match.group(1).strip()}\n{markers.end}"

                    # Replace existing thinking or prepend
                    if markers.start in content:
                        content = re.sub(pattern, new_thinking, content, flags=re.DOTALL)
                    else:
                        content = new_thinking + "\n\n" + content

                    conv["content"] = content
                break

        return improved

    def build_prompt_variables(self, example: Dict, judgment: Dict) -> Dict:
        """
        Build variables for thinking improvement template.

        Returns:
            {
                "current_content": <current thinking block>,
                "feedback": <judge feedback>,
                "system_prompt": <system prompt for context>,
                "user_request": <user request for context>
            }
        """
        # Extract current thinking
        current_content = self.scope_extractor.extract(example, "thinking")

        # Extract system and user for context
        conversations = example.get("conversations", [])
        system_prompt = ""
        user_request = ""

        for conv in conversations:
            role = conv.get("role")
            if role == "system":
                system_prompt = conv.get("content", "")
            elif role == "user":
                user_request = conv.get("content", "")

        return {
            "current_content": current_content or "",
            "system_prompt": system_prompt,
            "user_request": user_request,
        }
