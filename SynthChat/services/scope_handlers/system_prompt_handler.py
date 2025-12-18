"""System prompt handler - handles system message scope.

Single Responsibility: ALL system_prompt scope logic.
"""

import json
from typing import Dict, Optional
from .base import ScopeHandler


class SystemPromptHandler(ScopeHandler):
    """
    Handler for system_prompt scope.

    Applies improvements to system message.
    """

    def apply_improvement(
        self,
        example: Dict,
        improved_content: str,
        output_format: Optional[Dict] = None
    ) -> Dict:
        """
        Apply improved system message.

        Logic moved from ImprovementApplicator._apply_system()
        """
        improved = json.loads(json.dumps(example))  # Deep copy
        conversations = improved.get("conversations", [])

        # Find and replace system message
        for conv in conversations:
            if conv.get("role") == "system":
                conv["content"] = improved_content.strip()
                return improved

        # If no system message, add one at beginning
        improved.setdefault("conversations", []).insert(0, {
            "role": "system",
            "content": improved_content.strip()
        })

        return improved

    def build_prompt_variables(self, example: Dict, judgment: Dict) -> Dict:
        """
        Build variables for system_prompt improvement template.

        Returns:
            {
                "current_content": <current system prompt>,
                "feedback": <judge feedback>,
                "user_request": <user message for context>,
                "assistant_response": <assistant message for extracting tool paths>
            }
        """
        # Extract current system prompt
        current_content = self.scope_extractor.extract(example, "system_prompt")

        # Extract other messages for context
        conversations = example.get("conversations", [])
        user_request = ""
        assistant_response = ""

        for conv in conversations:
            role = conv.get("role")
            if role == "user":
                user_request = conv.get("content", "")
            elif role == "assistant":
                assistant_response = conv.get("content", "")

        return {
            "current_content": current_content or "",
            "user_request": user_request,
            "assistant_response": assistant_response,
        }
