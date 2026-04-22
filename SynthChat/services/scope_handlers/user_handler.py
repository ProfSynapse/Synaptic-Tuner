"""User message handler - handles user scope.

Single Responsibility: ALL user scope logic.

NOTE: User messages are typically not improved (they're input data).
This handler exists primarily for cross_scope validation -
comparing content in other scopes against what the user actually said.
"""

import json
from typing import Any, Dict, Optional
from .base import ScopeHandler


class UserHandler(ScopeHandler):
    """
    Handler for user scope.

    Primarily used for extracting user content for cross_scope validation.
    """

    def apply_improvement(
        self,
        example: Dict,
        improved_content: str,
        output_format: Optional[Dict] = None
    ) -> Dict:
        """
        Apply improved user message.

        NOTE: User messages are typically not improved, but this is here
        for completeness and future extensibility.
        """
        improved = json.loads(json.dumps(example))  # Deep copy
        conversations = improved.get("conversations", [])

        # Find and replace user message
        for conv in conversations:
            if conv.get("role") == "user":
                conv["content"] = improved_content.strip()
                return improved

        # If no user message exists (unusual), add one
        improved.setdefault("conversations", []).append({
            "role": "user",
            "content": improved_content.strip()
        })

        return improved

    def build_prompt_variables(
        self,
        example: Dict,
        judgment: Dict,
        prompt_context: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """
        Build variables for user scope template.

        Returns:
            {
                "current_content": <current user message>,
                "system_prompt": <system message for context>,
                "assistant_response": <assistant message for context>
            }
        """
        # Extract current user message
        current_content = self.scope_extractor.extract(example, "user")

        # Extract other messages for context
        conversations = example.get("conversations", [])
        system_prompt = ""
        assistant_response = ""

        for conv in conversations:
            role = conv.get("role")
            if role == "system":
                system_prompt = conv.get("content", "")
            elif role == "assistant":
                assistant_response = conv.get("content", "")

        return {
            "current_content": current_content or "",
            "system_prompt": system_prompt,
            "assistant_response": assistant_response,
        }
