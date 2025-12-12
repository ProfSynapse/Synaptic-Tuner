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

        Handles both:
        - Content with <thinking> tags
        - Raw JSON (possibly wrapped in markdown code blocks)
        """
        improved = json.loads(json.dumps(example))  # Deep copy
        conversations = improved.get("conversations", [])

        # Get scope definition for markers
        scope_def = self.scope_config.get_scope("thinking")
        if not scope_def:
            return improved

        markers = scope_def.markers

        # First, try to extract thinking content
        thinking_json = None

        # Try 1: Look for <thinking> tags
        pattern = f"{re.escape(markers.start)}(.*?){re.escape(markers.end)}"
        match = re.search(pattern, improved_content, re.DOTALL)
        if match:
            thinking_json = match.group(1).strip()

        # Try 2: Look for markdown code blocks (```json ... ``` or ``` ... ```)
        if not thinking_json:
            code_block_pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
            match = re.search(code_block_pattern, improved_content)
            if match:
                thinking_json = match.group(1).strip()

        # Try 3: Look for raw JSON object
        if not thinking_json:
            json_pattern = r'(\{[\s\S]*\})'
            match = re.search(json_pattern, improved_content)
            if match:
                try:
                    json.loads(match.group(1))  # Validate it's valid JSON
                    thinking_json = match.group(1).strip()
                except json.JSONDecodeError:
                    pass

        if not thinking_json:
            # No valid thinking content found
            return improved

        # Apply the thinking content
        for conv in conversations:
            if conv.get("role") == "assistant":
                content = conv.get("content", "")

                # Wrap in thinking tags
                new_thinking = f"{markers.start}\n{thinking_json}\n{markers.end}"

                # Replace existing thinking or prepend
                existing_pattern = f"{re.escape(markers.start)}.*?{re.escape(markers.end)}"
                if re.search(existing_pattern, content, re.DOTALL):
                    content = re.sub(existing_pattern, new_thinking, content, flags=re.DOTALL)
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
