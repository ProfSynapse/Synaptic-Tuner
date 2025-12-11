"""Response handler - handles assistant response scope.

Single Responsibility: ALL response scope logic (text + tool calls).
"""

import json
import re
from typing import Dict
from .base import ScopeHandler


class ResponseHandler(ScopeHandler):
    """
    Handler for response scope.

    Applies improvements to assistant response text and tool calls.
    """

    def apply_improvement(self, example: Dict, improved_content: str) -> Dict:
        """
        Apply improved response (text + tool calls).

        Logic moved from ImprovementApplicator._apply_response()
        """
        improved = json.loads(json.dumps(example))  # Deep copy
        conversations = improved.get("conversations", [])

        for conv in conversations:
            if conv.get("role") == "assistant":
                # Extract tool calls from new content
                new_tool_calls = self.scope_extractor.extract_tool_calls_from_text(improved_content)

                # Remove tool call text block from content
                clean_content = self._remove_tool_call_text(improved_content)

                # Preserve existing thinking block if present
                thinking_scope = self.scope_config.get_scope("thinking")
                if thinking_scope:
                    old_thinking_pattern = f"{re.escape(thinking_scope.markers.start)}.*?{re.escape(thinking_scope.markers.end)}"
                    old_thinking_match = re.search(old_thinking_pattern, conv.get("content", ""), re.DOTALL)

                    if old_thinking_match and thinking_scope.markers.start not in clean_content:
                        old_thinking = old_thinking_match.group(0)
                        clean_content = old_thinking + "\n\n" + clean_content

                conv["content"] = clean_content

                # Update tool_calls field
                if new_tool_calls:
                    conv["tool_calls"] = new_tool_calls
                elif "tool_calls" in conv:
                    del conv["tool_calls"]

                break

        return improved

    def build_prompt_variables(self, example: Dict, judgment: Dict) -> Dict:
        """
        Build variables for response improvement template.

        Returns:
            {
                "current_content": <current response text>,
                "feedback": <judge feedback>,
                "system_prompt": <system prompt for context>,
                "user_request": <user request>,
                "original_tool_calls": <original tool calls for format reference>
            }
        """
        # Extract current response (text only, no thinking)
        current_content = self.scope_extractor.extract(example, "response")

        # Extract context messages
        conversations = example.get("conversations", [])
        system_prompt = ""
        user_request = ""
        original_tool_calls = None

        for conv in conversations:
            role = conv.get("role")
            if role == "system":
                system_prompt = conv.get("content", "")
            elif role == "user":
                user_request = conv.get("content", "")
            elif role == "assistant" and "tool_calls" in conv:
                original_tool_calls = conv["tool_calls"]

        return {
            "current_content": current_content or "",
            "system_prompt": system_prompt,
            "user_request": user_request,
            "original_tool_calls": original_tool_calls,
        }

    def _remove_tool_call_text(self, content: str) -> str:
        """
        Remove tool call text blocks from content.

        Logic moved from ImprovementApplicator._remove_tool_call_text()
        """
        tool_scope = self.scope_config.get_scope("tool_calls")
        if not tool_scope:
            return content

        markers = tool_scope.markers
        lines = content.split("\n")
        clean_lines = []
        skip_mode = False

        for line in lines:
            stripped = line.strip()

            # Check for tool call start markers
            if any([
                markers.block_start and markers.block_start in line,
                markers.block_start_alt and markers.block_start_alt in line,
                markers.name_prefix and line.startswith(markers.name_prefix),
                markers.name_prefix_alt and line.startswith(markers.name_prefix_alt)
            ]):
                skip_mode = True
                continue

            # Check for end of tool call block
            if skip_mode:
                if stripped.startswith("}") or (stripped == "" and clean_lines and clean_lines[-1].strip() == "}"):
                    skip_mode = False
                continue

            clean_lines.append(line)

        return "\n".join(clean_lines).strip()
