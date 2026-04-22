"""Response handler - handles assistant response scope.

Single Responsibility: ALL response scope logic (text + tool calls).
"""

import json
import re
from typing import Any, Dict, Optional
from .base import ScopeHandler


class ResponseHandler(ScopeHandler):
    """
    Handler for response scope.

    Applies improvements to assistant response text and tool calls.
    Output format is controlled by rubric config, not hardcoded detection.
    """

    def apply_improvement(
        self,
        example: Dict,
        improved_content: str,
        output_format: Optional[Dict] = None
    ) -> Dict:
        """
        Apply improved response based on output_format config.

        Args:
            example: Original example
            improved_content: Improved content from LLM
            output_format: Config from rubric defining how to apply:
                          {type: "assistant_message" | "content_only" | "tool_calls_only"}

        Returns:
            Updated example (deep copy)
        """
        improved = json.loads(json.dumps(example))  # Deep copy
        conversations = improved.get("conversations", [])

        # Get format type (default to content_only for backward compatibility)
        format_type = (output_format or {}).get("type", "content_only")

        for conv in conversations:
            if conv.get("role") == "assistant":
                if format_type == "assistant_message":
                    # Parse improved content as full assistant message JSON
                    # Expected: {"content": null, "tool_calls": [...]}
                    self._apply_assistant_message(conv, improved_content)

                elif format_type == "tool_calls_only":
                    # Parse improved content as tool_calls array/object
                    # Set content to null
                    self._apply_tool_calls_only(conv, improved_content)

                else:  # content_only (default)
                    # Apply as text content, preserve/extract tool calls from text
                    self._apply_content_only(conv, improved_content)

                break

        return improved

    def _apply_assistant_message(self, conv: Dict, improved_content: str) -> None:
        """
        Apply improved content as full assistant message.

        Expected format: {"content": null|string, "tool_calls": [...]}
        """
        try:
            # Strip code fences if present
            stripped = improved_content.strip()
            if stripped.startswith("```"):
                stripped = re.sub(r'^```\w*\n?', '', stripped)
                stripped = re.sub(r'\n?```$', '', stripped)
                stripped = stripped.strip()

            parsed = json.loads(stripped)

            # Apply content (can be null or string)
            if "content" in parsed:
                conv["content"] = parsed["content"]

            # Apply tool_calls
            if "tool_calls" in parsed and parsed["tool_calls"]:
                conv["tool_calls"] = parsed["tool_calls"]
            elif "tool_calls" in conv:
                # Remove tool_calls if not in improved output
                del conv["tool_calls"]

        except json.JSONDecodeError as e:
            if self.logger:
                self.logger.warning(f"Failed to parse assistant_message format: {e}")
            # Fallback: apply as content_only
            self._apply_content_only(conv, improved_content)

    def _apply_tool_calls_only(self, conv: Dict, improved_content: str) -> None:
        """
        Apply improved content as tool_calls only, set content to null.

        Expected format: {"tool_calls": [...]} or just [...]
        """
        try:
            stripped = improved_content.strip()
            if stripped.startswith("```"):
                stripped = re.sub(r'^```\w*\n?', '', stripped)
                stripped = re.sub(r'\n?```$', '', stripped)
                stripped = stripped.strip()

            parsed = json.loads(stripped)

            # Handle both {"tool_calls": [...]} and bare [...]
            if isinstance(parsed, dict) and "tool_calls" in parsed:
                conv["tool_calls"] = parsed["tool_calls"]
            elif isinstance(parsed, list):
                conv["tool_calls"] = parsed
            else:
                raise ValueError("Expected tool_calls array or object")

            conv["content"] = None

        except (json.JSONDecodeError, ValueError) as e:
            if self.logger:
                self.logger.warning(f"Failed to parse tool_calls_only format: {e}")
            # Fallback: try to extract tool calls from text
            tool_calls = self.scope_extractor.extract_tool_calls_from_text(improved_content)
            if tool_calls:
                conv["tool_calls"] = tool_calls
                conv["content"] = None

    def _apply_content_only(self, conv: Dict, improved_content: str) -> None:
        """
        Apply improved content as text, extracting tool calls from text if present.

        This is the legacy behavior for backward compatibility.
        """
        # Extract tool calls from text if present
        new_tool_calls = self.scope_extractor.extract_tool_calls_from_text(improved_content)

        # Remove tool call text block from content
        clean_content = self._remove_tool_call_text(improved_content)

        # Preserve existing thinking block if present
        thinking_scope = self.scope_config.get_scope("thinking")
        if thinking_scope:
            old_thinking_pattern = f"{re.escape(thinking_scope.markers.start)}.*?{re.escape(thinking_scope.markers.end)}"
            old_thinking_match = re.search(old_thinking_pattern, conv.get("content") or "", re.DOTALL)

            if old_thinking_match and thinking_scope.markers.start not in clean_content:
                old_thinking = old_thinking_match.group(0)
                clean_content = old_thinking + "\n\n" + clean_content

        conv["content"] = clean_content

        # Update tool_calls field
        if new_tool_calls:
            conv["tool_calls"] = new_tool_calls
        elif not self._has_tool_markers(improved_content):
            # Text-only response - remove tool_calls
            if "tool_calls" in conv:
                del conv["tool_calls"]

    def _has_tool_markers(self, content: str) -> bool:
        """Check if content has tool call markers."""
        return "```tool" in content or '"name":' in content

    def build_prompt_variables(
        self,
        example: Dict,
        judgment: Dict,
        prompt_context: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """
        Build variables for response improvement template.

        Returns:
            {
                "current_content": <current response text>,
                "feedback": <judge feedback>,
                "system_prompt": <system prompt for context>,
                "user_request": <user request>,
                "original_tool_calls": <original tool calls for format reference>,
                "thinking_content": <thinking block for checking assessment>
            }
        """
        # Extract current response (text only, no thinking)
        current_content = self.scope_extractor.extract(example, "response")

        # Extract thinking block (for checking assessment in response improvers)
        thinking_content = self.scope_extractor.extract(example, "thinking")

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

        # Format tool calls as JSON string for template
        tool_calls_json = json.dumps(original_tool_calls, indent=2) if original_tool_calls else "[]"

        return {
            "current_content": current_content or "",
            "system_prompt": system_prompt,
            "user_request": user_request,
            "original_tool_calls": tool_calls_json,
            "thinking_content": thinking_content or "",
            "environment_result_json": json.dumps((prompt_context or {}).get("environment_result") or {}, indent=2),
            "environment_issue_summary": str((prompt_context or {}).get("environment_issue_summary") or ""),
            "environment_passed": (
                str((prompt_context or {}).get("environment_passed"))
                if (prompt_context or {}).get("environment_passed") is not None
                else ""
            ),
        }

    def _is_text_only_response(self, content: str) -> bool:
        """
        Detect if improved content is intentionally text-only (no tool execution intended).

        Simple logic: If the improved content doesn't contain tool call markers,
        it's text-only. The decision to ask for clarification vs. execute tools
        is made by the LLM in the prompt, not by keyword detection.

        Returns:
            True if this appears to be intentional text-only response
        """
        # Check for tool block markers
        has_tool_markers = "```tool" in content or '"name":' in content

        # Text-only if no tool markers present
        return not has_tool_markers

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
