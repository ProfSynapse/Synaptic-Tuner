"""Generic improver service for fixing examples based on judge feedback."""

import json
from typing import Dict, Optional

from ..utils.logger import ImproveLogger
from .schema_validator import ThinkingSchemaValidator
from .tool_schema_validator import ToolSchemaValidator


class ImproverService:
    """
    LLM that applies improvements based on judge feedback.

    Supports multiple scopes:
    - "response": Improve full assistant response
    - "thinking": Improve only thinking block
    - "tool_calls": Improve only tool call arguments
    - "text": Improve only text response
    """

    def __init__(self, rubric: Dict, llm_client, logger: Optional[ImproveLogger] = None):
        """
        Initialize improver service.

        Args:
            rubric: Rubric configuration dict
            llm_client: LLM client (from shared.llm)
            logger: Logger instance
        """
        self.rubric = rubric
        self.llm_client = llm_client
        self.logger = logger or ImproveLogger()

        # Initialize schema validators
        try:
            self.thinking_validator = ThinkingSchemaValidator(logger)
            self.tool_validator = ToolSchemaValidator(logger)
        except Exception as e:
            self.logger.warning(f"Could not load schema validators: {e}")
            self.thinking_validator = None
            self.tool_validator = None

    def improve(self, example: Dict, feedback: str) -> Dict:
        """
        Improve example based on judge feedback.

        Args:
            example: Original example dict
            feedback: Improvement feedback from judge

        Returns:
            Improved example dict
        """
        try:
            scope = self.rubric["scope"]

            # Extract components based on scope
            prompt_vars = self._extract_components_for_scope(example, feedback)

            # Build improvement prompt from rubric template
            improve_prompt = self.rubric["improver_prompt"].format(**prompt_vars)

            # Add strict output instruction
            if scope == "thinking":
                improve_prompt += "\n\n**CRITICAL INSTRUCTION:**\nRespond with ONLY the fixed thinking block as valid JSON.\nNo explanations, no markdown, no additional text.\nStart your response with { and end with }."

            # Get improved content from LLM
            improved_content = self.llm_client.chat(
                messages=[{"role": "user", "content": improve_prompt}],
                temperature=0.3
            )

            # Handle response format (might be dict or string)
            if isinstance(improved_content, dict):
                improved_content = improved_content.get("content", "")

            # Apply improvement based on scope
            if scope == "response":
                return self._replace_assistant_response(example, improved_content)
            elif scope == "thinking":
                return self._replace_thinking_block(example, improved_content)
            elif scope == "tool_calls":
                return self._replace_tool_calls(example, improved_content)
            elif scope == "text":
                return self._replace_text_response(example, improved_content)
            else:
                raise ValueError(f"Unknown scope: {scope}")

        except Exception as e:
            self.logger.error(f"Error during improvement: {e}")
            # Return original on error
            return example

    def _extract_components_for_scope(self, example: Dict, feedback: str) -> Dict:
        """Extract components from example based on rubric scope."""
        scope = self.rubric["scope"]

        # Extract conversations
        system_prompt, user_request, assistant_response = self._extract_conversations(example)

        if scope == "response":
            # Full assistant response
            return {
                "system_prompt": system_prompt,
                "user_request": user_request,
                "assistant_response": assistant_response,
                "feedback": feedback
            }

        elif scope == "thinking":
            # Only thinking block
            thinking_block = self._extract_thinking_block(assistant_response)
            return {
                "thinking_block": json.dumps(thinking_block, indent=2, ensure_ascii=False),
                "feedback": feedback
            }

        elif scope == "tool_calls":
            # Only tool calls
            tool_calls = self._extract_tool_calls(assistant_response)
            return {
                "system_prompt": system_prompt,
                "user_request": user_request,
                "tool_calls": json.dumps(tool_calls, indent=2, ensure_ascii=False),
                "feedback": feedback
            }

        elif scope == "text":
            # Only text response
            text_response = self._extract_text_response(assistant_response)
            return {
                "system_prompt": system_prompt,
                "user_request": user_request,
                "text_response": text_response,
                "feedback": feedback
            }

        else:
            raise ValueError(f"Unknown scope: {scope}")

    def _extract_conversations(self, example: Dict):
        """Extract system prompt, user request, assistant response."""
        system_prompt = ""
        user_request = ""
        assistant_response = ""

        for conv in example.get("conversations", []):
            role = conv.get("role", "")
            content = conv.get("content", "")

            if role == "system":
                system_prompt = content
            elif role == "user":
                user_request = content
            elif role == "assistant":
                assistant_response = content

        return system_prompt, user_request, assistant_response

    def _extract_thinking_block(self, assistant_response: str) -> Optional[Dict]:
        """Extract thinking block from assistant response."""
        if "<thinking>" not in assistant_response or "</thinking>" not in assistant_response:
            return {}

        start = assistant_response.index("<thinking>") + len("<thinking>")
        end = assistant_response.index("</thinking>")
        thinking_str = assistant_response[start:end].strip()

        try:
            return json.loads(thinking_str)
        except json.JSONDecodeError:
            return {}

    def _extract_tool_calls(self, assistant_response: str):
        """Extract tool calls from assistant response."""
        # Similar to judge_service
        if "tool_call:" not in assistant_response:
            return []

        lines = assistant_response.split("\n")
        tool_calls = []

        for i, line in enumerate(lines):
            if line.startswith("tool_call:"):
                tool_name = line.split("tool_call:")[1].strip()
                if i + 1 < len(lines) and lines[i + 1].startswith("arguments:"):
                    args_str = lines[i + 1].split("arguments:")[1].strip()
                    try:
                        args = json.loads(args_str)
                        tool_calls.append({
                            "function": tool_name,
                            "arguments": args
                        })
                    except json.JSONDecodeError:
                        pass

        return tool_calls

    def _extract_text_response(self, assistant_response: str) -> str:
        """Extract plain text response."""
        # Similar to judge_service
        text = assistant_response
        if "<thinking>" in text and "</thinking>" in text:
            start = text.index("<thinking>")
            end = text.index("</thinking>") + len("</thinking>")
            text = text[:start] + text[end:]

        lines = text.split("\n")
        clean_lines = []
        skip_next = False

        for line in lines:
            if line.startswith("tool_call:"):
                skip_next = True
                continue
            if skip_next and line.startswith("arguments:"):
                skip_next = False
                continue
            clean_lines.append(line)

        return "\n".join(clean_lines).strip()

    def _replace_assistant_response(self, example: Dict, new_response: str) -> Dict:
        """Replace entire assistant response (content + tool_calls)."""
        improved = example.copy()
        improved["conversations"] = [c.copy() for c in example["conversations"]]

        for conv in improved["conversations"]:
            if conv.get("role") == "assistant":
                # Parse the improved response to extract content and tool_calls
                content, tool_calls = self._parse_full_response(new_response)

                # Update content
                conv["content"] = content.strip()

                # Update tool_calls if present in improved response
                if tool_calls:
                    conv["tool_calls"] = tool_calls
                elif "tool_calls" in conv:
                    # If no tool_calls in improved response, it means the LLM
                    # determined the tool call was based on hallucinated info.
                    # Remove it and let the text response ask for clarification.
                    self.logger.info("Improved response intentionally has no tool calls (removed hallucinated call)")
                    del conv["tool_calls"]

        return improved

    def _parse_full_response(self, response: str) -> tuple:
        """
        Parse full response into content and tool_calls.

        Returns:
            (content_str, tool_calls_list)
        """
        # Check if response has [TOOL_CALLS] section
        if "[TOOL_CALLS]" not in response:
            return response, None

        # Split into content and tool calls
        parts = response.split("[TOOL_CALLS]")
        content = parts[0].strip()
        tool_calls_text = parts[1].strip() if len(parts) > 1 else ""

        # Parse tool calls from text format
        tool_calls = []
        lines = tool_calls_text.split("\n")
        current_tool = None

        for line in lines:
            line = line.strip()
            if line.startswith("tool_call:"):
                if current_tool:
                    tool_calls.append(current_tool)
                tool_name = line.split("tool_call:")[1].strip()
                current_tool = {
                    "id": f"call_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": "{}"
                    }
                }
            elif line.startswith("arguments:") and current_tool:
                args_str = line.split("arguments:")[1].strip()
                current_tool["function"]["arguments"] = args_str

        if current_tool:
            tool_calls.append(current_tool)

        return content, tool_calls if tool_calls else None

    def _replace_thinking_block(self, example: Dict, improved_thinking_str: str) -> Dict:
        """Replace only the thinking block within assistant response."""
        improved = example.copy()
        improved["conversations"] = example["conversations"].copy()

        # Parse improved thinking block
        try:
            # Extract JSON from response (might be wrapped in markdown)
            if "```json" in improved_thinking_str:
                start = improved_thinking_str.index("```json") + 7
                end = improved_thinking_str.index("```", start)
                improved_thinking_str = improved_thinking_str[start:end].strip()
            elif "```" in improved_thinking_str:
                start = improved_thinking_str.index("```") + 3
                end = improved_thinking_str.index("```", start)
                improved_thinking_str = improved_thinking_str[start:end].strip()

            improved_thinking = json.loads(improved_thinking_str)
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Could not parse improved thinking: {e}")
            return example

        # VALIDATE THINKING SCHEMA
        if self.thinking_validator:
            # Get original thinking for fallback
            original_thinking = self._extract_thinking_block(
                [c["content"] for c in example["conversations"] if c.get("role") == "assistant"][0]
            ) or {}

            # Validate improved thinking
            is_valid, errors = self.thinking_validator.validate(improved_thinking)

            if not is_valid:
                self.logger.warning(f"Thinking schema validation failed: {errors}")
                # Enforce schema using original as fallback
                improved_thinking = self.thinking_validator.enforce_schema_on_dict(
                    improved_thinking, original_thinking
                )
                self.logger.info("Schema enforced using original values as fallback")

        # Replace thinking block in assistant response
        for conv in improved["conversations"]:
            if conv.get("role") == "assistant":
                content = conv["content"]

                if "<thinking>" in content and "</thinking>" in content:
                    start = content.index("<thinking>")
                    end = content.index("</thinking>") + len("</thinking>")

                    # Build new thinking block
                    new_thinking_str = json.dumps(improved_thinking, ensure_ascii=False)
                    new_content = (
                        content[:start] +
                        "<thinking>\n" + new_thinking_str + "\n</thinking>" +
                        content[end:]
                    )

                    conv["content"] = new_content

        return improved

    def _replace_tool_calls(self, example: Dict, improved_tool_calls_str: str) -> Dict:
        """Replace tool calls in assistant response."""
        # For now, return original (can implement if needed)
        self.logger.warning("Tool call replacement not fully implemented yet")
        return example

    def _replace_text_response(self, example: Dict, improved_text: str) -> Dict:
        """Replace text response in assistant response."""
        # For now, return original (can implement if needed)
        self.logger.warning("Text response replacement not fully implemented yet")
        return example
