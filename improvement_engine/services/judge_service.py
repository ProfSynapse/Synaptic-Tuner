"""Generic judge service for evaluating examples against rubrics."""

import json
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from ..utils.logger import ImproveLogger


@dataclass
class JudgmentResult:
    """Result from judge evaluation."""
    score: float
    passed: bool
    feedback: str
    details: Dict


class JudgeService:
    """
    LLM-as-judge that scores examples against rubrics.

    Supports multiple scopes:
    - "response": Evaluate full assistant response
    - "thinking": Evaluate only thinking block
    - "tool_calls": Evaluate only tool call arguments
    - "text": Evaluate only text response
    """

    def __init__(self, rubric: Dict, llm_client, logger: Optional[ImproveLogger] = None):
        """
        Initialize judge service.

        Args:
            rubric: Rubric configuration dict
            llm_client: LLM client (from shared.llm)
            logger: Logger instance
        """
        self.rubric = rubric
        self.llm_client = llm_client
        self.logger = logger or ImproveLogger()

    def evaluate(self, example: Dict) -> JudgmentResult:
        """
        Evaluate example against rubric.

        Args:
            example: Example dict with conversations

        Returns:
            JudgmentResult with score, pass/fail, feedback
        """
        try:
            # Extract components based on scope
            prompt_vars = self._extract_components_for_scope(example)

            # Build judge prompt from rubric template
            judge_prompt = self.rubric["judge_prompt"].format(**prompt_vars)

            # Add strict JSON-only instruction
            judge_prompt += "\n\n**CRITICAL INSTRUCTION:**\nRespond with ONLY the JSON object. No explanations, no markdown, no additional text.\nStart your response with { and end with }.\nDo not wrap it in ```json or any other formatting."

            # Get structured judgment from LLM
            judgment = self.llm_client.structured_output(
                messages=[{"role": "user", "content": judge_prompt}],
                schema=self.rubric["output_schema"]
            )

            # Determine pass/fail
            # Find score field (could be "score", "factuality_score", "confidence_score", etc.)
            score = 0.0
            for key in judgment.keys():
                if 'score' in key.lower() and isinstance(judgment[key], (int, float)):
                    score = float(judgment[key])
                    break

            passed = score >= self.rubric["pass_threshold"]

            # Extract feedback (try multiple field names)
            feedback = (judgment.get("improvement_feedback") or
                       judgment.get("factuality_feedback") or
                       judgment.get("confidence_feedback") or
                       judgment.get("feedback") or
                       "")

            return JudgmentResult(
                score=score,
                passed=passed,
                feedback=feedback,
                details=judgment
            )

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error in judge evaluation: {e}")
            self.logger.error(f"Attempted to parse: {str(e)[:500]}")
            # Return failed judgment on error
            return JudgmentResult(
                score=0.0,
                passed=False,
                feedback=f"JSON parsing error: {str(e)}",
                details={"error": str(e), "error_type": "json_decode"}
            )
        except Exception as e:
            self.logger.error(f"Error during judge evaluation: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Return failed judgment on error
            return JudgmentResult(
                score=0.0,
                passed=False,
                feedback=f"Evaluation error: {str(e)}",
                details={"error": str(e)}
            )

    def _extract_components_for_scope(self, example: Dict) -> Dict:
        """
        Extract components from example based on rubric scope.

        Args:
            example: Example dict with conversations

        Returns:
            Dict of template variables for judge prompt
        """
        scope = self.rubric["scope"]

        # Extract conversations
        system_prompt, user_request, assistant_response = self._extract_conversations(example)

        if scope == "response":
            # Full assistant response
            return {
                "system_prompt": system_prompt,
                "user_request": user_request,
                "assistant_response": assistant_response
            }

        elif scope == "thinking":
            # Only thinking block
            thinking_block = self._extract_thinking_block(assistant_response)
            return {
                "thinking_block": json.dumps(thinking_block, indent=2, ensure_ascii=False)
            }

        elif scope == "tool_calls":
            # Only tool calls
            tool_calls = self._extract_tool_calls(assistant_response)
            return {
                "system_prompt": system_prompt,
                "user_request": user_request,
                "tool_calls": json.dumps(tool_calls, indent=2, ensure_ascii=False)
            }

        elif scope == "text":
            # Only text response
            text_response = self._extract_text_response(assistant_response)
            return {
                "system_prompt": system_prompt,
                "user_request": user_request,
                "text_response": text_response
            }

        else:
            raise ValueError(f"Unknown scope: {scope}")

    def _extract_conversations(self, example: Dict) -> Tuple[str, str, str]:
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
                # For scope="response", include EVERYTHING: content + tool_calls
                assistant_response = content

                # Add tool_calls if present
                if "tool_calls" in conv and conv["tool_calls"]:
                    assistant_response += "\n\n[TOOL_CALLS]\n"
                    for tool_call in conv["tool_calls"]:
                        func = tool_call.get("function", {})
                        tool_name = func.get("name", "unknown")
                        arguments = func.get("arguments", "{}")
                        assistant_response += f"\ntool_call: {tool_name}\n"
                        assistant_response += f"arguments: {arguments}\n"

        return system_prompt, user_request, assistant_response

    def _extract_thinking_block(self, assistant_response: str) -> Optional[Dict]:
        """Extract thinking block from assistant response."""
        if "<thinking>" not in assistant_response or "</thinking>" not in assistant_response:
            return None

        start = assistant_response.index("<thinking>") + len("<thinking>")
        end = assistant_response.index("</thinking>")
        thinking_str = assistant_response[start:end].strip()

        try:
            return json.loads(thinking_str)
        except json.JSONDecodeError:
            return None

    def _extract_tool_calls(self, assistant_response: str) -> Optional[Dict]:
        """Extract tool calls from assistant response."""
        # Look for tool_call pattern
        if "tool_call:" not in assistant_response:
            return None

        # Simple extraction - could be improved
        lines = assistant_response.split("\n")
        tool_calls = []

        for i, line in enumerate(lines):
            if line.startswith("tool_call:"):
                tool_name = line.split("tool_call:")[1].strip()
                # Next line should be arguments
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

        return tool_calls if tool_calls else None

    def _extract_text_response(self, assistant_response: str) -> str:
        """Extract plain text response (excluding thinking and tool calls)."""
        # Remove thinking block
        text = assistant_response
        if "<thinking>" in text and "</thinking>" in text:
            start = text.index("<thinking>")
            end = text.index("</thinking>") + len("</thinking>")
            text = text[:start] + text[end:]

        # Remove tool call lines
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
