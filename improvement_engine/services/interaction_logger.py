"""Logger for judge/improver interactions to create fine-tuning dataset."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

from ..utils.logger import ImproveLogger

if TYPE_CHECKING:
    from .rubric_runner import JudgmentResult


class InteractionLogger:
    """Logs judge/improver interactions for creating fine-tuning datasets with KTO labels."""

    def __init__(
        self,
        output_path: Path,
        logger: Optional[ImproveLogger] = None
    ):
        """
        Initialize interaction logger.

        Args:
            output_path: Path to write interaction JSONL
            logger: Logger instance
        """
        self.output_path = output_path
        self.logger = logger or ImproveLogger()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Open file in append mode
        self.file_handle = open(self.output_path, "a", encoding="utf-8")

        # Track last improver interaction for retroactive labeling
        self.last_improver_interaction = None

        self.logger.info(f"Interaction logging enabled: {output_path}")

    def log_judge_interaction(
        self,
        example_id: str,
        iteration: int,
        rubric_keys: List[str],
        example: Dict,
        judgment: "JudgmentResult",
    ):
        """
        Label the previous improver interaction based on judge result.

        If judgment passed, the previous improve gets labeled True.
        If judgment failed, the previous improve gets labeled False.

        Args:
            example_id: Unique example identifier
            iteration: Current iteration number
            rubric_keys: List of rubrics being evaluated
            example: Full example being judged
            judgment: Judge output
        """
        # Retroactively label the previous improver interaction
        if self.last_improver_interaction is not None:
            # Label based on whether this judge passed
            self.last_improver_interaction["label"] = judgment.passed
            self._write_interaction(self.last_improver_interaction)
            self.last_improver_interaction = None

        # Note: We don't log judge interactions - only improver interactions are needed for training

    def log_improver_interaction(
        self,
        example_id: str,
        iteration: int,
        scope: str,
        improve_prompt: str,
        regenerated_content: str,
        rubric_keys: List[str],
        feedback: str,
        rubric_judge_prompts: Dict[str, str],
    ):
        """
        Log an improver interaction (label will be set retroactively).

        The label is determined by the NEXT judge call:
        - True if next judge passes
        - False if next judge fails

        Args:
            example_id: Unique example identifier
            iteration: Current iteration number
            scope: Scope being regenerated (system_prompt, thinking, response)
            improve_prompt: Full improve prompt sent to LLM
            regenerated_content: LLM's regenerated content
            rubric_keys: List of rubrics being used
            feedback: Feedback from judge that triggered improvement
            rubric_judge_prompts: Dict of rubric configs/judge prompts
        """
        # System: The rubric judge prompt (config)
        # Use first rubric's judge prompt as system
        system_content = ""
        if rubric_judge_prompts:
            first_rubric = list(rubric_judge_prompts.values())[0]
            system_content = first_rubric

        # Strip code fences from regenerated content (```xml, ```json, etc.)
        cleaned_content = self._strip_code_fences(regenerated_content)

        # Store interaction for retroactive labeling (don't write yet)
        # Format: system=rubric config, user=judge feedback, assistant=fix
        self.last_improver_interaction = {
            "conversations": [
                {
                    "role": "system",
                    "content": system_content,
                },
                {
                    "role": "user",
                    "content": feedback,  # Judge's feedback on what's wrong
                },
                {
                    "role": "assistant",
                    "content": cleaned_content,  # The fix (cleaned)
                }
            ],
            # Label will be set in next log_judge_interaction() call
        }

    def _strip_code_fences(self, content: str) -> str:
        """Strip code fences (```xml, ```json, etc.) from content."""
        import re
        # Remove opening fence (```xml, ```json, etc.)
        content = re.sub(r'^```\w*\n?', '', content.strip(), flags=re.MULTILINE)
        # Remove closing fence
        content = re.sub(r'\n?```$', '', content.strip(), flags=re.MULTILINE)
        return content.strip()

    def _write_interaction(self, interaction: Dict):
        """Write interaction to JSONL file."""
        json_line = json.dumps(interaction, ensure_ascii=False)
        self.file_handle.write(json_line + "\n")
        self.file_handle.flush()

    def _extract_system_prompt(self, example: Dict) -> Optional[str]:
        """Extract system prompt from example."""
        for conv in example.get("conversations", []):
            if conv.get("role") == "system":
                return conv.get("content", "")
        return None

    def _extract_user_request(self, example: Dict) -> Optional[str]:
        """Extract user request from example."""
        for conv in example.get("conversations", []):
            if conv.get("role") == "user":
                return conv.get("content", "")
        return None

    def _extract_assistant_response(self, example: Dict) -> Dict:
        """Extract assistant response (text + tool calls) from example."""
        for conv in example.get("conversations", []):
            if conv.get("role") == "assistant":
                return {
                    "content": conv.get("content", ""),
                    "tool_calls": conv.get("tool_calls", []),
                }
        return {}

    def close(self):
        """Close the file handle and write any pending interactions."""
        # If there's a pending improver interaction, label it as False (didn't converge)
        if self.last_improver_interaction is not None:
            self.last_improver_interaction["label"] = False
            self._write_interaction(self.last_improver_interaction)
            self.last_improver_interaction = None

        if hasattr(self, "file_handle") and self.file_handle:
            self.file_handle.close()
            self.logger.info(f"Interaction log closed: {self.output_path}")

    def __enter__(self):
        """Context manager enter."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
