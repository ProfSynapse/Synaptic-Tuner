"""Interaction logger - captures judge/improver interactions for KTO training.

Logs interactions in ChatML format suitable for fine-tuning.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from ..utils.logger import ImproveLogger


class InteractionLogger:
    """
    Logs judge and improver interactions in ChatML format.

    Purpose: Capture training data for KTO fine-tuning so we can train
    a small local model to do improvement more effectively.

    Format:
    {
        "conversations": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "judge prompt"},
            {"role": "assistant", "content": "judge response"}
        ],
        "label": true  # true if improvement passed, false if failed
    }
    """

    def __init__(
        self,
        output_dir: Path,
        enabled: bool = True,
        logger: Optional[ImproveLogger] = None,
        dataset_name: Optional[str] = None
    ):
        """
        Initialize interaction logger.

        Args:
            output_dir: Directory to write interaction logs
            enabled: Whether logging is enabled
            logger: Logger instance
            dataset_name: Optional dataset identifier (parent_folder/filename)
        """
        self.output_dir = Path(output_dir)
        self.enabled = enabled
        self.logger = logger or ImproveLogger()

        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Create timestamped log file with dataset name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if dataset_name:
                # Format: interactions_parentFolder_filename_timestamp.jsonl
                self.log_file = self.output_dir / f"interactions_{dataset_name}_{timestamp}.jsonl"
            else:
                self.log_file = self.output_dir / f"interactions_{timestamp}.jsonl"
            self.logger.info(f"Interaction logging enabled: {self.log_file}")

    def log_judge_interaction(
        self,
        judge_prompt: str,
        judge_response: str,
        rubric_name: str,
        score: float,
        passed: bool,
        example_id: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> None:
        """
        Log a judge interaction.

        Args:
            judge_prompt: The USER prompt sent to judge (example to evaluate)
            judge_response: The judge's response
            rubric_name: Name of the rubric
            score: Score returned by judge
            passed: Whether the example passed
            example_id: Optional example identifier
            system_prompt: Optional SYSTEM prompt with criteria (if not provided, creates generic)
        """
        if not self.enabled:
            return

        # Use provided system prompt or create generic fallback
        if not system_prompt:
            system_prompt = (
                f"You are a quality judge for the '{rubric_name}' rubric. "
                f"Evaluate examples and provide scores from 0.0 (poor) to 1.0 (excellent). "
                f"Return JSON with the score field."
            )

        interaction = {
            "conversations": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": judge_prompt},
                {"role": "assistant", "content": judge_response}
            ],
            "label": passed,  # KTO label: true if passed, false if failed
            "metadata": {
                "type": "judge",
                "rubric": rubric_name,
                "score": score,
                "passed": passed,
                "example_id": example_id,
                "timestamp": datetime.now().isoformat()
            }
        }

        self._write_interaction(interaction)

    def log_improver_interaction(
        self,
        improver_prompt: str,
        improver_response: str,
        rubric_name: str,
        scope: str,
        before_score: float,
        after_score: float,
        improved: bool,
        example_id: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> None:
        """
        Log an improver interaction.

        Args:
            improver_prompt: The prompt sent to improver
            improver_response: The improver's response
            rubric_name: Name of the rubric
            scope: Scope being improved (thinking, system_prompt, response)
            before_score: Score before improvement
            after_score: Score after improvement
            improved: Whether score improved
            example_id: Optional example identifier
            system_prompt: Optional system prompt (will create default if not provided)
        """
        if not self.enabled:
            return

        # Use provided system prompt or create default
        if not system_prompt:
            system_prompt = (
                f"You are a quality improver for the '{rubric_name}' rubric. "
                f"Your task is to improve {scope} content based on feedback. "
                f"Follow the format specification and address all issues."
            )

        interaction = {
            "conversations": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": improver_prompt},
                {"role": "assistant", "content": improver_response}
            ],
            "label": improved,  # KTO label: true if improved, false if not
            "metadata": {
                "type": "improver",
                "rubric": rubric_name,
                "scope": scope,
                "before_score": before_score,
                "after_score": after_score,
                "delta": after_score - before_score,
                "improved": improved,
                "example_id": example_id,
                "timestamp": datetime.now().isoformat()
            }
        }

        self._write_interaction(interaction)

    def log_full_iteration(
        self,
        example: Dict,
        rubric_name: str,
        scope: str,
        judge_prompt: str,
        judge_response: str,
        before_score: float,
        improver_prompt: str,
        improver_response: str,
        after_score: float,
        passed: bool,
        example_id: Optional[str] = None
    ) -> None:
        """
        Log a complete judge + improve iteration.

        This creates a multi-turn conversation showing the full improvement flow.

        Args:
            example: The example being improved
            rubric_name: Name of the rubric
            scope: Scope being improved
            judge_prompt: Initial judge prompt
            judge_response: Initial judge response
            before_score: Score before improvement
            improver_prompt: Improver prompt
            improver_response: Improver response
            after_score: Score after improvement
            passed: Whether final score passed threshold
            example_id: Optional example identifier
        """
        if not self.enabled:
            return

        system_prompt = (
            f"You are working with the '{rubric_name}' rubric to improve training data. "
            f"First judge the quality, then improve the {scope} content if needed."
        )

        interaction = {
            "conversations": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"**JUDGE TASK:**\n\n{judge_prompt}"},
                {"role": "assistant", "content": judge_response},
                {"role": "user", "content": f"**IMPROVE TASK:**\n\n{improver_prompt}"},
                {"role": "assistant", "content": improver_response}
            ],
            "label": passed and (after_score > before_score),  # True if improved AND passed
            "metadata": {
                "type": "full_iteration",
                "rubric": rubric_name,
                "scope": scope,
                "before_score": before_score,
                "after_score": after_score,
                "delta": after_score - before_score,
                "passed": passed,
                "example_id": example_id,
                "timestamp": datetime.now().isoformat()
            }
        }

        self._write_interaction(interaction)

    def _write_interaction(self, interaction: Dict) -> None:
        """Write interaction to JSONL file."""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(interaction, ensure_ascii=False) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write interaction: {e}")

    def get_stats(self) -> Dict:
        """
        Get statistics about logged interactions.

        Returns:
            Dict with counts by type, labels, etc.
        """
        if not self.enabled or not self.log_file.exists():
            return {"total": 0}

        stats = {
            "total": 0,
            "by_type": {},
            "by_label": {"true": 0, "false": 0},
            "by_rubric": {}
        }

        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    interaction = json.loads(line)
                    stats["total"] += 1

                    # Count by type
                    interaction_type = interaction.get("metadata", {}).get("type", "unknown")
                    stats["by_type"][interaction_type] = stats["by_type"].get(interaction_type, 0) + 1

                    # Count by label
                    label = "true" if interaction.get("label") else "false"
                    stats["by_label"][label] += 1

                    # Count by rubric
                    rubric = interaction.get("metadata", {}).get("rubric", "unknown")
                    stats["by_rubric"][rubric] = stats["by_rubric"].get(rubric, 0) + 1

                except Exception:
                    continue

        return stats
