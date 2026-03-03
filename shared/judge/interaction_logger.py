"""Interaction logger -- thread-safe JSONL logging for KTO training.

Location: shared/judge/interaction_logger.py
Summary: Logs judge interactions in ChatML-compatible JSONL format suitable
         for KTO fine-tuning. Thread-safe via threading.Lock. Configurable
         prefix allows reuse across consumers (Evaluator uses "judge",
         SynthChat could use "improvement"). Mirrors SynthChat's
         InteractionLogger pattern but is generic.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class InteractionLogger:
    """Log judge interactions in JSONL format for KTO training.

    Each log entry follows ChatML format with system/user/assistant turns
    and a KTO-compatible label (true=passed, false=failed).

    Args:
        output_dir: Directory to write interaction log files.
        enabled: Whether logging is active. When False, all log calls are no-ops.
        prefix: Filename prefix (e.g., "judge" -> "judge_20260302_184500.jsonl").
    """

    def __init__(
        self,
        output_dir: Path,
        enabled: bool = True,
        prefix: str = "judge",
    ):
        self.output_dir = Path(output_dir)
        self.enabled = enabled
        self._write_lock = Lock()
        self._entry_count = 0
        self.log_file: Optional[Path] = None

        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = self.output_dir / f"{prefix}_{timestamp}.jsonl"
            logger.info("Interaction logging enabled: %s", self.log_file)

    def log_judge_interaction(
        self,
        judge_prompt: str,
        judge_response_raw: str,
        rubric_name: str,
        scores: Dict[str, float],
        passed: bool,
        case_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        judge_mode: Optional[str] = None,
        eval_model: Optional[str] = None,
        judge_model: Optional[str] = None,
    ) -> None:
        """Log a single judge interaction for KTO training.

        Args:
            judge_prompt: The user prompt sent to the judge LLM.
            judge_response_raw: Raw JSON string from the judge response.
            rubric_name: Rubric(s) evaluated (human-readable name).
            scores: Dict of {score_field_name: score_value}.
            passed: Overall pass/fail for KTO label.
            case_id: Evaluation case identifier for lineage tracking.
            system_prompt: System prompt used for the judge call.
            judge_mode: Composition mode used ("and", "or", "judge_only").
            eval_model: Model being evaluated.
            judge_model: Model used as judge.
        """
        if not self.enabled:
            return

        # Build a generic system prompt if none was provided
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
                {"role": "assistant", "content": judge_response_raw},
            ],
            "label": passed,
            "metadata": {
                "type": "judge",
                "rubric": rubric_name,
                "scores": scores,
                "passed": passed,
                "case_id": case_id,
                "judge_mode": judge_mode,
                "eval_model": eval_model,
                "judge_model": judge_model,
                "timestamp": datetime.now().isoformat(),
            },
        }

        self._write_interaction(interaction)

    def get_stats(self) -> Dict:
        """Return statistics about logged interactions.

        Returns:
            Dict with total count and breakdowns by label and rubric.
        """
        if not self.enabled or self.log_file is None or not self.log_file.exists():
            return {"total": 0}

        stats: Dict = {
            "total": 0,
            "by_label": {"true": 0, "false": 0},
            "by_rubric": {},
        }

        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    stats["total"] += 1

                    label_key = "true" if entry.get("label") else "false"
                    stats["by_label"][label_key] += 1

                    rubric = entry.get("metadata", {}).get("rubric", "unknown")
                    stats["by_rubric"][rubric] = stats["by_rubric"].get(rubric, 0) + 1
                except (json.JSONDecodeError, KeyError):
                    continue

        return stats

    def _write_interaction(self, interaction: Dict) -> None:
        """Write an interaction entry to the JSONL file (thread-safe)."""
        if self.log_file is None:
            return

        try:
            with self._write_lock:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(interaction, ensure_ascii=False) + "\n")
                self._entry_count += 1
        except Exception as exc:
            logger.error("Failed to write interaction: %s", exc)
