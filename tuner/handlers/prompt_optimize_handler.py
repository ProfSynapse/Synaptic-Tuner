"""Prompt optimization CLI handler."""

from __future__ import annotations

from argparse import Namespace
from typing import Optional

from tuner.handlers.base import BaseHandler


class PromptOptimizeHandler(BaseHandler):
    """Run deterministic config-first prompt optimization."""

    def __init__(self, args: Optional[Namespace] = None):
        super().__init__(args=args)

    @property
    def name(self) -> str:
        return "prompt-optimize"

    def can_handle_direct_mode(self) -> bool:
        return True

    def handle(self) -> int:
        config_path = getattr(self.args, "prompt_opt_config", None)
        if not config_path:
            self.output_error(
                "Missing --prompt-opt-config path.",
                code="MISSING_PROMPT_OPT_CONFIG",
            )
            return 1

        overrides = {}
        output_dir = getattr(self.args, "prompt_opt_output_dir", None)
        if output_dir:
            overrides["output_dir"] = output_dir

        try:
            from shared.prompt_optimization import PromptOptimizationService

            result = PromptOptimizationService.from_config(
                config_path,
                overrides=overrides or None,
            ).run()
        except Exception as exc:
            self.output_error(
                f"Prompt optimization failed: {exc}",
                code="PROMPT_OPTIMIZATION_FAILED",
            )
            return 1

        best = result.best_candidate
        payload = {
            "run_id": result.run_id,
            "output_dir": result.output_dir,
            "candidate_count": result.candidate_count,
            "best_candidate_id": best.get("id"),
            "best_score": best.get("score"),
            "artifact_paths": result.artifact_paths,
        }
        for field in ["schema_version", "strategy", "generation_count", "stop_reason", "best_score"]:
            value = getattr(result, field, None)
            if value is not None:
                payload[field] = value
        self.output(
            payload,
            (
                "Prompt optimization complete. "
                f"Best candidate: {best.get('id')} "
                f"(score={best.get('score')}). Output: {result.output_dir}"
            ),
        )
        return 0
