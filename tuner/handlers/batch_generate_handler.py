"""Handler for the ``batch-generate`` verb.

Location: tuner/handlers/batch_generate_handler.py
Purpose: Parse CLI args and drive ``tuner.batch.runner.run_batch_generate``.
    Prompts in (JSONL), completions out (JSONL), with crash-safe incremental
    persistence, resume, OOM auto-halving, and the generic sync hook.
"""

from __future__ import annotations

from argparse import Namespace
from typing import Optional

from tuner.handlers.base import BaseHandler


class BatchGenerateHandler(BaseHandler):
    """Batched prompts-in / completions-out generation."""

    def __init__(self, args: Optional[Namespace] = None):
        super().__init__(args=args)

    @property
    def name(self) -> str:
        return "batch-generate"

    def can_handle_direct_mode(self) -> bool:
        return True

    def handle(self) -> int:
        from tuner.batch.runner import run_batch_generate

        args = self.args
        prompts = getattr(args, "prompts", None)
        out_dir = getattr(args, "out_dir", None)
        model = getattr(args, "model", None)
        if not prompts or not out_dir or not model:
            self.output_error(
                "batch-generate requires --prompts, --model, and --out-dir.",
                code="MISSING_ARGS",
            )
            return 1

        stop = list(getattr(args, "stop_strings", None) or []) or None

        try:
            summary = run_batch_generate(
                prompts_path=prompts,
                out_dir=out_dir,
                model=model,
                engine=getattr(args, "engine", "hf-batched"),
                max_new_tokens=getattr(args, "max_new_tokens", 48),
                batch_size=getattr(args, "batch_size", 16),
                do_sample=getattr(args, "do_sample", False),
                temperature=getattr(args, "temperature", 1.0),
                top_p=getattr(args, "top_p", 1.0),
                seed=getattr(args, "seed", None),
                stop=stop,
                resume=getattr(args, "resume", False),
                sync_every=getattr(args, "sync_every", 0) or 0,
                sync_cmd=getattr(args, "sync_cmd", None),
                dtype=getattr(args, "compute_dtype", None),
                log=(lambda m: None) if self.json_mode else print,
            )
        except Exception as exc:  # noqa: BLE001 - surface as a clean CLI error
            self.output_error(str(exc), code="BATCH_GENERATE_FAILED")
            return 1

        self.output(summary, human_readable=(
            f"batch-generate complete: {summary['newly_processed']} new rows "
            f"-> {summary['artifact']}{summary.get('gpu_peak_suffix', '')}"
        ))
        return 0
