"""Handler for the ``batch-capture`` verb.

Location: tuner/handlers/batch_capture_handler.py
Purpose: Parse CLI args and drive ``tuner.batch.runner.run_batch_capture``.
    Sequences in (JSONL), per-layer hidden states at named token positions out
    (per-row safetensors + a capture.jsonl index), with the same crash-safe
    incremental persistence, resume, OOM auto-halving, and sync hook.
"""

from __future__ import annotations

from argparse import Namespace
from typing import Optional

from tuner.handlers.base import BaseHandler


class BatchCaptureHandler(BaseHandler):
    """Batched sequences-in / hidden-states-out capture."""

    def __init__(self, args: Optional[Namespace] = None):
        super().__init__(args=args)

    @property
    def name(self) -> str:
        return "batch-capture"

    def can_handle_direct_mode(self) -> bool:
        return True

    def handle(self) -> int:
        from tuner.batch.runner import run_batch_capture

        args = self.args
        rows = getattr(args, "rows", None)
        out_dir = getattr(args, "out_dir", None)
        model = getattr(args, "model", None)
        if not rows or not out_dir or not model:
            self.output_error(
                "batch-capture requires --rows, --model, and --out-dir.",
                code="MISSING_ARGS",
            )
            return 1

        try:
            summary = run_batch_capture(
                rows_path=rows,
                out_dir=out_dir,
                model=model,
                engine=getattr(args, "engine", "hf-batched"),
                layers=getattr(args, "layers", "all"),
                batch_size=getattr(args, "batch_size", 16),
                persist_dtype=getattr(args, "persist_dtype", "float32"),
                resume=getattr(args, "resume", False),
                sync_every=getattr(args, "sync_every", 0) or 0,
                sync_cmd=getattr(args, "sync_cmd", None),
                dtype=getattr(args, "compute_dtype", None),
                log=(lambda m: None) if self.json_mode else print,
            )
        except Exception as exc:  # noqa: BLE001 - surface as a clean CLI error
            self.output_error(str(exc), code="BATCH_CAPTURE_FAILED")
            return 1

        self.output(summary, human_readable=(
            f"batch-capture complete: {summary['newly_processed']} new rows "
            f"-> {summary['artifact']}"
        ))
        return 0
