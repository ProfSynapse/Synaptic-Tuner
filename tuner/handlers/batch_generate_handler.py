"""Handler for the ``batch-generate`` verb.

Location: tuner/handlers/batch_generate_handler.py
Purpose: Parse CLI args and drive ``tuner.batch.runner.run_batch_generate``.
    Prompts in (JSONL), completions out (JSONL), with crash-safe incremental
    persistence, resume, OOM auto-halving, and the generic sync hook.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
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
        json_schema = None
        json_schema_path = getattr(args, "json_schema", None)
        if json_schema_path:
            try:
                json_schema = json.loads(Path(json_schema_path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                self.output_error(
                    f"Cannot read --json-schema {json_schema_path!r}: {exc}",
                    code="INVALID_JSON_SCHEMA",
                )
                return 1
            if not isinstance(json_schema, dict):
                self.output_error(
                    "--json-schema must contain a JSON object.",
                    code="INVALID_JSON_SCHEMA",
                )
                return 1
        limit_mm_per_prompt = None
        limit_mm_raw = getattr(args, "limit_mm_per_prompt", None)
        if limit_mm_raw:
            try:
                limit_mm_per_prompt = json.loads(limit_mm_raw)
            except json.JSONDecodeError as exc:
                self.output_error(
                    f"Cannot parse --limit-mm-per-prompt JSON: {exc}",
                    code="INVALID_MM_LIMITS",
                )
                return 1
            if not isinstance(limit_mm_per_prompt, dict):
                self.output_error(
                    "--limit-mm-per-prompt must be a JSON object.",
                    code="INVALID_MM_LIMITS",
                )
                return 1

        try:
            summary = run_batch_generate(
                prompts_path=prompts,
                out_dir=out_dir,
                model=model,
                model_revision=getattr(args, "model_revision", None),
                tokenizer_revision=getattr(args, "tokenizer_revision", None),
                engine=getattr(args, "engine", "hf-batched"),
                max_new_tokens=getattr(args, "max_new_tokens", 48),
                min_new_tokens=getattr(args, "min_new_tokens", 0),
                batch_size=getattr(args, "batch_size", 16),
                do_sample=getattr(args, "do_sample", False),
                temperature=getattr(args, "temperature", 1.0),
                top_p=getattr(args, "top_p", 1.0),
                seed=getattr(args, "seed", None),
                extra_eos_tokens=getattr(args, "extra_eos_tokens", None),
                stop=stop,
                json_schema=json_schema,
                structured_output_backend=getattr(
                    args, "structured_output_backend", "auto"
                ),
                structured_output_disable_any_whitespace=getattr(
                    args, "structured_output_disable_any_whitespace", False
                ),
                expected_vllm_version=getattr(args, "expected_vllm_version", None),
                min_compute_capability=getattr(args, "min_compute_capability", None),
                tensor_parallel_size=getattr(args, "tensor_parallel_size", 1),
                max_num_seqs=getattr(args, "max_num_seqs", None),
                max_num_batched_tokens=getattr(args, "max_num_batched_tokens", None),
                max_model_len=getattr(args, "max_model_len", None),
                limit_mm_per_prompt=limit_mm_per_prompt,
                gpu_memory_utilization=getattr(args, "gpu_memory_utilization", None),
                resume=getattr(args, "resume", False),
                sync_every=getattr(args, "sync_every", 0) or 0,
                sync_cmd=getattr(args, "sync_cmd", None),
                dtype=getattr(args, "compute_dtype", None),
                trust_remote_code=getattr(args, "trust_remote_code", False),
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
