"""
Inspect HF Jobs cloud evaluation artifacts stored in Hugging Face Buckets.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Optional

from tuner.handlers.base import BaseHandler
from tuner.handlers.cloud_eval_handler import CloudEvalHandler
from tuner.ui import BOX, print_config, print_error, print_header, print_info, print_menu
from shared.utilities.env import get_hf_token


class CloudInspectHandler(BaseHandler):
    """Inspect saved HF Jobs cloud evaluation results."""

    def __init__(self, args: Optional[Namespace] = None):
        super().__init__(args=args)
        self._cloud_eval = CloudEvalHandler(args=args)
        self._cloud_eval._repo_root = self.repo_root

    @property
    def name(self) -> str:
        return "cloud-inspect"

    def can_handle_direct_mode(self) -> bool:
        return True

    def _list_eval_runs(self, huggingface_hub, bucket_id: str, run_prefix: str) -> List[Dict[str, str]]:
        api = huggingface_hub.HfApi(token=get_hf_token())
        prefix = f"{run_prefix.strip('/')}/evaluations/vllm"
        eval_runs: List[Dict[str, str]] = []

        for item in api.list_bucket_tree(
            bucket_id,
            prefix=prefix,
            recursive=False,
            token=get_hf_token(),
        ):
            item_type = getattr(item, "type", None)
            item_path = getattr(item, "path", "")
            if item_type != "directory":
                continue
            suffix = item_path[len(prefix):].strip("/")
            if not suffix or "/" in suffix:
                continue
            eval_runs.append({"slug": suffix, "prefix": item_path})

        eval_runs.sort(key=lambda run: run["slug"], reverse=True)
        return eval_runs

    def _select_eval_run(self, eval_runs: List[Dict[str, str]], requested_eval: Optional[str]) -> Dict[str, str]:
        if not eval_runs:
            raise RuntimeError("No cloud evaluation runs found for the selected training run.")

        if requested_eval in (None, "", "latest"):
            if requested_eval == "latest":
                return eval_runs[0]
            choice = print_menu(
                [(run["prefix"], f"{BOX['bullet']} {run['slug']}") for run in eval_runs[:20]],
                "Select cloud evaluation run to inspect:",
            )
            if not choice:
                raise RuntimeError("Cloud inspection cancelled.")
            for run in eval_runs:
                if run["prefix"] == choice:
                    return run
            raise RuntimeError(f"Unknown evaluation selection: {choice}")

        normalized = requested_eval.strip("/")
        for run in eval_runs:
            if run["slug"] == normalized or run["prefix"] == normalized:
                return run

        raise RuntimeError(f"Cloud evaluation run not found: {requested_eval}")

    def _download_eval_payload(self, bucket_id: str, eval_prefix: str) -> Optional[Dict]:
        try:
            from huggingface_hub import sync_bucket
        except ImportError:
            return None

        local_root = Path(tempfile.mkdtemp(prefix="cloud-inspect-"))
        try:
            sync_bucket(
                f"hf://buckets/{bucket_id}/{eval_prefix.strip('/')}",
                str(local_root),
            )
            results_path = local_root / "evaluation_results.json"
            if not results_path.exists():
                return None
            return json.loads(results_path.read_text(encoding="utf-8"))
        finally:
            shutil.rmtree(local_root, ignore_errors=True)

    def _extract_failure_reason(self, record: Dict) -> str:
        error = record.get("error")
        if error:
            return str(error)

        validator = record.get("validator") or {}
        for issue in validator.get("issues", []) or []:
            if issue.get("message"):
                return str(issue["message"])

        environment = record.get("environment") or {}
        for issue in environment.get("issues", []) or []:
            if issue.get("message"):
                return str(issue["message"])

        behavior = record.get("behavior") or {}
        for issue in behavior.get("issues", []) or []:
            if issue.get("message"):
                return str(issue["message"])

        judge = record.get("judge") or {}
        judge_result = judge.get("judge_result") or {}
        for score in judge_result.get("scores", []) or []:
            if score.get("feedback"):
                return str(score["feedback"])

        return "No failure reason captured."

    def _print_failures(self, payload: Dict) -> None:
        records = payload.get("records") or []
        failed = [
            record for record in records
            if not bool(record.get("passed", False))
        ]

        if not failed:
            print_info("No failed cases in this evaluation run.")
            return

        print()
        print_header("FAILURES", f"Showing {len(failed)} failed/warned cases")
        for index, record in enumerate(failed[:10], start=1):
            case_id = record.get("case_id") or f"case-{index}"
            reason = self._extract_failure_reason(record)
            response_text = str(record.get("response_text") or "").strip()
            if len(response_text) > 400:
                response_text = response_text[:400] + "..."
            print_config(
                {
                    "Case": case_id,
                    "Tags": ", ".join(record.get("tags") or []),
                    "Why": reason,
                    "LLM Response": response_text or "(empty)",
                },
                f"Failure {index}",
            )

    def handle(self) -> int:
        print_header("CLOUD INSPECT", "Inspect saved HF cloud evaluation results")

        try:
            huggingface_hub = self._cloud_eval._validate_environment()
            hf_settings = self._cloud_eval._hf_jobs_settings()
            bucket_id = self._cloud_eval._resolve_bucket_id(
                huggingface_hub,
                getattr(self.args, "bucket", None) or hf_settings.get("artifact_identifier", ""),
            )
            runs = self._cloud_eval._list_remote_runs(
                huggingface_hub,
                bucket_id=bucket_id,
                method=getattr(self.args, "method", None),
            )
            selected_run = self._cloud_eval._select_run(runs, getattr(self.args, "run", None))
            eval_runs = self._list_eval_runs(huggingface_hub, bucket_id, selected_run["prefix"])
            selected_eval = self._select_eval_run(eval_runs, getattr(self.args, "eval_run", None))
            payload = self._download_eval_payload(bucket_id, selected_eval["prefix"])
            if not payload:
                print_error("evaluation_results.json not found for the selected evaluation run.")
                return 1
        except Exception as exc:
            print_error(str(exc))
            return 1

        summary = payload.get("summary") or {}
        print_config(
            {
                "Bucket": bucket_id,
                "Training Run": selected_run["prefix"],
                "Evaluation Run": selected_eval["prefix"],
                "Passed": str(summary.get("passed", 0)),
                "Warned": str(summary.get("warned", 0)),
                "Failed": str(summary.get("failed", 0)),
                "Total": str(summary.get("total", 0)),
                "Request Errors": str(summary.get("request_errors", 0)),
            },
            "Cloud Evaluation Summary",
        )

        self._print_failures(payload)
        return 0
