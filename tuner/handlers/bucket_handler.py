"""
Bucket artifact read/list handler.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Optional

from shared.utilities.bucket_artifacts import list_artifacts, pull_artifacts, push_artifacts, read_artifact
from shared.utilities.env import get_hf_token
from tuner.backends.training.cloud.base_cloud import load_cloud_config
from tuner.cloud import load_huggingface_hub, resolve_hf_bucket_id
from tuner.core.exceptions import CloudProviderError
from tuner.handlers.base import BaseHandler


class BucketHandler(BaseHandler):
    """Read or list local/HF bucket-backed artifacts from the CLI."""

    @property
    def name(self) -> str:
        return "bucket"

    def can_handle_direct_mode(self) -> bool:
        return True

    def _cloud_config_path(self) -> Path:
        return self.repo_root / "Trainers" / "cloud" / "cloud_config.yaml"

    def _default_bucket_id(self) -> Optional[str]:
        settings = load_cloud_config(self._cloud_config_path()).get("hf_jobs", {})
        configured = str(settings.get("artifact_identifier", "")).strip()
        if not configured:
            return None
        if "/" in configured:
            return configured.strip("/")
        hf_token = get_hf_token()
        if not hf_token:
            return configured.strip("/")
        hub = load_huggingface_hub(require_apis=("create_bucket", "HfApi"))
        return resolve_hf_bucket_id(hub, configured, token=hf_token, private=True)

    def _selected_bucket_id(self) -> Optional[str]:
        explicit = str(getattr(self.args, "bucket", "") or "").strip()
        if explicit:
            return explicit.strip("/")
        return self._default_bucket_id()

    def _require_bucket_id(self) -> str:
        bucket_id = self._selected_bucket_id()
        if not bucket_id:
            raise CloudProviderError("Bucket ID is required. Use --bucket or configure HF artifact_identifier.")
        return bucket_id

    def _require_path(self) -> str:
        path = str(getattr(self.args, "path", "") or "").strip()
        if not path:
            raise CloudProviderError("Bucket path is required. Use --path <bucket-relative path or hf:// URI>.")
        return path

    def _bucket_for_path(self, path: str) -> Optional[str]:
        if path.startswith("hf://"):
            return None
        local_path = Path(path)
        if local_path.exists() or path.startswith("./") or path.startswith("../") or path.startswith("/"):
            return None
        return self._selected_bucket_id()

    def _normalize_bucket_path(self, path: str) -> str:
        normalized = str(path or "").strip()
        if normalized.startswith("hf://"):
            return normalized
        bucket_id = self._selected_bucket_id()
        if bucket_id and normalized.startswith(f"buckets/{bucket_id}/"):
            return normalized[len(f"buckets/{bucket_id}/") :]
        return normalized

    def _read_json(self, path: str) -> dict[str, Any]:
        normalized_path = self._normalize_bucket_path(path)
        bucket_id = self._bucket_for_path(normalized_path)
        contents = read_artifact(normalized_path, bucket_id=bucket_id)
        return json.loads(contents)

    @staticmethod
    def _artifact_name(path: str) -> str:
        return Path(str(path).rstrip("/")).name

    def _eval_candidates(self, run_prefix: str) -> list[str]:
        eval_root = f"{run_prefix.rstrip('/')}/evaluations/vllm"
        entries = list_artifacts(eval_root, bucket_id=self._bucket_for_path(eval_root))
        return [
            self._normalize_bucket_path(entry["path"]).rstrip("/")
            for entry in entries
            if entry.get("type") == "directory"
        ]

    def _latest_eval_prefix(self, run_prefix: str) -> Optional[str]:
        candidates = self._eval_candidates(run_prefix)
        if not candidates:
            return None
        return max(candidates, key=self._artifact_name)

    @staticmethod
    def _summarize_training(training: dict[str, Any]) -> dict[str, Any]:
        training_cfg = training.get("training", {})
        capacity = training.get("capacity_profile", {})
        runtime = training.get("runtime", {})
        pricing = training.get("pricing", {})
        results = training.get("results", {})
        hardware = training.get("hardware", {})
        return {
            "status": runtime.get("status"),
            "duration_seconds": runtime.get("duration_seconds"),
            "estimated_cost_usd": pricing.get("estimated_cost_usd"),
            "hardware_flavor": hardware.get("cloud_gpu_type"),
            "batch_size": training_cfg.get("batch_size"),
            "gradient_accumulation_steps": training_cfg.get("gradient_accumulation_steps"),
            "effective_batch_size": training_cfg.get("effective_batch_size"),
            "final_loss": results.get("final_loss"),
            "peak_reserved_vram_gb": capacity.get("peak_gpu_memory_reserved_gb"),
            "reserved_headroom_gb": capacity.get("min_gpu_memory_reserved_headroom_gb"),
            "oom_risk_level": capacity.get("oom_risk_level"),
        }

    @staticmethod
    def _summarize_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
        runtime = evaluation.get("runtime", {})
        pricing = evaluation.get("pricing", {})
        execution = evaluation.get("execution", {})
        results = evaluation.get("results_summary", {})
        behavior = evaluation.get("behavior_results", {})
        return {
            "status": runtime.get("status"),
            "duration_seconds": runtime.get("duration_seconds"),
            "estimated_cost_usd": pricing.get("estimated_cost_usd"),
            "hardware_flavor": execution.get("hardware_flavor"),
            "backend": runtime.get("backend"),
            "passed": results.get("passed"),
            "failed": results.get("failed"),
            "request_errors": results.get("request_errors"),
            "overall_pass_rate": results.get("overall_pass_rate"),
            "schema_pass_rate": results.get("schema_pass_rate"),
            "behavior_pass_rate": behavior.get("pass_rate"),
            "top_failure_reasons": evaluation.get("top_failure_reasons", [])[:5],
        }

    @staticmethod
    def _summarize_loss(loss: dict[str, Any]) -> dict[str, Any]:
        runtime = loss.get("runtime", {})
        pricing = loss.get("pricing", {})
        execution = loss.get("execution", {})
        results = loss.get("results", {})
        return {
            "status": loss.get("status"),
            "duration_seconds": runtime.get("duration_seconds"),
            "estimated_cost_usd": pricing.get("estimated_cost_usd"),
            "hardware_flavor": execution.get("hardware_flavor"),
            "worker_count": execution.get("worker_count"),
            "row_count": results.get("row_count"),
            "mean_loss": results.get("mean_loss"),
            "median_loss": results.get("median_loss"),
            "p95_loss": results.get("p95_loss"),
            "max_loss": results.get("max_loss"),
        }

    @staticmethod
    def _derive_signals(
        training: Optional[dict[str, Any]],
        evaluation: Optional[dict[str, Any]],
        loss: Optional[dict[str, Any]],
    ) -> list[str]:
        signals: list[str] = []
        if training:
            headroom = training.get("reserved_headroom_gb")
            if isinstance(headroom, (int, float)) and headroom >= 20:
                signals.append("training appears underpacked; reserved VRAM headroom remained very high")
        if evaluation:
            schema = evaluation.get("schema_pass_rate")
            behavior = evaluation.get("behavior_pass_rate")
            if isinstance(schema, (int, float)) and isinstance(behavior, (int, float)) and schema > behavior:
                signals.append("tool-call structure is stronger than higher-level behavior; focus next on restraint/clarification")
        if evaluation:
            reasons = evaluation.get("top_failure_reasons") or []
            if reasons:
                top_reason = reasons[0]
                reason_text = top_reason.get("reason") if isinstance(top_reason, dict) else None
                if isinstance(reason_text, str) and "TEXT_ONLY" in reason_text:
                    signals.append("top eval failures still prefer text-only or clarify-first behavior over tool use")
        if loss:
            mean_loss = loss.get("mean_loss")
            p95_loss = loss.get("p95_loss")
            if isinstance(mean_loss, (int, float)) and isinstance(p95_loss, (int, float)) and p95_loss > mean_loss * 3:
                signals.append("loss distribution still has a sharp high-loss tail worth slicing before the next SFT run")
        return signals

    def _handle_analyze(self) -> int:
        run_prefix = self._require_path().rstrip("/")
        training_summary = None
        evaluation_summary = None
        loss_summary = None
        eval_prefix = None
        signals: list[str] = []

        try:
            training_summary = self._summarize_training(
                self._read_json(f"{run_prefix}/training_lineage.json")
            )
        except Exception:
            training_summary = None

        try:
            explicit_eval = str(getattr(self.args, "eval_path", "") or "").strip()
            if explicit_eval:
                normalized = self._normalize_bucket_path(explicit_eval)
                eval_prefix = normalized[:-len("/evaluation_lineage.json")] if normalized.endswith("/evaluation_lineage.json") else normalized.rstrip("/")
            else:
                candidates = self._eval_candidates(run_prefix)
                if len(candidates) > 1:
                    signals.append("multiple evaluation prefixes found; latest prefix was selected automatically")
                eval_prefix = max(candidates, key=self._artifact_name) if candidates else None
            if eval_prefix:
                evaluation_summary = self._summarize_evaluation(
                    self._read_json(f"{eval_prefix}/evaluation_lineage.json")
                )
        except Exception:
            evaluation_summary = None

        try:
            explicit_loss = str(getattr(self.args, "loss_path", "") or "").strip()
            loss_path = (
                self._normalize_bucket_path(explicit_loss)
                if explicit_loss
                else f"{run_prefix}/analysis/loss/loss_lineage.json"
            )
            if explicit_loss and not loss_path.endswith("/loss_lineage.json"):
                loss_path = f"{loss_path.rstrip('/')}/loss_lineage.json"
            loss_summary = self._summarize_loss(
                self._read_json(loss_path)
            )
        except Exception:
            loss_summary = None

        payload = {
            "run_prefix": run_prefix,
            "bucket": self._bucket_for_path(run_prefix),
            "artifacts": {
                "training_lineage": f"{run_prefix}/training_lineage.json" if training_summary else None,
                "evaluation_prefix": eval_prefix,
                "loss_lineage": loss_path if loss_summary else None,
            },
            "training": training_summary,
            "evaluation": evaluation_summary,
            "loss": loss_summary,
            "signals": signals + self._derive_signals(training_summary, evaluation_summary, loss_summary),
        }
        if self.json_mode:
            self.output(payload)
            return 0
        print(json.dumps(payload, indent=2))
        return 0

    def _handle_read(self) -> int:
        path = self._require_path()
        bucket_id = self._bucket_for_path(path)
        output = read_artifact(
            path,
            bucket_id=bucket_id,
            tail=getattr(self.args, "tail", None),
            jsonl_latest=bool(getattr(self.args, "jsonl_latest", False)),
            pretty=bool(getattr(self.args, "pretty", False)),
        )
        if self.json_mode:
            self.output(
                {
                    "path": path,
                    "bucket": bucket_id,
                    "content": output,
                }
            )
        elif output:
            print(output, end="" if output.endswith("\n") else "\n")
        return 0

    def _handle_list(self) -> int:
        path = self._require_path()
        bucket_id = self._bucket_for_path(path)
        entries = list_artifacts(
            path,
            bucket_id=bucket_id,
            recursive=bool(getattr(self.args, "recursive", False)),
            files_only=bool(getattr(self.args, "files_only", False)),
            dirs_only=bool(getattr(self.args, "dirs_only", False)),
        )
        limit = getattr(self.args, "limit", None)
        if limit:
            entries = entries[:limit]
        if self.json_mode:
            self.output(
                {
                    "path": path,
                    "bucket": bucket_id,
                    "entries": entries,
                }
            )
            return 0
        for entry in entries:
            size = entry.get("size")
            suffix = f" ({size} bytes)" if isinstance(size, int) else ""
            print(f"{entry['type']}: {entry['path']}{suffix}")
        return 0

    def _handle_pull(self) -> int:
        path = self._require_path()
        bucket_id = self._bucket_for_path(path)
        destination = str(getattr(self.args, "dest", "") or ".").strip() or "."
        local_path = pull_artifacts(path, bucket_id=bucket_id, destination=destination)
        payload = {
            "path": path,
            "bucket": bucket_id,
            "destination": str(Path(destination).resolve()),
            "local_path": str(local_path),
        }
        if self.json_mode:
            self.output(payload)
        else:
            print(f"Pulled to {local_path}")
        return 0

    def _handle_push(self) -> int:
        path = self._require_path()
        source = Path(path)
        if not source.exists():
            raise CloudProviderError(f"Local source path not found: {path}")
        bucket_id = self._require_bucket_id()
        destination = str(getattr(self.args, "dest", "") or "").strip() or None
        remote_uri = push_artifacts(path, bucket_id=bucket_id, destination=destination)
        payload = {
            "path": str(source.resolve()),
            "bucket": bucket_id,
            "destination": destination,
            "remote_uri": remote_uri,
        }
        if self.json_mode:
            self.output(payload)
        else:
            print(f"Pushed to {remote_uri}")
        return 0

    def handle(self) -> int:
        try:
            subcommand = str(getattr(self.args, "subcommand", "") or "").strip().lower()
            if subcommand == "read":
                return self._handle_read()
            if subcommand == "list":
                return self._handle_list()
            if subcommand == "pull":
                return self._handle_pull()
            if subcommand == "push":
                return self._handle_push()
            if subcommand == "analyze":
                return self._handle_analyze()
            raise CloudProviderError("Bucket command requires subcommand 'read', 'list', 'pull', 'push', or 'analyze'.")
        except Exception as exc:
            self.output_error(str(exc), code="BUCKET_ERROR")
            return 1
