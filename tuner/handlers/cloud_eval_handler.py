"""
Cloud evaluation workflow handler.

Launches a Hugging Face Job that downloads a bucketed training run,
starts vLLM remotely, and runs the evaluator against it.
"""

from __future__ import annotations

import json
import logging
import shutil
import shlex
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from shared.cloud_artifacts import normalize_hf_bucket_id
from shared.utilities.env import get_hf_token
from Evaluator.config_loader import ConfigLoader
from tuner.handlers.base import BaseHandler
from tuner.handlers.cloud_eval_dashboard import (
    CloudEvalDashboardWatcher,
    CloudEvalProviderAdapter,
    CloudEvalWatchResult,
    can_render_cloud_eval_dashboard,
)
from tuner.ui import (
    BOX,
    confirm,
    print_config,
    print_error,
    print_header,
    print_info,
    print_menu,
)
from tuner.backends.training.cloud.base_cloud import load_cloud_config, load_project_deps, resolve_repo_source
from tuner.core.exceptions import CloudProviderError

logger = logging.getLogger(__name__)

_HF_EVAL_OVERLAY = "/tmp/hf-eval-site"
_HF_EVAL_PIP_PACKAGES = [
    "-r",
    "Evaluator/requirements.txt",
    "huggingface_hub>=1.5.0",
    "hf_transfer",
]


class HFJobsCloudEvalAdapter(CloudEvalProviderAdapter):
    """HF Jobs implementation of the cloud evaluation dashboard adapter."""

    def __init__(self, huggingface_hub, *, bucket_id: str, eval_prefix: str, job_id: str, token: Optional[str]) -> None:
        self.huggingface_hub = huggingface_hub
        self.bucket_id = bucket_id
        self.eval_prefix = eval_prefix.strip("/")
        self.job_id = job_id
        self.token = token

    def fetch_status(self) -> str:
        job_info = self.huggingface_hub.inspect_job(job_id=self.job_id)
        status_obj = getattr(job_info, "status", None)
        return status_obj.stage if status_obj and hasattr(status_obj, "stage") else str(status_obj or "UNKNOWN")

    def sync_progress(self, local_dir: Path) -> None:
        from huggingface_hub import sync_bucket

        sync_bucket(
            f"hf://buckets/{self.bucket_id}/{self.eval_prefix}/logs",
            str(local_dir),
            token=self.token,
        )

    def sync_results(self, local_dir: Path) -> None:
        from huggingface_hub import sync_bucket

        sync_bucket(
            f"hf://buckets/{self.bucket_id}/{self.eval_prefix}",
            str(local_dir),
            token=self.token,
        )


class CloudEvalHandler(BaseHandler):
    """Submit a cloud evaluation run to Hugging Face Jobs."""

    def __init__(self, args: Optional[Namespace] = None):
        super().__init__(args=args)

    @property
    def name(self) -> str:
        return "cloud-eval"

    def can_handle_direct_mode(self) -> bool:
        return True

    def _cloud_config_path(self) -> Path:
        return self.repo_root / "Trainers" / "cloud" / "cloud_config.yaml"

    def _load_cloud_config(self) -> dict:
        return load_cloud_config(self._cloud_config_path())

    def _hf_jobs_settings(self) -> dict:
        return self._load_cloud_config().get("hf_jobs", {})

    def _validate_environment(self):
        hf_token = get_hf_token()
        if not hf_token:
            raise CloudProviderError("HF_TOKEN not set. Required for cloud evaluation.")

        try:
            import huggingface_hub
        except ImportError as exc:
            raise CloudProviderError(
                "huggingface_hub not installed. Install with: pip install -r requirements-cloud.txt"
            ) from exc

        missing = [name for name in ("run_job", "create_bucket", "HfApi") if not hasattr(huggingface_hub, name)]
        if missing:
            version = getattr(huggingface_hub, "__version__", "unknown")
            raise CloudProviderError(
                f"huggingface_hub {version} is missing required APIs for cloud evaluation: {', '.join(missing)}"
            )
        return huggingface_hub

    def _resolve_bucket_id(self, huggingface_hub, bucket_id: str) -> str:
        requested_bucket_id = normalize_hf_bucket_id(bucket_id)
        hf_token = get_hf_token()
        try:
            try:
                bucket_info = huggingface_hub.create_bucket(
                    requested_bucket_id,
                    exist_ok=True,
                    private=True,
                    token=hf_token,
                )
            except TypeError:
                bucket_info = huggingface_hub.create_bucket(
                    requested_bucket_id,
                    exist_ok=True,
                    token=hf_token,
                )
        except Exception as exc:
            raise CloudProviderError(f"Failed to resolve HF bucket '{requested_bucket_id}': {exc}") from exc

        resolved = getattr(bucket_info, "bucket_id", None) or getattr(bucket_info, "id", None) or requested_bucket_id
        return normalize_hf_bucket_id(str(resolved))

    def _list_remote_runs(self, huggingface_hub, bucket_id: str, method: Optional[str]) -> List[Dict[str, str]]:
        api = huggingface_hub.HfApi(token=get_hf_token())
        methods = [method] if method else ["sft", "kto"]
        runs: List[Dict[str, str]] = []

        for current_method in methods:
            for item in api.list_bucket_tree(
                bucket_id,
                prefix=f"runs/hf_jobs/{current_method}",
                recursive=False,
                token=get_hf_token(),
            ):
                item_type = getattr(item, "type", None)
                item_path = getattr(item, "path", "")
                if item_type != "directory":
                    continue
                if item_path.count("/") != 3:
                    continue
                runs.append(
                    {
                        "method": current_method,
                        "slug": item_path.split("/")[-1],
                        "prefix": item_path,
                    }
                )

        runs.sort(key=lambda run: run["slug"], reverse=True)
        return runs

    def _select_run(self, runs: List[Dict[str, str]], requested_run: Optional[str]) -> Dict[str, str]:
        if not runs:
            raise CloudProviderError("No cloud training runs found in the configured HF bucket.")

        if requested_run in (None, ""):
            options = [
                (
                    run["prefix"],
                    f"{BOX['bullet']} {run['method'].upper()} {run['slug']}",
                )
                for run in runs[:20]
            ]
            choice = print_menu(options, "Select cloud training run to evaluate:")
            if not choice:
                raise CloudProviderError("Cloud evaluation cancelled.")
            for run in runs:
                if run["prefix"] == choice:
                    return run
            raise CloudProviderError(f"Unknown run selection: {choice}")

        if requested_run == "latest":
            return runs[0]

        normalized = requested_run.strip("/")
        for run in runs:
            if run["slug"] == normalized or run["prefix"] == normalized:
                return run

        raise CloudProviderError(f"Cloud run not found: {requested_run}")

    def _build_eval_prefix(self, run_prefix: str, timestamp: str) -> str:
        return f"{run_prefix.strip('/')}/evaluations/vllm/{timestamp}"

    def _resolve_display_scenarios(
        self,
        *,
        preset: Optional[str],
        scenarios: Optional[List[str]],
    ) -> List[str]:
        if scenarios:
            return list(scenarios)
        if not preset:
            return []

        config_dir = self.repo_root / "Evaluator" / "config"
        try:
            run_config = ConfigLoader(config_dir).load_eval_run(preset)
        except Exception:
            return []
        return list(run_config.scenarios or [])

    def _build_eval_command(
        self,
        *,
        bucket_id: str,
        run_prefix: str,
        eval_prefix: str,
        preset: Optional[str],
        scenarios: Optional[List[str]],
        tags: Optional[str],
        upload_to_hf: Optional[str],
        update_model_card: bool,
    ) -> str:
        repo_source = resolve_repo_source(self.repo_root)
        cloud_config_path = self._cloud_config_path()
        project_deps = load_project_deps(cloud_config_path)
        quoted_project_deps = " ".join(shlex.quote(dep) for dep in project_deps)
        quoted_eval_deps = " ".join(shlex.quote(dep) for dep in _HF_EVAL_PIP_PACKAGES)
        quoted_repo_url = shlex.quote(repo_source.url)
        quoted_branch = shlex.quote(repo_source.branch)
        quoted_commit = shlex.quote(repo_source.commit)

        parts = [
            f"git clone --branch {quoted_branch} --depth 1 {quoted_repo_url} /workspace/repo",
            f"cd /workspace/repo && git fetch --depth 1 origin {quoted_commit} && git checkout {quoted_commit}",
            f"cd /workspace/repo && python -m pip install --upgrade {quoted_project_deps}",
            f"mkdir -p {_HF_EVAL_OVERLAY}",
            f"cd /workspace/repo && python -m pip install --upgrade --target {_HF_EVAL_OVERLAY} {quoted_eval_deps}",
            "export HF_BUCKET_SYNC_PYTHON=$(command -v python)",
            f"export HF_BUCKET_SYNC_PYTHONPATH={_HF_EVAL_OVERLAY}",
            "export HF_HUB_ENABLE_HF_TRANSFER=1",
        ]

        eval_cmd = [
            "python",
            "-m",
            "Evaluator.cloud_hf_job",
            "--bucket-id",
            bucket_id,
            "--run-prefix",
            run_prefix,
            "--eval-prefix",
            eval_prefix,
            "--config-dir",
            "Evaluator/config",
        ]
        if preset:
            eval_cmd.extend(["--preset", preset])
        if scenarios:
            for scenario in scenarios:
                eval_cmd.extend(["--scenario", scenario])
        if tags:
            eval_cmd.extend(["--tags", tags])
        if upload_to_hf:
            eval_cmd.extend(["--upload-to-hf", upload_to_hf])
        if update_model_card:
            eval_cmd.append("--update-model-card")

        parts.append("cd /workspace/repo && " + " ".join(shlex.quote(arg) for arg in eval_cmd))
        return " && ".join(parts)

    def _poll_job(self, huggingface_hub, job_id: str, timeout_hours: float) -> int:
        timeout_seconds = int(timeout_hours * 3600)
        elapsed = 0
        poll_interval = 30
        last_log_offset = 0

        while elapsed < timeout_seconds:
            try:
                job_info = huggingface_hub.inspect_job(job_id=job_id)
                status_obj = getattr(job_info, "status", None)
                status = status_obj.stage if status_obj and hasattr(status_obj, "stage") else str(status_obj or "UNKNOWN")
            except Exception as exc:
                logger.warning("Status check failed: %s", exc)
                status = "UNKNOWN"

            try:
                logs = huggingface_hub.fetch_job_logs(job_id=job_id) or ""
                if len(logs) > last_log_offset:
                    print(logs[last_log_offset:], end="", flush=True)
                    last_log_offset = len(logs)
            except Exception:
                pass

            if status in ("completed", "COMPLETED"):
                return 0
            if status in ("error", "ERROR", "failed", "FAILED", "cancelled", "CANCELLED"):
                print()
                print(f"  Evaluation job {job_id} failed with status: {status}")
                return 1

            import time

            time.sleep(poll_interval)
            elapsed += poll_interval

        print()
        print(f"  Evaluation job {job_id} failed: Timeout exceeded")
        return 1

    def _should_use_live_dashboard(self) -> bool:
        return can_render_cloud_eval_dashboard() and sys.stdin.isatty() and sys.stdout.isatty()

    def _watch_job_with_dashboard(
        self,
        *,
        huggingface_hub,
        bucket_id: str,
        eval_prefix: str,
        job_id: str,
        timeout_hours: float,
    ) -> CloudEvalWatchResult:
        watcher = CloudEvalDashboardWatcher(
            HFJobsCloudEvalAdapter(
                huggingface_hub,
                bucket_id=bucket_id,
                eval_prefix=eval_prefix,
                job_id=job_id,
                token=get_hf_token(),
            ),
            title="Cloud Evaluation",
            poll_interval=15,
        )
        return watcher.watch(timeout_seconds=int(timeout_hours * 3600))

    def _download_eval_results(self, bucket_id: str, eval_prefix: str) -> Optional[Path]:
        try:
            from huggingface_hub import sync_bucket
        except ImportError:
            return None

        import tempfile

        local_root = Path(tempfile.mkdtemp(prefix="cloud-eval-results-"))
        try:
            sync_bucket(
                f"hf://buckets/{bucket_id}/{eval_prefix.strip('/')}",
                str(local_root),
                token=get_hf_token(),
            )
        except Exception:
            shutil.rmtree(local_root, ignore_errors=True)
            return None
        return local_root

    def _load_eval_summary(self, results_dir: Optional[Path]) -> Optional[Dict[str, int]]:
        payload = self._load_eval_payload(results_dir)
        if not payload:
            return None

        summary = payload.get("summary")
        return summary if isinstance(summary, dict) else None

    def _load_eval_payload(self, results_dir: Optional[Path]) -> Optional[Dict]:
        if results_dir is None:
            return None

        results_path = Path(results_dir) / "evaluation_results.json"
        if not results_path.exists():
            return None

        payload = json.loads(results_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None

    def _print_eval_summary(self, summary: Optional[Dict[str, int]]) -> None:
        if not summary:
            return

        print()
        print_info(
            "Summary: "
            f"{summary.get('passed', 0)} passed, "
            f"{summary.get('warned', 0)} warned, "
            f"{summary.get('failed', 0)} failed "
            f"out of {summary.get('total', 0)}"
        )

    def _extract_failure_reason(self, record: Dict) -> str:
        error = record.get("error")
        if error:
            return str(error)

        for section_name in ("validator", "environment", "behavior"):
            section = record.get(section_name) or {}
            for issue in section.get("issues", []) or []:
                if issue.get("message"):
                    return str(issue["message"])

        judge = record.get("judge") or {}
        judge_result = judge.get("judge_result") or {}
        for score in judge_result.get("scores", []) or []:
            if score.get("feedback"):
                return str(score["feedback"])

        return "No failure reason captured."

    def _print_eval_failure_preview(self, payload: Optional[Dict]) -> None:
        if not payload:
            return

        failed_records = [
            record for record in (payload.get("records") or [])
            if not bool(record.get("passed", False))
        ]
        if not failed_records:
            return

        print()
        print_info(f"Failure preview: showing {min(len(failed_records), 3)} of {len(failed_records)} failed/warned cases")
        for index, record in enumerate(failed_records[:3], start=1):
            case_id = record.get("case_id") or f"case-{index}"
            reason = self._extract_failure_reason(record)
            response_text = str(record.get("response_text") or "").strip()
            if len(response_text) > 240:
                response_text = response_text[:240] + "..."
            print_config(
                {
                    "Case": case_id,
                    "Why": reason,
                    "LLM Response": response_text or "(empty)",
                },
                f"Failure {index}",
            )

    def _auto_confirm(self) -> bool:
        return bool(getattr(self.args, "auto_confirm", False))

    def handle(self) -> int:
        if self.json_mode:
            try:
                huggingface_hub = self._validate_environment()
                hf_settings = self._hf_jobs_settings()
                bucket_id = self._resolve_bucket_id(
                    huggingface_hub,
                    getattr(self.args, "bucket", None) or hf_settings.get("artifact_identifier", ""),
                )
                runs = self._list_remote_runs(
                    huggingface_hub,
                    bucket_id=bucket_id,
                    method=getattr(self.args, "method", None),
                )
                self.output(
                    {
                        "provider": "hf_jobs",
                        "bucket_id": bucket_id,
                        "runs": runs[:20],
                    }
                )
                return 0
            except Exception as exc:
                self.output_error(str(exc), code="CLOUD_EVAL_ERROR")
                return 1

        print_header("CLOUD EVALUATION", "Run vLLM evaluation on Hugging Face Jobs")

        try:
            huggingface_hub = self._validate_environment()
            hf_settings = self._hf_jobs_settings()
            bucket_id = self._resolve_bucket_id(
                huggingface_hub,
                getattr(self.args, "bucket", None) or hf_settings.get("artifact_identifier", ""),
            )
            runs = self._list_remote_runs(
                huggingface_hub,
                bucket_id=bucket_id,
                method=getattr(self.args, "method", None),
            )
            selected_run = self._select_run(runs, getattr(self.args, "run", None))
        except Exception as exc:
            print_error(str(exc))
            return 1

        preset = getattr(self.args, "preset", None) or ("full" if not getattr(self.args, "scenario", None) else None)
        scenarios = getattr(self.args, "scenario", None)
        display_scenarios = self._resolve_display_scenarios(preset=preset, scenarios=scenarios)
        tags = getattr(self.args, "tags", None)
        upload_to_hf = getattr(self.args, "upload_to_hf", None)
        update_model_card = bool(getattr(self.args, "update_model_card", False))
        flavor = getattr(self.args, "gpu", None) or hf_settings.get("flavor", "a10g-small")
        timeout_hours = float(getattr(self.args, "timeout_hours", None) or 4.0)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        eval_prefix = self._build_eval_prefix(selected_run["prefix"], timestamp)
        command = self._build_eval_command(
            bucket_id=bucket_id,
            run_prefix=selected_run["prefix"],
            eval_prefix=eval_prefix,
            preset=preset,
            scenarios=scenarios,
            tags=tags,
            upload_to_hf=upload_to_hf,
            update_model_card=update_model_card,
        )

        print_config(
            {
                "Provider": "hf_jobs",
                "Run": selected_run["prefix"],
                "Bucket": bucket_id,
                "Backend": "unsloth",
                "Preset": preset or "-",
                "Scenarios": ", ".join(display_scenarios) if display_scenarios else "-",
                "Tags": tags or "-",
                "GPU": flavor,
                "Timeout": f"{timeout_hours:.1f}h",
                "Results": f"hf://buckets/{bucket_id}/{eval_prefix}",
            },
            "Cloud Evaluation Configuration",
        )

        if not self._auto_confirm() and not confirm("Start cloud evaluation with this configuration?"):
            print_info("Cloud evaluation cancelled.")
            return 0

        try:
            hf_token = get_hf_token()
            job = huggingface_hub.run_job(
                image=hf_settings.get("image"),
                command=["bash", "-c", command],
                flavor=flavor,
                timeout=f"{timeout_hours}h",
                secrets=(
                    {
                        "HF_TOKEN": hf_token,
                        "HF_API_KEY": hf_token,
                    }
                    if hf_token
                    else None
                ),
            )
        except Exception as exc:
            print_error(f"Failed to submit HF cloud evaluation: {exc}")
            return 1

        job_id = job.id if hasattr(job, "id") else str(job)
        job_url = getattr(job, "url", None)
        print_info(f"Evaluation job submitted: {job_id}")
        if job_url:
            print_info(f"Monitor at: {job_url}")
        print_info(f"Evaluation artifacts will sync to: hf://buckets/{bucket_id}/{eval_prefix}")
        print()

        results_dir = None
        local_root = None
        failure_status = None
        if self._should_use_live_dashboard():
            watch_result = self._watch_job_with_dashboard(
                huggingface_hub=huggingface_hub,
                bucket_id=bucket_id,
                eval_prefix=eval_prefix,
                job_id=job_id,
                timeout_hours=timeout_hours,
            )
            exit_code = watch_result.exit_code
            results_dir = watch_result.results_dir
            local_root = watch_result.local_root
            failure_status = watch_result.status
        else:
            exit_code = self._poll_job(huggingface_hub, job_id, timeout_hours)
            if exit_code == 0:
                results_dir = self._download_eval_results(bucket_id, eval_prefix)
                local_root = results_dir
            else:
                recovered_results_dir = self._download_eval_results(bucket_id, eval_prefix)
                if self._load_eval_payload(recovered_results_dir):
                    exit_code = 0
                    results_dir = recovered_results_dir
                    local_root = recovered_results_dir

        if exit_code == 0:
            print()
            print_info("Cloud evaluation completed successfully.")
            print_info(f"Results: hf://buckets/{bucket_id}/{eval_prefix}")
            payload = self._load_eval_payload(results_dir)
            self._print_eval_summary(self._load_eval_summary(results_dir))
            self._print_eval_failure_preview(payload)
        elif failure_status:
            print_error(f"Cloud evaluation failed with status: {failure_status}")
        if local_root is not None:
            shutil.rmtree(local_root, ignore_errors=True)
        return exit_code
