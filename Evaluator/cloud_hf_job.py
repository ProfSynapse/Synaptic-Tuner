"""
Run a cloud evaluation inside a Hugging Face Job against a bucketed training run.

The current stable path uses direct Unsloth-backed evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Evaluator.cli import main as evaluator_main
from shared.cloud_stage_logging import StageLogger, apply_stage_logging_env, detect_cloud_job_ref
from shared.cloud_eval_progress import EVAL_PROGRESS_LOG_FILENAME
from shared.utilities.env import get_hf_token


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a bucketed training run on HF Jobs using vLLM."
    )
    parser.add_argument("--bucket-id", required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--eval-prefix", required=True)
    parser.add_argument("--config-dir", default="Evaluator/config")
    parser.add_argument("--output-root", default="/workspace/eval_outputs")
    parser.add_argument("--preset")
    parser.add_argument("--scenario", action="append", dest="scenarios")
    parser.add_argument("--tags")
    parser.add_argument("--env-backend", choices=["none", "local", "e2b"], default="none")
    parser.add_argument("--env-template")
    parser.add_argument("--env-tool-schema")
    parser.add_argument("--env-exec-config")
    parser.add_argument("--upload-to-hf")
    parser.add_argument("--update-model-card", action="store_true")
    parser.add_argument("--with-loss", action="store_true")
    parser.add_argument("--loss-dataset-path")
    parser.add_argument("--loss-dataset-name")
    parser.add_argument("--loss-dataset-file")
    parser.add_argument("--loss-max-seq-length", type=int, default=2048)
    parser.add_argument("--loss-no-completion-only", action="store_true")
    return parser.parse_args()


def _sync_from_bucket(bucket_id: str, remote_prefix: str, local_dir: Path, token: Optional[str]) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    _sync_bucket(
        f"hf://buckets/{bucket_id}/{remote_prefix.strip('/')}",
        str(local_dir),
        token=token,
    )


def _load_base_model_name(final_model_dir: Path) -> str:
    adapter_config_path = final_model_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        raise RuntimeError(f"adapter_config.json not found in {final_model_dir}")

    payload = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    base_model = str(payload.get("base_model_name_or_path", "")).strip()
    if not base_model:
        raise RuntimeError("adapter_config.json is missing base_model_name_or_path")
    if base_model.startswith("/") or base_model.startswith("."):
        raise RuntimeError(
            "base_model_name_or_path points to a local path. "
            "Cloud vLLM evaluation currently requires a hub-accessible base model."
        )
    return base_model


def _sync_bucket(source_path: str, destination_path: str, token: Optional[str]) -> None:
    helper_python = os.environ.get("HF_BUCKET_SYNC_PYTHON", "").strip() or sys.executable
    helper_pythonpath = os.environ.get("HF_BUCKET_SYNC_PYTHONPATH", "").strip()
    env = dict(os.environ)
    normalized_token = token.strip() if token else ""
    if normalized_token:
        env["HF_TOKEN"] = normalized_token
        env["HF_API_KEY"] = normalized_token
    else:
        env.pop("HF_TOKEN", None)
        env.pop("HF_API_KEY", None)
    if helper_pythonpath:
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{helper_pythonpath}:{existing_pythonpath}"
            if existing_pythonpath
            else helper_pythonpath
        )

    subprocess.run(
        [
            helper_python,
            str(_REPO_ROOT / "shared" / "hf_bucket_sync_helper.py"),
            source_path,
            destination_path,
        ],
        check=True,
        env=env,
    )


def _finalize_cloud_exit_code(exit_code: int, output_json: Path) -> int:
    """Treat completed evaluations as successful cloud jobs even if test cases failed.

    The evaluator CLI uses non-zero exit codes to signal quality failures
    (for example, failing test cases). For cloud orchestration we only want
    runtime/infrastructure failures to mark the HF job as ERROR. If the final
    structured results exist, the evaluation completed and the cloud job should
    be considered successful.
    """
    if exit_code == 0:
        return 0
    return 0 if output_json.exists() else exit_code


class _PeriodicBucketSyncer:
    """Best-effort periodic sync for incremental evaluation progress."""

    def __init__(
        self,
        local_dir: Path,
        bucket_id: str,
        remote_prefix: str,
        token: str,
        interval_seconds: int = 15,
        stage_logger: Optional[StageLogger] = None,
    ):
        self.local_dir = Path(local_dir)
        self.bucket_id = bucket_id
        self.remote_prefix = remote_prefix.strip("/")
        self.token = token
        self.interval_seconds = max(5, int(interval_seconds))
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.stage_logger = stage_logger

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=self.interval_seconds + 5)

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self.sync_once()

    def sync_once(self) -> None:
        if not self.local_dir.exists():
            return
        try:
            _sync_bucket(
                str(self.local_dir),
                f"hf://buckets/{self.bucket_id}/{self.remote_prefix}",
                token=self.token,
            )
            if self.stage_logger is not None:
                self.stage_logger.emit_sync(path=f"hf://buckets/{self.bucket_id}/{self.remote_prefix}")
        except Exception as exc:
            print(f"Progress sync warning: {exc}")
            if self.stage_logger is not None:
                self.stage_logger.emit(
                    "artifacts_synced",
                    message=f"Progress sync warning: {exc}",
                    details={
                        "last_sync_path": f"hf://buckets/{self.bucket_id}/{self.remote_prefix}",
                        "sync_warning": str(exc),
                    },
                )


def _log_runtime_versions() -> None:
    """Log the key eval runtime package versions for cloud debugging."""
    package_names = ["torch", "unsloth", "transformers", "tokenizers", "huggingface_hub"]
    versions = []
    for package_name in package_names:
        try:
            module = __import__(package_name)
            versions.append(f"{package_name}={getattr(module, '__version__', 'unknown')}")
        except Exception as exc:
            versions.append(f"{package_name}=unavailable({exc})")
    print("Eval runtime versions: " + ", ".join(versions))


def main() -> int:
    args = _parse_args()
    hf_token = get_hf_token()
    if not hf_token:
        raise RuntimeError("HF_TOKEN or HF_API_KEY is required inside the cloud evaluation job.")

    _log_runtime_versions()

    output_root = Path(args.output_root)
    model_dir = output_root / "model"
    results_dir = output_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    progress_dir = results_dir / "logs"
    progress_log_path = progress_dir / EVAL_PROGRESS_LOG_FILENAME
    stage_logger = StageLogger(
        progress_dir,
        stage="evaluation",
        provider="hf_jobs",
        job_ref=detect_cloud_job_ref(),
        run_prefix=args.run_prefix,
        bucket_id=args.bucket_id,
    )
    apply_stage_logging_env(
        stage="evaluation",
        provider="hf_jobs",
        run_prefix=args.run_prefix,
        job_ref=stage_logger.job_ref,
        bucket_id=args.bucket_id,
    )
    partial_output_json = results_dir / "evaluation_results.partial.json"
    partial_output_md = results_dir / "evaluation_results.partial.md"
    failure_json = results_dir / "evaluation_failure.json"
    analysis_dir = results_dir / "analysis"

    # Download only the adapter payload needed for direct Unsloth LoRA evaluation.
    stage_logger.emit("bootstrap_started", message="Cloud eval bootstrap started")
    _sync_from_bucket(args.bucket_id, f"{args.run_prefix}/final_model", model_dir, hf_token)
    stage_logger.emit(
        "artifacts_downloaded",
        message="Downloaded final model artifacts",
        details={"last_sync_path": f"hf://buckets/{args.bucket_id}/{args.run_prefix.strip('/')}/final_model"},
    )

    output_json = results_dir / "evaluation_results.json"
    output_md = results_dir / "evaluation_results.md"
    lineage_json = results_dir / "evaluation_lineage.json"

    cli_args = [
        "--backend",
        "unsloth",
        "--model",
        str(model_dir),
        "--config-dir",
        args.config_dir,
        "--output",
        str(output_json),
        "--markdown",
        str(output_md),
        "--lineage",
        str(lineage_json),
        "--partial-output-json",
        str(partial_output_json),
        "--partial-markdown",
        str(partial_output_md),
        "--failure-json",
        str(failure_json),
    ]

    if args.preset:
        cli_args.extend(["--preset", args.preset])
    if args.scenarios:
        for scenario in args.scenarios:
            cli_args.extend(["--scenario", scenario])
    if args.tags:
        cli_args.extend(["--tags", args.tags])
    if args.env_backend and args.env_backend != "none":
        cli_args.extend(["--env-backend", args.env_backend])
    if args.env_template:
        cli_args.extend(["--env-template", args.env_template])
    if args.env_tool_schema:
        cli_args.extend(["--env-tool-schema", args.env_tool_schema])
    if args.env_exec_config:
        cli_args.extend(["--env-exec-config", args.env_exec_config])
    if args.upload_to_hf:
        cli_args.extend(["--upload-to-hf", args.upload_to_hf, "--hf-token", hf_token])
        if args.update_model_card:
            cli_args.append("--update-model-card")
    if args.with_loss:
        cli_args.append("--with-loss")
        if args.loss_dataset_path:
            cli_args.extend(["--loss-dataset-path", args.loss_dataset_path])
        if args.loss_dataset_name:
            cli_args.extend(["--loss-dataset-name", args.loss_dataset_name])
        if args.loss_dataset_file:
            cli_args.extend(["--loss-dataset-file", args.loss_dataset_file])
        cli_args.extend(["--loss-output-jsonl", str(analysis_dir / "per_example_losses.jsonl")])
        cli_args.extend(["--loss-feature-jsonl", str(analysis_dir / "feature_dataset.jsonl")])
        cli_args.extend(["--loss-feature-csv", str(analysis_dir / "feature_dataset.csv")])
        cli_args.extend(["--loss-high-loss-jsonl", str(analysis_dir / "failure_slices" / "high_loss_examples.jsonl")])
        cli_args.extend(["--loss-summary-json", str(analysis_dir / "loss_summary.json")])
        if args.loss_max_seq_length:
            cli_args.extend(["--loss-max-seq-length", str(args.loss_max_seq_length)])
        if args.loss_no_completion_only:
            cli_args.append("--loss-no-completion-only")
    cli_args.extend(["--progress-jsonl", str(progress_log_path), "--no-dashboard"])

    progress_syncer = _PeriodicBucketSyncer(
        results_dir,
        args.bucket_id,
        args.eval_prefix,
        hf_token,
        interval_seconds=15,
        stage_logger=stage_logger,
    )

    try:
        stage_logger.emit("runtime_ready", message="Evaluation runtime ready", details={"backend": "unsloth"})
        progress_syncer.start()
        exit_code = evaluator_main(cli_args)
    except Exception as exc:
        traceback.print_exc()
        stage_logger.emit_failure(exc, message="Evaluation job failed")
        failure_payload = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        failure_json.write_text(json.dumps(failure_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        progress_syncer.sync_once()
        _sync_bucket(
            str(results_dir),
            f"hf://buckets/{args.bucket_id}/{args.eval_prefix.strip('/')}",
            token=hf_token,
        )
        final_exit_code = _finalize_cloud_exit_code(1, output_json)
        if final_exit_code == 0:
            print(
                "Evaluation artifacts were written before optional post-processing failed. "
                "Returning success for cloud job status."
            )
        return final_exit_code
    finally:
        progress_syncer.stop()
        progress_syncer.sync_once()

    _sync_bucket(
        str(results_dir),
        f"hf://buckets/{args.bucket_id}/{args.eval_prefix.strip('/')}",
        token=hf_token,
    )
    final_exit_code = _finalize_cloud_exit_code(exit_code, output_json)
    stage_logger.emit_sync(path=f"hf://buckets/{args.bucket_id}/{args.eval_prefix.strip('/')}")
    if final_exit_code == 0:
        stage_logger.emit("completed", status="completed", message="Evaluation artifacts written successfully")
    print(f"Evaluation artifacts synced to: hf://buckets/{args.bucket_id}/{args.eval_prefix.strip('/')}")
    if final_exit_code == 0 and exit_code != 0:
        print(
            "Evaluation completed with failed cases, but final artifacts were written. "
            "Returning success for cloud job status."
        )
    return final_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
