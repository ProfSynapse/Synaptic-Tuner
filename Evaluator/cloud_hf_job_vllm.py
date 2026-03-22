"""
Run a cloud evaluation inside a Hugging Face Job against a bucketed training run.

This runtime uses vLLM for the evaluation pass, then optionally computes exact
per-example loss with Transformers in the same job.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Evaluator.cli import main as evaluator_main
from Evaluator.vllm_setup import start_vllm_server, stop_vllm_server
from shared.cloud_eval_progress import EVAL_PROGRESS_LOG_FILENAME
from shared.experiment_tracking.per_example_loss import compute_per_example_losses_parallel
from shared.utilities.env import get_hf_token

from .cloud_hf_job import _PeriodicBucketSyncer, _finalize_cloud_exit_code


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a bucketed training run on HF Jobs using vLLM, with optional exact loss afterward."
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
    parser.add_argument("--vllm-host", default="127.0.0.1")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--vllm-timeout", type=int, default=600)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=0)
    parser.add_argument("--loss-workers", type=int, default=0)
    return parser.parse_args()


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
            "Cloud vLLM evaluation requires a hub-accessible base model."
        )
    return base_model


def _resolve_loss_dataset_path(
    *,
    dataset_path: str | None,
    dataset_name: str | None,
    dataset_file: str | None,
    token: str | None,
) -> Path:
    if dataset_path:
        return Path(dataset_path)
    if dataset_name and dataset_file:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=dataset_name,
            filename=dataset_file,
            repo_type="dataset",
            token=token,
        )
        return Path(downloaded)
    raise ValueError("Loss computation requires --loss-dataset-path or both --loss-dataset-name and --loss-dataset-file.")


def _write_loss_summary(path: Path, *, dataset_path: Path, row_count: int, completion_only: bool, max_seq_length: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_path": str(dataset_path),
        "row_count": row_count,
        "completion_only": completion_only,
        "max_seq_length": max_seq_length,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _compute_exact_loss_outputs(
    *,
    args: argparse.Namespace,
    model_dir: Path,
    results_dir: Path,
    hf_token: str,
    progress_syncer: _PeriodicBucketSyncer,
) -> None:
    analysis_dir = results_dir / "analysis"
    dataset_path = _resolve_loss_dataset_path(
        dataset_path=args.loss_dataset_path,
        dataset_name=args.loss_dataset_name,
        dataset_file=args.loss_dataset_file,
        token=hf_token,
    )

    def _on_aggregate(_aggregate_root: Path) -> None:
        progress_syncer.sync_once()

    losses = compute_per_example_losses_parallel(
        model_dir=model_dir,
        dataset_path=dataset_path,
        output_root=analysis_dir,
        max_seq_length=args.loss_max_seq_length or 2048,
        completion_only=not args.loss_no_completion_only,
        num_workers=(args.loss_workers or None),
        on_aggregate=_on_aggregate,
    )

    feature_rows = [
        {
            "index": item.index,
            "loss": item.loss,
            "num_completion_tokens": item.num_completion_tokens,
            "num_total_tokens": item.num_total_tokens,
            "jsonl_hash": item.jsonl_hash,
        }
        for item in losses
    ]
    feature_jsonl = analysis_dir / "feature_dataset.jsonl"
    feature_csv = analysis_dir / "feature_dataset.csv"
    high_loss_path = analysis_dir / "failure_slices" / "high_loss_examples.jsonl"
    summary_path = analysis_dir / "loss_summary.json"
    feature_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with feature_jsonl.open("w", encoding="utf-8") as handle:
        for row in feature_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with feature_csv.open("w", encoding="utf-8") as handle:
        handle.write("index,loss,num_completion_tokens,num_total_tokens,jsonl_hash\n")
        for row in feature_rows:
            handle.write(
                f"{row['index']},{row['loss']},{row['num_completion_tokens']},{row['num_total_tokens']},{row['jsonl_hash']}\n"
            )
    high_loss_path.parent.mkdir(parents=True, exist_ok=True)
    with high_loss_path.open("w", encoding="utf-8") as handle:
        for row in sorted(feature_rows, key=lambda row: row["loss"], reverse=True)[:25]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_loss_summary(
        summary_path,
        dataset_path=dataset_path,
        row_count=len(feature_rows),
        completion_only=not args.loss_no_completion_only,
        max_seq_length=args.loss_max_seq_length or 2048,
    )
    progress_syncer.sync_once()


def _log_runtime_versions() -> None:
    package_names = ["torch", "transformers", "tokenizers", "huggingface_hub", "vllm", "peft"]
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
    partial_output_json = results_dir / "evaluation_results.partial.json"
    partial_output_md = results_dir / "evaluation_results.partial.md"
    failure_json = results_dir / "evaluation_failure.json"

    _sync_from_bucket(args.bucket_id, f"{args.run_prefix}/final_model", model_dir, hf_token)
    base_model_name = _load_base_model_name(model_dir)

    output_json = results_dir / "evaluation_results.json"
    output_md = results_dir / "evaluation_results.md"
    lineage_json = results_dir / "evaluation_lineage.json"

    progress_syncer = _PeriodicBucketSyncer(
        results_dir,
        args.bucket_id,
        args.eval_prefix,
        hf_token,
        interval_seconds=15,
    )

    server_started = False
    try:
        progress_syncer.start()
        server_started = start_vllm_server(
            model=base_model_name,
            host=args.vllm_host,
            port=args.vllm_port,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            lora_modules={"finetuned": str(model_dir)},
            timeout=args.vllm_timeout,
            show_logs=True,
        )
        if not server_started:
            raise RuntimeError("Failed to start the vLLM server in the cloud eval job.")

        cli_args = [
            "--backend",
            "vllm",
            "--model",
            "finetuned",
            "--host",
            args.vllm_host,
            "--port",
            str(args.vllm_port),
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
            "--progress-jsonl",
            str(progress_log_path),
            "--no-dashboard",
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

        exit_code = evaluator_main(cli_args)
        if args.with_loss:
            stop_vllm_server()
            server_started = False
            _compute_exact_loss_outputs(
                args=args,
                model_dir=model_dir,
                results_dir=results_dir,
                hf_token=hf_token,
                progress_syncer=progress_syncer,
            )
    except Exception as exc:
        traceback.print_exc()
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
        if server_started:
            stop_vllm_server()

    _sync_bucket(
        str(results_dir),
        f"hf://buckets/{args.bucket_id}/{args.eval_prefix.strip('/')}",
        token=hf_token,
    )
    print(f"Evaluation artifacts synced to: hf://buckets/{args.bucket_id}/{args.eval_prefix.strip('/')}")
    final_exit_code = _finalize_cloud_exit_code(exit_code, output_json)
    if final_exit_code == 0 and exit_code != 0:
        print(
            "Evaluation completed with failed cases, but final artifacts were written. "
            "Returning success for cloud job status."
        )
    return final_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
