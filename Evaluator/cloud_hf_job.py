"""
Run a vLLM-backed evaluation inside a Hugging Face Job against a bucketed training run.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path
from typing import Optional

from huggingface_hub import sync_bucket

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Evaluator.cli import main as evaluator_main
from Evaluator.vllm_setup import start_vllm_server, stop_vllm_server
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
    parser.add_argument("--upload-to-hf")
    parser.add_argument("--update-model-card", action="store_true")
    return parser.parse_args()


def _sync_from_bucket(bucket_id: str, remote_prefix: str, local_dir: Path, token: Optional[str]) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    sync_bucket(
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


class _PeriodicBucketSyncer:
    """Best-effort periodic sync for incremental evaluation progress."""

    def __init__(self, local_dir: Path, bucket_id: str, remote_prefix: str, token: str, interval_seconds: int = 15):
        self.local_dir = Path(local_dir)
        self.bucket_id = bucket_id
        self.remote_prefix = remote_prefix.strip("/")
        self.token = token
        self.interval_seconds = max(5, int(interval_seconds))
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

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
            sync_bucket(
                str(self.local_dir),
                f"hf://buckets/{self.bucket_id}/{self.remote_prefix}",
                token=self.token,
            )
        except Exception as exc:
            print(f"Progress sync warning: {exc}")


def _log_runtime_versions() -> None:
    """Log the key eval runtime package versions for cloud debugging."""
    package_names = ["torch", "vllm", "transformers", "tokenizers", "huggingface_hub"]
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

    # Download only the adapter payload needed for vLLM LoRA evaluation.
    _sync_from_bucket(args.bucket_id, f"{args.run_prefix}/final_model", model_dir, hf_token)
    base_model = _load_base_model_name(model_dir)

    started = start_vllm_server(
        model=base_model,
        host="127.0.0.1",
        port=8000,
        lora_modules={"finetuned": str(model_dir)},
        timeout=600,
        show_logs=True,
    )
    if not started:
        raise RuntimeError("Failed to start vLLM server inside the HF evaluation job.")

    output_json = results_dir / "evaluation_results.json"
    output_md = results_dir / "evaluation_results.md"
    lineage_json = results_dir / "evaluation_lineage.json"

    cli_args = [
        "--backend",
        "vllm",
        "--model",
        "finetuned",
        "--config-dir",
        args.config_dir,
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--output",
        str(output_json),
        "--markdown",
        str(output_md),
        "--lineage",
        str(lineage_json),
    ]

    if args.preset:
        cli_args.extend(["--preset", args.preset])
    if args.scenarios:
        for scenario in args.scenarios:
            cli_args.extend(["--scenario", scenario])
    if args.tags:
        cli_args.extend(["--tags", args.tags])
    if args.upload_to_hf:
        cli_args.extend(["--upload-to-hf", args.upload_to_hf, "--hf-token", hf_token])
        if args.update_model_card:
            cli_args.append("--update-model-card")
    cli_args.extend(["--progress-jsonl", str(progress_log_path), "--no-dashboard"])

    progress_syncer = _PeriodicBucketSyncer(
        progress_dir,
        args.bucket_id,
        f"{args.eval_prefix}/logs",
        hf_token,
        interval_seconds=15,
    )

    try:
        progress_syncer.start()
        exit_code = evaluator_main(cli_args)
    finally:
        progress_syncer.stop()
        progress_syncer.sync_once()
        stop_vllm_server()

    sync_bucket(
        str(results_dir),
        f"hf://buckets/{args.bucket_id}/{args.eval_prefix.strip('/')}",
        token=hf_token,
    )
    print(f"Evaluation artifacts synced to: hf://buckets/{args.bucket_id}/{args.eval_prefix.strip('/')}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
