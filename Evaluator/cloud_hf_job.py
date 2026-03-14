"""
Run a vLLM-backed evaluation inside a Hugging Face Job against a bucketed training run.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from huggingface_hub import sync_bucket

from Evaluator.cli import main as evaluator_main
from Evaluator.vllm_setup import start_vllm_server, stop_vllm_server
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


def main() -> int:
    args = _parse_args()
    hf_token = get_hf_token()
    if not hf_token:
        raise RuntimeError("HF_TOKEN or HF_API_KEY is required inside the cloud evaluation job.")

    output_root = Path(args.output_root)
    model_dir = output_root / "model"
    results_dir = output_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

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

    try:
        exit_code = evaluator_main(cli_args)
    finally:
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
