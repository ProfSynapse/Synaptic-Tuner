"""
Run a loss computation step inside a Hugging Face Job against a bucketed training run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.utilities.env import get_hf_token

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-example losses for a bucketed training run on HF Jobs."
    )
    parser.add_argument("--bucket-id", required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--dataset-path", help="Relative path to training dataset inside repo or absolute if local")
    parser.add_argument("--dataset-name", help="Hugging Face dataset repo to load when --dataset-path is not provided")
    parser.add_argument("--dataset-file", help="Dataset file inside the Hugging Face dataset repo")
    parser.add_argument("--results-prefix", help="Bucket prefix where loss artifacts should be uploaded")
    parser.add_argument("--output-root", default="/workspace/loss_outputs")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--no-completion-only", action="store_true", help="Disable completion-only masking")
    return parser.parse_args()

def _sync_bucket(source_path: str, destination_path: str, token: Optional[str]) -> None:
    helper_python = os.environ.get("HF_BUCKET_SYNC_PYTHON", "").strip() or sys.executable
    helper_pythonpath = os.environ.get("HF_BUCKET_SYNC_PYTHONPATH", "").strip()
    env = dict(os.environ)
    if token:
        env["HF_TOKEN"] = token
        env["HF_API_KEY"] = token
    else:
        env.pop("HF_TOKEN", None)
        env.pop("HF_API_KEY", None)
    if helper_pythonpath:
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{helper_pythonpath}:{existing_pythonpath}" if existing_pythonpath else helper_pythonpath

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
    _sync_bucket(f"hf://buckets/{bucket_id}/{remote_prefix.strip('/')}", str(local_dir), token)


def _materialize_hf_dataset(dataset_name: str, dataset_file: str, output_root: Path) -> Path:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, data_files=dataset_file, split="train")
    dataset_path = output_root / "dataset.jsonl"
    with dataset_path.open("w", encoding="utf-8") as handle:
        for row in dataset:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return dataset_path

def main() -> int:
    args = _parse_args()
    hf_token = get_hf_token()
    if not hf_token:
        raise RuntimeError("HF_TOKEN or HF_API_KEY is required inside the cloud evaluation job.")

    output_root = Path(args.output_root)
    model_dir = output_root / "model"
    results_dir = output_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model from bucket: {args.run_prefix}/final_model")
    _sync_from_bucket(args.bucket_id, f"{args.run_prefix}/final_model", model_dir, hf_token)

    if args.dataset_path:
        dataset_path = Path(_REPO_ROOT) / args.dataset_path
        if not dataset_path.exists():
            dataset_path = Path(args.dataset_path)
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")
    elif args.dataset_name and args.dataset_file:
        dataset_path = _materialize_hf_dataset(args.dataset_name, args.dataset_file, output_root)
    else:
        raise ValueError("Provide either --dataset-path or both --dataset-name and --dataset-file.")

    print("Loading model and tokenizer...")
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_dir),
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)

    from shared.experiment_tracking.per_example_loss import compute_per_example_losses, save_losses
    print("Computing per-example losses...")
    losses = compute_per_example_losses(
        model=model,
        tokenizer=tokenizer,
        dataset_path=str(dataset_path),
        max_seq_length=args.max_seq_length,
        completion_only=not args.no_completion_only,
    )

    losses_path = results_dir / "per_example_losses.jsonl"
    print(f"Saving losses to {losses_path}")
    save_losses(losses, losses_path)

    destination_prefix = (args.results_prefix or args.run_prefix).strip("/")
    print(f"Syncing artifacts to bucket: {destination_prefix}")
    _sync_bucket(
        str(results_dir),
        f"hf://buckets/{args.bucket_id}/{destination_prefix}",
        token=hf_token,
    )
    print("Done computing losses in cloud.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
