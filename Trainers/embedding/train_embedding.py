#!/usr/bin/env python3
"""
Embedding training entry point (sentence-transformers + optional Unsloth fast path).

Location: Trainers/embedding/train_embedding.py
Purpose:  Train an embedding model (bi-encoder) from a registry base + a local
          triplet/pairs JSONL, using the modern sentence-transformers
          SentenceTransformerTrainer. Mirrors the Trainers/sft/train_sft.py
          bootstrap shape (sys.path + init_trainer_env before torch/unsloth) and
          the canonical run-dir layout. The training loop is loader-agnostic: it
          consumes whatever LoadedEmbeddingModel.load_embedding_model returns
          (fast or fallback) identically.
Used by:  The `embedding` training method (CLI/recipe), wired by WU-C.

Contract: docs/architecture/embedding-reranker-phase1/01_CONTRACTS.md §6.

Usage:
    python train_embedding.py
    python train_embedding.py --config configs/config.yaml --max-steps 10
    python train_embedding.py --registry-name qwen3-embedding-0.6b --adapter-mode lora
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add this trainer's src + the repo root to sys.path BEFORE the env bootstrap /
# torch / unsloth imports (mirrors train_sft.py).
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Environment bootstrap — must run before importing torch/unsloth/sentence-transformers.
from shared.env_bootstrap import init_trainer_env  # noqa: E402

init_trainer_env()

import yaml  # noqa: E402

from registry import get_spec  # noqa: E402
from model_loader import load_embedding_model  # noqa: E402
from data_loader import load_embedding_dataset  # noqa: E402
from losses import build_loss, select_batch_sampler  # noqa: E402
from callbacks import EmbeddingMetricsCallback  # noqa: E402


def parse_args(argv=None):
    """Parse CLI arguments. Flags override the loaded config values."""
    parser = argparse.ArgumentParser(description="Embedding (bi-encoder) training")

    parser.add_argument("--config", type=str, default=str(Path(__file__).parent / "configs" / "config.yaml"),
                        help="Path to the embedding run config YAML.")
    parser.add_argument("--registry-name", type=str,
                        help="Override model.registry_name (key into model_registry.yaml).")
    parser.add_argument("--adapter-mode", choices=["full", "lora", "frozen_head"],
                        help="Override model.adapter_mode (NO qlora — R8).")
    parser.add_argument("--loss", type=str, help="Override training.loss.")
    parser.add_argument("--batch-size", type=int, help="Override training.batch_size.")
    parser.add_argument("--num-epochs", type=int, help="Override training.epochs.")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Cap training steps (overrides epochs for smoke runs).")
    parser.add_argument("--learning-rate", type=float, help="Override training.learning_rate.")
    parser.add_argument("--local-file", type=str, help="Override dataset.local_file.")
    parser.add_argument("--output-root", type=str,
                        help="Root for training outputs (default: ./embedding_output).")
    parser.add_argument("--no-fast-path", action="store_true",
                        help="Disable the Unsloth fast path (force the SentenceTransformer baseline).")
    parser.add_argument("--seed", type=int, default=42, help="Training seed.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Set up (load model + dataset) but do not train.")

    return parser.parse_args(argv)


def _load_config(path: str) -> dict[str, Any]:
    """Load the run config YAML into a nested dict."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Apply CLI overrides onto the loaded config in place.

    Uses `is not None` so an explicit falsy value is honored, never silently
    dropped to the config default (the provenance discipline the SFT trainer uses).
    """
    model = config.setdefault("model", {})
    training = config.setdefault("training", {})
    dataset = config.setdefault("dataset", {})

    if args.registry_name is not None:
        model["registry_name"] = args.registry_name
    if args.adapter_mode is not None:
        model["adapter_mode"] = args.adapter_mode
    if args.loss is not None:
        training["loss"] = args.loss
    if args.batch_size is not None:
        training["batch_size"] = args.batch_size
    if args.num_epochs is not None:
        training["epochs"] = args.num_epochs
    if args.learning_rate is not None:
        training["learning_rate"] = args.learning_rate
    if args.local_file is not None:
        dataset["local_file"] = args.local_file


def _build_run_dirs(output_root: str | None) -> tuple[Path, Path, Path, Path, Path]:
    """Create the canonical embedding_output/<timestamp>/ run layout.

    Returns (run_dir, checkpoints_dir, logs_dir, final_model_dir, lineage_path).
    The "embedding_output/" convention is asserted by WU-C adding "embedding" to
    TRAINING_METHODS (§5.1); here we honor the same layout SFT uses.
    """
    base = Path(output_root) if output_root else Path("./embedding_output")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / timestamp
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    final_model_dir = run_dir / "final_model"
    lineage_path = run_dir / "training_lineage.json"
    for path in (checkpoints_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    return run_dir, checkpoints_dir, logs_dir, final_model_dir, lineage_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute an embedding training run with the provided CLI arguments."""
    config = _load_config(args.config)
    _apply_overrides(config, args)

    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    dataset_cfg = config.get("dataset", {})

    registry_name = model_cfg.get("registry_name")
    adapter_mode = model_cfg.get("adapter_mode", "lora")
    loss_name = training_cfg.get("loss", "multiple_negatives_ranking")

    if not registry_name:
        raise ValueError("config.model.registry_name is required")

    # Resolve the base model spec from the registry (the SSOT). This also runs
    # registry validation, so a bad spec fails before any model load.
    spec = get_spec(registry_name)

    print("\n" + "=" * 80)
    print("EMBEDDING TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Registry model:  {spec.name} ({spec.hf_id})")
    print(f"Family/pooling:  {spec.family} / {spec.pooling}")
    print(f"Adapter mode:    {adapter_mode}")
    print(f"Loss:            {loss_name}")
    print(f"Matryoshka dims: {spec.matryoshka_dims or '(disabled)'}")
    print("=" * 80 + "\n")

    # Load the model (fallback-primary; fast path is opt-in + never load-bearing).
    loaded = load_embedding_model(
        spec,
        adapter_mode,
        lora_config=config.get("lora"),
        allow_fast_path=not args.no_fast_path,
    )
    print(f"[loader] path={loaded.loader_path} | {loaded.capabilities.reason}")

    # Build the dataset (triplets/pairs JSONL -> ST Dataset + prompt prefixing).
    local_file = dataset_cfg.get("local_file")
    if not local_file:
        raise ValueError("config.dataset.local_file is required")
    train_dataset, eval_dataset = load_embedding_dataset(
        local_file,
        spec,
        eval_split=float(dataset_cfg.get("eval_split", 0.0) or 0.0),
        seed=args.seed,
    )
    print(f"[data] train={len(train_dataset)} eval={len(eval_dataset) if eval_dataset else 0}")

    run_dir, checkpoints_dir, logs_dir, final_model_dir, lineage_path = _build_run_dirs(args.output_root)
    print(f"Run directory: {run_dir}")

    if args.dry_run:
        print("[OK] Dry run completed (model + dataset loaded). Exiting without training.")
        return {
            "run_dir": str(run_dir),
            "loader_path": loaded.loader_path,
            "train_size": len(train_dataset),
            "eval_size": len(eval_dataset) if eval_dataset else 0,
        }

    # Import the ST trainer machinery only when actually training (keeps --dry-run
    # and import-only smoke fast and torch/ST-light where possible).
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments

    loss = build_loss(loaded.model, loss_name, spec)
    batch_sampler = select_batch_sampler(loss_name)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(checkpoints_dir),
        num_train_epochs=int(training_cfg.get("epochs", 1)),
        max_steps=int(args.max_steps) if args.max_steps else -1,
        per_device_train_batch_size=int(training_cfg.get("batch_size", 64)),
        learning_rate=float(training_cfg.get("learning_rate", 2.0e-5)),
        warmup_ratio=float(training_cfg.get("warmup_ratio", 0.1)),
        batch_sampler=batch_sampler,
        seed=args.seed,
        logging_steps=int(training_cfg.get("logging_steps", 5)),
        eval_strategy="steps" if eval_dataset is not None else "no",
        report_to="none",
    )

    trainer = SentenceTransformerTrainer(
        model=loaded.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        callbacks=[EmbeddingMetricsCallback(
            log_every_n_steps=int(training_cfg.get("logging_steps", 5)),
            output_dir=str(run_dir),
        )],
    )

    print("\n" + "=" * 80)
    print("STARTING EMBEDDING TRAINING")
    print("=" * 80 + "\n")
    trainer.train()

    print(f"\nSaving final model to: {final_model_dir}")
    loaded.model.save_pretrained(str(final_model_dir))

    print(f"[OK] Embedding training complete. Model: {final_model_dir}")
    return {
        "run_dir": str(run_dir),
        "final_model_dir": str(final_model_dir),
        "loader_path": loaded.loader_path,
        "train_size": len(train_dataset),
        "eval_size": len(eval_dataset) if eval_dataset else 0,
    }


def main(argv=None):
    """Main embedding training function."""
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
