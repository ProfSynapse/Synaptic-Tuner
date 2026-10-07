#!/usr/bin/env python3
"""
Decision-model training entry point (Jev-style typed decisions).

Location: Trainers/decision/train_decision.py
Purpose:  Train a "System One" decision model: a causal-LM torso + LoRA + a
          readout that answers typed questions about a state -- ``noul``
          (yes/no), ``choice`` (one of N) and ``score`` (ordered levels) -- with
          calibrated option probabilities in one forward pass, no generation.

          Stages (``--stage all`` runs them in order):
            train      LoRA + readout on decision rows, options re-shuffled per draw
            calibrate  per-kind temperature fitted on held-out rows (NLL)
            evaluate   accuracy / NLL / Brier / ECE overall, per kind, per task,
                       plus answer-change rate under option reordering

          Readouts (``model.readout``):
            pointer        strands-decider style ~1M-param head: <answer> state
                           attends over each option line's last-token state
            letter_logits  OpenJev style: the LM's own logits for the option
                           markers at <answer>; no new parameters

Used by:  The ``decision`` training method (TRAINING_METHODS SSOT in
          shared/utilities/paths.py); local Docker via
          Trainers/recipes/decision_*.yaml (explicit run.command).

Usage:
    python Trainers/decision/train_decision.py --config Trainers/decision/configs/config.yaml --dry-run
    python Trainers/decision/train_decision.py --config Trainers/decision/configs/config.yaml
    python Trainers/decision/train_decision.py --stage evaluate --checkpoint decision_output/<run>/final_model
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

TRAINER_DIR = Path(__file__).resolve().parent
REPO_ROOT = TRAINER_DIR.parent.parent
sys.path.insert(0, str(TRAINER_DIR))
sys.path.insert(0, str(REPO_ROOT))

# Import-light modules only (yaml/json): --dry-run works without torch.
from decision_core.config import DecisionRunConfig, load_run_config, validate_run_config  # noqa: E402
from decision_core.examples import DecisionExample, load_examples, split_by_task  # noqa: E402
from decision_core.model_config import DecisionModelConfig  # noqa: E402
from decision_core.prompting import render_prompt  # noqa: E402
from decision_core.registry import DEFAULT_REGISTRY_PATH, TorsoSpec, get_spec  # noqa: E402

STAGES = ("all", "train", "calibrate", "evaluate")
logger = logging.getLogger("train_decision")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Decision-model (typed choice/yes-no/score) training")
    p.add_argument("--config", default=str(TRAINER_DIR / "configs" / "config.yaml"))
    p.add_argument("--stage", choices=STAGES, default="all")
    p.add_argument("--checkpoint", default=None,
                   help="Existing final_model dir for --stage calibrate/evaluate.")
    p.add_argument("--model", default=None,
                   help="Override model.registry_name (a key in configs/model_registry.yaml).")
    p.add_argument("--readout", choices=("pointer", "letter_logits"), default=None)
    p.add_argument("--train-file", action="append", default=None,
                   help="Replace data.train_files (repeatable).")
    p.add_argument("--eval-file", action="append", default=None,
                   help="Replace data.eval_files (repeatable).")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--max-train-examples", type=int, default=None)
    p.add_argument("--max-eval-examples", type=int, default=None)
    p.add_argument("--output-root", default=None)
    p.add_argument("--run-timestamp", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="Validate config + data and render a sample prompt; no model load.")
    return p.parse_args(argv)


def apply_overrides(cfg: DecisionRunConfig, args: argparse.Namespace) -> DecisionRunConfig:
    if args.model is not None:
        cfg.model.registry_name = args.model
    if args.readout is not None:
        cfg.model.readout = args.readout
    if args.train_file:
        cfg.data.train_files = list(args.train_file)
    if args.eval_file:
        cfg.data.eval_files = list(args.eval_file)
    if args.max_steps is not None:
        cfg.training.max_steps = args.max_steps
    if args.max_train_examples is not None:
        cfg.data.max_train_examples = args.max_train_examples
    if args.max_eval_examples is not None:
        cfg.data.max_eval_examples = args.max_eval_examples
    if args.output_root is not None:
        cfg.output.output_root = args.output_root
    validate_run_config(cfg)
    return cfg


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)


def resolve_model_config(cfg: DecisionRunConfig) -> tuple[DecisionModelConfig, TorsoSpec]:
    """Registry entry + run-config overrides -> the self-contained model config."""
    registry = _resolve(cfg.model.registry_path) if cfg.model.registry_path else DEFAULT_REGISTRY_PATH
    spec = get_spec(cfg.model.registry_name, registry)
    m = cfg.model
    mcfg = DecisionModelConfig(
        hf_id=spec.hf_id,
        lora_targets=list(cfg.lora.target_modules or spec.lora_target_modules),
        registry_name=spec.name,
        revision=m.revision or spec.revision,
        loader=spec.loader,
        causal_lm_class=spec.causal_lm_class,
        readout=m.readout,
        marker_style=cfg.prompt.marker_style,
        headers=dict(cfg.prompt.headers),
        pointer_dim=m.pointer_dim,
        head_dropout=m.head_dropout,
        max_length=m.max_length or spec.max_length,
        torch_dtype=m.torch_dtype or spec.torch_dtype,
        attn_implementation=m.attn_implementation or spec.attn_implementation,
        use_lora=cfg.lora.enabled,
        lora_r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        ordinal_smoothing=cfg.augmentation.ordinal_smoothing,
    )
    return mcfg, spec


def _load(paths: list[str], limit: int | None, seed: int) -> list[DecisionExample]:
    missing = [p for p in paths if not _resolve(p).exists()]
    if missing:
        raise SystemExit(
            f"decision data file(s) not found: {missing}. Build the corpus first:\n"
            "  python tuner.py local-run --job-config Trainers/recipes/decision_strands_corpus_build.yaml --yes"
        )
    rows = load_examples([_resolve(p) for p in paths])
    if limit and len(rows) > limit:
        # Corpus files are grouped by task, so a head slice would keep one or two
        # tasks; a seeded sample keeps the task mix.
        rows = random.Random(seed).sample(rows, limit)
    return rows


def load_data(cfg: DecisionRunConfig, stage: str) -> dict[str, list[DecisionExample]]:
    data: dict[str, list[DecisionExample]] = {"train": [], "val": [], "calibration": [], "eval": []}
    if stage in ("all", "train"):
        if not cfg.data.train_files:
            raise ValueError("data.train_files is empty")
        rows = _load(cfg.data.train_files, cfg.data.max_train_examples, cfg.data.seed)
        data["train"], data["val"] = split_by_task(rows, val_fraction=cfg.data.val_fraction, seed=cfg.data.seed)

    need_eval = stage in ("all", "calibrate", "evaluate")
    if need_eval and cfg.data.eval_files:
        eval_rows = _load(cfg.data.eval_files, cfg.data.max_eval_examples, cfg.data.seed)
        if cfg.data.calibration_files:
            data["calibration"] = _load(cfg.data.calibration_files, cfg.data.max_eval_examples, cfg.data.seed)
            data["eval"] = eval_rows
        else:
            data["eval"], data["calibration"] = split_by_task(
                eval_rows, val_fraction=cfg.calibration.holdout_fraction, seed=cfg.data.seed
            )
    return data


def summarize(name: str, rows: list[DecisionExample]) -> dict[str, Any]:
    kinds = Counter(ex.kind for ex in rows)
    tasks = Counter(ex.task for ex in rows)
    max_opts = max((ex.n_options for ex in rows), default=0)
    print(f"  {name:<12} {len(rows):>8,} rows | kinds {dict(kinds)} | {len(tasks)} tasks | max options {max_opts}")
    return {"rows": len(rows), "kinds": dict(kinds), "tasks": dict(tasks), "max_options": max_opts}


def dry_run(cfg: DecisionRunConfig, mcfg: DecisionModelConfig, spec: TorsoSpec,
            data: dict[str, list[DecisionExample]]) -> int:
    print("=" * 72)
    print("DECISION TRAINING -- DRY RUN")
    print("=" * 72)
    loader = mcfg.loader + (f" ({mcfg.causal_lm_class})" if mcfg.causal_lm_class else "")
    print(f"  model        {spec.name}: {mcfg.hf_id} @ {mcfg.revision or '<default branch>'}")
    print(f"  loader       {loader}, {mcfg.torch_dtype}, max_length {mcfg.max_length}")
    print(f"  lora         r={mcfg.lora_r} alpha={mcfg.lora_alpha} on {len(mcfg.lora_targets)} module names")
    if spec.notes:
        print(f"  notes        {spec.notes}")
    print(f"  readout      {mcfg.readout} (markers: {mcfg.marker_style})")
    print(f"  loss         ce + {cfg.loss.brier_weight} brier + {cfg.loss.kl_frozen_weight} kl_frozen")
    for name, rows in data.items():
        summarize(name, rows)
    if mcfg.readout == "letter_logits":
        cap = 9 if mcfg.marker_style == "numbers" else 26
        over = sum(ex.n_options > cap for ex in data["train"])
        print(f"  letter_logits caps options at {cap}: {over:,} train rows would be dropped")
    sample = next(iter(data["train"] or data["eval"]), None)
    if sample is not None:
        rendered = render_prompt(sample.state, sample.kind, sample.instructions, sample.options,
                                 marker_style=mcfg.marker_style, headers=mcfg.headers)
        text = rendered.text if len(rendered.text) < 1500 else rendered.text[:600] + "\n...\n" + rendered.text[-700:]
        print("-" * 72)
        print(f"Sample prompt (task={sample.task}, gold={sample.options[sample.label][0]!r}):")
        print(text)
    print("=" * 72)
    print("Dry run OK.")
    return 0


# ---------------------------------------------------------------------------
# Real run
# ---------------------------------------------------------------------------

def filter_trainable(model: Any, rows: list[DecisionExample], name: str
                     ) -> tuple[list[DecisionExample], list[int]]:
    """Drop rows the readout cannot score and rows too long for the window.

    Training drops, never truncates: a truncated state can remove the evidence the
    gold label depends on. Returns kept rows and their canonical token lengths.
    """
    from decision_core.collate import CollatorConfig, DecisionCollator

    mcfg = model.decision_config
    collator = DecisionCollator(
        model.tokenizer,
        CollatorConfig(max_length=10**9, marker_style=mcfg.marker_style, headers=dict(mcfg.headers)),
        train=False,
    )
    cap = model.max_marker_options if mcfg.readout == "letter_logits" else None
    kept: list[DecisionExample] = []
    lengths: list[int] = []
    too_many = too_long = 0
    for ex in rows:
        if cap is not None and ex.n_options > cap:
            too_many += 1
            continue
        longest = max(ex.all_instructions(), key=len)
        probe = DecisionExample(ex.kind, ex.state, longest, ex.options, ex.label, ex.task)
        n_tok = len(collator.encode(probe).input_ids)
        if n_tok > mcfg.max_length:
            too_long += 1
            continue
        kept.append(ex)
        lengths.append(n_tok)
    if too_many or too_long:
        print(f"  {name}: dropped {too_many:,} rows over the marker cap and {too_long:,} over "
              f"{mcfg.max_length} tokens; kept {len(kept):,}")
    return kept, lengths


def build_model(mcfg: DecisionModelConfig) -> Any:
    from decision_core.modeling import DecisionModel

    return DecisionModel.from_base(mcfg)


def warmup_kwargs(training_args_cls: Any, ratio: float) -> dict[str, float]:
    """transformers 5 folded warmup_ratio into warmup_steps (a float in [0, 1) is a ratio)."""
    import inspect

    if "warmup_ratio" in inspect.signature(training_args_cls.__init__).parameters:
        return {"warmup_ratio": ratio}
    return {"warmup_steps": float(ratio)}


def run_training(model: Any, cfg: DecisionRunConfig, data: dict[str, list[DecisionExample]],
                 run_dir: Path) -> Any:
    import torch
    from transformers import TrainingArguments

    from decision_core.callbacks import DecisionMetricsCallback
    from decision_core.collate import CollatorConfig, DecisionCollator
    from decision_core.losses import LossConfig
    from decision_core.trainer import DecisionTrainer, ExampleDataset

    train_rows, lengths = filter_trainable(model, data["train"], "train")
    val_rows, _ = filter_trainable(model, data["val"], "val")
    if not train_rows:
        raise SystemExit("no trainable rows left after filtering")

    aug = cfg.augmentation
    mcfg = model.decision_config
    coll_cfg = CollatorConfig(
        max_length=mcfg.max_length,
        marker_style=mcfg.marker_style,
        headers=dict(mcfg.headers),
        shuffle_options=aug.shuffle_options,
        reverse_score_prob=aug.reverse_score_prob,
        ordinal_smoothing=aug.ordinal_smoothing,
        vary_instructions=aug.vary_instructions,
        seed=cfg.training.seed,
    )
    t = cfg.training
    if t.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    use_bf16 = bool(t.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    has_val = bool(val_rows) and t.eval_steps > 0
    args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=t.num_epochs,
        max_steps=t.max_steps,
        per_device_train_batch_size=t.per_device_train_batch_size,
        per_device_eval_batch_size=t.per_device_eval_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        learning_rate=t.learning_rate,
        weight_decay=t.weight_decay,
        **warmup_kwargs(TrainingArguments, t.warmup_ratio),
        max_grad_norm=t.max_grad_norm,
        lr_scheduler_type=t.lr_scheduler_type,
        logging_steps=t.logging_steps,
        eval_strategy="steps" if has_val else "no",
        eval_steps=t.eval_steps if has_val else None,
        save_strategy="steps",
        save_steps=t.save_steps,
        save_total_limit=t.save_total_limit,
        bf16=use_bf16,
        seed=t.seed,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        gradient_checkpointing=False,  # enabled on the torso above (non-reentrant)
    )
    trainer = DecisionTrainer(
        model=model,
        args=args,
        train_dataset=ExampleDataset(train_rows),
        eval_dataset=ExampleDataset(val_rows) if has_val else None,
        data_collator=DecisionCollator(model.tokenizer, coll_cfg, train=True),
        loss_config=LossConfig(brier_weight=cfg.loss.brier_weight, kl_frozen_weight=cfg.loss.kl_frozen_weight),
        head_learning_rate=t.head_learning_rate,
        eval_collator=DecisionCollator(model.tokenizer, coll_cfg, train=False),
        train_lengths=lengths if t.group_by_length else None,
        callbacks=[DecisionMetricsCallback(log_every_n_steps=t.logging_steps, output_dir=str(run_dir))],
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable:,} / {total:,} ({100 * trainable / max(total, 1):.2f}%)")
    print(f"Train rows: {len(train_rows):,} | val rows: {len(val_rows):,}\n")
    trainer.train()
    return trainer


def run_calibration(model: Any, cfg: DecisionRunConfig, rows: list[DecisionExample]) -> dict[str, float]:
    from decision_core.evaluate import fit_temperatures, predict

    rows, _ = filter_trainable(model, rows, "calibration")
    if not rows:
        print("No calibration rows; temperatures stay at 1.0.")
        return {}
    pred = predict(model, rows, max_length=model.decision_config.max_length,
                   batch_size=cfg.evaluation.batch_size)
    temps = fit_temperatures(pred, min_rows_per_kind=cfg.calibration.min_rows_per_kind)
    print(f"Calibrated temperatures on {len(rows):,} rows: {temps}")
    return temps


def run_evaluation(model: Any, cfg: DecisionRunConfig, rows: list[DecisionExample],
                   temps: dict[str, float]) -> dict[str, Any]:
    from decision_core.evaluate import evaluate

    rows, _ = filter_trainable(model, rows, "eval")
    if not rows:
        print("No evaluation rows.")
        return {}
    report = evaluate(
        model, rows, temps,
        max_length=model.decision_config.max_length,
        batch_size=cfg.evaluation.batch_size,
        shuffle_trials=cfg.evaluation.shuffle_trials,
        ece_bins=cfg.evaluation.ece_bins,
        seed=cfg.data.seed,
    )
    c = report["calibrated"]
    print(f"Eval on {report['n_examples']:,} rows: acc {c['accuracy']:.4f} | NLL {c['nll']:.4f} | "
          f"Brier {c['brier']:.4f} | ECE {c['ece']:.4f}")
    if "order_consistency" in report:
        print(f"  answer change rate under option reordering: "
              f"{report['order_consistency']['answer_change_rate']:.4f}")
    return report


def build_lineage(cfg: DecisionRunConfig, mcfg: DecisionModelConfig, trainer: Any, run_dir: Path, data: dict[str, list[DecisionExample]],
                  temps: dict[str, float], report: dict[str, Any], elapsed: float, args: argparse.Namespace
                  ) -> dict[str, Any]:
    from shared.experiment_tracking.lineage_enrichment import enrich_training_lineage
    from shared.training_utils import build_base_lineage

    t = cfg.training
    lineage = build_base_lineage(
        training_type="DECISION",
        model_info={
            "base_model": mcfg.hf_id,
            "base_revision": mcfg.revision,
            "registry_name": mcfg.registry_name,
            "max_seq_length": mcfg.max_length,
            "load_in_4bit": False,
            "dtype": mcfg.torch_dtype,
        },
        lora_info={
            "rank": cfg.lora.r,
            "alpha": cfg.lora.alpha,
            "dropout": cfg.lora.dropout,
            "target_modules": mcfg.lora_targets,
            "bias": "none",
        },
        training_info={
            "batch_size": t.per_device_train_batch_size,
            "gradient_accumulation_steps": t.gradient_accumulation_steps,
            "effective_batch_size": t.per_device_train_batch_size * t.gradient_accumulation_steps,
            "learning_rate": t.learning_rate,
            "head_learning_rate": t.head_learning_rate,
            "num_epochs": t.num_epochs,
            "max_steps": t.max_steps,
            "warmup_ratio": t.warmup_ratio,
            "lr_scheduler": t.lr_scheduler_type,
            "optimizer": "adamw_torch",
            "max_grad_norm": t.max_grad_norm,
            "gradient_checkpointing": t.gradient_checkpointing,
            "bf16": t.bf16,
            "seed": t.seed,
        },
        dataset_info={
            "source": ",".join(cfg.data.train_files),
            "train_examples": len(data["train"]),
            "eval_examples": len(data["val"]),
            "calibration_examples": len(data["calibration"]),
            "heldout_eval_examples": len(data["eval"]),
        },
        run_dir=run_dir,
        trainer=trainer,
        training_time_seconds=elapsed,
    )
    lineage["decision"] = {
        "readout": mcfg.readout,
        "marker_style": mcfg.marker_style,
        "prompt_headers": mcfg.headers,
        "pointer_dim": mcfg.pointer_dim if mcfg.readout == "pointer" else None,
        "loss": {"brier_weight": cfg.loss.brier_weight, "kl_frozen_weight": cfg.loss.kl_frozen_weight},
        "augmentation": vars(cfg.augmentation),
        "temperature_by_kind": temps,
        "evaluation": {k: report.get(k) for k in ("n_examples", "calibrated", "uncalibrated",
                                                    "by_kind", "order_consistency")} if report else None,
    }
    if report:
        lineage["results"]["eval_accuracy"] = report["calibrated"]["accuracy"]
        lineage["results"]["eval_ece"] = report["calibrated"]["ece"]
    return enrich_training_lineage(lineage, args=args)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = apply_overrides(load_run_config(args.config), args)
    data = load_data(cfg, args.stage)
    mcfg, spec = resolve_model_config(cfg)

    if args.dry_run:
        return dry_run(cfg, mcfg, spec, data)

    from shared.env_bootstrap import init_trainer_env

    init_trainer_env(apply_windows_patches=False)
    logging.basicConfig(level=logging.INFO)

    from decision_core.modeling import DecisionModel

    run_timestamp = args.run_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _resolve(cfg.output.output_root) / run_timestamp
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "evaluation").mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    final_dir = run_dir / "final_model"

    trainer = None
    elapsed = 0.0
    if args.stage in ("all", "train"):
        model = build_model(mcfg)
        start = time.time()
        trainer = run_training(model, cfg, data, run_dir)
        elapsed = time.time() - start
        model.save(final_dir)
        print(f"Saved adapter + readout to {final_dir}")
    else:
        if not args.checkpoint:
            raise SystemExit(f"--stage {args.stage} needs --checkpoint <final_model dir>")
        import torch

        final_dir = _resolve(args.checkpoint)
        model = DecisionModel.load(final_dir, device="cuda" if torch.cuda.is_available() else "cpu")

    temps = dict(model.decision_config.temperature_by_kind)
    if args.stage in ("all", "calibrate") and cfg.calibration.enabled:
        temps = run_calibration(model, cfg, data["calibration"])
        model.decision_config.temperature_by_kind = temps
        # Temperatures belong to the checkpoint: rewrite only its config file.
        (final_dir / "decision_config.json").write_text(
            json.dumps(model.decision_config.to_dict(), indent=2), encoding="utf-8")

    report: dict[str, Any] = {}
    if args.stage in ("all", "evaluate") and cfg.evaluation.enabled:
        report = run_evaluation(model, cfg, data["eval"], temps)
        if report:
            out = run_dir / "evaluation" / "decision_eval.json"
            out.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"Evaluation report: {out}")

    if trainer is not None:
        from shared.training_utils import save_training_lineage

        lineage = build_lineage(cfg, mcfg, trainer, run_dir, data, temps, report, elapsed, args)
        save_training_lineage(lineage, run_dir)
        try:
            from shared.experiment_tracking.adapters import decision_lineage_to_run_record
            from shared.experiment_tracking.registry import RunRegistry

            record = decision_lineage_to_run_record(lineage, str(run_dir))
            RunRegistry().register_run(record)
            logger.info("Run registered: %s", record.run_id)
        except Exception as exc:  # tracking is best-effort, as in the other trainers
            logger.warning("Unified tracking registration failed (non-fatal): %s", exc)

    print(f"\nRun directory: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
