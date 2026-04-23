"""Checkpoint configuration banner, hoisted from sft + kto (byte-identical)."""

from __future__ import annotations

from transformers import TrainerCallback


class CheckpointMonitorCallback(TrainerCallback):
    """Display checkpoint configuration at start."""

    def on_train_begin(self, args, state, control, **kwargs):
        print(f"\nCheckpoint Configuration:")
        print(f"  Save every: {args.save_steps} steps")
        print(f"  Keep last: {args.save_total_limit} checkpoints")
        print(f"  Output dir: {args.output_dir}")
        print()
