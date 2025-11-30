"""
Checkpoint discovery and analysis service.

Location: /mnt/f/Code/Toolset-Training/tuner/discovery/checkpoints.py
Purpose: Discover checkpoints and load their training metrics
Used by: Upload handler to display checkpoint metrics and allow checkpoint selection

This module implements the CheckpointDiscovery service which scans training run
directories for checkpoints, parses training logs to extract metrics, and returns
structured CheckpointInfo objects with all relevant metadata.

Pattern migrated from: tuner.py lines 330-364 (_load_checkpoint_metrics and _detect_training_type)
"""

import json
from pathlib import Path
from typing import List, Dict

from tuner.core.config import CheckpointInfo


class CheckpointDiscovery:
    """
    Discover and analyze training checkpoints.

    This service scans a training run directory for checkpoints, parses training
    logs to extract metrics for each checkpoint, and returns structured CheckpointInfo
    objects that include both the checkpoint path and its associated metrics.

    Example:
        from tuner.discovery import CheckpointDiscovery

        # Discover checkpoints in a run
        discovery = CheckpointDiscovery()
        run_dir = Path('/path/to/sft_output_rtx3090/20251122_143000')

        # Load metrics from logs
        metrics = discovery.load_metrics(run_dir)
        print(f"Metrics for steps: {list(metrics.keys())}")

        # Discover all checkpoints with metrics
        checkpoints = discovery.discover(run_dir)
        for cp in checkpoints:
            if cp.is_final:
                print(f"Final model: {cp.path}")
            else:
                print(f"Checkpoint {cp.step}: loss={cp.metrics.get('loss', 'N/A')}")
    """

    @staticmethod
    def load_metrics(run_dir: Path) -> Dict[int, Dict]:
        """
        Load training metrics from JSONL logs.

        Parses the training logs to extract metrics for each training step.
        Returns a dictionary mapping step numbers to their metric dictionaries.

        Args:
            run_dir: Path to training run directory (e.g., sft_output_rtx3090/20251122_143000)

        Returns:
            Dictionary mapping step number (int) to metrics dictionary.
            Returns empty dict if no logs found or parsing fails.

        Example:
            metrics = CheckpointDiscovery.load_metrics(run_dir)

            # Get metrics for step 100
            if 100 in metrics:
                step_100 = metrics[100]
                print(f"Loss: {step_100.get('loss', 'N/A')}")
                print(f"LR: {step_100.get('learning_rate', 'N/A')}")

            # For KTO training
            if 'kl' in step_100:
                print(f"KL: {step_100['kl']}")
                print(f"Margin: {step_100.get('rewards/margins', 'N/A')}")

        Log format expected:
            logs/training_*.jsonl with entries like:
            {"step": 5, "loss": 0.5, "learning_rate": 2e-4, ...}
            {"step": 10, "loss": 0.45, "learning_rate": 2e-4, ...}
            ...
        """
        metrics = {}
        logs_dir = run_dir / "logs"

        # Return empty dict if logs directory doesn't exist
        if not logs_dir.exists():
            return metrics

        # Find training log files (training_*.jsonl)
        log_files = list(logs_dir.glob("training_*.jsonl"))
        if not log_files:
            return metrics

        # Parse the first log file found (usually only one)
        try:
            with open(log_files[0]) as f:
                for line in f:
                    if not line.strip():
                        continue

                    try:
                        entry = json.loads(line)
                        step = entry.get("step", 0)
                        metrics[step] = entry
                    except json.JSONDecodeError:
                        # Skip malformed lines
                        continue
        except Exception:
            # Return whatever we've parsed so far
            pass

        return metrics

    @staticmethod
    def discover(run_dir: Path) -> List[CheckpointInfo]:
        """
        Discover all checkpoints in a training run.

        Scans the run directory for final_model and checkpoint-* directories,
        loads metrics from training logs, and returns CheckpointInfo objects
        for each checkpoint found.

        Args:
            run_dir: Path to training run directory

        Returns:
            List of CheckpointInfo objects, sorted by step number.
            Final model (step=-1) is first, followed by checkpoints in ascending order.
            Returns empty list if no checkpoints found.

        Example:
            checkpoints = CheckpointDiscovery.discover(run_dir)

            # Filter to only final model
            final = [cp for cp in checkpoints if cp.is_final]

            # Filter to only intermediate checkpoints
            intermediate = [cp for cp in checkpoints if not cp.is_final]

            # Find best checkpoint by score (for KTO)
            best = max(checkpoints, key=lambda cp: cp.score('kto'))

            # Display checkpoint table
            for cp in checkpoints:
                name = "final_model" if cp.is_final else f"checkpoint-{cp.step}"
                loss = cp.metrics.get('loss', 'N/A')
                print(f"{name}: loss={loss}")

        Directory structure expected:
            run_dir/
            ├── final_model/           <- CheckpointInfo(step=-1, is_final=True)
            ├── checkpoints/
            │   ├── checkpoint-50/     <- CheckpointInfo(step=50, is_final=False)
            │   ├── checkpoint-100/    <- CheckpointInfo(step=100, is_final=False)
            │   └── checkpoint-150/    <- CheckpointInfo(step=150, is_final=False)
            └── logs/
                └── training_*.jsonl   <- Metrics source
        """
        checkpoints = []

        # Load metrics from logs
        metrics = CheckpointDiscovery.load_metrics(run_dir)

        # Check for final_model
        final_model = run_dir / "final_model"
        if final_model.exists():
            checkpoints.append(
                CheckpointInfo(
                    path=final_model,
                    step=-1,
                    metrics={},  # Final model doesn't have associated step metrics
                    is_final=True,
                )
            )

        # Check for individual checkpoints
        checkpoints_dir = run_dir / "checkpoints"
        if checkpoints_dir.exists():
            for cp_dir in sorted(checkpoints_dir.iterdir()):
                if not cp_dir.is_dir():
                    continue

                # Parse checkpoint name (e.g., "checkpoint-100")
                if cp_dir.name.startswith("checkpoint-"):
                    try:
                        step = int(cp_dir.name.split("-")[1])
                        checkpoints.append(
                            CheckpointInfo(
                                path=cp_dir,
                                step=step,
                                metrics=metrics.get(step, {}),
                                is_final=False,
                            )
                        )
                    except (ValueError, IndexError):
                        # Skip malformed checkpoint directory names
                        continue

        # Sort checkpoints: final model first, then by step number
        checkpoints.sort(key=lambda cp: cp.step if not cp.is_final else float('-inf'))

        return checkpoints
