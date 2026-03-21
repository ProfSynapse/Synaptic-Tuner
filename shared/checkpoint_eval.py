"""Evaluate training checkpoints to find the best one.

Pre-filters checkpoints by training loss (cheapest signal), then runs
full evaluation on the top N to find the actual best checkpoint.
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from shared.eval_backend import EvalBackend, EvalResult

logger = logging.getLogger(__name__)


@dataclass
class CheckpointInfo:
    """Metadata about a single checkpoint."""

    path: Path
    step: int
    training_loss: float | None = None


@dataclass
class CheckpointResult:
    """Evaluation result for a single checkpoint."""

    path: Path
    step: int
    training_loss: float | None
    eval_score: float
    rank: int = 0


@dataclass
class CheckpointReport:
    """Summary of checkpoint evaluation."""

    checkpoints_evaluated: int
    best: CheckpointResult
    final_model_rank: int
    results: list[CheckpointResult]


class CheckpointEvaluator:
    """Evaluate training checkpoints and select the best one.

    Strategy:
    1. Discover checkpoints in run directory
    2. Read training log to get per-checkpoint loss
    3. Pre-filter to top N by lowest training loss
    4. Run full evaluation on survivors
    5. Rank by eval score, copy best
    """

    def __init__(self, run_dir: str, eval_backend: EvalBackend, eval_scenario: str):
        self.run_dir = Path(run_dir)
        self.eval_backend = eval_backend
        self.eval_scenario = eval_scenario

    def discover_checkpoints(self) -> list[CheckpointInfo]:
        """Find all saved checkpoints in the run directory."""
        checkpoints_dir = self.run_dir / "checkpoints"
        if not checkpoints_dir.exists():
            logger.warning(f"No checkpoints directory found at {checkpoints_dir}")
            return []

        checkpoints = []
        for entry in sorted(checkpoints_dir.iterdir()):
            if entry.is_dir() and entry.name.startswith("checkpoint-"):
                try:
                    step = int(entry.name.split("-")[1])
                    checkpoints.append(CheckpointInfo(path=entry, step=step))
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse step from {entry.name}")

        return checkpoints

    def read_checkpoint_losses(
        self, checkpoints: list[CheckpointInfo]
    ) -> list[CheckpointInfo]:
        """Read training log to get per-checkpoint training loss."""
        log_path = self.run_dir / "logs" / "training_latest.jsonl"
        if not log_path.exists():
            logger.warning(f"Training log not found at {log_path}")
            return checkpoints

        # Build step -> loss mapping from training log
        step_losses: dict[int, float] = {}
        try:
            with open(log_path) as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        step = entry.get("step") or entry.get("global_step")
                        loss = entry.get("loss") or entry.get("train_loss")
                        if step is not None and loss is not None:
                            step_losses[int(step)] = float(loss)
                    except (json.JSONDecodeError, ValueError):
                        continue
        except OSError as e:
            logger.warning(f"Could not read training log: {e}")
            return checkpoints

        # Match checkpoints to their training loss
        for ckpt in checkpoints:
            if ckpt.step in step_losses:
                ckpt.training_loss = step_losses[ckpt.step]
            else:
                # Find closest step
                closest_step = min(
                    step_losses.keys(),
                    key=lambda s: abs(s - ckpt.step),
                    default=None,
                )
                if closest_step is not None:
                    ckpt.training_loss = step_losses[closest_step]

        return checkpoints

    def prefilter_by_loss(
        self, checkpoints: list[CheckpointInfo], top_n: int
    ) -> list[CheckpointInfo]:
        """Select top N checkpoints by lowest training loss."""
        # Separate those with loss info from those without
        with_loss = [c for c in checkpoints if c.training_loss is not None]
        without_loss = [c for c in checkpoints if c.training_loss is None]

        # Sort by loss ascending (lowest = best)
        with_loss.sort(key=lambda c: c.training_loss)

        # Take top N from those with loss, then fill with unknowns
        selected = with_loss[:top_n]
        remaining_slots = top_n - len(selected)
        if remaining_slots > 0:
            selected.extend(without_loss[:remaining_slots])

        logger.info(
            f"Pre-filtered {len(checkpoints)} checkpoints to {len(selected)} "
            f"by lowest training loss"
        )
        for c in selected:
            loss_str = (
                f"{c.training_loss:.4f}" if c.training_loss is not None else "unknown"
            )
            logger.info(f"  step {c.step}: loss={loss_str}")

        return selected

    async def evaluate_checkpoints(self, top_n: int = 5) -> CheckpointReport:
        """Evaluate top N checkpoints by loss, find the best by eval score.

        Args:
            top_n: Number of checkpoints to evaluate (pre-filtered by lowest
                   training loss). Set to 0 to evaluate ALL checkpoints.
        """
        # 1. Discover checkpoints
        checkpoints = self.discover_checkpoints()
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {self.run_dir}")

        # 2. Read training losses
        checkpoints = self.read_checkpoint_losses(checkpoints)

        # 3. Pre-filter by loss (if top_n > 0)
        if top_n > 0:
            checkpoints = self.prefilter_by_loss(checkpoints, top_n)

        # 4. Always include final_model if it exists
        final_model = self.run_dir / "final_model"
        final_included = False
        if final_model.exists():
            # Check if final_model is already in the list (by path)
            if not any(c.path == final_model for c in checkpoints):
                # Get final model's loss from the last training log entry
                final_loss = None
                log_path = self.run_dir / "logs" / "training_latest.jsonl"
                if log_path.exists():
                    try:
                        last_line = None
                        with open(log_path) as f:
                            for line in f:
                                stripped = line.strip()
                                if stripped:
                                    last_line = stripped
                        if last_line:
                            last_entry = json.loads(last_line)
                            final_loss = last_entry.get("loss") or last_entry.get(
                                "train_loss"
                            )
                            if final_loss is not None:
                                final_loss = float(final_loss)
                    except (json.JSONDecodeError, OSError, ValueError):
                        pass
                checkpoints.append(
                    CheckpointInfo(
                        path=final_model, step=-1, training_loss=final_loss
                    )
                )
            final_included = True

        # 5. Run evaluation on each checkpoint
        results: list[CheckpointResult] = []
        for ckpt in checkpoints:
            logger.info(
                f"Evaluating checkpoint: {ckpt.path.name} (step {ckpt.step})"
            )
            try:
                eval_result = await self.eval_backend.run_eval(
                    str(ckpt.path), self.eval_scenario
                )
                results.append(
                    CheckpointResult(
                        path=ckpt.path,
                        step=ckpt.step,
                        training_loss=ckpt.training_loss,
                        eval_score=eval_result.eval_score,
                    )
                )
            except Exception as e:
                logger.error(f"Failed to evaluate {ckpt.path.name}: {e}")
                results.append(
                    CheckpointResult(
                        path=ckpt.path,
                        step=ckpt.step,
                        training_loss=ckpt.training_loss,
                        eval_score=0.0,
                    )
                )

        # 6. Rank by eval score
        results.sort(key=lambda r: r.eval_score, reverse=True)
        for i, r in enumerate(results):
            r.rank = i + 1

        best = results[0]
        logger.info(
            f"Best checkpoint: {best.path.name} "
            f"(step={best.step}, score={best.eval_score:.4f})"
        )

        # Check if final model is best
        final_rank = -1
        if final_included:
            for r in results:
                if r.path == final_model or r.step == -1:
                    final_rank = r.rank
                    if r.rank > 1:
                        logger.info(
                            f"Final model ranked #{r.rank} "
                            f"(score={r.eval_score:.4f}). "
                            f"Best is step {best.step} "
                            f"(score={best.eval_score:.4f}), "
                            f"delta={best.eval_score - r.eval_score:+.4f}"
                        )
                    break

        # 7. Copy best to best_checkpoint/
        best_dir = self.run_dir / "best_checkpoint"
        if best.path != best_dir:
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(best.path, best_dir)
            logger.info(f"Copied best checkpoint to {best_dir}")

        # 8. Write results TSV
        self._write_results_tsv(results)

        return CheckpointReport(
            checkpoints_evaluated=len(results),
            best=best,
            final_model_rank=final_rank,
            results=results,
        )

    def _write_results_tsv(self, results: list[CheckpointResult]) -> None:
        """Write evaluation results as TSV."""
        tsv_path = self.run_dir / "checkpoint_eval_results.tsv"
        with open(tsv_path, "w") as f:
            f.write("rank\tstep\tpath\ttraining_loss\teval_score\n")
            for r in results:
                loss_str = (
                    f"{r.training_loss:.4f}" if r.training_loss is not None else ""
                )
                f.write(
                    f"{r.rank}\t{r.step}\t{r.path.name}\t{loss_str}"
                    f"\t{r.eval_score:.4f}\n"
                )
        logger.info(f"Results written to {tsv_path}")
