"""
File: /Users/jrosenbaum/Documents/Code/Synthetic Conversations/code/mistral_lora_mac/src/trainer.py

Training Engine for MLX Fine-Tuning System

This module implements the complete training loop:
- Forward/backward passes with gradient computation
- Optimizer updates (AdamW)
- Learning rate scheduling (cosine with warmup)
- Gradient accumulation
- Gradient clipping
- Checkpoint management
- Metrics tracking and logging
- Training state management
- Interruption recovery

Dependencies:
- mlx.core: Core operations
- mlx.nn: Neural network modules
- mlx.optimizers: Optimizers

Related Files:
- src/model_manager.py: Provides model
- src/data_pipeline.py: Provides data loaders
- src/utils.py: Logging and memory monitoring
- config/config_manager.py: Training configuration
"""

import time
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import json


@dataclass
class TrainingState:
    """Complete training state for checkpointing."""
    epoch: int
    global_step: int
    best_val_loss: float
    train_loss_history: list
    val_loss_history: list
    learning_rates: list


@dataclass
class StepMetrics:
    """Metrics for a single training step."""
    loss: float
    learning_rate: float
    step_time: float
    tokens_per_sec: float


@dataclass
class EpochMetrics:
    """Metrics for an epoch."""
    epoch: int
    train_loss: float
    val_loss: float
    epoch_time: float
    steps: int


class CosineWarmupScheduler:
    """
    Cosine learning rate schedule with linear warmup.

    Warmup: LR increases linearly from 0 to max_lr over warmup_steps
    Decay: LR decreases following cosine curve from max_lr to 0 over remaining steps
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0
    ):
        """
        Initialize scheduler.

        Args:
            optimizer: MLX optimizer
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            min_lr: Minimum learning rate after decay
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.learning_rate
        self.current_step = 0

    def step(self):
        """Update learning rate for next step."""
        self.current_step += 1

        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        self.optimizer.learning_rate = lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return float(self.optimizer.learning_rate)


class GradientAccumulator:
    """Accumulates gradients over multiple steps."""

    def __init__(self, accumulation_steps: int):
        """
        Initialize accumulator.

        Args:
            accumulation_steps: Number of steps to accumulate gradients
        """
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        self.accumulated_grads = {}

    def accumulate(self, grads: Dict[str, mx.array]):
        """
        Accumulate gradients.

        Args:
            grads: Dictionary of gradients (can be nested)
        """
        if not self.accumulated_grads:
            # First accumulation - deep copy the gradients
            self.accumulated_grads = self._deep_copy_grads(grads)
        else:
            # Add to existing - recursive addition for nested dicts
            self.accumulated_grads = self._add_grad_dicts(self.accumulated_grads, grads)

        self.current_step += 1

    def _deep_copy_grads(self, grads: Dict) -> Dict:
        """Deep copy gradients, handling nested dictionaries."""
        result = {}
        for k, v in grads.items():
            if isinstance(v, dict):
                result[k] = self._deep_copy_grads(v)
            else:
                result[k] = v
        return result

    def _add_grad_dicts(self, acc: Dict, grads: Dict) -> Dict:
        """Recursively add gradient dictionaries."""
        result = {}
        for k in acc:
            if k in grads:
                if isinstance(acc[k], dict) and isinstance(grads[k], dict):
                    result[k] = self._add_grad_dicts(acc[k], grads[k])
                elif isinstance(acc[k], list) or isinstance(grads[k], list):
                    # Skip lists - just use the accumulated value
                    result[k] = acc[k]
                else:
                    # Add mx.arrays
                    result[k] = acc[k] + grads[k]
            else:
                result[k] = acc[k]
        return result

    def should_update(self) -> bool:
        """Check if we should update parameters."""
        return self.current_step >= self.accumulation_steps

    def get_gradients(self) -> Dict[str, mx.array]:
        """
        Get averaged gradients.

        Returns:
            Dictionary of averaged gradients
        """
        # Average gradients (recursive for nested dicts)
        return self._avg_grad_dicts(self.accumulated_grads)

    def _avg_grad_dicts(self, grads: Dict) -> Dict:
        """Recursively average gradient dictionaries."""
        result = {}
        for k, v in grads.items():
            if isinstance(v, dict):
                result[k] = self._avg_grad_dicts(v)
            elif isinstance(v, list):
                # Skip lists (these are not gradient arrays)
                result[k] = v
            else:
                # Divide mx.array by the accumulation steps
                result[k] = v / self.accumulation_steps
        return result

    def reset(self):
        """Reset accumulator."""
        self.accumulated_grads = {}
        self.current_step = 0


def compute_loss(logits: mx.array, labels: mx.array, attention_mask: mx.array) -> mx.array:
    """
    Compute cross-entropy loss.

    Args:
        logits: Model logits (batch_size, seq_length, vocab_size)
        labels: Target labels (batch_size, seq_length)
        attention_mask: Attention mask (batch_size, seq_length)

    Returns:
        Scalar loss value
    """
    # Flatten
    logits_flat = logits.reshape(-1, logits.shape[-1])
    labels_flat = labels.reshape(-1)

    # Compute cross-entropy loss (don't reduce yet)
    loss_per_token = nn.losses.cross_entropy(logits_flat, labels_flat, reduction='none')

    # Create mask for valid tokens (labels != -100 means it's not padding/masked)
    valid_mask = (labels_flat != -100).astype(loss_per_token.dtype)

    # Apply mask and compute mean over valid tokens only
    masked_loss = loss_per_token * valid_mask
    loss = mx.sum(masked_loss) / (mx.sum(valid_mask) + 1e-8)

    return loss


def clip_gradients(grads: Dict[str, mx.array], max_norm: float) -> Tuple[Dict[str, mx.array], float]:
    """
    Clip gradients by global norm.

    Args:
        grads: Dictionary of gradients (can be nested)
        max_norm: Maximum gradient norm

    Returns:
        Tuple of (clipped_gradients, total_norm)
    """
    # Compute total norm recursively
    def compute_norm(g):
        """Compute norm of a gradient value (handles nested dicts)."""
        if isinstance(g, dict):
            norm = 0.0
            for v in g.values():
                norm += compute_norm(v)
            return norm
        elif isinstance(g, (list, tuple)):
            return 0.0  # Skip non-tensor types
        else:
            # mx.array
            return mx.sum(g * g)

    total_norm = 0.0
    for g in grads.values():
        total_norm += compute_norm(g)
    total_norm = mx.sqrt(total_norm)

    # Clip if needed
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = mx.minimum(clip_coef, mx.array(1.0))

    # Apply clipping recursively
    def clip_grad(g, coef):
        """Apply clipping to a gradient value (handles nested dicts)."""
        if isinstance(g, dict):
            result = {}
            for k, v in g.items():
                result[k] = clip_grad(v, coef)
            return result
        elif isinstance(g, (list, tuple)):
            return g  # Return as-is
        else:
            # mx.array
            return g * coef

    clipped_grads = {}
    for k, g in grads.items():
        clipped_grads[k] = clip_grad(g, clip_coef)

    return clipped_grads, float(total_norm)


class Trainer:
    """
    Main training engine.

    Manages:
    - Training loop
    - Optimization
    - Checkpointing
    - Metrics tracking
    - State management
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config,
        logger,
        memory_monitor
    ):
        """
        Initialize trainer.

        Args:
            model: MLX model with LoRA
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Config object
            logger: StructuredLogger
            memory_monitor: MemoryMonitor
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger
        self.memory_monitor = memory_monitor

        # Training state
        self.state = TrainingState(
            epoch=0,
            global_step=0,
            best_val_loss=float('inf'),
            train_loss_history=[],
            val_loss_history=[],
            learning_rates=[]
        )

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Calculate total steps
        steps_per_epoch = len(train_loader)
        if config.training.max_steps > 0:
            self.total_steps = config.training.max_steps
        else:
            self.total_steps = steps_per_epoch * config.training.num_epochs

        # Setup scheduler
        self.scheduler = CosineWarmupScheduler(
            optimizer=self.optimizer,
            warmup_steps=config.training.warmup_steps,
            total_steps=self.total_steps
        )

        # Gradient accumulator
        self.grad_accumulator = GradientAccumulator(
            accumulation_steps=config.training.gradient_accumulation_steps
        )

        # Checkpoint directory
        self.checkpoint_dir = Path(config.output.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Trainer initialized:")
        self.logger.info(f"  Total steps: {self.total_steps}")
        self.logger.info(f"  Steps per epoch: {steps_per_epoch}")
        self.logger.info(f"  Warmup steps: {config.training.warmup_steps}")
        self.logger.info(f"  Gradient accumulation: {config.training.gradient_accumulation_steps}")
        self.logger.info(f"  Effective batch size: "
                        f"{config.training.per_device_batch_size * config.training.gradient_accumulation_steps}")

    def _create_optimizer(self):
        """Create AdamW optimizer."""
        # Get trainable parameters
        trainable_params = {}
        for name, param in self.model.parameters().items():
            if 'lora' in name.lower():
                trainable_params[name] = param

        self.logger.info(f"Optimizer will update {len(trainable_params)} parameter tensors")

        # Create AdamW optimizer
        optimizer = optim.AdamW(
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )

        return optimizer

    def loss_fn(self, model, batch):
        """
        Loss function for gradient computation.

        Args:
            model: Model
            batch: Training batch

        Returns:
            Loss value
        """
        # Forward pass - MLX models don't require attention_mask as a separate parameter
        # The model handles masking internally through the token embeddings
        logits = model(batch.input_ids)

        # Compute loss
        loss = compute_loss(logits, batch.labels, batch.attention_mask)

        return loss

    def train_step(self, batch) -> StepMetrics:
        """
        Execute single training step.

        Args:
            batch: Training batch

        Returns:
            StepMetrics object
        """
        step_start = time.time()

        # Define value and gradient function
        loss_and_grad_fn = nn.value_and_grad(self.model, self.loss_fn)

        # Forward and backward pass
        loss, grads = loss_and_grad_fn(self.model, batch)

        # Evaluate to get actual values
        mx.eval(loss)
        mx.eval(grads)

        # Accumulate gradients
        self.grad_accumulator.accumulate(grads)

        # Update if accumulation complete
        if self.grad_accumulator.should_update():
            # Get averaged gradients
            avg_grads = self.grad_accumulator.get_gradients()

            # Clip gradients
            clipped_grads, grad_norm = clip_gradients(
                avg_grads,
                self.config.training.max_grad_norm
            )

            # Optimizer step
            self.optimizer.update(self.model, clipped_grads)

            # Scheduler step
            self.scheduler.step()

            # Reset accumulator
            self.grad_accumulator.reset()

            # Increment global step
            self.state.global_step += 1

        # Calculate metrics
        step_time = time.time() - step_start
        learning_rate = self.scheduler.get_lr()

        # Estimate tokens per second
        batch_size = batch.input_ids.shape[0]
        seq_length = batch.input_ids.shape[1]
        tokens = batch_size * seq_length
        tokens_per_sec = tokens / step_time if step_time > 0 else 0

        metrics = StepMetrics(
            loss=float(loss),
            learning_rate=learning_rate,
            step_time=step_time,
            tokens_per_sec=tokens_per_sec
        )

        return metrics

    def evaluate(self) -> float:
        """
        Run validation evaluation.

        Returns:
            Average validation loss
        """
        self.logger.info("Running validation...")

        self.model.eval()  # Set to eval mode

        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            # Forward pass (no gradients)
            logits = self.model(batch.input_ids, batch.attention_mask)
            loss = compute_loss(logits, batch.labels, batch.attention_mask)

            mx.eval(loss)

            total_loss += float(loss)
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        self.model.train()  # Back to train mode

        self.logger.info(f"Validation loss: {avg_loss:.4f}")

        return avg_loss

    def save_checkpoint(self, is_best: bool = False):
        """
        Save training checkpoint.

        Args:
            is_best: Whether this is the best checkpoint so far
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{self.state.global_step}.npz"

        self.logger.info(f"Saving checkpoint: {checkpoint_path}")

        # Get model parameters (only LoRA)
        lora_params = {}
        for name, param in self.model.parameters().items():
            if 'lora' in name.lower():
                lora_params[name] = np.array(param)

        # Save state
        checkpoint_data = {
            'model_params': lora_params,
            'optimizer_lr': self.optimizer.learning_rate,
            'state': asdict(self.state)
        }

        # Save to file
        with open(checkpoint_path, 'wb') as f:
            np.savez(f, **{
                'model_params': checkpoint_data['model_params'],
                'optimizer_lr': checkpoint_data['optimizer_lr'],
                'state': json.dumps(checkpoint_data['state'])
            })

        # Save as best if needed
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.npz"
            import shutil
            shutil.copy(checkpoint_path, best_path)
            self.logger.info(f"Saved best checkpoint: {best_path}")

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only last N."""
        checkpoints = sorted(
            [f for f in self.checkpoint_dir.glob("checkpoint_step_*.npz")],
            key=lambda x: int(x.stem.split('_')[-1])
        )

        keep_n = self.config.output.keep_last_n_checkpoints
        if len(checkpoints) > keep_n:
            for checkpoint in checkpoints[:-keep_n]:
                checkpoint.unlink()
                self.logger.debug(f"Removed old checkpoint: {checkpoint}")

    def train(self):
        """
        Main training loop.

        Returns:
            Final training state
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting Training")
        self.logger.info("=" * 80)

        self.memory_monitor.log_memory("before_training")

        training_start_time = time.time()

        for epoch in range(self.config.training.num_epochs):
            self.state.epoch = epoch

            self.logger.info(f"\nEpoch {epoch + 1}/{self.config.training.num_epochs}")
            self.logger.info("-" * 80)

            epoch_start_time = time.time()
            epoch_loss = 0.0
            epoch_steps = 0

            # Training loop
            self.model.train()

            for batch_idx, batch in enumerate(self.train_loader):
                # Train step
                metrics = self.train_step(batch)

                epoch_loss += metrics.loss
                epoch_steps += 1

                # Logging
                if self.state.global_step % self.config.training.logging_steps == 0:
                    self.logger.metrics(
                        step=self.state.global_step,
                        metrics={
                            'loss': metrics.loss,
                            'lr': metrics.learning_rate,
                            'step_time': metrics.step_time,
                            'tokens_per_sec': metrics.tokens_per_sec
                        }
                    )

                    self.state.train_loss_history.append(metrics.loss)
                    self.state.learning_rates.append(metrics.learning_rate)

                # Evaluation
                if self.state.global_step % self.config.training.eval_steps == 0 and self.state.global_step > 0:
                    val_loss = self.evaluate()
                    self.state.val_loss_history.append(val_loss)

                    # Check if best
                    is_best = val_loss < self.state.best_val_loss
                    if is_best:
                        self.state.best_val_loss = val_loss
                        self.logger.info(f"New best validation loss: {val_loss:.4f}")

                # Checkpointing
                if self.state.global_step % self.config.training.save_steps == 0 and self.state.global_step > 0:
                    is_best = (self.state.val_loss_history and
                             self.state.val_loss_history[-1] == self.state.best_val_loss)
                    self.save_checkpoint(is_best=is_best)

                    self.memory_monitor.log_memory(f"step_{self.state.global_step}")

                # Check max steps
                if self.config.training.max_steps > 0 and self.state.global_step >= self.config.training.max_steps:
                    self.logger.info(f"Reached max steps: {self.config.training.max_steps}")
                    break

            # Epoch summary
            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0.0

            # Final validation for epoch
            val_loss = self.evaluate()

            epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': avg_epoch_loss,
                'val_loss': val_loss,
                'epoch_time': epoch_time,
                'steps': epoch_steps
            }

            self.logger.epoch_summary(epoch + 1, epoch_metrics)

            # Check if best and save
            is_best = val_loss < self.state.best_val_loss
            if is_best:
                self.state.best_val_loss = val_loss

            self.save_checkpoint(is_best=is_best)

        # Training complete
        total_time = time.time() - training_start_time
        self.logger.info("=" * 80)
        self.logger.info("Training Complete!")
        self.logger.info(f"Total time: {total_time / 3600:.2f} hours")
        self.logger.info(f"Best validation loss: {self.state.best_val_loss:.4f}")
        self.logger.info("=" * 80)

        self.memory_monitor.log_memory("after_training")

        peak_memory = self.memory_monitor.get_peak_usage()
        self.logger.info(f"Peak RAM usage: {peak_memory['peak_ram_gb']:.2f} GB")
        if peak_memory['peak_metal_gb']:
            self.logger.info(f"Peak Metal usage: {peak_memory['peak_metal_gb']:.2f} GB")

        return self.state
