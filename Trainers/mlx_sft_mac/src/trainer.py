"""
SFT Trainer for MLX.
Simple cross-entropy loss training with LoRA adapters.
"""

import time
import math
import json
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from data_loader import SFTDataLoader, SFTDataset


@dataclass
class TrainingState:
    """Training state for checkpointing."""
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float('inf')
    train_losses: list = None

    def __post_init__(self):
        if self.train_losses is None:
            self.train_losses = []


class CosineWarmupScheduler:
    """Cosine learning rate schedule with linear warmup."""

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0
    ):
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
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        self.optimizer.learning_rate = lr

    def get_lr(self) -> float:
        return float(self.optimizer.learning_rate)


def compute_cross_entropy_loss(logits: mx.array, labels: mx.array) -> mx.array:
    """
    Compute cross-entropy loss for language modeling.

    Args:
        logits: (batch_size, seq_length, vocab_size)
        labels: (batch_size, seq_length) with -100 for padding

    Returns:
        Scalar loss
    """
    batch_size, seq_length, vocab_size = logits.shape

    # Shift for next-token prediction
    # logits: predict token at position i
    # labels: target is token at position i+1
    shift_logits = logits[:, :-1, :]  # (batch, seq-1, vocab)
    shift_labels = labels[:, 1:]       # (batch, seq-1)

    # Flatten
    shift_logits = shift_logits.reshape(-1, vocab_size)
    shift_labels = shift_labels.reshape(-1)

    # Compute cross-entropy
    log_probs = nn.log_softmax(shift_logits, axis=-1)

    # Gather log probs for target tokens
    # Create indices for gathering
    indices = mx.arange(shift_labels.shape[0])
    target_log_probs = log_probs[indices, shift_labels]

    # Mask padding tokens (-100)
    mask = (shift_labels != -100).astype(target_log_probs.dtype)
    masked_log_probs = target_log_probs * mask

    # Mean over valid tokens
    loss = -mx.sum(masked_log_probs) / (mx.sum(mask) + 1e-8)

    return loss


def clip_gradients(grads: dict, max_norm: float) -> tuple:
    """Clip gradients by global norm."""

    def compute_norm(g):
        if isinstance(g, dict):
            return sum(compute_norm(v) for v in g.values())
        elif isinstance(g, (list, tuple)):
            return 0.0
        else:
            return mx.sum(g * g)

    total_norm_sq = sum(compute_norm(g) for g in grads.values())
    total_norm = mx.sqrt(total_norm_sq)

    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = mx.minimum(clip_coef, mx.array(1.0))

    def clip_grad(g, coef):
        if isinstance(g, dict):
            return {k: clip_grad(v, coef) for k, v in g.items()}
        elif isinstance(g, (list, tuple)):
            return g
        else:
            return g * coef

    clipped = {k: clip_grad(g, clip_coef) for k, g in grads.items()}
    return clipped, float(total_norm)


class SFTTrainer:
    """
    SFT Trainer for MLX.

    Uses standard cross-entropy loss for supervised fine-tuning.
    """

    def __init__(
        self,
        model,
        tokenizer,
        train_dataset: SFTDataset,
        eval_dataset: Optional[SFTDataset],
        config: dict,
        run_dir: Path,
        max_steps: Optional[int] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.run_dir = Path(run_dir)
        self.max_steps = max_steps

        # Training config
        train_cfg = config['training']
        self.batch_size = train_cfg['per_device_batch_size']
        self.grad_accum_steps = train_cfg['gradient_accumulation_steps']
        self.learning_rate = train_cfg['learning_rate']
        self.warmup_steps = train_cfg['warmup_steps']
        self.max_grad_norm = train_cfg['max_grad_norm']
        self.num_epochs = train_cfg['num_epochs']
        self.logging_steps = train_cfg['logging_steps']
        self.save_steps = train_cfg['save_steps']
        self.eval_steps = train_cfg.get('eval_steps', self.save_steps)

        # Create data loaders
        self.train_loader = SFTDataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            seed=config['data']['seed']
        )

        self.eval_loader = None
        if eval_dataset:
            self.eval_loader = SFTDataLoader(
                eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                seed=config['data']['seed']
            )

        # Calculate total steps
        steps_per_epoch = len(self.train_loader)
        if max_steps:
            self.total_steps = max_steps
        else:
            self.total_steps = steps_per_epoch * self.num_epochs

        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Total steps: {self.total_steps}")

        # Setup optimizer
        self.optimizer = optim.AdamW(
            learning_rate=self.learning_rate,
            weight_decay=config['training']['weight_decay']
        )

        # Setup scheduler
        self.scheduler = CosineWarmupScheduler(
            optimizer=self.optimizer,
            warmup_steps=self.warmup_steps,
            total_steps=self.total_steps
        )

        # Training state
        self.state = TrainingState()

        # Logging
        self.logs_dir = self.run_dir / config['output']['logs_dir']
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.logs_dir / "training.jsonl"

        # Checkpoints
        self.checkpoints_dir = self.run_dir / config['output']['checkpoint_dir']
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def compute_loss(self, model, input_ids, labels):
        """Compute loss for a batch - static method for gradient computation."""
        # Forward pass
        logits = model(input_ids)

        # Compute loss
        loss = compute_cross_entropy_loss(logits, labels)

        return loss

    def train_step(self, batch) -> dict:
        """Execute a single training step."""
        step_start = time.time()

        # Create loss function that takes model as first argument
        def loss_fn(model, input_ids, labels):
            logits = model(input_ids)
            return compute_cross_entropy_loss(logits, labels)

        # Value and gradient function
        loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)

        # Compute loss and gradients
        loss, grads = loss_and_grad_fn(self.model, batch.input_ids, batch.labels)

        # Evaluate to materialize values
        mx.eval(loss)
        mx.eval(grads)

        # Clip gradients
        grads, grad_norm = clip_gradients(grads, self.max_grad_norm)

        # Update model
        self.optimizer.update(self.model, grads)

        # Update scheduler
        self.scheduler.step()

        # Increment step
        self.state.global_step += 1

        step_time = time.time() - step_start

        return {
            'loss': float(loss),
            'grad_norm': grad_norm,
            'lr': self.scheduler.get_lr(),
            'step_time': step_time
        }

    def evaluate(self) -> float:
        """Run evaluation on validation set."""
        if self.eval_loader is None:
            return float('inf')

        total_loss = 0.0
        num_batches = 0

        for batch in self.eval_loader:
            logits = self.model(batch.input_ids)
            loss = compute_cross_entropy_loss(logits, batch.labels)
            mx.eval(loss)

            total_loss += float(loss)
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def log_metrics(self, metrics: dict):
        """Log metrics to file and console."""
        # Add timestamp and step
        log_entry = {
            'step': self.state.global_step,
            'epoch': self.state.epoch,
            **metrics
        }

        # Write to JSONL
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        # Console output
        metrics_str = " | ".join([
            f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in metrics.items()
        ])
        print(f"Step {self.state.global_step} | {metrics_str}")

    def save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint."""
        from model_loader import save_lora_adapters

        checkpoint_path = self.checkpoints_dir / f"checkpoint_{self.state.global_step}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save LoRA adapters
        save_lora_adapters(self.model, str(checkpoint_path))

        # Save training state
        state_path = checkpoint_path / "training_state.json"
        with open(state_path, 'w') as f:
            json.dump({
                'epoch': self.state.epoch,
                'global_step': self.state.global_step,
                'best_val_loss': self.state.best_val_loss,
            }, f)

        print(f"[OK] Saved checkpoint: {checkpoint_path}")

        if is_best:
            best_path = self.checkpoints_dir / "best"
            best_path.mkdir(parents=True, exist_ok=True)
            save_lora_adapters(self.model, str(best_path))
            print(f"[OK] Saved best checkpoint: {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and resume training."""
        from model_loader import load_lora_adapters

        checkpoint_path = Path(checkpoint_path)

        # Load LoRA adapters
        load_lora_adapters(self.model, str(checkpoint_path))

        # Load training state
        state_path = checkpoint_path / "training_state.json"
        if state_path.exists():
            with open(state_path, 'r') as f:
                state = json.load(f)
                self.state.epoch = state['epoch']
                self.state.global_step = state['global_step']
                self.state.best_val_loss = state['best_val_loss']

        print(f"[OK] Resumed from step {self.state.global_step}")

    def train(self):
        """Main training loop."""
        print(f"\nStarting training for {self.num_epochs} epochs ({self.total_steps} steps)")
        print(f"Logging every {self.logging_steps} steps")
        print(f"Saving every {self.save_steps} steps\n")

        training_start = time.time()

        for epoch in range(self.num_epochs):
            self.state.epoch = epoch
            epoch_loss = 0.0
            epoch_steps = 0

            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 60)

            for batch in self.train_loader:
                # Train step
                metrics = self.train_step(batch)

                epoch_loss += metrics['loss']
                epoch_steps += 1
                self.state.train_losses.append(metrics['loss'])

                # Logging
                if self.state.global_step % self.logging_steps == 0:
                    self.log_metrics(metrics)

                # Evaluation
                if self.eval_loader and self.state.global_step % self.eval_steps == 0:
                    val_loss = self.evaluate()
                    print(f"  Validation loss: {val_loss:.6f}")

                    is_best = val_loss < self.state.best_val_loss
                    if is_best:
                        self.state.best_val_loss = val_loss
                        print(f"  New best validation loss!")

                # Checkpointing
                if self.state.global_step % self.save_steps == 0:
                    is_best = (self.eval_loader is not None and
                              self.state.best_val_loss < float('inf'))
                    self.save_checkpoint(is_best=is_best)

                # Check max steps
                if self.max_steps and self.state.global_step >= self.max_steps:
                    print(f"\nReached max_steps ({self.max_steps})")
                    break

            # Epoch summary
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            print(f"\nEpoch {epoch + 1} complete: avg loss = {avg_epoch_loss:.6f}")

            # Check max steps
            if self.max_steps and self.state.global_step >= self.max_steps:
                break

        # Training complete
        total_time = time.time() - training_start
        print(f"\nTraining complete!")
        print(f"  Total steps: {self.state.global_step}")
        print(f"  Total time: {total_time / 60:.1f} minutes")
        if self.state.best_val_loss < float('inf'):
            print(f"  Best validation loss: {self.state.best_val_loss:.6f}")

        # Save final checkpoint
        self.save_checkpoint(is_best=False)

    def save_model(self, save_path: str):
        """Save the final model."""
        from model_loader import save_lora_adapters

        save_lora_adapters(self.model, save_path)

        # Also save tokenizer
        tokenizer_path = Path(save_path) / "tokenizer"
        tokenizer_path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(str(tokenizer_path))

        print(f"[OK] Model and tokenizer saved to: {save_path}")
