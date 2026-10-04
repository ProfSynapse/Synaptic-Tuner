"""
HF Trainer subclass for decision models.

Location: Trainers/decision/decision_core/trainer.py
Used by:  train_decision.py.

Riding the HF Trainer keeps the repo's shared callbacks, schedulers, gradient
accumulation, checkpointing and ``trainer.state`` lineage extraction. The
deltas are:

- ``compute_loss`` runs the decision forward and ``losses.decision_loss``.
- Two optimizer groups: LoRA at ``learning_rate``, the randomly initialised
  pointer head at ``head_learning_rate`` (a head that starts as noise needs a
  bigger step or the torso adapts to noise).
- Length-grouped sampling from precomputed prompt lengths.
- ``_save`` writes only the adapter + head + config (never the 2B base weights).
- The eval loader uses a canonical (unshuffled) collator.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer

from .collate import DecisionCollator
from .losses import LossConfig, decision_loss

TENSOR_KEYS = ("input_ids", "attention_mask", "option_index", "answer_index", "n_options")


class ExampleDataset(Dataset):
    def __init__(self, examples: list[Any]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> Any:
        return self.examples[i]


class FrozenReadout:
    """Callable wrapper so losses.frozen_kl can query the base torso's marker readout."""

    def __init__(self, model: Any):
        self.model = model
        self.max_options = model.max_marker_options

    def __call__(self, **batch: torch.Tensor) -> torch.Tensor:
        return self.model.frozen_marker_logits(**batch)


class DecisionTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        loss_config: LossConfig,
        head_learning_rate: float,
        eval_collator: DecisionCollator | None = None,
        train_lengths: list[int] | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.loss_config = loss_config
        self.head_learning_rate = head_learning_rate
        self.eval_collator = eval_collator
        self.train_lengths = train_lengths
        # compute_loss returns a per-micro-batch mean; let the Trainer scale it
        # by gradient_accumulation_steps as it does for ordinary losses.
        self.model_accepts_loss_kwargs = False
        self._part_sums: dict[str, float] = defaultdict(float)
        self._part_count = 0

    # ---- loss ---------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        tensors = {k: inputs[k] for k in TENSOR_KEYS}
        logits = model(**tensors)
        base = self.accelerator.unwrap_model(model)
        frozen = FrozenReadout(base) if self.loss_config.kl_frozen_weight > 0 and model.training else None
        loss, parts = decision_loss(logits, inputs, self.loss_config, frozen)
        if model.training:
            for key, value in parts.items():
                self._part_sums[key] += value
            self._part_count += 1
        return (loss, logits) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            tensors = {k: inputs[k] for k in TENSOR_KEYS}
            logits = model(**tensors)
            loss, _ = decision_loss(logits, inputs, LossConfig(brier_weight=self.loss_config.brier_weight))
        return loss.detach(), None, None

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
        if self._part_count and "loss" in logs:
            for key, total in self._part_sums.items():
                logs[f"loss_{key}"] = round(total / self._part_count, 6)
            self._part_sums.clear()
            self._part_count = 0
        super().log(logs, *args, **kwargs)

    # ---- optimizer ----------------------------------------------------------

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        model = self.accelerator.unwrap_model(self.model) if hasattr(self, "accelerator") else self.model
        head_params = {id(p) for p in (model.head.parameters() if model.head is not None else [])}
        decay, no_decay, head = [], [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in head_params:
                head.append(p)
            elif p.ndim < 2 or "norm" in name or name.endswith(".bias"):
                no_decay.append(p)
            else:
                decay.append(p)
        groups = [
            {"params": decay, "weight_decay": self.args.weight_decay, "lr": self.args.learning_rate},
            {"params": no_decay, "weight_decay": 0.0, "lr": self.args.learning_rate},
            {"params": head, "weight_decay": self.args.weight_decay, "lr": self.head_learning_rate},
        ]
        groups = [g for g in groups if g["params"]]
        self.optimizer = torch.optim.AdamW(groups, betas=(self.args.adam_beta1, self.args.adam_beta2),
                                           eps=self.args.adam_epsilon)
        return self.optimizer

    # ---- data ---------------------------------------------------------------

    def _get_train_sampler(self, *args: Any, **kwargs: Any):
        if self.train_lengths is None:
            return super()._get_train_sampler(*args, **kwargs)
        from transformers.trainer_pt_utils import LengthGroupedSampler

        return LengthGroupedSampler(
            self.args.train_batch_size * self.args.gradient_accumulation_steps,
            lengths=self.train_lengths,
            generator=torch.Generator().manual_seed(self.args.seed),
        )

    def get_eval_dataloader(self, eval_dataset: Any = None) -> DataLoader:
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if self.eval_collator is None or dataset is None:
            return super().get_eval_dataloader(eval_dataset)
        loader = DataLoader(
            dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self.eval_collator,
            num_workers=0,
        )
        return self.accelerator.prepare(loader)

    # ---- checkpoints --------------------------------------------------------

    def _save(self, output_dir: str | None = None, state_dict: Any = None) -> None:
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        self.accelerator.unwrap_model(self.model).save(output_dir)
        torch.save(self.args, f"{output_dir}/training_args.bin")
