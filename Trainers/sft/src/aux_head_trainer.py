"""
Trainer subclass that trains an auxiliary scalar readout head (Phase A).

Location: Trainers/sft/src/aux_head_trainer.py

Summary
-------
``AuxHeadTrainer`` subclasses the stock ``transformers.Trainer`` used by the SFT
path. It freezes the base (and any LoRA) so only the appended :class:`AuxHead`
trains, enables hidden states inside ``compute_loss``, reduces the configured
layer at the configured token position, and supervises the head with a
proper-scoring loss against the per-row ``aux_target`` carried by the collator.

How it is used
--------------
``train_sft.run`` constructs this in place of ``Trainer`` only when
``config.aux_head.enabled`` is true; otherwise the off-path is byte-identical to
the stock trainer. After ``train()`` the caller saves the head via
``aux_head.save_aux_head``.

Phase A vs Phase B
------------------
Phase A loss IS the head loss alone (no LM term). The base + LoRA are frozen and
the optimizer is built over the head's parameters only. The single Phase-B seam
(``loss = outputs.loss + lm_loss_weight * head_loss``) is marked with a one-line
comment in ``compute_loss``; flipping ``freeze_base``/``lm_loss_weight`` is the
only change Phase B needs here.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from transformers import Trainer

from aux_head import AuxHead, compute_aux_head_loss, reduce_hidden_states


class AuxHeadTrainer(Trainer):
    """Stock ``Trainer`` that trains only an :class:`AuxHead` over a frozen base.

    Args (beyond the stock ``Trainer`` kwargs):
        aux_head: the head module to train.
        aux_head_config: the resolved ``AuxHeadConfig`` (layer, token_position,
            target_field, loss, freeze_base, lm_loss_weight, head_lr).
    """

    def __init__(self, *args: Any, aux_head: AuxHead, aux_head_config: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.aux_head = aux_head
        self.aux_head_config = aux_head_config

        # Place the head on the base model's device (dtype stays the head's own,
        # so a bf16 base can feed an fp32 head; AuxHead.forward casts the input).
        try:
            ref = next(self.model.parameters())
            self.aux_head = self.aux_head.to(device=ref.device)
        except StopIteration:
            pass

        if getattr(aux_head_config, "freeze_base", True):
            self._freeze_base_keep_head()

        self._log_trainable_param_accounting()

    def _freeze_base_keep_head(self) -> None:
        """Freeze every base/LoRA parameter so ONLY the head accumulates gradient.

        Mirrors the embedding ``frozen_head`` precedent's requires_grad mechanics:
        freeze all of ``self.model``; the head (held separately, not a submodule
        of ``model``) stays trainable.
        """
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.aux_head.parameters():
            param.requires_grad = True

    def _log_trainable_param_accounting(self) -> None:
        """Log trainable-param counts (mirror model_loader accounting) for audit."""
        base_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        head_trainable = sum(p.numel() for p in self.aux_head.parameters() if p.requires_grad)
        if getattr(self.aux_head_config, "freeze_base", True) and base_trainable != 0:
            # Loud: Phase A must train the head ONLY.
            print(
                f"[aux_head][WARN] freeze_base=true but base reports {base_trainable} trainable "
                f"params (expected 0). Only the head should train in Phase A."
            )
        print(
            f"[aux_head] trainable params -> head: {head_trainable:,} | base: {base_trainable:,} "
            f"(freeze_base={getattr(self.aux_head_config, 'freeze_base', True)})"
        )

    def create_optimizer(self, model=None):  # type: ignore[override]
        """Build the optimizer over the HEAD's parameters.

        The head is not a submodule of ``self.model``, so stock
        ``create_optimizer`` (which walks ``model.parameters()``) would never see
        it. We override to register the head's params explicitly, optionally on a
        dedicated ``head_lr`` learning rate. Phase B (``freeze_base=false``) would
        additionally add the unfrozen LoRA params here.
        """
        if self.optimizer is not None:
            return self.optimizer

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        head_lr = getattr(self.aux_head_config, "head_lr", None)
        head_params = [p for p in self.aux_head.parameters() if p.requires_grad]
        param_group = {"params": head_params}
        if head_lr is not None:
            param_group["lr"] = head_lr
        self.optimizer = optimizer_cls([param_group], **optimizer_kwargs)
        return self.optimizer

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        """Phase A: loss = proper_score(head(reduced hidden state), aux_target)."""
        cfg = self.aux_head_config

        # The collator stacks the per-row target under "aux_target"; pop it so it
        # is never passed into the language model forward.
        aux_target = inputs.pop("aux_target", None)
        if aux_target is None:
            raise ValueError(
                "AuxHeadTrainer.compute_loss received a batch without 'aux_target'. "
                "Ensure aux_head.target_field is set and the preprocessing/collator "
                "plumbing carried the per-row target through."
            )

        # Enable hidden states HERE (no global flag); base is frozen so this
        # forward accrues gradient only through the head below.
        outputs = model(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states[cfg.layer]
        reduced = reduce_hidden_states(hidden, inputs["attention_mask"], cfg.token_position)
        pred = self.aux_head(reduced)

        target = aux_target.to(pred.device)
        head_loss = compute_aux_head_loss(pred, target, cfg.loss)

        # Phase A: the head loss is the ENTIRE loss (no LM term).
        # Phase B seam (do NOT enable here): loss = outputs.loss + cfg.lm_loss_weight * head_loss
        loss = head_loss

        return (loss, outputs) if return_outputs else loss
