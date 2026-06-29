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
Phase A (``freeze_base=true``, ``lm_loss_weight=0``): the base + LoRA are frozen,
the optimizer is built over the head's parameters only, and the loss IS the head
loss alone. Phase B (``freeze_base=false``, ``lm_loss_weight>0``): the base stays
unfrozen (PEFT's LoRA params remain trainable), the optimizer adds those params
as a second group, and the loss is ``outputs.loss + lm_loss_weight * head_loss``,
co-training the base with the head. The Phase-A path is byte-identical regardless
of the Phase-B code being present (both branches gate on the config values).
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from transformers import Trainer

from aux_head import (
    AuxHead,
    compute_aux_head_loss,
    prompt_end_indices,
    reduce_hidden_states,
)


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

        # Fail loud on the load-bearing dependency, not deep inside compute_loss.
        # The per-row ``aux_target`` column survives into the collator ONLY because
        # the SFT path sets remove_unused_columns=False; HF would otherwise strip
        # any column not in the model's forward signature. Flipping that flag would
        # surface as an opaque "batch without 'aux_target'" error far from the
        # toggle, so assert it here where the cause is obvious.
        if self.args.remove_unused_columns is not False:
            raise ValueError(
                "AuxHeadTrainer requires TrainingArguments.remove_unused_columns=False "
                "so the per-row 'aux_target' column survives into the data collator and "
                f"compute_loss; got remove_unused_columns={self.args.remove_unused_columns!r}. "
                "Do not flip this when aux_head is enabled."
            )

        # Place the head on the base model's device (dtype stays the head's own,
        # so a bf16 base can feed an fp32 head; AuxHead.forward casts the input).
        try:
            ref = next(self.model.parameters())
            self.aux_head = self.aux_head.to(device=ref.device)
        except StopIteration:
            pass

        if getattr(aux_head_config, "freeze_base", True):
            self._freeze_base_keep_head()
        else:
            self._prepare_unfrozen_base_keep_head()

        self._log_trainable_param_accounting()

    def train(self, resume_from_checkpoint=None, *args: Any, **kwargs: Any):  # type: ignore[override]
        """Fail loud on resume — Phase A does not support ``resume_from_checkpoint``.

        The head is held OUTSIDE ``self.model`` and persisted ONLY as a post-train
        sidecar (``save_aux_head``). HF's per-step checkpoints serialize
        ``self.model`` + ``optimizer.pt`` + ``scheduler.pt`` but NOT the head module,
        so resuming would reconstruct the head fresh (random) in ``run()`` while
        reloading STALE head-optimizer momentum onto it — silently corrupting the
        head. Until the head is hooked into checkpoint save/load (out of Phase-A
        scope), refuse to resume rather than train a corrupted head.

        The guard lives on the trainer (not only at the call site) so it protects
        every caller, and it is observed HERE because ``resume_from_checkpoint``
        arrives as a ``train()`` argument, not a ``TrainingArguments`` field.
        """
        if resume_from_checkpoint:
            raise RuntimeError(
                "AuxHeadTrainer does not support resume_from_checkpoint: the aux_head is "
                "not captured by HF per-step checkpoints (it is sidecar-saved after "
                "train() only), so resuming would reinitialize the head while reapplying "
                "stale head-optimizer state. Phase A does not support resume — restart "
                "training from scratch or clear resume_from_checkpoint."
            )
        return super().train(resume_from_checkpoint, *args, **kwargs)

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

    def _prepare_unfrozen_base_keep_head(self) -> None:
        """Phase B: co-train the unfrozen base alongside the head.

        Leave the base/adapter ``requires_grad`` exactly as PEFT set them (the
        LoRA params stay trainable) — do NOT freeze. Only ensure the head's params
        are trainable. ``create_optimizer`` then registers both the head group and
        the base's trainable (LoRA) params.

        Defensive belt for the gradient-checkpointing path: ``output_hidden_states``
        + checkpointing + LoRA can detach the recomputed hidden state the head
        reads. ``enable_input_require_grads`` registers a forward hook that keeps
        the input-embedding output requiring grad, so the autograd graph reaches
        the adapter through the recompute boundary. It is a no-op under vanilla
        ``torch`` checkpointing and harmless otherwise; ``use_cache`` is left to
        the framework.
        """
        for param in self.aux_head.parameters():
            param.requires_grad = True
        enable_input_require_grads = getattr(self.model, "enable_input_require_grads", None)
        if callable(enable_input_require_grads):
            enable_input_require_grads()

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
        dedicated ``head_lr`` learning rate. Phase B (``freeze_base=false``)
        additionally adds the base's trainable (LoRA) params as a second group at
        the trainer learning rate.
        """
        if self.optimizer is not None:
            return self.optimizer

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        head_lr = getattr(self.aux_head_config, "head_lr", None)
        head_params = [p for p in self.aux_head.parameters() if p.requires_grad]
        head_group = {"params": head_params}
        if head_lr is not None:
            head_group["lr"] = head_lr
        param_groups = [head_group]

        # Phase B: also optimize the unfrozen base (the LoRA params PEFT left
        # trainable). No explicit lr ⇒ this group inherits the optimizer default
        # (the trainer learning_rate carried in optimizer_kwargs). The head group
        # keeps its optional head_lr override.
        if not getattr(self.aux_head_config, "freeze_base", True):
            base_params = [p for p in self.model.parameters() if p.requires_grad]
            if base_params:
                param_groups.append({"params": base_params})

        self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
        return self.optimizer

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        """Head loss, optionally combined with the LM loss (Phase B).

        Phase A (``lm_loss_weight == 0``): ``loss = proper_score(head, aux_target)``
        alone. Phase B (``lm_loss_weight > 0``): ``loss = outputs.loss +
        lm_loss_weight * head_loss``, co-training the unfrozen base with the head.
        """
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

        # For "end_of_prompt", recover the per-row prompt/completion boundary from
        # the labels (prompt tokens are masked to -100). Computed HERE because the
        # batch still carries "labels"; passed into the reduction below.
        prompt_end_idx = None
        if cfg.token_position == "end_of_prompt":
            labels = inputs.get("labels")
            if labels is None:
                raise ValueError(
                    "token_position='end_of_prompt' requires 'labels' in the batch to "
                    "locate the prompt/completion boundary; none were present."
                )
            prompt_end_idx = prompt_end_indices(labels, inputs["attention_mask"])

        # Enable hidden states HERE (no global flag). In Phase A the base is frozen
        # so this forward accrues gradient only through the head; in Phase B the
        # unfrozen base co-trains through both the head loss and the LM loss.
        outputs = model(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states[cfg.layer]
        reduced = reduce_hidden_states(
            hidden, inputs["attention_mask"], cfg.token_position, prompt_end_idx=prompt_end_idx
        )
        pred = self.aux_head(reduced)

        target = aux_target.to(pred.device)
        head_loss = compute_aux_head_loss(pred, target, cfg.loss)

        # Phase A (lm_loss_weight == 0): the head loss is the ENTIRE loss, leaving
        # the off-path byte-identical. Phase B (> 0): add the weighted LM loss
        # (already computed because "labels" stayed in the forward).
        lm_loss_weight = getattr(cfg, "lm_loss_weight", 0.0)
        if lm_loss_weight > 0:
            loss = outputs.loss + lm_loss_weight * head_loss
        else:
            loss = head_loss

        return (loss, outputs) if return_outputs else loss
