"""Two-stage LR reduction callback, hoisted from sft + kto (byte-identical)."""

from __future__ import annotations

from transformers import TrainerCallback


class TwoStageLRCallback(TrainerCallback):
    """Reduce LR at a specified step to prevent optimization instability.

    Example:
        Steps 1-50:  LR = 5e-7 (fast early learning)
        Steps 51+:   LR = 2.5e-7 (reduced to prevent overshoot/instability)
    """

    def __init__(self, initial_lr: float, reduced_lr: float, reduction_step: int):
        self.initial_lr = initial_lr
        self.reduced_lr = reduced_lr
        self.reduction_step = reduction_step
        self.lr_reduced = False

    def on_train_begin(self, args, state, control, **kwargs):
        print("\n" + "=" * 100)
        print("TWO-STAGE LEARNING RATE SCHEDULE")
        print("=" * 100)
        print(f"  Steps 1-{self.reduction_step}:  LR = {self.initial_lr:.2e} (fast early learning)")
        print(f"  Steps {self.reduction_step + 1}+:     LR = {self.reduced_lr:.2e} ({(self.reduced_lr / self.initial_lr) * 100:.0f}% of initial, prevents instability)")
        print(f"  Reduction ratio: {self.reduced_lr / self.initial_lr:.1%}")
        print("=" * 100 + "\n")

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == self.reduction_step and not self.lr_reduced:
            optimizer = kwargs.get("optimizer")
            if optimizer is not None:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = self.reduced_lr
                self.lr_reduced = True
                print("\n" + "!" * 100)
                print(f"🔧 LEARNING RATE REDUCED at step {state.global_step}")
                print(f"   {self.initial_lr:.2e} → {self.reduced_lr:.2e} (50% reduction)")
                print(f"   Reason: Preemptive intervention before step 60 instability zone")
                print("!" * 100 + "\n")
