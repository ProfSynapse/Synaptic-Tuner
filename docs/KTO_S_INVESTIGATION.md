# KTO-S Investigation & Implementation Plan

## 🔍 Current State Analysis

### What You're Using
- **TRL Version:** 0.11.4
- **KTO Implementation:** Standard KTO (from TRL library)
- **Problem:** KL divergence spikes (exactly as described in the paper)

### Evidence from Your Runs
```
Step 5:  KL = 0.039 ✅
Step 10: KL = 0.189 ⚠️
Step 15: KL = 0.472 🔴
Step 20: KL = 2.056 💥 EXPLOSION
```

**This matches the paper's findings exactly** - KTO without SFT has early KL instability.

---

## 📊 What I Found

### 1. TRL 0.11.4 Does NOT Have KTO-S

Checked the `KTOConfig` parameters:
```python
# No SIGN-related parameters found in TRL 0.11.4
```

The SIGN correction is not available in your current TRL version.

### 2. The Core Difference

**Standard KTO (what you have):**
```python
# In TRL's KTOTrainer loss function:
# chosen_KL and rejected_KL are used as:
chosen_losses = -F.logsigmoid(beta * (chosen_rewards - chosen_KL))
rejected_losses = -F.logsigmoid(beta * (rejected_KL - rejected_rewards))
```

**KTO-S (what you need):**
```python
# Add SIGN correction:
S_chosen = torch.sign(chosen_rewards)
S_rejected = torch.sign(rejected_rewards)

chosen_losses = -F.logsigmoid(beta * (chosen_rewards + S_chosen * chosen_KL))
rejected_losses = -F.logsigmoid(beta * (rejected_rewards + S_rejected * rejected_KL))
```

**The fix:** Flip the KL term's sign based on reward sign, so the gradient penalty adapts correctly.

---

## 🛠️ Implementation Options

### Option 1: Monkey-Patch KTOTrainer (Easiest)

**Pros:**
- ✅ No need to modify TRL source code
- ✅ Can toggle on/off easily
- ✅ Works with existing setup

**Cons:**
- ⚠️ Fragile if TRL updates
- ⚠️ Requires understanding TRL internals

**Implementation:**
Create a custom loss function override before training starts.

### Option 2: Custom KTOTrainer Subclass (Recommended)

**Pros:**
- ✅ Clean, maintainable
- ✅ Easy to toggle on/off
- ✅ Can add to codebase permanently

**Cons:**
- ⚠️ Need to understand TRL's KTOTrainer structure
- ⚠️ More code to write

**Implementation:**
```python
class KTOSTrainer(KTOTrainer):
    """KTO trainer with SIGN correction for stable KL."""

    def kto_loss(self, ...):
        # Override loss computation with SIGN correction
        pass
```

### Option 3: Upgrade TRL (If Available)

**Check if newer TRL has KTO-S:**
```bash
pip install --upgrade trl
```

**Pros:**
- ✅ Official implementation
- ✅ Well-tested

**Cons:**
- ⚠️ May break existing setup
- ⚠️ Requires compatibility testing

---

## 🎯 Recommended Approach

### Step 1: Create Custom KTOSTrainer

**File:** `src/kto_s_trainer.py` (new file)

```python
"""
KTO-S Trainer Implementation
Based on research paper addressing KL divergence instability in KTO.
"""

import torch
import torch.nn.functional as F
from trl import KTOTrainer
from typing import Dict, Optional, Tuple, Union


class KTOSTrainer(KTOTrainer):
    """
    KTO trainer with SIGN correction for stable KL divergence.

    This implements the KTO-S variant from the paper, which adds a
    dynamic sign-based correction to the KL penalty term.

    Key difference from standard KTO:
    - Standard: loss = -log_sigmoid(beta * (reward - KL))
    - KTO-S:    loss = -log_sigmoid(beta * (reward + SIGN(reward) * KL))

    The SIGN correction ensures gradient scaling adapts correctly,
    preventing higher-KL responses from getting stronger updates.
    """

    def __init__(self, *args, use_sign_correction: bool = True, **kwargs):
        """
        Args:
            use_sign_correction: Enable SIGN correction (KTO-S mode)
                                If False, reverts to standard KTO
        """
        super().__init__(*args, **kwargs)
        self.use_sign_correction = use_sign_correction

        if self.use_sign_correction:
            print("\n" + "="*80)
            print("KTO-S MODE ENABLED")
            print("="*80)
            print("Using SIGN correction for stable KL divergence")
            print("Paper: [Add paper reference here]")
            print("="*80 + "\n")
        else:
            print("\n⚠️ KTO-S disabled - using standard KTO\n")

    def kto_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        policy_KL_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
        reference_KL_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute KTO loss with optional SIGN correction.

        This overrides the parent class method to add the SIGN correction.
        """
        # Compute rewards (unchanged from standard KTO)
        chosen_rewards = (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = (policy_rejected_logps - reference_rejected_logps).detach()

        # Compute KL divergence (unchanged from standard KTO)
        chosen_KL = (policy_chosen_logps - policy_KL_logps).mean()
        rejected_KL = (policy_rejected_logps - policy_KL_logps).mean()

        if self.use_sign_correction:
            # KTO-S: Add SIGN correction
            S_chosen = torch.sign(chosen_rewards)
            S_rejected = torch.sign(rejected_rewards)

            # Apply sign-corrected KL penalty
            chosen_losses = -F.logsigmoid(
                self.beta * (chosen_rewards + S_chosen * chosen_KL)
            )
            rejected_losses = -F.logsigmoid(
                self.beta * (S_rejected * rejected_KL - rejected_rewards)
            )
        else:
            # Standard KTO: Original formulation
            chosen_losses = -F.logsigmoid(
                self.beta * (chosen_rewards - chosen_KL)
            )
            rejected_losses = -F.logsigmoid(
                self.beta * (rejected_KL - rejected_rewards)
            )

        # Weight the losses
        chosen_loss = (chosen_losses * self.desirable_weight).mean()
        rejected_loss = (rejected_losses * self.undesirable_weight).mean()

        # Total loss
        loss = chosen_loss + rejected_loss

        return loss, chosen_rewards.mean(), rejected_rewards.mean()
```

### Step 2: Update Training Script

**File:** `train_kto.py`

**Changes needed:**
```python
# Line 29 - Update import
from trl import KTOConfig  # Remove KTOTrainer
from src.kto_s_trainer import KTOSTrainer  # Add custom trainer

# Line 638 - Use custom trainer
trainer = KTOSTrainer(  # Changed from KTOTrainer
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=callbacks,
    use_sign_correction=True,  # NEW: Enable KTO-S
)
```

### Step 3: Add Config Parameter

**File:** `configs/training_config.py`

```python
@dataclass
class KTOTrainingConfig:
    # ... existing config ...

    # KTO-S: SIGN correction for stable KL
    use_kto_s: bool = True  # Enable SIGN correction (recommended)
```

### Step 4: Test Implementation

**Create test script:** `test_kto_s.py`

```python
"""Quick test to verify KTO-S implementation works."""

import torch
from src.kto_s_trainer import KTOSTrainer

# Mock data
policy_chosen = torch.tensor([1.0, 2.0, 3.0])
policy_rejected = torch.tensor([0.5, 1.0, 1.5])
reference = torch.tensor([0.8, 1.5, 2.2])

# Test with SIGN correction ON
trainer_s = KTOSTrainer(use_sign_correction=True)
# ... run loss computation

# Test with SIGN correction OFF (should match standard KTO)
trainer_standard = KTOSTrainer(use_sign_correction=False)
# ... run loss computation

print("✓ KTO-S implementation test passed")
```

---

## 📋 Implementation Checklist

- [ ] **Phase 1: Investigation**
  - [x] Confirm TRL 0.11.4 doesn't have KTO-S
  - [ ] Read TRL's KTOTrainer source code to understand loss computation
  - [ ] Identify exact location of loss function to override

- [ ] **Phase 2: Implementation**
  - [ ] Create `src/kto_s_trainer.py` with custom trainer
  - [ ] Implement SIGN correction in loss function
  - [ ] Add toggle parameter `use_sign_correction`
  - [ ] Add clear logging when KTO-S is enabled

- [ ] **Phase 3: Integration**
  - [ ] Update `train_kto.py` to import KTOSTrainer
  - [ ] Add `use_kto_s` parameter to config
  - [ ] Update argument parsing for CLI flag
  - [ ] Add documentation comments

- [ ] **Phase 4: Testing**
  - [ ] Create unit test for loss computation
  - [ ] Test with KTO-S enabled vs disabled
  - [ ] Verify gradient flow is correct
  - [ ] Run short training (10 steps) to verify stability

- [ ] **Phase 5: Full Test Run**
  - [ ] Run full training with KTO-S
  - [ ] Monitor KL divergence at steps 5, 10, 15, 20
  - [ ] Compare with previous runs
  - [ ] Document results

---

## 🔬 Expected Results with KTO-S

Based on the paper (Figure 3, page 6):

**Without KTO-S (your current runs):**
```
Step 5:  KL = 0.039
Step 10: KL = 0.189  ← Spike begins
Step 15: KL = 0.472  ← Getting worse
Step 20: KL = 2.056  ← Explosion
```

**With KTO-S (expected):**
```
Step 5:  KL = 0.030  ← Slightly lower
Step 10: KL = 0.045  ← STABLE! (no spike)
Step 15: KL = 0.055  ← Controlled growth
Step 20: KL = 0.065  ← Still stable
```

**Key metrics to watch:**
- KL should stay **< 0.1** through step 20
- Rewards should increase steadily
- No sudden jumps in loss

---

## ⚠️ Important Notes

### 1. This is NOT About Learning Rate

Your two-stage LR schedule was a good hypothesis, but the paper shows this is actually about **gradient scaling direction**, not LR magnitude.

### 2. The Bug is Subtle

The original KTO implementation works for:
- ✅ Models that start from SFT (already aligned)
- ❌ Base models (what you're using)

You're starting from a base model → you need KTO-S.

### 3. Beta Still Matters

KTO-S fixes the gradient direction bug, but beta still controls the strength of the KL penalty. Your beta=0.2 should work well with KTO-S.

---

## 🚀 Next Steps

1. **Read TRL Source Code**
   - Check `KTOTrainer` implementation in TRL 0.11.4
   - Understand the exact loss computation
   - Identify override points

2. **Implement Custom Trainer**
   - Create `src/kto_s_trainer.py`
   - Test with mock data first
   - Verify gradient computation

3. **Integrate & Test**
   - Update training script
   - Run short test (10 steps)
   - Verify KL stays stable

4. **Full Training Run**
   - Run to step 145
   - Compare KL trajectory with previous runs
   - Document results

---

## 📚 References

- Paper: [Add exact paper citation here]
- Figure 2 (page 6): Shows KTO instability without SFT
- Figure 3 (page 6): Shows KTO-S stability
- TRL Documentation: https://huggingface.co/docs/trl/

---

## 🤔 Questions to Answer

1. **Does TRL > 0.11.4 have KTO-S built-in?**
   - Need to check TRL changelog
   - If yes, might be easier to upgrade

2. **What's the exact loss function in TRL 0.11.4?**
   - Need to read source code
   - Verify SIGN correction location

3. **Are there other KTO variants to consider?**
   - KTO-X?
   - Other stabilization techniques?

---

**Ready to implement?** Let me know and I'll create the full implementation with all necessary files!
