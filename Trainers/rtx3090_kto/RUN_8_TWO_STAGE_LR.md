# Run 8: Two-Stage Learning Rate Schedule

## 🎯 Hypothesis

**The step 60 spike is caused by optimization momentum overshoot, not individual hyperparameter values.**

After systematically eliminating beta tuning (Runs 1-3, 6), batch configuration (Run 4), learning rate reduction alone (Run 4), and gradient clipping (Run 7), the spike persists at step 60 regardless of these parameters.

**Solution:** Implement a two-stage learning rate schedule that preemptively reduces LR at step 50, giving the optimizer "brakes" before the instability zone.

---

## 📊 Configuration

### Run 8 Settings

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Beta** | 0.2 | Optimal from Run 5/6/7 |
| **Batch Size** | 8 | Required configuration |
| **Gradient Accumulation** | 4 | Effective batch = 32 |
| **Initial Learning Rate** | 5e-7 | Proven effective for steps 1-50 |
| **LR Reduction Step** | 50 | **NEW**: Reduce before step 60 spike |
| **LR Reduction Factor** | 0.5 | **NEW**: Cut LR to 2.5e-7 (50% reduction) |
| **Gradient Clipping** | 1.0 | Back to default (0.5 had no effect) |
| **Max Steps** | 145 | Full run to completion |

### Two-Stage Learning Rate Timeline

```
Steps 1-50:   LR = 5.00e-07  ← Fast early learning (proven effective)
              ↓
Step 50:      🔧 REDUCTION TRIGGERED
              ↓
Steps 51-145: LR = 2.50e-07  ← Slower optimization (prevents overshoot)
```

---

## 🔬 What Changed

### 1. New Code: `TwoStageLRCallback`

**File:** `src/training_callbacks.py`

Added a custom training callback that monitors training steps and automatically reduces learning rate at a specified step:

```python
class TwoStageLRCallback(TrainerCallback):
    """
    Implements a two-stage learning rate schedule.

    Reduces LR at specified step to prevent optimization instability
    while maintaining fast early learning.
    """

    def __init__(self, initial_lr: float, reduced_lr: float, reduction_step: int):
        self.initial_lr = initial_lr
        self.reduced_lr = reduced_lr
        self.reduction_step = reduction_step
        self.lr_reduced = False

    def on_step_begin(self, args, state, control, **kwargs):
        """Reduce learning rate when reaching reduction_step."""
        if state.global_step == self.reduction_step and not self.lr_reduced:
            optimizer = kwargs.get('optimizer')
            if optimizer is not None:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.reduced_lr
                self.lr_reduced = True
                # Logs clear notification to console
```

**Key Features:**
- ✅ Automatically reduces LR at specified step
- ✅ No manual intervention needed
- ✅ Clear console notification when reduction occurs
- ✅ Works with any optimizer

### 2. Configuration Parameters

**File:** `configs/training_config.py`

Added three new parameters to `KTOTrainingConfig`:

```python
# Two-stage learning rate schedule (optional)
use_two_stage_lr: bool = False          # Enable/disable two-stage LR
lr_reduction_step: int = 50             # Step at which to reduce LR
lr_reduction_factor: float = 0.5        # Multiply LR by this factor
```

**Easy tuning:**
- Set `use_two_stage_lr = True` to enable
- Adjust `lr_reduction_step` to change when reduction happens
- Adjust `lr_reduction_factor` to control how much reduction (0.5 = 50%, 0.25 = 25%, etc.)

### 3. Training Script Integration

**File:** `train_kto.py`

- Imports `TwoStageLRCallback`
- Conditionally adds callback when `use_two_stage_lr = True`
- Displays two-stage LR config in training setup
- Shows clear notification at step 50 when LR reduces

---

## 🚀 How to Run

### Method 1: Config File (Recommended)

**Edit `configs/training_config.py`:**

```python
@dataclass
class KTOTrainingConfig:
    # ... existing config ...

    # Two-stage learning rate schedule
    use_two_stage_lr: bool = True        # ← Enable for Run 8
    lr_reduction_step: int = 50          # ← Reduce at step 50
    lr_reduction_factor: float = 0.5     # ← 50% reduction
```

**Then just run:**

```bash
python train_kto.py --model-size 7b --max-steps 145
```

### Method 2: Command-Line Flags

```bash
python train_kto.py \
    --model-size 7b \
    --beta 0.2 \
    --batch-size 8 \
    --gradient-accumulation 4 \
    --learning-rate 5e-7 \
    --two-stage-lr \
    --lr-reduction-step 50 \
    --lr-reduction-factor 0.5 \
    --max-steps 145
```

---

## 📈 Expected Results

### If Hypothesis is CORRECT:

| Step | Metric | Expected Value | Status |
|------|--------|----------------|--------|
| 30 | KL Divergence | ~0.36 | Same as Run 5/7 |
| 30 | Margins | ~0.67 | Early learning on track |
| 50 | KL Divergence | ~0.50 | Self-correction complete |
| **50** | **LR** | **5e-7 → 2.5e-7** | **🔧 REDUCTION** |
| **60** | **KL Divergence** | **< 0.8** | **✅ NO SPIKE!** |
| 60 | Margins | ~0.80 | Stable improvement |
| 145 | KL Divergence | < 1.0 | Full run completes |

**Success criteria:** KL < 0.8 at step 60 (no spike)

### If Hypothesis is INCORRECT:

| Scenario | What It Means | Next Step |
|----------|---------------|-----------|
| Spike still at step 60 | LR reduction too late | Try `lr_reduction_step = 45` |
| Spike at step 55 | Reduction triggered it | Try gentler reduction (`lr_reduction_factor = 0.7`) |
| Spike delayed to step 70 | Partial success | Try more aggressive reduction (`lr_reduction_factor = 0.25`) |

---

## 🎛️ Easy Tuning Guide

### Common Variations

**Run 8a: Earlier Intervention**
```python
lr_reduction_step: int = 45
lr_reduction_factor: float = 0.5
```

**Run 8b: More Aggressive Reduction**
```python
lr_reduction_step: int = 50
lr_reduction_factor: float = 0.25  # 75% reduction
```

**Run 8c: Gentler Reduction**
```python
lr_reduction_step: int = 50
lr_reduction_factor: float = 0.75  # 25% reduction
```

**Run 8d: Multiple Stages** (requires code modification)
```python
# Future enhancement: reduce at multiple points
# Step 40: 5e-7 → 3e-7
# Step 60: 3e-7 → 1.5e-7
```

---

## 📝 What We've Eliminated

| Hypothesis | Runs Tested | Configuration | Result |
|------------|-------------|---------------|--------|
| **Beta tuning** | 1, 2, 3, 6 | 0.1, 0.15, 0.2, 0.3 | ❌ Beta 0.2 optimal, spike persists |
| **Batch config** | 4, 5 | 4×8 vs 8×4 | ✅ 8×4 required, spike persists |
| **LR reduction alone** | 4 | 5e-7 vs 1e-6 | ❌ Lower LR worse |
| **Gradient clipping** | 7 | 0.5 vs 1.0 | ❌ No effect on spike |
| **Optimization trajectory** | **8** | **Two-stage LR** | **🔬 TESTING** |

---

## 🔍 Key Observations from Previous Runs

### The Step 60 Phenomenon

**Highly reproducible pattern across all runs:**

1. ✅ **Steps 1-30:** Normal early learning (KL ~0.36, margins ~0.67)
2. ✅ **Steps 30-45:** KL increases to ~0.74 (expected behavior)
3. ✅ **Steps 45-50:** Self-correction! KL drops to ~0.52 (30% reduction)
4. 🔴 **Step 60:** SPIKE! KL jumps to ~1.18 (+130% from step 50)
5. ❓ **After step 60:** Unknown (runs terminated)

**Key insight:** The spike occurs REGARDLESS of:
- Beta value (tested 0.1, 0.15, 0.2, 0.3)
- Gradient clipping (tested 0.5, 1.0)
- Batch configuration (once fixed to 8×4)

**This strongly suggests:** The problem is the **optimization trajectory** itself, not individual hyperparameter values.

---

## 🛠️ Implementation Details

### Console Output During Training

**At training start:**
```
====================================================================================================
TWO-STAGE LEARNING RATE SCHEDULE
====================================================================================================
  Steps 1-50:  LR = 5.00e-07 (fast early learning)
  Steps 51+:   LR = 2.50e-07 (50% of initial, prevents instability)
  Reduction ratio: 50.0%
====================================================================================================
```

**At step 50:**
```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
🔧 LEARNING RATE REDUCED at step 50
   5.00e-07 → 2.50e-07 (50% reduction)
   Reason: Preemptive intervention before step 60 instability zone
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

### Files Modified

1. **`src/training_callbacks.py`** - Added `TwoStageLRCallback` class
2. **`configs/training_config.py`** - Added three new config parameters
3. **`train_kto.py`** - Integrated callback, added CLI flags, display logic

---

## 📊 Comparison: Run 5 vs Run 7 vs Run 8

| Metric | Run 5 | Run 7 | Run 8 |
|--------|-------|-------|-------|
| **Beta** | 0.2 | 0.2 | 0.2 |
| **Batch Config** | 8×4 | 8×4 | 8×4 |
| **Learning Rate** | 5e-7 (constant) | 5e-7 (constant) | **5e-7 → 2.5e-7** |
| **Grad Clipping** | 1.0 | **0.5** | 1.0 |
| **LR Schedule** | Cosine | Cosine | **Two-stage** |
| **Step 60 KL** | 1.187 | 1.178 | **TBD** |
| **Result** | Spike | Spike | **Testing** |

**The key difference:** Run 8 is the first to address the **optimization trajectory** rather than individual parameters.

---

## 🎓 Scientific Rationale

### Why Two-Stage LR Should Work

1. **Momentum Buildup (Steps 1-50):**
   - Optimizer accumulates momentum through early training
   - Fast LR (5e-7) enables rapid learning
   - Self-correction at steps 45-50 shows model can stabilize

2. **Momentum Overshoot (Step 60):**
   - Built-up momentum causes optimizer to overshoot
   - Fast LR + momentum = instability spike
   - Occurs consistently at same point (step 60)

3. **Preemptive Reduction (Step 50):**
   - Reduce LR BEFORE the instability zone
   - Slower LR reduces momentum's impact
   - Maintains gains from fast early learning

### Alternative Explanations (if this fails)

- **Data ordering:** Certain examples at step 60 trigger instability
- **Warmup interaction:** Warmup ends around step 15 (10% of 145), momentum builds afterward
- **KTO-specific:** Beta parameter interacts with LR in unexpected way
- **Model-specific:** Mistral-7B architecture sensitive to this pattern

---

## 📁 Run Artifacts

**Location:** `kto_output_rtx3090/[timestamp]/`

- `checkpoints/` - Model checkpoints every 50 steps
- `logs/training_[timestamp].jsonl` - Detailed metrics (every step)
- `final_model/` - Final trained model (if run completes)

**Key metrics to monitor:**
- `rewards/margins` - Should increase steadily
- `logps/rejected` - KL divergence proxy (watch for spike)
- `learning_rate` - Should drop at step 50

---

## ✅ Success Criteria

**Run 8 is successful if:**

1. ✅ Step 30: KL ~0.36 (normal early learning)
2. ✅ Step 50: LR reduces from 5e-7 to 2.5e-7
3. ✅ **Step 60: KL < 0.8** (no spike, < 60% increase from step 50)
4. ✅ Step 145: Run completes without crashes
5. ✅ Final KL < 1.0 (model learning effectively)

**If successful:** We've identified optimization trajectory as the root cause and have a working solution.

**If unsuccessful:** Move to continuous LR decay or investigate alternative root causes.

---

## 🔄 Next Steps If Run 8 Fails

### Option A: Earlier Reduction
```python
lr_reduction_step: int = 45  # Reduce 5 steps earlier
```

### Option B: More Aggressive Reduction
```python
lr_reduction_factor: float = 0.25  # Reduce to 25% of initial
```

### Option C: Continuous Decay (requires new callback)
```python
# Linear decay from step 50 onward
# Steps 50-145: LR decays from 5e-7 to 1e-7
```

### Option D: Investigate Data
- Examine examples processed at step 60
- Check for unusual prompt/completion patterns
- Test with shuffled dataset

---

## 📚 References

- Previous runs: See individual run logs in `kto_output_rtx3090/`
- KTO paper: [Kahneman-Tversky Optimization](https://arxiv.org/abs/2402.01306)
- Learning rate schedules: Transformers documentation
- This implementation: Custom callback based on HuggingFace `TrainerCallback`
