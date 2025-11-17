# Run 8: Quick Start Guide

## 🚀 Easiest Way to Run

```bash
./train_run8.sh
```

That's it! All settings are in the config file.

---

## ⚙️ Tuning Parameters

**Edit:** `configs/training_config.py`

### Current Run 8 Settings

```python
# Two-stage learning rate schedule
use_two_stage_lr: bool = True        # ← Toggle on/off here
lr_reduction_step: int = 50          # ← When to reduce LR
lr_reduction_factor: float = 0.5     # ← How much to reduce (0.5 = 50%)
```

### Common Adjustments

**To disable two-stage LR:**
```python
use_two_stage_lr: bool = False
```

**To reduce earlier (step 45):**
```python
lr_reduction_step: int = 45
```

**More aggressive reduction (75%):**
```python
lr_reduction_factor: float = 0.25  # Reduces to 25% of original
```

**Gentler reduction (25%):**
```python
lr_reduction_factor: float = 0.75  # Reduces to 75% of original
```

### Other Key Parameters

```python
# From configs/training_config.py

beta: float = 0.2                    # KTO beta (0.1-0.3)
learning_rate: float = 5e-7          # Initial learning rate
max_grad_norm: float = 1.0           # Gradient clipping
per_device_train_batch_size: int = 8 # Batch size
gradient_accumulation_steps: int = 4 # Gradient accumulation
```

---

## 📊 What to Watch For

### Step 50 - LR Reduction
```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
🔧 LEARNING RATE REDUCED at step 50
   5.00e-07 → 2.50e-07 (50% reduction)
   Reason: Preemptive intervention before step 60 instability zone
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

### Step 60 - Critical Checkpoint
**Success:** KL < 0.8 (no spike!)
**Failure:** KL > 1.1 (spike still occurs)

### Metrics Table
```
   Step      |   Loss   |    LR     | Chosen | Reject | Margin | GPU Mem  | Samp/sec |    ETA
 30/145      |   0.xxxx | 5.00e-07  |  0.xxx |  0.xxx |  0.xxx |   20.xGB |      x.x |      xxs
 50/145      |   0.xxxx | 2.50e-07  |  0.xxx |  0.xxx |  0.xxx |   20.xGB |      x.x |      xxs  ← LR reduced
 60/145      |   0.xxxx | 2.50e-07  |  0.xxx |  0.xxx |  0.xxx |   20.xGB |      x.x |      xxs  ← Watch this!
```

---

## 📁 Logs Location

Training creates timestamped directory:
```
kto_output_rtx3090/
└── 20251115_HHMMSS/          ← Your run
    ├── checkpoints/
    │   ├── checkpoint-50/
    │   └── checkpoint-100/
    ├── logs/
    │   └── training_20251115_HHMMSS.jsonl
    └── final_model/
```

**View logs in real-time:**
```bash
tail -f kto_output_rtx3090/[timestamp]/logs/logs/training_*.jsonl
```

---

## 🔄 Quick Variations to Test

### Run 8a: Earlier Reduction
```python
lr_reduction_step: int = 45
lr_reduction_factor: float = 0.5
```

### Run 8b: More Aggressive
```python
lr_reduction_step: int = 50
lr_reduction_factor: float = 0.25  # 75% reduction!
```

### Run 8c: Gentler
```python
lr_reduction_step: int = 50
lr_reduction_factor: float = 0.75  # Only 25% reduction
```

Just edit the config and run `./train_run8.sh` again!

---

## 📖 Full Documentation

See **RUN_8_TWO_STAGE_LR.md** for:
- Complete technical details
- Implementation explanation
- Expected results
- Comparison with previous runs
- Scientific rationale

---

## 🆘 Troubleshooting

**If training fails to start:**
```bash
# Make script executable
chmod +x train_run8.sh

# Or run directly:
python train_kto.py --model-size 7b --max-steps 145
```

**If you see the old NoneType error:**
- You're on an old version of the code
- Pull latest changes
- The fix is in train_kto.py line 551

**To run with different model:**
```bash
python train_kto.py --model-size 13b --max-steps 145  # For 13B models
python train_kto.py --model-size 3b --max-steps 145   # For 3B models
```

All two-stage LR settings still come from the config!
