# KTO Training Metrics Guide

## Understanding Your Training Metrics

This guide explains what each KTO metric means and what "good" values look like during training.

## Where Metrics Are Logged

**Clean Table (CLI)**: Shows summary metrics every 5 steps
**Detailed Log File**: `./kto_output_rtx3090/logs/training_metrics.jsonl` - ALL metrics every step

Your logs are organized in the output directory:
```
kto_output_rtx3090/
├── logs/
│   └── training_metrics.jsonl  ← All detailed metrics here
├── checkpoint-50/
├── checkpoint-100/
└── final_model/
```

## Key Metrics Explained

### 1. Loss
**What it is**: Overall training loss (lower is better)

**Good values**:
- Start: 0.6-0.7 (typical for KTO)
- Mid-training: 0.4-0.6
- End: 0.3-0.5

**Healthy trend**: Steadily decreasing

**Warning signs**:
- Not decreasing after 20-30 steps → Learning rate too low
- Dropping too fast (< 0.1 in first 10 steps) → Learning rate too high
- Increasing → Diverging, revert to checkpoint
- NaN/Inf → Training collapsed, restart

### 2. Rewards/Chosen (Table: "Chosen")
**What it is**: Model's reward for chosen (desirable=True) responses

**Good values**:
- Start: -0.5 to 0.5
- Target: Increasing toward 0 to 2.0
- End: 0.5 to 2.0

**Healthy trend**: Gradually increasing

**Warning signs**:
- Staying very negative (< -1.0) → Model not learning preferences
- Near zero and not moving → Reward collapse

### 3. Rewards/Rejected (Table: "Reject")
**What it is**: Model's reward for rejected (desirable=False) responses

**Good values**:
- Start: -0.5 to 0.5
- Target: Decreasing or staying low
- End: -1.0 to 0.5 (should be lower than chosen)

**Healthy trend**: Decreasing or stable (lower than chosen)

**Warning signs**:
- Higher than chosen rewards → Model inverted, restart
- Near zero and not moving → Reward collapse

### 4. Rewards/Margins (Table: "Margin")
**What it is**: Difference between chosen and rejected rewards (chosen - rejected)

**Good values**:
- Start: -0.1 to 0.1
- Target: Increasing
- End: 0.3 to 2.0

**Healthy trend**: **Steadily increasing** (MOST IMPORTANT METRIC!)

**Warning signs**:
- Negative and decreasing → Model learning backwards, stop immediately!
- Stuck at ~0 → Not learning, check dataset
- Very large (> 5.0) → Overfitting

**🎯 This is THE metric to watch! It should go up throughout training.**

### 5. Grad Norm
**What it is**: Magnitude of gradients (stability indicator)

**Good values**: 1.0 to 50.0

**Warning signs**:
- > 100 → Training unstable, reduce learning rate
- > 1000 → About to explode, stop training!
- Very small (< 0.01) → Learning rate too low

### 6. Learning Rate
**What it is**: Current learning rate (with warmup)

**Good values**:
- Warmup (first 10% steps): 0 → 5e-7
- Training: ~5e-7 (configured value)
- End: ~5e-7 (constant in our config)

**Healthy trend**: Ramps up during warmup, then stable

### 7. KL Divergence
**What it is**: How much model diverges from reference

**Good values**: 0.01 to 0.5

**Warning signs**:
- Very high (> 1.0) → Model diverging too much from base
- Very low (< 0.001) → Model not changing enough

## Example Healthy Training

```
Step     Loss    Chosen  Reject  Margin
5        0.650   -0.045  0.033   -0.078   ← Starting
10       0.620   0.012   -0.015  0.027    ← Margin turning positive ✓
25       0.580   0.089   -0.112  0.201    ← Margin increasing ✓
50       0.520   0.234   -0.189  0.423    ← Good progress!
100      0.450   0.512   -0.298  0.810    ← Strong separation ✓
145      0.420   0.678   -0.445  1.123    ← Excellent! ✓✓
```

**What to look for**:
- Loss: 0.65 → 0.42 (decreasing ✓)
- Margin: -0.078 → 1.123 (INCREASING ✓✓✓)
- Chosen > Rejected throughout (✓)

## Example Problematic Training

```
Step     Loss    Chosen  Reject  Margin
5        0.650   -0.045  0.033   -0.078
10       0.720   -0.112  0.089   -0.201   ← Loss increasing ✗
25       0.850   -0.234  0.198   -0.432   ← Margin MORE negative ✗✗
```

**Problems**:
- Loss increasing → Diverging
- Margin becoming more negative → Learning backwards!

**Action**: Stop training, revert to checkpoint, reduce learning rate by 50%

## Warning System

The training monitor automatically checks for issues and will display warnings like:

```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ⚠ Very negative margin: -0.432 (chosen model may be worse than reference)
  Consider: reducing learning rate or reverting to last checkpoint
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

## When to Stop and Revert

Stop training and go back to last checkpoint if:

1. **Margin goes negative** after initial warmup (first 10-20 steps)
2. **Loss increases** for 10+ consecutive steps
3. **Grad norm > 100** persistently
4. **NaN/Inf values** appear

## When to Reduce Learning Rate

Reduce by 50% (5e-7 → 2.5e-7) if:

1. **Loss not decreasing** after 30-50 steps
2. **High grad norms** (50-100) frequently
3. **Unstable metrics** (jumping up and down)

Restart training from last checkpoint with new LR:
```bash
python train_kto.py --model-size 7b --learning-rate 2.5e-7 --resume-from ./kto_output_rtx3090/checkpoint-50
```

## Analyzing Your Training Log

View detailed metrics:
```bash
# Watch metrics in real-time
tail -f ./kto_output_rtx3090/logs/training_metrics.jsonl

# Show last 10 steps (formatted)
tail -n 10 ./kto_output_rtx3090/logs/training_metrics.jsonl | python -m json.tool

# Check specific metric over time
cat ./kto_output_rtx3090/logs/training_metrics.jsonl | jq '.["rewards/margins"]' -c
```

## Quick Metrics Checker

```python
import json

def check_training_health(log_file="./kto_output_rtx3090/logs/training_metrics.jsonl"):
    """Quick health check of training metrics."""

    with open(log_file, "r") as f:
        logs = [json.loads(line) for line in f]

    if len(logs) < 10:
        print("Not enough data yet")
        return

    latest = logs[-1]
    previous = logs[max(0, len(logs)-11):-1]  # Last 10 steps

    # Check margin trend
    margins = [log.get('rewards/margins', 0) for log in previous] + [latest.get('rewards/margins', 0)]
    margin_trend = margins[-1] - margins[0]

    print(f"Latest step: {latest['step']}")
    print(f"Current margin: {latest.get('rewards/margins', 0):.4f}")
    print(f"Margin trend (last 10 steps): {margin_trend:+.4f}")
    print(f"Loss: {latest.get('loss', 0):.4f}")

    if margin_trend > 0:
        print("✓ Training is healthy! Margin increasing.")
    else:
        print("⚠ Warning: Margin not increasing. Consider checking training.")

# Run it
check_training_health()
```

## Target Metrics by Training Stage

| Stage | Steps | Loss | Margin | Chosen | Rejected |
|-------|-------|------|--------|--------|----------|
| Start | 1-10 | 0.6-0.7 | -0.1 to 0.1 | -0.5 to 0.5 | -0.5 to 0.5 |
| Early | 11-50 | 0.5-0.6 | 0.1 to 0.5 | 0.0 to 0.5 | -0.5 to 0.0 |
| Mid | 51-100 | 0.4-0.5 | 0.5 to 1.0 | 0.3 to 0.8 | -0.5 to -0.2 |
| Late | 101-145 | 0.3-0.5 | 1.0 to 2.0 | 0.5 to 1.0 | -0.8 to -0.3 |

## Summary: The One Metric That Matters

**Rewards/Margins** (Margin column) should **steadily increase** from ~0 to 1.0+

If margin is increasing: ✓ Training is working!
If margin is flat/decreasing: ✗ Something is wrong

Everything else is secondary to this metric.

## Additional Resources

- **Detailed logs**: `./kto_output_rtx3090/logs/training_metrics.jsonl`
- **Checkpoints**: `./kto_output_rtx3090/checkpoint-*`
- **Final model**: `./kto_output_rtx3090/final_model`

## Quick Access Commands

```bash
# Monitor live
tail -f ./kto_output_rtx3090/logs/training_metrics.jsonl

# Check current margin
tail -n 1 ./kto_output_rtx3090/logs/training_metrics.jsonl | python -c "import sys, json; d=json.load(sys.stdin); print(f\"Step {d['step']}: Margin={d.get('rewards/margins', 0):.4f}\")"

# Count steps completed
wc -l < ./kto_output_rtx3090/logs/training_metrics.jsonl
```

Happy training! 🚀
