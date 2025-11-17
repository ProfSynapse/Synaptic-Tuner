# Weights & Biases (W&B) Setup Guide

## What is W&B?

Weights & Biases provides beautiful real-time training dashboards with interactive charts. Instead of reading JSONL log files, you get visual learning curves and can monitor training from anywhere.

## Benefits for KTO Training

1. **Visual metrics** - See your margin increasing in real-time charts
2. **Compare runs** - Try different hyperparameters and compare side-by-side
3. **Remote monitoring** - Check progress from phone/laptop anywhere
4. **Automatic tracking** - GPU, memory, system metrics all logged
5. **Free tier** - 100GB storage free for personal use

## 🚀 AUTOMATIC SETUP (New!)

W&B is now **automatically enabled** when you add your API key to the `.env` file. No need for `--wandb` flags or manual login!

### 1. Get Your W&B API Key

1. Go to https://wandb.ai/authorize
2. Sign up (free) or login
3. Copy your API key

### 2. Add Key to .env

Open your `.env` file and add:
```bash
WANDB_API_KEY=your_actual_key_here
```

That's it! Now just run training normally:

```bash
python train_kto.py --model-size 7b
```

W&B will automatically:
- ✓ Login using your API key from `.env`
- ✓ Enable tracking for all runs
- ✓ Auto-generate project/run names
- ✓ Upload all metrics in real-time

## Manual Setup (Old Way - Still Supported)

If you prefer not to use `.env`, you can still use flags:

```bash
python train_kto.py \
  --model-size 7b \
  --wandb \
  --wandb-project "claudesidian-tools" \
  --wandb-run-name "mistral-7b-kto-v1"
```

## What Gets Tracked Automatically

### Training Metrics (every 5 steps)
- `loss` - Overall training loss
- `rewards/chosen` - Chosen response rewards
- `rewards/rejected` - Rejected response rewards
- `rewards/margins` - THE key metric (chosen - rejected)
- `grad_norm` - Gradient magnitude
- `learning_rate` - Current LR with warmup
- `kl` - KL divergence from reference

### System Metrics (automatic)
- GPU utilization %
- GPU memory usage
- CPU usage
- System memory
- Network I/O
- Disk I/O

### Run Info (automatic)
- All hyperparameters
- Model architecture
- Dataset info
- Git commit hash
- Environment details

## Viewing Your Dashboard

After training starts, you'll see:
```
wandb: 🚀 View run at https://wandb.ai/your-username/claudesidian-tools/runs/abc123
```

Click that link to see:
1. **Overview** - Run status, duration, system info
2. **Charts** - Interactive plots of all metrics
3. **System** - GPU/CPU/memory usage
4. **Logs** - Console output
5. **Files** - Model checkpoints (if configured)

## Best Charts to Watch

### 1. Rewards/Margins (Most Important!)
This should steadily increase from ~0 to 1.0+. If it's not going up, something's wrong.

### 2. Loss
Should decrease steadily. If it increases, training is diverging.

### 3. Rewards: Chosen vs Rejected
Two lines that should separate over time:
- Chosen (blue) going up
- Rejected (red) going down

### 4. GPU Memory
Should show ~23GB usage (confirming optimization is working)

## Comparing Multiple Runs

Run multiple experiments:
```bash
# Run 1: Default LR
python train_kto.py --model-size 7b --wandb --wandb-run-name "lr-5e-7"

# Run 2: Lower LR
python train_kto.py --model-size 7b --learning-rate 2.5e-7 --wandb --wandb-run-name "lr-2.5e-7"

# Run 3: Different model
python train_kto.py --model-size 13b --wandb --wandb-run-name "13b-baseline"
```

Then in W&B dashboard:
1. Click "Compare" tab
2. Select your runs
3. See all metrics side-by-side

## Custom Dashboards

You can create custom dashboards with:
- Multi-run comparisons
- Custom metric combinations
- Alerts on metric thresholds
- Sweeps for hyperparameter tuning

## Privacy & Data

**Local logs still work**: Your JSONL files are still saved locally as backup

**What's uploaded**:
- Metrics (numbers only)
- Hyperparameters
- System info
- Console logs

**NOT uploaded**:
- Your training data
- Model weights (unless you configure it)
- Code (unless you enable git tracking)

**Privacy options**:
- Make projects private (free tier supports this)
- Self-hosted W&B server (for enterprises)
- Offline mode (logs locally, upload later)

## Offline Mode

If you want to log locally first, then upload later:

```bash
# Run in offline mode
wandb offline
python train_kto.py --wandb --wandb-run-name "test-run"

# Later, sync to cloud
wandb sync ./wandb/offline-run-*
```

## Advanced: Hyperparameter Sweeps

W&B can automatically try different hyperparameters to find the best:

```yaml
# sweep.yaml
program: train_kto.py
method: grid
parameters:
  learning_rate:
    values: [5e-7, 2.5e-7, 1e-7]
  batch_size:
    values: [6, 8, 10]
```

```bash
wandb sweep sweep.yaml
wandb agent your-username/project/sweep-id
```

It will automatically run all combinations and show which works best!

## Cost

**Free Tier (Personal)**:
- Unlimited runs
- 100GB storage
- 1 team member
- Private projects ✓

**Pro Tier** ($50/month):
- Unlimited storage
- More team members
- Advanced features

For your use case, free tier is perfect!

## Integration with Your Current Setup

Your training already has:
1. ✓ Clean CLI table (keeps working)
2. ✓ JSONL logs (keeps working as backup)
3. ✓ Health warnings (keeps working)
4. ✓ Checkpoint saves (keeps working)

W&B just **adds** the web dashboard on top. Nothing changes!

## Example Dashboard

After a few steps, you'd see charts like:

```
┌─────────────────────────────────────────┐
│  Rewards/Margins                        │
│   1.2 ┤                            ╭─   │
│   1.0 ┤                      ╭────╯     │
│   0.8 ┤                ╭────╯           │
│   0.6 ┤          ╭────╯                 │
│   0.4 ┤    ╭────╯                       │
│   0.2 ┤───╯                             │
│     0 ┤                                 │
│       └─────────────────────────────────│
│         0   20   40   60   80   100     │
│                  Steps                  │
└─────────────────────────────────────────┘
```

But interactive! Hover for values, zoom, compare runs, etc.

## Quick Start Commands

### First Time Setup
```bash
pip install wandb
wandb login
```

### Start Training with W&B
```bash
python train_kto.py \
  --model-size 7b \
  --batch-size 8 \
  --gradient-accumulation 4 \
  --wandb \
  --wandb-project "my-kto-training"
```

### Your Current Run (Already Started)
Since your current run is already going, you can:
1. Let it finish without W&B
2. **OR** start a new run in parallel with W&B to test it
3. Next run, add `--wandb` flag

## Recommendation

**For your next training run**: Definitely use W&B!

The visual dashboards make it SO much easier to:
- Spot problems early
- Share results
- Compare experiments
- Monitor remotely

Plus it's free and takes 5 minutes to setup.

## Quick Test (Optional)

Test W&B without running full training:

```bash
# Install
pip install wandb

# Login
wandb login

# Quick test with dry-run
python train_kto.py --model-size 7b --dry-run --wandb --wandb-run-name "test"
```

This will verify W&B is working without starting actual training.

## Summary

**Should you use W&B?** → **YES!**

**When?** → Next training run (current one is already 41% done)

**Setup time** → 5 minutes

**Cost** → Free

**What you get** → Beautiful visual dashboards instead of reading JSONL files

Perfect for seeing that "sexy learning curve" you mentioned! 📈

---

**Next Steps:**
1. `pip install wandb`
2. `wandb login`
3. Add `--wandb` to your next training command

That's it! 🚀
