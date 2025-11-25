# Quick Start Guide - Fresh Terminal

This guide walks you through starting KTO training from a completely fresh terminal session.

## Step-by-Step Instructions

### 1. Open New Terminal
Open your WSL terminal (you should be in Windows Terminal with Ubuntu).

### 2. Navigate to Project Directory
```bash
cd /mnt/c/Users/Joseph/Documents/Code/Toolset-Training/code/rtx3090_kto
```

### 3. Activate Conda Environment
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ./venv
```

You should see `(venv)` prefix in your prompt.

### 4. Verify Installation (Optional)
Quick check that everything is installed:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

Should output: `PyTorch: 2.4.1+cu121, CUDA: True`

### 5. Start Training

**Option A: Use Config Defaults (batch_size=8, fully optimized)**
```bash
python train_kto.py --model-size 7b
```

**Option B: Force Batch Size Explicitly (safest)**
```bash
python train_kto.py --model-size 7b --batch-size 8 --gradient-accumulation 4
```

**Option C: Conservative (if you want less VRAM usage)**
```bash
python train_kto.py --model-size 7b --batch-size 4 --gradient-accumulation 8
```

## What You Should See

### Initialization (first ~30 seconds)
```
============================================================
RTX 3090 KTO TRAINING
============================================================
PyTorch version: 2.4.1+cu121
CUDA available: True
GPU: NVIDIA GeForce RTX 3090
GPU Memory: 25.8 GB
BFloat16 supported: True
```

### Configuration Confirmation
Look for this section to confirm batch size:
```
Batch configuration:
  Batch size: 8              ← Should be 8 for optimized
  Gradient accumulation: 4    ← Should be 4 for optimized
  Effective batch size: 32
```

### Training Starts
```
====================================================================================================
                                      TRAINING METRICS
====================================================================================================
   Step      |   Loss   |    LR     | Chosen | Reject | Margin | GPU Mem  | Samp/sec |    ETA
----------------------------------------------------------------------------------------------------
        5/145 |  0.5123  |  2.50e-07 |  0.234 | -0.123 |  0.357 |   23.3GB |     1.2  |  2h 15m
```

**KEY INDICATOR**: GPU Mem should show **18-23GB** (not 5GB!)

## Expected Performance

| Metric | Value |
|--------|-------|
| GPU Memory | 18-23 GB (optimized) or 5-6 GB (conservative) |
| Samples/sec | ~1-2 (depends on sequence length) |
| Time per step | ~45-60 seconds |
| Total training time | ~2-3 hours for 145 steps |

## Monitoring Training

### Check GPU Usage
In another terminal:
```bash
watch -n 1 nvidia-smi
```

Should show:
- GPU utilization: 90-100%
- Memory usage: 23310 MiB / 25434 MiB (~23GB)

### Check Metrics Table
The table updates every 5 steps automatically. Watch for:
- Loss decreasing over time
- GPU Mem staying around 18-23GB
- Samples/sec consistent
- ETA counting down

## Checkpoints

Training automatically saves checkpoints every 50 steps:
```
----------------------------------------------------------------------------------------------------
>> CHECKPOINT SAVED at step 50 -> ./kto_output_rtx3090/checkpoint-50
----------------------------------------------------------------------------------------------------
```

Keeps last 3 checkpoints automatically.

## If Training is Slow or Stuck

### Training Taking Forever?
Check:
1. Is GPU utilization at 90-100%? (`nvidia-smi`)
2. Is GPU memory at 18-23GB? (not 5GB)
3. Are you using batch_size=8?

### GPU Memory Still at 5GB?
Something went wrong. Try:
1. Stop training (Ctrl+C)
2. Clear Python cache:
   ```bash
   find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
   find . -name "*.pyc" -delete 2>/dev/null
   ```
3. Restart with explicit flags:
   ```bash
   python train_kto.py --model-size 7b --batch-size 8 --gradient-accumulation 4
   ```

### Out of Memory Error?
Reduce batch size:
```bash
python train_kto.py --model-size 7b --batch-size 6 --gradient-accumulation 5
```

## Stopping Training

To stop training gracefully:
1. Press `Ctrl+C` in the terminal
2. Wait for it to print "Training interrupted"
3. Last checkpoint is saved automatically

## After Training Completes

Training will automatically:
1. Save final model to `./kto_output_rtx3090/final_model`
2. Print completion message
3. Show where model was saved

To upload to HuggingFace:
```python
# In Python:
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./kto_output_rtx3090/final_model",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

model.push_to_hub_merged(
    "your-username/your-model-name",
    tokenizer,
    save_method="merged_16bit"
)
```

## Files Created During Training

```
kto_output_rtx3090/
├── checkpoint-50/          # First checkpoint
├── checkpoint-100/         # Second checkpoint
├── checkpoint-145/         # Last checkpoint (if completed)
└── final_model/           # Final merged model (at completion)
```

## Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| GPU Mem shows 5GB | Restart with `--batch-size 8 --gradient-accumulation 4` |
| OOM Error | Use `--batch-size 6 --gradient-accumulation 5` |
| Training very slow | Check `nvidia-smi` - GPU should be 90-100% |
| Import errors | Reactivate environment: `conda activate ./venv` |
| Can't find train_kto.py | Make sure you're in `code/rtx3090_kto` directory |

## Summary Commands

```bash
# Complete fresh start:
cd /mnt/c/Users/Joseph/Documents/Code/Toolset-Training/code/rtx3090_kto
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ./venv
python train_kto.py --model-size 7b --batch-size 8 --gradient-accumulation 4

# Monitor in another terminal:
watch -n 1 nvidia-smi

# Verify config:
python check_config.py
```

## Important Notes

1. **Always use explicit batch size** (`--batch-size 8`) for first run in new terminal
2. **Config defaults are optimized** but explicit flags are safer
3. **GPU memory at 18-23GB = working correctly**
4. **GPU memory at 5-6GB = something wrong, restart with explicit flags**
5. **Training takes ~2-3 hours** for full dataset
6. **Checkpoints save every 50 steps** automatically

## Configuration Reference

Current optimized settings (from `configs/training_config.py`):
- Model: Mistral 7B (4-bit quantized)
- Batch size: 8
- Gradient accumulation: 4
- Effective batch size: 32
- Max sequence length: 2048
- LoRA rank: 64
- Learning rate: 5e-7
- Optimizer: adamw_8bit
- BF16: Enabled
- Checkpointing: Every 50 steps
- Metrics table: Every 5 steps

Ready to train!
