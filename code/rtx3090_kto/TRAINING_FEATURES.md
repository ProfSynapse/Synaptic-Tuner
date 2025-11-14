# Training Features - Checkpointing & Metrics Tracking

This document describes the training features implemented for KTO fine-tuning on RTX 3090.

## Automatic Checkpointing

The training script automatically saves checkpoints during training to prevent data loss and enable resuming interrupted training sessions.

### Configuration

**Default Settings (7B models):**
- Save checkpoint every: **50 steps**
- Keep last: **3 checkpoints** (auto-deletes older ones to save disk space)
- Output directory: `./kto_output_rtx3090/`

**Checkpoint Structure:**
```
kto_output_rtx3090/
├── checkpoint-50/
│   ├── adapter_model.safetensors  # LoRA weights
│   ├── adapter_config.json
│   └── training_args.bin
├── checkpoint-100/
├── checkpoint-150/
└── final_model/  # Saved at end of training
```

### Customizing Checkpoints

You can override checkpoint settings via command line:

```bash
# Save every 100 steps instead of 50
python train_kto.py --model-size 7b --save-steps 100

# Keep last 5 checkpoints instead of 3
python train_kto.py --model-size 7b --save-total-limit 5

# Change output directory
python train_kto.py --model-size 7b --output-dir ./my_checkpoints
```

### Resuming from Checkpoint

To resume training from a checkpoint (feature to be implemented):

```bash
python train_kto.py --model-size 7b --resume-from-checkpoint ./kto_output_rtx3090/checkpoint-150
```

## Real-Time Metrics Table

The training script displays a live metrics table every 5 steps to track training progress.

### Example Output

```
====================================================================================================
TRAINING STARTED
====================================================================================================

Checkpoint Configuration:
  Save every: 50 steps
  Keep last: 3 checkpoints
  Output dir: ./kto_output_rtx3090

┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                        TRAINING METRICS                                          │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│    Step      │   Loss   │    LR     │ Chosen │ Reject │ Margin │ GPU Mem  │ Samp/sec │    ETA     │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│         5/500 │  0.6823  │  2.50e-07 │  0.342 │ -0.128 │  0.470 │   18.2GB │     12.5 │   40m 12s  │
│        10/500 │  0.6421  │  5.00e-07 │  0.389 │ -0.156 │  0.545 │   18.3GB │     12.6 │   38m 54s  │
│        15/500 │  0.6012  │  5.00e-07 │  0.421 │ -0.189 │  0.610 │   18.3GB │     12.6 │   38m 28s  │
│        20/500 │  0.5687  │  5.00e-07 │  0.456 │ -0.213 │  0.669 │   18.3GB │     12.7 │   37m 45s  │
│        25/500 │  0.5432  │  5.00e-07 │  0.482 │ -0.231 │  0.713 │   18.3GB │     12.7 │   37m 21s  │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 💾 CHECKPOINT SAVED at step 50 → ./kto_output_rtx3090/checkpoint-50                             │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│        50/500 │  0.4523  │  5.00e-07 │  0.567 │ -0.289 │  0.856 │   18.3GB │     12.8 │   35m 04s  │
...
└──────────────────────────────────────────────────────────────────────────────────────────────────┘

====================================================================================================
TRAINING COMPLETED
====================================================================================================
Total time: 1h 2m
Total steps: 500
Average speed: 8.05 steps/sec
====================================================================================================
```

### Metrics Explained

| Metric | Description |
|--------|-------------|
| **Step** | Current training step / Total steps |
| **Loss** | Training loss (lower is better) |
| **LR** | Current learning rate (changes with warmup/schedule) |
| **Chosen** | Average reward for chosen (desirable) responses |
| **Reject** | Average reward for rejected (undesirable) responses |
| **Margin** | Difference between chosen and rejected (higher is better) |
| **GPU Mem** | Current GPU memory usage |
| **Samp/sec** | Samples processed per second (throughput) |
| **ETA** | Estimated time remaining |

### What to Watch For

**Good Training Indicators:**
- Loss steadily decreasing
- Margin steadily increasing (chosen > rejected)
- Chosen rewards increasing
- Rejected rewards decreasing

**Warning Signs:**
- Loss increasing or oscillating wildly → reduce learning rate
- Margin decreasing → model is not learning preference correctly
- GPU memory errors → reduce batch size or sequence length

## Customizing Metrics Display

The metrics table is generated by `src/training_callbacks.py`.

To change the logging frequency (default is every 5 steps):

```python
# In train_kto.py, line 334
callbacks = [
    MetricsTableCallback(log_every_n_steps=10),  # Change to 10 steps
    CheckpointMonitorCallback()
]
```

Or modify `configs/training_config.py`:

```python
logging_steps: int = 10  # Change from 5 to 10
```

## Integration with Weights & Biases (Optional)

If you want cloud-based experiment tracking in addition to the table:

```bash
pip install wandb
wandb login

# Enable W&B logging
python train_kto.py --model-size 7b --wandb --wandb-project my-project --wandb-run-name mistral-7b-run-1
```

This will send metrics to both:
1. The terminal table (local, real-time)
2. W&B dashboard (cloud, with graphs and history)

## File Locations

**Callbacks:** `src/training_callbacks.py`
- `MetricsTableCallback` - Table display
- `CheckpointMonitorCallback` - Checkpoint info

**Configuration:** `configs/training_config.py`
- `logging_steps` - How often to log metrics (5)
- `save_steps` - How often to save checkpoints (50)
- `save_total_limit` - How many checkpoints to keep (3)

**Training Script:** `train_kto.py`
- Integrates callbacks with KTOTrainer

## Best Practices

1. **Monitor the table during training** - Don't just let it run blindly
2. **Check checkpoints are being saved** - Look for the 💾 messages
3. **Watch GPU memory** - If it's maxed out, reduce batch size
4. **Compare margin trends** - Should increase steadily
5. **Save the terminal output** - Use `tee` to log to file:
   ```bash
   python train_kto.py --model-size 7b 2>&1 | tee training.log
   ```

## Troubleshooting

**Table not showing:**
- Check that `src/training_callbacks.py` exists
- Verify callbacks are imported in `train_kto.py`

**Checkpoints not saving:**
- Check disk space
- Verify write permissions on output directory
- Check `save_steps` in config

**OOM (Out of Memory) errors:**
- Reduce `per_device_train_batch_size` in config
- Reduce `max_seq_length` in config
- Enable `gradient_checkpointing` for 13B+ models

Last Updated: November 14, 2025
