# Troubleshooting Reference

Common issues and fixes for training.

---

## CUDA Out of Memory (OOM)

**Symptoms:** `CUDA error: out of memory`, training crashes

**Fixes (in order of impact):**
1. Reduce `--batch-size` (e.g., 2 → 1)
2. Increase `gradient_accumulation_steps` (maintain effective batch)
3. Reduce `--max-seq-length` (e.g., 2048 → 1024)
4. Use smaller model (`--model-size 3b`)
5. Verify `load_in_4bit: true` in config
6. Verify `optim: adamw_8bit` in config
7. Check no other GPU processes: `nvidia-smi`

**KTO-specific:** If using explicit reference model, switch to implicit:
```yaml
# Don't set USE_EXPLICIT_REF_MODEL=true (saves ~8GB)
```

---

## NaN/Inf Loss

**Symptoms:** Loss shows NaN or Inf, training diverges

**Fixes:**
1. Reduce learning rate (`--learning-rate 1e-5`)
2. Reduce `max_grad_norm` (e.g., 1.0 → 0.5)
3. Enable gradient checkpointing
4. Check dataset for malformed examples
5. For KTO: increase `beta` (more conservative)

---

## KTO Crashes on Homogeneous Batches

**Symptoms:** `CUDA error: invalid configuration argument`

**Cause:** TRL bug — crashes when batch has all True or all False labels

**Fix:** Ensure dataset has both labels — data loader auto-interleaves. If still crashing:
- Check your dataset actually has both `label: true` and `label: false`
- Verify labels aren't strings: must be boolean `true`/`false`, not `"true"`/`"false"`

---

## Training Stalls / Hangs

**Symptoms:** No progress, 0 steps/sec

**Fixes:**
1. Set `dataloader_num_workers: 0` (required on WSL2)
2. Set `group_by_length: false` (can hang with multiprocessing)
3. Set `num_proc: 1` in dataset config
4. Check GPU isn't throttling: `nvidia-smi -l 1`

---

## High Gradient Norms

**Symptoms:** Warnings about gradient norm > 100

**Fixes:**
1. Reduce learning rate
2. Ensure `max_grad_norm: 1.0` is set (gradient clipping)
3. Check dataset quality — outlier examples can cause spikes
4. For GRPO: high gradient norms may indicate reward function issues

---

## Loss Not Decreasing (SFT)

**Symptoms:** Loss flat after multiple epochs

**Fixes:**
1. Increase learning rate (try 5e-4)
2. Increase epochs
3. Verify `completion_only_loss: true` — without it, loss includes user tokens
4. Check dataset isn't too small (< 100 examples may need more epochs)
5. Verify packing is working: look for "packing enabled" in output

---

## KTO Margins Not Increasing

**Symptoms:** `rewards/margins` stays near 0 or decreases

**Fixes:**
1. Check dataset balance (True vs False count)
2. Try `use_kto_s: true` for more stable training
3. Enable two-stage LR schedule
4. Reduce beta for more exploration (try 0.05)
5. Verify examples are meaningfully different (not trivial positive/negative)

---

## GRPO Rewards Stay Low

**Symptoms:** Mean reward doesn't improve

**Fixes:**
1. Check reward rubrics in `configs/rewards/` — verify they match your data format
2. Increase `num_generations` (more candidates = better exploration)
3. Reduce beta (allow more divergence)
4. Verify ground truth format matches model output format
5. Try different temperature (1.0 → 0.7 for more focused generation)

---

## Checkpoint Resume Fails

**Symptoms:** Error when using `--resume-from-checkpoint`

**Fixes:**
1. Verify checkpoint path exists and contains `trainer_state.json`
2. Use same config as original run
3. If config changed, start fresh instead of resuming

---

## WSL-Specific Issues

| Issue | Fix |
|-------|-----|
| Slow file I/O | Keep data on Linux filesystem, not `/mnt/c/` |
| GPU not detected | Install NVIDIA drivers on Windows host |
| Memory pressure | Close Windows apps, set WSL memory limit in `.wslconfig` |
| Symlinks fail | Use `cp` instead of `ln -s` if filesystem doesn't support |

---

## Platform Compatibility

| Feature | Linux | WSL2 | Windows (native) |
|---------|-------|------|-------------------|
| SFT | Full | Full | Limited |
| KTO | Full | Full | Limited |
| GRPO | Full | Full | Not supported |
| Flash Attention | Full | Full | Not supported |
| Multiprocessing | Full | Limited (num_workers=0) | Limited |

---

## Quick Diagnostics

```bash
# Check GPU
nvidia-smi

# Check CUDA
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Check Unsloth
python -c "from unsloth import FastLanguageModel; print('OK')"

# Check TRL version
python -c "import trl; print(trl.__version__)"

# Validate dataset
python Tools/validate_syngen.py Datasets/my_data.jsonl
```
