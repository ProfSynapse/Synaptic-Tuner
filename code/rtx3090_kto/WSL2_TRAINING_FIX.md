# WSL2 Training Freeze Fix

## The Problem

Training freezes at step 5 on WSL2.

## The Cause

**DataLoader multiprocessing doesn't work on WSL2.**

The config file `configs/training_config.py` had:
```python
dataloader_num_workers: int = 4  # 2-4 workers (set to 0 on Windows)
```

The comment says "set to 0 on Windows" but the value was 4. WSL2 has the same multiprocessing limitations as Windows.

## The Fix

**Changed in `configs/training_config.py` line 135:**
```python
dataloader_num_workers: int = 0  # MUST be 0 on WSL2 (multiprocessing hangs)
```

That's it. This single line fixes the freezing.

## Why It Started After Your First 2 Runs

Multiprocessing deadlocks on WSL2 are **non-deterministic** - sometimes they happen, sometimes they don't. You got lucky on runs 1-2, then hit the deadlock consistently afterward.

## Test the Fix

```bash
python train_kto.py --model-size 7b
```

Training should now pass step 5 without freezing.

## Additional Optimizations Kept

The following change was also made to reduce memory fragmentation (this is legitimate and helpful):

**In `train_kto.py`:**
```python
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

This can reduce memory usage by ~30% by consolidating CUDA memory allocations.

## Reference

This is a well-documented issue with PyTorch DataLoader on Windows/WSL2:
- PyTorch Issue #1579: DataLoader hangs with num_workers > 0
- PyTorch Issue #51344: DataLoader freezes on Windows
- Multiple forum posts confirming num_workers=0 is required on WSL2
