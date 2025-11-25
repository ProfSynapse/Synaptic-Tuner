# KTO Training Reference: TRL Bug & Dataset Requirements

**Version:** 1.0
**Date:** 2025-11-09
**Status:** ✅ Training Working

---

## TL;DR: The Fix

**Problem**: TRL's KTOTrainer crashes with `CUDA error: invalid configuration argument`
**Root Cause**: Bug in `forward()` method when batches contain only one label type
**Solution**: Use **interleaved dataset** with strict True/False/True/False pattern
**Result**: 100% mixed batches, no CUDA errors

---

## The Bug

### Location
- **Library**: HuggingFace TRL (Transformer Reinforcement Learning)
- **File**: `trl/trainer/kto_trainer.py`
- **Method**: `forward()` (around line 800)
- **Affects**: All KTO training (Unsloth doesn't patch this)

### What Happens
```python
# TRL's forward method
chosen_idx = [i for i in range(batch_size) if batch["label"][i] is True]
rejected_idx = [i for i in range(batch_size) if batch["label"][i] is False]

chosen_logps = completion_logps[chosen_idx, ...]   # ← Crashes if chosen_idx = []
rejected_logps = completion_logps[rejected_idx, ...] # ← Crashes if rejected_idx = []
```

**When batch = `[True, True]`:**
- `rejected_idx = []` (empty)
- Indexing with empty list → CUDA error

**Error Message:**
```
AcceleratorError: CUDA error: invalid configuration argument
```

### Why Weights Don't Help
`desirable_weight` and `undesirable_weight` are applied **after** the forward pass during loss computation. The bug occurs **before** weights are even considered.

---

## The Solution: Interleaved Datasets

### Structure
Create dataset with **alternating labels**:
```
Example 0: label = True   (desirable)
Example 1: label = False  (undesirable)
Example 2: label = True   (desirable)
Example 3: label = False  (undesirable)
...
```

### Why It Works
With `batch_size=2` and sequential sampling:
- Batch 0: `[True, False]` ✓ Mixed
- Batch 1: `[True, False]` ✓ Mixed
- Batch 2: `[True, False]` ✓ Mixed
- **Result**: 100% mixed batches, 0% homogeneous

### Implementation Steps

1. **Collect Examples**
   ```python
   desirable = [ex for ex in dataset if ex["label"] == True]
   undesirable = [ex for ex in dataset if ex["label"] == False]
   ```

2. **Balance Counts**
   ```python
   # Take smaller count
   min_count = min(len(desirable), len(undesirable))
   desirable = random.sample(desirable, min_count)
   ```

3. **Shuffle Each Group**
   ```python
   random.seed(42)
   random.shuffle(desirable)
   random.shuffle(undesirable)
   ```

4. **Interleave**
   ```python
   interleaved = []
   for i in range(min_count):
       interleaved.append(desirable[i])    # True
       interleaved.append(undesirable[i])  # False
   ```

5. **Verify Pattern**
   ```python
   labels = [ex["label"] for ex in interleaved]
   assert all(labels[i] != labels[i+1] for i in range(len(labels)-1))
   ```

---

## Reference Implementation

### Dataset File
**File**: `syngen_toolset_v1.0.0_claude_balanced_interleaved.jsonl`
**Location**: https://huggingface.co/datasets/professorsynapse/claudesidian-synthetic-dataset
**Examples**: 508 (254 desirable + 254 undesirable)
**Pattern**: Perfect True/False alternation

### Loading in Notebook
```python
raw_datasets = load_dataset(
    "professorsynapse/claudesidian-synthetic-dataset",
    data_files="syngen_toolset_v1.0.0_claude_balanced_interleaved.jsonl"
)
```

### Training Configuration
```python
KTOConfig(
    per_device_train_batch_size=2,
    desirable_weight=1.0,      # Equal weights for balanced data
    undesirable_weight=1.0,
    # ... other params
)
```

---

## Comparison with TRL Example

### TRL Official: `kto-mix-14k`
- 13,500 examples
- Perfect 1:1 balance (6,750:6,750)
- **Interleaved pattern**: True, False, True, False...
- Uses conversational format

### Our Dataset: `balanced_interleaved`
- 508 examples
- Perfect 1:1 balance (254:254)
- **Interleaved pattern**: True, False, True, False...
- Uses standard format (ChatML converted)

**Key Insight**: TRL's official example uses the same interleaving strategy we discovered!

---

## What We Tried (That Didn't Work)

### ❌ Approach 1: Weighted Loss
- **Tried**: `undesirable_weight = 2.94` for 746:254 imbalanced data
- **Failed**: Weights apply after forward pass; bug occurs before

### ❌ Approach 2: Random Shuffling
- **Tried**: Shuffled balanced dataset hoping for mixed batches
- **Failed**: Still got 50% homogeneous batches due to local clustering

### ❌ Approach 3: Balanced Subset (Not Interleaved)
- **Tried**: Created 254:254 balanced dataset, shuffled
- **Failed**: Sequential sampling still created homogeneous batches

### ✅ Approach 4: Interleaved Pattern
- **Success**: Guarantees mixed batches with sequential sampling
- **Matches**: TRL's official example strategy

---

## Training Verification

### Batch Inspector Output (Success)
```
First 30 labels from dataset:
[True, False, True, False, True, False, ...]

Testing first 10 batches (batch_size=2):
Batch 0: [True, False] ✓ MIXED
Batch 1: [True, False] ✓ MIXED
Batch 2: [True, False] ✓ MIXED
...

Summary:
  Mixed (both True & False): 10
  Homogeneous (all same): 0
  Percentage mixed: 100.0%
```

---

## Key Takeaways

1. **TRL KTO has a bug**: Cannot handle homogeneous batches
2. **Unsloth doesn't fix it**: `PatchKTOTrainer()` is empty
3. **Interleaving is the solution**: True/False/True/False pattern
4. **This is intentional in TRL examples**: Their official dataset uses interleaving
5. **For production**: Report bug to TRL, but use interleaving workaround now

---

## Future Work

### If Training Full Imbalanced Dataset (746:254)
Options:
1. **Custom Batch Sampler**: Guarantee mixed batches with imbalanced data
2. **Oversample Minority**: Duplicate undesirable examples to reach 746:746
3. **Switch Methods**: Use DPO or ORPO (paired preference methods)
4. **Wait for Fix**: Report bug and wait for TRL patch

### Reporting the Bug
**To**: HuggingFace TRL GitHub Issues
**Include**:
- Minimal reproduction case
- Link to this document
- Proposed fix: Add empty index check in `forward()`

---

## Related Documentation

- `finetuning-strategy.md` - Section 2.3 (updated with interleaving requirement)
- `kto_colab_notebook.ipynb` - Working implementation
- Original issue research: Session logs 2025-11-09

---

**Status**: ✅ KTO training working with interleaved dataset approach
