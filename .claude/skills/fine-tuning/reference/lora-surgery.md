# LoRA Weight Surgery Reference

Eval-guided post-training optimization of LoRA adapters. Tries weight operations, evaluates each, keeps improvements.

---

## Quick Start

```bash
# Interactive mode
python tuner.py surgery

# With config
python tuner.py surgery --config configs/lora_surgery.yaml
```

---

## How It Works

1. **Copy** — Adapter copied to temp directory (original untouched)
2. **Modify** — Apply a weight operation (e.g., scale layer weights)
3. **Evaluate** — Run eval scenarios against modified adapter
4. **Decide** — Keep if score improves by `min_improvement` (default 0.005), discard otherwise
5. **Repeat** — Next operation starts from the best state so far

Operations run cheapest-first by default.

---

## Configuration

**File:** `configs/lora_surgery.yaml`

```yaml
surgery:
  min_improvement: 0.005        # Min eval score gain to keep a change
  eval_backend: "local"         # "local" or "cloud"
  cloud_provider: "hf_jobs"     # Used when eval_backend is "cloud"

  # Operations to run, in order
  operations:
    - alpha_sweep
    - layer_scaling
    - module_ablation
    - checkpoint_interpolation
    - dare_drop_rescale
    - svd_rank_reduction
    - attention_mlp_ablation

  # Per-operation parameters
  alpha_sweep:
    multipliers: [0.5, 0.75, 1.25, 1.5, 2.0]

  layer_scaling:
    scales: [0.0, 0.5, 0.75, 1.25, 1.5]

  dare:
    drop_rates: [0.1, 0.2, 0.3, 0.5]

  checkpoint_interpolation:
    blend_ratios: [0.25, 0.5, 0.75]

  svd_rank_reduction:
    rank_fractions: [0.25, 0.5, 0.75]
```

---

## Operations

### Alpha Sweep
Scales the global `lora_alpha` by different multipliers. Quick way to find if the adapter is over- or under-weighted.
- **Multipliers:** `[0.5, 0.75, 1.25, 1.5, 2.0]`
- **Cost:** Low (single weight multiplication)
- **When to use:** First — cheapest operation, often yields quick wins

### Layer Scaling
Scales individual layer weights by different factors. Finds layers that help or hurt.
- **Scales:** `[0.0, 0.5, 0.75, 1.25, 1.5]`
- `0.0` effectively removes a layer's contribution
- **Cost:** Medium (one eval per layer × scale)
- **When to use:** When some layers may have overfit

### Module Ablation
Zeros out entire module types (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj).
- Tests which projection types the adapter actually needs
- **Cost:** Low (one eval per module type)
- **When to use:** To identify unnecessary adapter modules for pruning

### Checkpoint Interpolation
Blends two checkpoints at different ratios (e.g., 75% primary + 25% secondary).
- **Ratios:** `[0.25, 0.5, 0.75]`
- Requires a second checkpoint path
- **Cost:** Low (one eval per ratio)
- **When to use:** When you have multiple checkpoints from the same run

### DARE Drop-and-Rescale
Randomly drops adapter weights, then rescales survivors to preserve magnitude. From the DARE paper.
- **Drop rates:** `[0.1, 0.2, 0.3, 0.5]`
- Guards against division by zero when drop rate approaches 1.0
- **Cost:** Low (one eval per drop rate)
- **When to use:** To reduce adapter density / improve generalization

### Metrics-Weighted Merge
Merges N checkpoints, weighting each by its eval score.
- Requires multiple checkpoint paths
- **Cost:** Low (single merge + eval)
- **When to use:** When you have several checkpoints and want a single best-of-all

### SVD Rank Reduction
Compresses LoRA via truncated SVD to a fraction of original rank.
- **Rank fractions:** `[0.25, 0.5, 0.75]`
- Reduces adapter size while preserving most important directions
- **Cost:** Medium (SVD computation + eval per fraction)
- **When to use:** When adapter is over-parameterized or you need smaller deployment

### Attention/MLP Ablation
Zeros all attention weights vs. all MLP weights separately.
- Tests whether the adapter's value is in attention or MLP layers
- **Cost:** Low (two evals)
- **When to use:** Diagnostic — understand where the adapter learned

---

## CLI Integration

The `surgery` command is available via the tuner CLI:

```bash
# Interactive — prompts for adapter path and scenarios
python tuner.py surgery

# Also accessible from the interactive menu
python tuner.py
# Select: Surgery
```

The handler (`tuner/handlers/surgery_handler.py`) guides you through:
1. Selecting an adapter path (or loading from config)
2. Selecting eval scenarios
3. Configuring which operations to run
4. Running surgery asynchronously
5. Displaying results with improvements

---

## Tips

- Run `alpha_sweep` and `module_ablation` first — cheapest and most informative
- Use `--dry-run` on your training first to confirm the adapter loads correctly
- Surgery works on any LoRA adapter directory (PeftModel format)
- The original adapter is never modified — all operations work on copies
- Set `min_improvement: 0.0` to see all results (even regressions) for diagnostics
- For large models, use `eval_backend: "cloud"` to offload evaluation
