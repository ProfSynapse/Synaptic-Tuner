# GRPO Training Reference

Group Relative Policy Optimization — optimizes model behavior using reward functions against generated completions.

---

## Overview

GRPO generates multiple completions per prompt, scores them with deterministic reward functions, and trains the model to favor higher-reward responses. No LLM judge needed.

**Also supports GSPO** (Group Sampling Policy Optimization) via `use_gspo: true`.

---

## Configuration

GRPO is configured entirely via YAML: `Trainers/rtx3090_grpo/configs/config.yaml`

```bash
# Run GRPO training
cd Trainers/rtx3090_grpo
python train_grpo.py
```

**No CLI flag overrides** — all configuration via `config.yaml`.

---

## Key Config Settings

```yaml
model:
  model_name: "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
  lora_path: "../rtx3090_sft/sft_output_rtx3090/.../checkpoint-1150"  # Optional

training:
  per_device_train_batch_size: 6
  gradient_accumulation_steps: 6
  num_generations: 4              # Completions per prompt
  max_prompt_length: 1024
  max_completion_length: 512
  temperature: 1.0
  learning_rate: 5e-6
  beta: 0.1                       # KL penalty (higher = more stable)
  use_gspo: false                 # Toggle GSPO mode
  num_train_epochs: 1

dataset:
  local_file: "../../Datasets/my_grpo_data.jsonl"
  prompt_column: "prompt"
```

---

## Dataset Format

GRPO requires prompts with ground truth for reward scoring:

```json
{
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "ground_truth_tool": "useTools",
  "ground_truth_args_json": "{\"workspaceId\":\"default\",\"sessionId\":\"session_123\",\"memory\":\"Need to inspect and reorganize notes.\",\"goal\":\"Move a note and then read it back.\",\"constraints\":\"Do not touch unrelated files.\",\"tool\":\"storage open \\\"notes/example.md\\\"\",\"strategy\":\"serial\"}"
}
```

Ground truth uses the same `useTools` wrapper structure as model output.

---

## Reward System (YAML-Driven)

All rewards are deterministic YAML rubrics in `configs/rewards/`:

| Reward | Weight | What It Scores |
|--------|--------|----------------|
| `args_match.yaml` | 1.0 | Field-by-field comparison against ground truth |
| `json_structure.yaml` | 0.3 | Valid JSON parsing |
| `format.yaml` | 0.2 | Correct `useTools` wrapper format |
| `fitness.yaml` | 0.3 | Structural fitness via FitnessEvaluator |
| `context_completeness.yaml` | — | Context field presence |
| `tool_selection.yaml` | — | Correct tool name |

### How Rewards Work
1. Model generates `num_generations` (4) completions per prompt
2. Each completion scored by all reward rubrics
3. Scores combined with weights → single reward per completion
4. GRPO uses relative ranking within the group to compute policy gradient

### Reward Scoring Strategies

In reward YAML files:
- `binary` — 1.0 if pass, 0.0 if fail
- `proportional` — Score based on how many checks pass
- `tiered` — Different scores for different levels
- `weighted` — Weighted field mapping with per-field scores

### Structural Fitness Reward (FitnessEvaluator)

The `fitness.yaml` reward uses `FitnessEvaluator` from `shared/validation/fitness.py` to score structural correctness:

```yaml
# configs/rewards/fitness.yaml
name: structural_fitness
weight: 0.3
type: fitness_evaluator
config_path: "configs/flywheel/fitness_rules.yaml"
```

Checks performed:
- Does the tool call parse correctly?
- Is the JSON valid?
- Are required fields present?

This complements semantic rewards (`args_match`, `json_structure`) by validating the response structure against configurable rules in `fitness_rules.yaml`. Useful when training tool-calling models where structural correctness is critical.

**Tuning weight:**
- Higher weight (0.5+) when structural errors are common early in training
- Lower weight (0.1-0.2) when the model already produces valid structure

### Adding Custom Rewards

Create a new YAML in `configs/rewards/`:
```yaml
name: my_reward
weight: 0.5
strategy: binary
checks:
  - type: contains
    field: response
    value: "expected text"
```

Or load from Python:
```yaml
name: custom_reward
weight: 0.5
source: module  # or "file"
module: my_rewards.custom_fn
```

---

## Continuing from SFT

To start GRPO from an SFT checkpoint:

1. Set `model.lora_path` in config to point at SFT checkpoint
2. The trainer auto-merges the SFT LoRA into base weights
3. New LoRA adapters are applied for GRPO training

```yaml
model:
  model_name: "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
  lora_path: "../rtx3090_sft/sft_output_rtx3090/20250114/checkpoint-1150"
```

---

## GSPO Variant

Toggle in config:
```yaml
training:
  use_gspo: true
```

**GSPO workflow:**
1. Split dataset: 67% for SFT, 33% for GSPO
2. Train SFT on 67% first
3. Run GSPO on held-out 33%

Use `scripts/split_for_gspo.py` to split datasets.

---

## Key Metrics

| Metric | What It Shows | Healthy Trend |
|--------|--------------|---------------|
| `reward` | Mean reward across batch | Increasing |
| `reward_std` | Reward variance | Decreasing (model converging) |
| `kl_penalty` | KL divergence from reference | Stable, < 0.1 |
| `advantage` | Relative reward within group | Positive |
| `loss` | Policy gradient loss | Decreasing |

---

## SFT vs KTO vs GRPO

| Aspect | SFT | KTO | GRPO |
|--------|-----|-----|------|
| Purpose | Teach format | Refine preferences | Optimize rewards |
| Dataset | Positive only | Interleaved T/F | Prompts + ground truth |
| Learning rate | 2e-4 | 1e-6 | 5e-6 |
| Generations/prompt | 1 | 1 | 4 |
| Reward source | N/A | Human labels | Deterministic rubrics |
| Key metric | Loss | Margins | Reward |

---

## PivotRL (Variance-Gated Data Selection)

PivotRL profiles SFT trajectory turns to find "pivots" — turns where the model shows mixed success and failure across multiple rollouts. These high-variance turns sit at the model's decision boundary: sometimes the model gets them right, sometimes wrong. Standard GRPO wastes compute on turns the model already handles consistently (low variance) or never gets right (zero reward). Pivots provide the strongest gradient signal because the model can learn from the contrast within each group.

Training only on pivot-filtered data achieves ~4x compute reduction with comparable accuracy (per NVIDIA's PivotRL paper, arXiv:2603.21383). This matters especially on RTX 3090 where rollout generation is the bottleneck. The profiling step can run overnight, and the resulting pivot dataset trains much faster than the full SFT source.

The functional equivalence reward component ships alongside PivotRL, replacing brittle string matching with normalized tool-call comparison. This handles argument reordering, type coercion (string `"true"` → bool), path separator normalization, and whitespace differences — scoring functionally identical tool calls as equivalent even when their string representations differ.

### When to Use

- **SFT → GRPO refinement**: You have SFT trajectories and want GRPO to sharpen the model on its weakest points
- **Compute-limited training**: RTX 3090 — pivot filtering reduces the number of examples that need rollouts
- **OOD degradation from SFT**: Standard SFT can overfit to in-distribution patterns; PivotRL-style GRPO preserves out-of-distribution capabilities by focusing on decision-boundary turns

### Quick Start

```bash
# Step 1: Profile SFT data to find pivots (can run overnight)
cd Trainers/grpo
python train_grpo.py --config configs/pivot_config.yaml --pivot-profile-only

# Step 2: Train on pivot-filtered data
python train_grpo.py --config configs/pivot_config.yaml
```

Profiling results are cached to `Datasets/grpo/.pivot_cache/`. Changed rewards or SFT data invalidate the cache automatically (key = file hash + model name + reward config hash).

### Config Reference

Add the `pivot:` section to any GRPO config YAML. Omit it or set `enabled: false` for standard GRPO — zero behavior change.

```yaml
pivot:
  enabled: true
  sft_source: null              # SFT JSONL to profile (null = use dataset.local_file)
  profiled_file: null           # Pre-profiled pivot dataset (skip profiling if set)
  profiling:
    n_rollouts: 8               # Rollouts per candidate turn (4-16 recommended)
    temperature: 1.0            # Sampling temperature during profiling
    max_completion_length: 512  # Max tokens per rollout
    batch_size: 16              # Inference batch size
  filtering:
    variance_threshold: 0.1     # Min reward std to qualify as pivot (0.05-0.2 typical)
    min_candidates: 50          # Warning if fewer pivots found
    max_candidates: null        # Optional cap
    mean_reward_range: null     # Optional [min, max] band (e.g., [0.2, 0.8])
  cache:
    enabled: true
    cache_dir: null             # Default: Datasets/grpo/.pivot_cache/
```

| Field | Default | Notes |
|-------|---------|-------|
| `sft_source` | `null` | Falls back to `dataset.local_file` |
| `profiled_file` | `null` | Point at a pre-profiled JSONL to skip profiling entirely |
| `n_rollouts` | 8 | Higher = more accurate variance estimate, slower profiling |
| `variance_threshold` | 0.1 | Lower = more pivots (lenient), higher = fewer pivots (strict) |
| `min_candidates` | 50 | Logs a warning if fewer pivots pass the filter |
| `max_candidates` | `null` | Optional hard cap, takes highest-variance first |
| `mean_reward_range` | `null` | Optional band filter, e.g. `[0.2, 0.8]` to exclude trivial/impossible turns |

### Functional Equivalence Reward

Added via the `rewards` section or as a standalone YAML in `configs/rewards/functional_equivalence.yaml`:

```yaml
name: functional_equivalence
type: custom
module_file: "../src/functional_verifier.py"
function_name: "functional_equivalence_reward"
default_weight: 0.5
```

Normalization handles: argument key reordering, type coercion (`"true"` → `true`, `"42"` → `42`), path separator normalization (`\` → `/`), and whitespace stripping. Scoring: 1.0 (fully equivalent), partial (same tool, partial arg match), 0.0 (wrong tool or unparseable).

### Key Metrics to Watch

| Metric | What to Check | Action |
|--------|---------------|--------|
| **Pivot coverage** | % of SFT turns that qualified as pivots | 10-40% is typical |
| **Variance distribution** | Check profiling output log for reward std stats | Bimodal = good signal |
| **Filtered count** | Total pivots after filtering | Too few → lower `variance_threshold`; too many → raise it |
| **Mean reward band** | Distribution of pivot mean rewards | All near 0.0 or 1.0 = threshold too low |

### Relationship to Other Systems

- **Loss pipeline** (`prune_dataset_from_loss.py`): Post-hoc difficulty analysis after training. PivotRL does pre-training difficulty analysis. Complementary — chain them: PivotRL for data selection → loss analysis after training for next-iteration cleanup.
- **Evolutionary model** (`shared/evolutionary/`): Same generate-score-select pattern at the weight-update level. PivotRL applies it at the data-selection level.
- **Env-GRPO** (`train_env_grpo.py`): Architectural precedent for config-activated mode. PivotRL follows the same pattern: separate config preset, conditional branch in trainer.

---

## Platform Note

GRPO requires **WSL/Linux only** — native Windows is not supported.
