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
  "ground_truth_tool": "vaultManager_openNote",
  "ground_truth_args_json": "{\"context\": {...}, \"calls\": [{...}]}"
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

Use `tools/split_for_gspo.py` to split datasets.

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

## Platform Note

GRPO requires **WSL/Linux only** — native Windows is not supported.
