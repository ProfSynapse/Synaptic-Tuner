# GRPO Training Reference

Group Relative Policy Optimization — optimizes model behavior using reward functions against generated completions.

---

## Overview

GRPO generates multiple completions per prompt, scores them with deterministic reward functions, and trains the model to favor higher-reward responses. No LLM judge needed.

**Also supports GSPO** (Group Sampling Policy Optimization) via `use_gspo: true`.

---

## Configuration

GRPO is configured entirely via YAML: `Trainers/grpo/configs/config.yaml`

```bash
# Run GRPO training
cd Trainers/grpo
python train_grpo.py
```

**No CLI flag overrides** — all configuration via `config.yaml`.

---

## Key Config Settings

```yaml
model:
  model_name: "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
  lora_path: "../sft/sft_output/.../checkpoint-1150"  # Optional

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

Or load Python functions from a file via the trainer config (`rewards.custom`;
module-path loading and per-item `params` are not supported and are refused):
```yaml
rewards:
  custom:
    enabled: true
    file: "./my_rewards.py"        # relative to Trainers/grpo/
    functions:
      - name: custom_fn
        weight: 0.5
```

Both GRPO entrypoints refuse YAML keys they do not read (`GRPO_CONFIG_SCHEMA` /
`ENV_GRPO_CONFIG_SCHEMA`), and any `training.*` / `training.extra_args.*`
setting the installed TRL `GRPOConfig` does not accept raises instead of being
dropped (e.g. TRL 0.28 removed `max_prompt_length`).

---

## Continuing from SFT

To start GRPO from an SFT checkpoint:

1. Set `model.lora_path` in config to point at SFT checkpoint
2. The trainer auto-merges the SFT LoRA into base weights
3. New LoRA adapters are applied for GRPO training

```yaml
model:
  model_name: "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
  lora_path: "../sft/sft_output/20250114/checkpoint-1150"
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

## Token-Faithful Multi-Turn Rollout (Env-GRPO)

Environment-backed GRPO (`train_env_grpo.py`) rolls out multi-turn episodes: the
model emits a tool call, the environment executes it, the result is fed back, the
model responds again, and so on. By default the trainer now serializes these
episodes **token-faithfully** (POLAR-style, arXiv:2605.24220): GRPO trains on
exactly the tokens the model sampled while still conditioning on the real
tool-result / user-feedback context.

**Why it matters.** The legacy serialization flattened every assistant turn's
tokens into one contiguous completion and dropped the intermediate tool-result /
user-feedback tokens entirely. GRPO then optimized a fictional stream that never
existed at sampling time — a train/inference mismatch. Faithful mode keeps the
full interleaved sequence and supplies a per-token `env_mask` (model token = 1,
external context token = 0) that TRL multiplies into the completion loss mask: the
model attends to the real context but is trained only on its own tokens.

### Config

In `Trainers/grpo/configs/env_config.yaml` under `env_training:`:

```yaml
env_training:
  token_faithful: true          # emit faithful sequence + env_mask (default)
  context_token_policy: mask     # "mask" = faithful; "drop" = legacy flattened (A/B only)
```

- **Single-turn episodes are byte-identical** under either policy — the change
  only affects episodes with more than one assistant turn.
- Requires **`trl>=0.28.0`** (when `env_mask` became a real loss mask). The rollout
  builder probes the installed TRL via `detect_openenv_runtime_support()['has_env_mask']`
  and, if the runtime can't honor `env_mask`, **auto-falls back to the legacy
  flattened path** (with a warning) rather than train on context tokens.
- Use `context_token_policy: drop` only to A/B against the old behavior.

Related keys (not under `env_training`):

```yaml
model:
  model_revision: null          # Hub commit/branch/tag for model AND tokenizer (CLI: --model-revision)
training:
  chat_template_kwargs: null    # e.g. {enable_thinking: false} for Qwen3/Qwen3.5
```

- `model.model_revision` pins `AutoTokenizer.from_pretrained(..., revision=)` and
  the trainer's model load (`GRPOConfig.model_init_kwargs.revision`). A different
  `revision` in `training.extra_args.model_init_kwargs` is refused.
- `training.chat_template_kwargs` uses the same key and validator as the SFT/GRPO
  trainers. It is applied to the dataset prompt render and to **every** chat-template
  render inside the rollout (first prompt and each per-turn context suffix).

**TRL version note.** TRL 1.9 removed `trl.experimental.openenv.generate_rollout_completions`.
The rollout imports it only when `training.use_vllm: true`, so the transformers
rollout path works on any TRL `>=0.28` (including 1.9+). The vLLM path needs a TRL
that still ships the helper (0.28–1.8). It sends token-id prompts: colocate mode
works on all of those, while server mode needs a `VLLMClient.generate` that accepts
token-id lists (TRL 1.x).

### How the sequence is built (prefix-stable)

The rollout never re-renders earlier turns. Turn 1's prompt is the chat-template
render of the episode's initial messages. Every later prompt is built from ids:

```
prompt(t+1) = prompt(t) + completion(t) + suffix(t)
```

`suffix(t)` holds only what the model sees next: the assistant end-of-turn marker
if the sampled completion lacks one, the new tool-result / feedback / final-text
messages, and the generation prompt. It is computed with the technique TRL's own
multi-turn tool loop uses (`GRPOTrainer._get_tool_suffix_ids`): render a fixed
minimal conversation ending in an assistant turn with the new messages and
generation prompt appended, and keep the ids after that assistant turn, aligned at
the end-of-turn (EOS) id. TRL locates that boundary by rendering the dummy a second
time without the new messages, which only works on prefix-preserving templates
(TRL swaps in its own training template otherwise). Here the dummy assistant
content is a sentinel and the suffix is the tokenized text after it, so templates
that render the dummy differently depending on what follows (Qwen3.5) still work.
If the completion ended with a stop id (tokenizer / generation-config EOS), the
template's marker up to and including the first stop id is dropped. If it did not
(truncated at `max_completion_length`, no EOS sampled), the whole marker is kept as
masked context. Nothing is specific to one template.

The prompt ids go to the generator directly (transformers `generate(input_ids=…)`;
vLLM `{"prompt_token_ids": …}` in colocate mode, an id list in server mode), so the
model conditions on exactly these ids. The decoded completion text is still
appended to `messages` for response parsing and the environment; it is never
re-tokenized into the prompt.

Why: re-rendering the whole conversation each turn diverges from what the model
saw. Qwen3.5's template drops `<think>` from every assistant turn before the latest
user message, so with re-rendering every multi-turn episode would be truncated to
turn 1. With prefix-stable construction the model keeps seeing its own earlier
reasoning, exactly as sampled.

The assembled episode is:

- `prompt_ids` = the first turn's prompt.
- `completion_ids` = assistant turn 1 ++ suffix(1) ++ assistant turn 2 ++ …. The
  context ids are the tail of the next turn's prompt past `prompt + completion`.
  Assistant spans are the raw sampled ids.
- `env_mask` = 1 on assistant tokens, 0 on context tokens (same length as
  `completion_ids`).
- `logprobs` = sampling log-probs on assistant tokens, 0.0 on context tokens.

### Prefix verification (safety net)

Every transition is still checked: turn `t`'s prompt must **start with**
`prompt(t-1) + completion(t-1)` (idea from agent-lightning's `ids_startswith`). The
rollout records the prompt ids the generator reports it conditioned on, so this
only fires if a backend altered them (for example by adding a BOS); with
prefix-stable construction it should never fire in normal operation. On the first
transition that fails (including a prompt shorter than that prefix), the episode's
faithful sequence is **truncated after the last consistent turn**, so every kept
token is exactly what the model saw or sampled. TRL accepts one row per episode, so
later turns are dropped rather than spliced in with the wrong context. The episode
reward is unchanged.

Observability (no config key; always on in faithful mode):

- `prefix_mismatch_count` per episode — on `EpisodeRolloutResult`, as an extra
  rollout output column (reaches reward functions as a kwarg), and in the
  `debug_rollouts_path` JSONL record.
- A per-episode `WARNING` from `env_rollout` with turn index, expected prefix
  length, prompt length and first divergence position, plus a per-batch
  `N/M episodes` summary.

Any non-zero `prefix_mismatch_count` is a bug in the generation backend or the
rollout, not a template quirk. Investigate it rather than ignoring the warning: the
multi-turn signal for those episodes is being truncated.

### What TRL does with it (verified against trl source)

`completion_mask` is all-ones over `completion_ids`, so the full interleaved
sequence (assistant + context) is **attended** — the model, the `old`/reference
logprobs, and the importance-sampling ratio are all computed conditioned on the
real context. The loss uses `mask = completion_mask * tool_mask`, and every loss
variant normalizes by `mask.sum()`, so context tokens contribute **zero to the
loss and are excluded from the denominator** (no dilution). Our supplied per-token
`logprobs` only feed the vLLM importance-sampling correction and are themselves
multiplied by the mask, so the `0.0` placeholders on context tokens are inert.
Reward grouping is unchanged (one record per episode → `rewards.view(-1,
num_generations)`), and length metrics count only model tokens (`sum(env_mask)`).

### Caveat — sequence-length budget

In faithful mode the tool-result / user-feedback tokens live **inside**
`completion_ids` (masked), so the trained completion spans all turns *plus* their
context. `max_completion_length` caps each turn's generation, not the episode
total, so the full sequence can be several times `max_completion_length`. Size
`max_seq_length` / VRAM for the whole interleaved episode. (Legacy `drop` mode
already concatenated assistant turns; faithful mode adds the context tokens on
top.)

### Verifying

`tests/trainers/grpo/test_env_rollout_faithful.py` covers single-turn parity, the
interleaved multi-turn sequence, mask/logprob length alignment, the capability
gate, the prefix check and the rollout_func output contract.
`tests/trainers/grpo/test_env_rollout_prefix_stable.py` runs 3-turn episodes
through a Qwen3.5-like template (zero mismatches, exact suffixes, truncated
completions without EOS, chat-template kwargs, token-id prompts, no TRL rollout
helper). `RUN_LIVE_HUB=1` adds the same check against the real Qwen3.5-4B tokenizer
at the smoke recipe's pinned revision (tokenizer files only).
`tests/trainers/grpo/test_env_grpo_model_revision.py` covers the revision pin and
chat-template kwargs wiring in `train_env_grpo.py`. Before a full launch, run a short
env-GRPO smoke job and confirm the run does not raise a length/logprob mismatch.

---

## Platform Note

GRPO requires **WSL/Linux only** — native Windows is not supported.
