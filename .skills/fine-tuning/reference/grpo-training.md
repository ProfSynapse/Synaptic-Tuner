# GRPO Training Reference

Group Relative Policy Optimization — optimizes model behavior using reward functions against generated completions.

## Public GRPO declarations and CPU preparation

The provider-neutral `TrainingInputV1` accepts explicit method `grpo` with
`synaptic-grpo-hyperparameters/v1` and rewards `{items, documents, schema}`.
This lane supports **CPU preparation, compilation and the ordinary Modal runner
join for an explicitly compatible host-supplied profile**. It does not establish
model/image compatibility or qualify GRPO training. The legacy trainer commands below remain separate. The generic Modal private-copy
helper can preserve an already verified GRPO prompt publication through these
ports. A strict normal Modal recipe can report a core-only preplan when the host
supplies an explicitly compatible named GRPO profile; no ordinary candidate
profile/image is shipped or qualified yet. Missing or incompatible profiles reject.
Ordinary execution and CPU qualification use the normal pinned host path.
Provider quotation and metrics qualification remain refused for GRPO.

The host material boundary can also validate an explicitly supplied packaged GRPO
configuration against the exact G1 core, original public `TrainingInputV1`, actual
runtime-release context and finite child limit. Its fixed lazy method selector
preserves the SFT projection and requires the original retained context for cold
material readback. This CPU material capability does not enable provider dispatch,
log/evaluation capability or live qualification. The default recipe registry
remains unchanged; the host request bridge lazily registers the existing GRPO
recipe only for an explicit GRPO request. Hosts must not fabricate a runtime release to
produce a final packaged plan before runtime qualification.

The fixed Modal parent worker can parse an exact signed GRPO dispatch and route
it through the isolated GRPO executor. Native verification must succeed before
it signs completion. The existing reader authenticates that completion and
streams the exact five artifacts with bounded size and hash checks. GRPO does
not accept SFT runner/evaluation callbacks, post-training evaluation or capture
configuration through this route. Its trainer environment keeps the existing
closed override policy; image PATH and the child's derived executable search
path are separate from a dispatch override.

The installed CPU control joins the actual owned child and native verifier to
real HMAC completion and actual reader streams, using explicit ML/image/provider
substitutes. Fresh reader objects preserve the original binding and key. This is
not cold public-host SQLite recovery or provider/GPU qualification. The ordinary
runner now joins this worker through its actual CPU qualification receipt and
public TrainingAPI. No logs, cancellation or reconciliation capability is
promoted by that join.

For the existing `train --job-config <recipe> --plan --json` surface, a GRPO recipe
uses `target: cloud`, `provider: modal`, `method: grpo`,
`run: {method: grpo, dry_run: false}`, existing bounded `job` fields, and the exact
canonical public GRPO object under `training_input`. Its `dataset` declares only
`local_file`, `schema_version: synaptic-grpo-prompts/v1`, `expected_digest` and
`expected_row_count`; no SFT split or target fields. `limits.child_max_seconds`
is a finite isolated-child budget distinct from the provider job timeout. Rewards
remain declared `{items, documents, schema}`; no ambient directory fallback.
Post-training evaluation, observation capture and environment rollout are outside
this initial recipe scope.

Planning checks the actual original JSONL bytes through the existing bounded
same-descriptor publication inspector and core admission. It reports only the
core fingerprint, immutable prepared identity, named build intent and child limit,
with an explicit runtime-binding-and-CPU-qualification-required status; it cannot report the final wrapped
workload/configuration digest before an actual release exists. The existing
`--prepare-local` mode remains local source-wheel/bootstrap preparation, subject
to the normal pinned launcher and accepted engine source requirements.

The internal final-material hook requires the genuine retained local preparation,
revalidates its original profile/source/intent, and compares the supplied actual
release's material exactly before deriving the final wrapper and public planning
inputs. Keep those local inputs open through final binding. CPU tests use explicit
synthetic host profile/runtime observations and a scripted wheel builder, then the
real public load/resolve/plan/preflight and SQLite planning-store close/reopen.
This proves saved plan/context readback only, not a restored provider host,
training resume or a working model/image. The normal runner retains these exact
local inputs through the actual CPU check and final binding, then uses public
prepare/load/resolve/plan/preflight/start and native-verified signed readback.

The existing internal CPU self-check selects SFT or GRPO only from the exact
validated release contracts. It verifies the corresponding fixed worker closure,
selected trainer source in the pinned wheel, and the isolated child's source
compile. The existing CPU protocol, sealed fixture and signed receipt stay the
same; unknown or mixed method contracts reject. Legacy SFT CPU schema remains
accepted. No caller method override selects a different closure.

Source controls use private test-owned captures of the same fixed member lists
and explicit package/image/device-namespace observations. The parent and real
isolated child retain actual environment, sealed-FD, private-root, selected import
guard and trainer-source checks. These controls do not prove container isolation,
GPU image compatibility or training. The installed runner join uses the actual
CPU child, signed qualification receipt/operator readback, public lifecycle,
owned isolated trainer, native verification, and signed five-artifact streaming.
Its deterministic provider/image/ML observations remain explicit test substitutes.
The same `train --job-config <recipe>` surface executes the joined path; `--qualify`
reports CPU-only evidence under `synaptic-modal-grpo-cpu-qualification/v1`, and a
successful ordinary run reports `synaptic-modal-grpo-run-result/v1`. The normal
pinned 3.11.14 launcher, clean source and named compatible profile are still
required. The first executor requires equal model/tokenizer revisions and does
not rewrite an unsupported selection. A corrupted CPU receipt stops before
training; a lost submission reply preserves the consumed claim and uncertain
workflow, even if the scripted provider completed. Cold public-host/fresh-process
GRPO recovery and live image/model qualification remain separate acceptance
steps. No new GRPO candidate profile is shipped by this runner join.

Use the existing generic preparation service with explicit host-owned roots:

```python
from pathlib import Path
from synaptic_tuner.api.v1.training_sources import (
    LocalTrainingInputPathV1, TrainingPreparationConfigV1,
)
from tuner.training.grpo_preparation import (
    GRPONormalizerConfigV1, GRPOTrainingInputNormalizerV1,
    GRPOPreparedInputFormatVerifierV1,
)
from tuner.training.input_preparation import (
    PosixPrivatePreparedRootAuthorityV1, TrainingInputPreparationServiceV1,
)

# source_root and prepared_root are explicit absolute caller-owned 0700 Paths.
normalizer = GRPOTrainingInputNormalizerV1(allowed_source_roots=(source_root,))
verifier = GRPOPreparedInputFormatVerifierV1()
service = TrainingInputPreparationServiceV1(
    prepared_root=prepared_root,
    normalizers={"synaptic.grpo-prompts/v1": normalizer},
    format_verifiers={verifier.format: verifier},
    root_authority=PosixPrivatePreparedRootAuthorityV1(),
)
result = service.prepare(
    LocalTrainingInputPathV1(source_root),
    TrainingPreparationConfigV1(
        request_id, project_ref, canonical_training_json,
        GRPONormalizerConfigV1(member_name, prompt_column),
    ),
)
```

`canonical_training_json` is the exact validated GRPO training input, not a
legacy trainer YAML document. Choose `prompt_column` explicitly in both its
hyperparameters and the normalizer config. The source is one declared private,
regular, single-link JSONL leaf beneath the exact admitted source root. This
adapter currently requires Linux retained-descriptor support.

Prepared format is `synaptic-grpo-prompts/v1`, with at most 64 KiB and 128 rows.
Original UTF-8 bytes and finite reward columns are preserved; BOM, duplicate
fields, nonfinite values and ambiguous reserved `prompt` column collisions
reject. There are no invented SFT targets or splits. A closed manifest binds
format, prompt column, byte/hash identity and row count. Fresh writes are
exclusive; independently verified identical publications may be reused across
legitimate model, reward and hyperparameter changes. Changed destinations reject.
Uncertain publication failures retain their owned stage/publication and report
failure.

The optional verifier input check validates the selected request through the
actual retained GRPO core using the same verified bytes, before the generic
service verifies the retained lease and returns. `result.prepared` is the public
identity/request; `result.retained_source` is host-only input custody. Keep that
source for later explicitly supported composition; do not serialize it or infer
execution authority from its identity.

For pure compilation, explicitly register `tuner.training.GRPORecipe` in an
existing `RecipeRegistry`. The resolved `CanonicalDocument` has exactly
`schema_version: synaptic-grpo-config/v1`, `method: grpo`,
`execution: {mode: developer_integration}`, `training_input` from the prepared
request, and `prepared_input` from its identity. Pass an actual authenticated
`ExecutionSourceV1` to existing `compile_execution_workload`. The default registry
remains SFT-only. The resulting `synaptic-grpo-workload/v1` commits normalized
config, original dataset bytes, model/tokenizer revisions and declarative reward
documents/schema. Compilation alone does not attest rows, reward dispersion,
parameter updates or a runnable packaged runtime.

The first declaration lane is native-template, step-limited, single-process
plain GRPO with raw-response binary regex rewards and explicit empty reward
schema. Custom Python, judges, environment rollout, GSPO, pivot, existing adapter
loading and checkpoint retention are unsupported here. Precision follows the
retained core's hardware selection; both precision flags false do not promise
FP32. See `docs/plans/public-grpo-compilation.md` (repository root)
for the exact fields, schemas and limits. Packaged dispatch and live qualification are
separate gates; do not use a provider-specific launcher to bypass them.

---

## Overview

GRPO generates multiple completions per prompt, scores them with deterministic reward functions, and trains the model to favor higher-reward responses. No LLM judge needed.

**Also supports GSPO** (Group Sampling Policy Optimization) via `use_gspo: true`.

---

## Packaged GRPO candidate execution

The isolated packaged candidate wraps the exact public GRPO core workload with
an authenticated runtime release and configured `child_max_seconds` (integer
1–86400). The budget starts once after model preparation, immediately before
child execution; supervision uses the same absolute deadline throughout. Up to
five additional seconds confirm cleanup. This is a child-only bound, not an
end-to-end provider deadline or public cancellation qualification. The first
executable candidate requires identical model and tokenizer revisions because
both consume one verified local snapshot.

GRPO has a separate fixed 131-member offline inventory and 13-member outer
closure. Existing SFT inventories remain unchanged. Maintenance uses the
checked-in command, offline first because the outer closure binds its bytes:

```bash
python scripts/regenerate_grpo_worker_closures.py --kind offline
python scripts/regenerate_grpo_worker_closures.py --kind outer
```

The default checks without writing. Use `--write` only for an intentionally
reviewed source-hash refresh; it preserves fixed membership and metadata.
`--initialize` creates a missing manifest exclusively and requires explicit
inventory/schema/policy review first. It must not overwrite or adopt resources.

CPU controls execute the real child, configuration, reward and save paths with
explicit model/trainer/image substitutes. Separate controls exercise the actual
selected import guard, full wheel Python checks, retained model/input custody,
mandatory adapter/tokenizer saves, native five-role archives, and closed cleanup
failures. The installed lane consumes wheel schemas and both manifests. These
checks do not qualify a GPU image, provider execution, parameter changes, reward
dispersion or model quality. Public provider routing and live candidate
qualification remain separate gates; step counts alone are not update evidence.

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

### How the sequence is built

For each episode the rollout records per-turn `(prompt_ids, completion_ids,
logprobs)` from `generate_rollout_completions`, then assembles:

- `prompt_ids` = the first turn's rendered prompt.
- `completion_ids` = assistant turn 1 ++ context(1→2) ++ assistant turn 2 ++ …,
  where context tokens are the tail of the next turn's rendered prompt (the same
  length-delta slicing TRL's internal tool loop uses). Assistant spans use the raw
  sampled token ids, so trained tokens stay byte-faithful.
- `env_mask` = 1 on assistant tokens, 0 on context tokens (same length as
  `completion_ids`).
- `logprobs` = sampling log-probs on assistant tokens, 0.0 on context tokens.

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
gate, and the rollout_func output contract. Before a full launch, run a short
env-GRPO smoke job and confirm the run does not raise a length/logprob mismatch.

---

## Platform Note

GRPO requires **WSL/Linux only** — native Windows is not supported.

## Optional bounded training measurements

Packaged GRPO limits may include `qualification_evidence` with exactly
`max_trainable_bytes` (1–268435456), `max_trainable_tensors` (1–4096), and
`max_reward_values` (1–1048576). Absence preserves existing workload bytes and
behavior. Presence measures the entire supported trainable parameter population
at actual trainer begin/end callbacks, transferring at most 1 MiB per CPU chunk
under the original child-only deadline. Raw weights and reward vectors are not
persisted. Unsupported tensor layouts, distributed populations, exhausted bounds,
and nonfinite values reject rather than sample or silently truncate.

The existing lineage and metrics roles retain parameter fingerprints and actual
reward callback statistics. Grouped variation requires complete pinned-TRL
`group`/`none` statistics with positive reward standard deviation and a zero-group
fraction below one; batch-wide statistics are inconclusive. Signed zero changes
do not count as parameter changes. Unchanged parameters, missing grouped evidence,
or zero variation yield truthful `inconclusive` measurements without retry.
These locked-child measurements are not independent reconstruction of historical
weights, native completion verification, GPU/image qualification, or model quality.

The permanent provider-free CPU lane explicitly selects
`SYNAPTIC_TEST_GRPO_CPU_EVIDENCE=1` and asserts official CPU Torch `2.9.0+cpu` with
no CUDA. Other lanes skip the tiny-tensor controls before importing Torch; base
consumers remain Torch-free. Scripted third-party trainer/model boundaries exercise
real CPU tensor updates, isolated child custody, and native five-role verification,
without qualifying the real GRPO training stack or provider execution.
