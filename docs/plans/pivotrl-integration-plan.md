# PivotRL Integration Plan

**Date:** 2026-03-27
**Status:** Proposed
**Paper:** [arXiv:2603.21383](https://arxiv.org/abs/2603.21383)
**Branch:** `claude/pivotrl-integration-research-RB3e9`

---

## Goal

Add PivotRL as an **optional, config-driven mode** for GRPO training. When enabled, the trainer profiles SFT trajectory turns to find "pivots" (high-variance turns) and trains only on those — delivering the same GRPO accuracy with ~4x fewer rollouts. A functional equivalence reward component is added alongside, replacing brittle string matching.

**Design principle:** Minimal new code. Reuse existing reward system, dataset loader, model loader, and config patterns. Follow the precedent set by `env_config.yaml` / `train_env_grpo.py` — a config preset activates a new mode through the same infrastructure.

---

## Why This Fits

We already have ~70% of PivotRL's infrastructure:

| PivotRL Concept | Existing Infrastructure | Gap |
|---|---|---|
| Generate N rollouts, score them | `per_example_loss.py` batch pattern, `rewards.py` scoring | Need generation (not just loss) per turn |
| Filter by difficulty signal | `prune_dataset_from_loss.py` (post-hoc), `filter_env_rollout_dataset` | Need pre-training variance filter |
| Functional verification | `shared/validation/`, `FitnessEvaluator`, `EnvironmentValidator` | Need normalized comparison reward |
| GRPO optimizer | `train_grpo.py` + TRL `GRPOTrainer` | Nothing — same algorithm |
| Config-driven rewards | `rewards.py` YAML rubrics + `custom` extension point | Nothing — just add a component |

---

## What Gets Built

| # | Artifact | Type | Est. Lines | Purpose |
|---|----------|------|-----------|---------|
| 1 | `Trainers/grpo/src/pivot_profiler.py` | New module | ~200 | Turn extraction, batch rollout, variance filtering |
| 2 | `Trainers/grpo/src/functional_verifier.py` | New module | ~120 | Functional equivalence reward function |
| 3 | `Trainers/grpo/configs/rewards/functional_equivalence.yaml` | New config | ~15 | Reward rubric definition |
| 4 | `Trainers/grpo/configs/pivot_config.yaml` | New preset | ~160 | Complete config with pivot enabled |
| 5 | `Trainers/grpo/train_grpo.py` | Minor edit | +~20 lines | Conditional pivot branch + `--pivot-profile-only` flag |
| 6 | `.skills/fine-tuning/reference/grpo-training.md` | Skill update | +~60 lines | PivotRL docs section |
| 7 | `.skills/fine-tuning/SKILL.md` | Skill update | +~5 lines | Quick reference entry |

**Total: ~335 lines of Python + ~240 lines of YAML/docs. 3 new files, 4 edited files.**

No new dependencies. No new training entrypoints. No changes to reward engine, dataset loader, model loader, or callbacks.

---

## Data Flow

```
SFT Trajectory JSONL (existing dataset)
  │
  ▼
┌──────────────────────────┐
│ 1. Extract Candidates    │  Walk conversations[], yield one
│    _extract_candidates() │  (state, action) pair per assistant turn.
└──────────┬───────────────┘  Multi-turn convos → multiple candidates.
           │
           ▼
┌──────────────────────────┐
│ 2. Batch Rollout         │  Generate N completions per candidate
│    _batch_rollout()      │  using frozen base model (temperature=1.0).
└──────────┬───────────────┘  Follow per_example_loss.py batch pattern.
           │
           ▼
┌──────────────────────────┐
│ 3. Score & Filter        │  Score each rollout with existing
│    _score_and_filter()   │  build_combined_reward_function().
│                          │  Compute per-candidate reward variance.
│                          │  Keep candidates where var > threshold.
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ 4. Pivot Dataset JSONL   │  Same schema as existing GRPO dataset
│    (cached output)       │  (prompt, ground_truth_*, pivot_metadata).
└──────────┬───────────────┘  Directly consumable by load_raw_dataset().
           │
           ▼
┌──────────────────────────┐
│ 5. Standard GRPO Train   │  train_grpo.py::run() — unchanged from
│    GRPOTrainer.train()   │  this point onward. Just better data in.
└──────────────────────────┘
```

**Caching:** Profiling results persist to `Datasets/grpo/.pivot_cache/<hash>.jsonl`. Cache key = SFT file hash + model name + reward config hash. Changed rewards → automatic re-profile.

---

## Config Schema

### New `pivot:` section in config.yaml

```yaml
# PivotRL: Variance-gated data selection for GRPO
# When enabled, profiles SFT turns and trains only on high-variance "pivots".
# Disabled by default — omit or set enabled: false for standard GRPO.
pivot:
  enabled: false

  # Source SFT file to extract candidates from.
  # If null, uses dataset.local_file as the source.
  sft_source: null

  # Pre-profiled pivot dataset. If provided, skip profiling entirely.
  # Use with --pivot-profile-only to separate profiling from training.
  profiled_file: null

  # Profiling parameters
  profiling:
    n_rollouts: 8               # Completions per candidate turn
    temperature: 1.0            # Sampling temp during profiling
    max_completion_length: 512  # Max tokens per rollout
    batch_size: 16              # Inference batch size

  # Filtering parameters
  filtering:
    variance_threshold: 0.1     # Min reward variance to qualify as pivot
    min_candidates: 50          # Fail-safe: error if fewer pivots found
    max_candidates: null        # Optional cap on pivot count

  # Cache settings
  cache:
    enabled: true
    cache_dir: null             # Default: Datasets/grpo/.pivot_cache/
```

### Functional equivalence reward (added to rewards: section)

```yaml
rewards:
  items:
    # ... existing items unchanged ...
    - name: "functional_equivalence"
      weight: 0.5
      params:
        fallback: "structural"  # "structural" | "args_match"

  custom:
    enabled: true
    file: "./src/functional_verifier.py"
    functions:
      - name: "functional_equivalence_reward"
        weight: 0.5
```

**Backward compatibility:** `pivot:` section omitted or `enabled: false` → zero behavior change. Functional equivalence reward only active when added to `rewards.items` with weight > 0.

---

## Implementation Phases

### Phase 1: Pivot Profiler (`pivot_profiler.py`)

**Delegate to:** `pact-backend-coder`

| Task | Details |
|------|---------|
| `_extract_candidates(sft_file)` | Walk `conversations[]`, yield `(state, action, ground_truth)` per assistant turn. State = all preceding messages. Ground truth = the assistant turn itself (for reward scoring). |
| `_batch_rollout(candidates, model, tokenizer, n_rollouts, ...)` | Load frozen model via existing `model_loader.py`. Generate N completions per candidate. Follow `per_example_loss.py` batch pattern: token-budgeted batching, incremental persistence, crash recovery. |
| `_score_and_filter(candidates, rollouts, reward_fn, filtering_cfg)` | Score each rollout with `build_combined_reward_function()`. Compute `mean(rewards)`, `std(rewards)` per candidate. Apply filtering: `std >= variance_threshold`, optional band filter on mean, optional top-% cap. |
| `profile_pivots(...)` | Public entry point. Orchestrates extract → rollout → score → filter. Returns HF `Dataset` or saves JSONL. Checks cache first. |
| Cache logic | SHA-1 key from (sft_file mtime, model_name, reward_config_hash). JSONL shard output with manifest for resumption. |

**Patterns to follow:**
- `per_example_loss.py`: `PreparedLossExample` → `PreparedPivotCandidate`, `IncrementalLossWriter` → incremental shard writer, `_iter_batches()` → token-budgeted batching
- `env_dataset.py`: `filter_env_rollout_dataset()` → `filter_pivot_dataset()` signature

**Tests:**
- Unit: candidate extraction from single-turn and multi-turn conversations
- Unit: variance computation and filtering thresholds
- Integration: profile a 10-example dataset, verify output schema matches GRPO expectations

---

### Phase 2: Functional Verifier (`functional_verifier.py`)

**Delegate to:** `pact-backend-coder`

| Task | Details |
|------|---------|
| `_normalize_tool_call(parsed)` | Sort argument keys, coerce types (str "true" → bool), strip whitespace, normalize paths. |
| `_compare_functional(predicted, ground_truth)` | Same tool name (exact). Same keys. Values equivalent after normalization. Order-independent for unordered collections. |
| `functional_equivalence_reward(completions, prompts, **kwargs)` | TRL reward signature. Parse completion → extract tool call (reuse `RewardRubric._extract_data` patterns from `rewards.py`). Parse ground truth from kwargs. Compare. Return 1.0 (equivalent), 0.5 (partial — same tool, some args match), 0.0 (different tool or unparseable). |
| Fallback chain | If parsing fails → fall back to `structural` comparison (key overlap). If that fails → fall back to `args_match` (string equality). Configurable via `params.fallback`. |

**Reuses:**
- `shared/validation/parsing/` for tool call extraction (Qwen/Mistral format detection)
- `RewardRubric._extract_data()` patterns from `rewards.py`

**Tests:**
- Unit: normalization cases (arg reorder, type coercion, path normalization)
- Unit: equivalent calls score 1.0, partial 0.5, wrong tool 0.0
- Unit: fallback chain triggers correctly

---

### Phase 3: Config & Wiring

**Delegate to:** `pact-backend-coder`

| Task | Details |
|------|---------|
| `configs/pivot_config.yaml` | Copy `config.yaml`, enable `pivot:` section, add functional_equivalence to rewards. Serves as a ready-to-use preset. |
| `configs/rewards/functional_equivalence.yaml` | Rubric YAML for the new reward component. |
| `train_grpo.py` edit | ~20 lines: after config load, check `pivot.enabled`. If true + no `profiled_file`, call `profile_pivots()`. If true + `profiled_file`, point dataset loader at it. Add `--pivot-profile-only` argparse flag. |
| Reward function ordering | When pivot is enabled, build reward function before dataset loading (reward_fn needed for profiling). Currently rewards are built after dataset. Move the call earlier when pivot is active — `build_combined_reward_function` has no dataset dependency. |

---

### Phase 4: Skill Documentation

**Delegate to:** `pact-backend-coder` (docs are in the skill, not standalone)

| Task | Details |
|------|---------|
| `.skills/fine-tuning/SKILL.md` | Add quick reference row: `Pivot-profile GRPO dataset` with command. |
| `.skills/fine-tuning/reference/grpo-training.md` | New section: "PivotRL (Variance-Gated Data Selection)". Cover: what it does, config fields, how to run profiling separately, when to use vs. standard GRPO, key metrics to watch. |
| Sync mirrors | Run `python3 .skills/scripts/sync_skill_trees.py` after edits. |

---

### Phase 5: Tests

**Delegate to:** `pact-test-engineer`

| Test | Type | What |
|------|------|------|
| `tests/trainers/grpo/test_pivot_profiler.py` | Unit | Candidate extraction (single/multi-turn), variance filtering, cache key computation |
| `tests/trainers/grpo/test_functional_verifier.py` | Unit | Normalization, equivalence comparison, fallback chain, TRL reward signature |
| `tests/trainers/grpo/test_pivot_config.py` | Integration | Load `pivot_config.yaml`, verify backward compat with base `config.yaml`, verify pivot disabled by default |
| Existing GRPO tests | Regression | Run existing test suite to confirm no breakage |

---

## Risk Mitigation

| Risk | Severity | Mitigation |
|------|----------|------------|
| Profiling too slow on RTX 3090 | Medium | Cache results. `--pivot-profile-only` for overnight runs. Default N=8 (not 16). |
| Too few pivots selected | Medium | `min_candidates: 50` fail-safe. Log full variance distribution for tuning. |
| Functional verifier false positives | Medium | Start conservative (same tool name required, exact key match, only normalize types/whitespace). |
| Breaking existing GRPO flow | Low | `pivot.enabled: false` by default. Entire path is a conditional branch. |
| Reward build order change | Low | `build_combined_reward_function` verified to have no dataset dependency. |

---

## Connection to Existing Systems

| Existing System | Relationship to PivotRL |
|---|---|
| **Loss pipeline** (`prune_dataset_from_loss.py`) | Post-hoc version of the same idea. PivotRL profiles *before* training. Could chain: PivotRL for GRPO data selection → loss analysis after training for next-iteration cleanup. |
| **Evolutionary model** (`shared/evolutionary/`) | Same generate-score-select loop at the weight-update level. PivotRL applies it at the data-selection level. Complementary — could combine both. |
| **Flywheel auto-routing** (`shared/flywheel/tagger.py`) | Future integration: add `pivot_score` to tagged records, route high-variance examples to GRPO automatically. Not in scope for this plan. |
| **Env-GRPO** (`train_env_grpo.py`) | Architectural precedent for "config-activated mode." PivotRL follows the same pattern: separate config preset, conditional branch in trainer. |

---

## Success Criteria

1. `python train_grpo.py --config configs/pivot_config.yaml --pivot-profile-only` profiles an SFT dataset and outputs a pivot JSONL
2. `python train_grpo.py --config configs/pivot_config.yaml` trains GRPO on the pivot-filtered dataset
3. `python train_grpo.py --config configs/config.yaml` (no pivot section) behaves identically to today
4. Functional equivalence reward scores `ls -la` and `ls -al` as equivalent (1.0)
5. All existing GRPO tests pass unchanged
6. Skill docs updated and mirrors synced
