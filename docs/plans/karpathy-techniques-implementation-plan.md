# Implementation Plan: Karpathy-Inspired Training Optimizations

> Generated from research session on 2026-03-21
> Research doc: `docs/research/karpathy_nanochat_autoresearch.md`
> Status: APPROVED → IN_PROGRESS

<!-- Status Lifecycle:
     PENDING APPROVAL → APPROVED → IN_PROGRESS → IMPLEMENTED
                    ↘ SUPERSEDED (if replaced by newer plan)
                    ↘ BLOCKED (if unresolved conflicts)
-->

## Summary

Six implementations inspired by Karpathy's nanochat/autoresearch projects and LoRA optimization research, organized as a DAG with clear dependency edges. The implementations build a 4-layer optimization stack: post-training weight surgery → per-step evolutionary training → per-run hyperparameter search → per-cycle dataset optimization (flywheel).

---

## Dependency DAG

```
                    ┌─────────────┐
                    │   1A        │
                    │ Fix Evo     │──────────────────────┐
                    │ Training    │                      │
                    └──────┬──────┘                      │
                           │                             │
                           │ (evo fixed →                │
                           │  experiment loop            │
                           │  can test evo configs)      │
                           │                             │
┌─────────────┐    ┌───────▼──────┐              ┌──────▼──────┐
│   1B        │    │   3          │              │   2         │
│ Complexity  │    │ Experiment   │              │ GRPO Reward │
│ Tiers       │    │ Loop         │              │ Wiring      │
└─────────────┘    └───────┬──────┘              └─────────────┘
                           │
      ┌─────────────┐      │ (experiment loop
      │   1C        │      │  uses checkpoint eval)
      │ Checkpoint  │──────┘
      │ Eval        │──────┐
      └──────┬──────┘      │
             │             │ (best checkpoint +
             │             │  final → interpolation
             │             │  candidates)
             │             │
      ┌──────▼─────────────▼──┐
      │   4                    │
      │ LoRA Weight Surgery    │
      │ (eval-guided)          │
      └────────────────────────┘
```

### Dependency Edges

| From | To | Reason |
|------|----|--------|
| **1A** → **3** | Fix evolutionary training before the experiment loop tries to auto-tune evolutionary configs |
| **1C** → **3** | Experiment loop reuses checkpoint eval infrastructure to score short training runs |
| **1C** → **4** | Checkpoint eval discovers best vs final checkpoint; surgery interpolates between them |
| **1B** → **3** | Complexity tiers provide the search space presets for the experiment loop |

### Parallelism Opportunities

| Can Run in Parallel | Why |
|---------------------|-----|
| **1A** ∥ **1B** ∥ **1C** | No dependencies between them |
| **1A** ∥ **2** | Completely independent subsystems |
| **1B** ∥ **2** | Completely independent subsystems |
| **1C** ∥ **2** | Completely independent subsystems |

---

## Stage 1: Foundation (No Dependencies — All Parallel)

### 1A: Fix Evolutionary Training

**Agent**: `pact-backend-coder`
**Effort**: Small (config + ~50 lines code)
**Risk**: Low

**Deliverables**:
1. Gradient norm clipping in `candidate_generator.py`
2. Adaptive noise scaling in `strategies/gradient_noise.py`
3. Tuned config defaults in `Trainers/sft/configs/config.yaml`
4. Gradient norm diagnostic logging in `trainer_wrapper.py`
5. Unit tests in `tests/test_evolutionary/`

**Files to modify**:
- `shared/evolutionary/candidate_generator.py` — add `clip_gradients()` static method
- `shared/evolutionary/strategies/gradient_noise.py` — cap noise to `min(grad.norm(), max_grad_norm)`
- `shared/evolutionary/trainer_wrapper.py` — log avg/max grad norms per step
- `shared/evolutionary/config.py` — add `max_grad_norm: float = 1.0` field
- `Trainers/sft/configs/config.yaml` — update evolutionary section defaults

**Config changes**:
```yaml
evolutionary:
  enabled: true                    # Was: false
  candidates: 4
  eval_batch_size: 2
  validation_config: "configs/fitness/tool_calling.yaml"
  strategy:
    type: "gradient_noise"
    params:
      noise_scale: 0.03            # Was: 0.1
      max_grad_norm: 1.0           # NEW
  selection:
    method: "best"
    min_improvement: 0.01          # Was: 0.0
  eval_frequency: 5                # Was: 1
  warmup_steps: 200                # Was: 0
```

**Acceptance criteria**:
- [ ] Gradient norms stay below `max_grad_norm` after clipping
- [ ] Fitness scores distribute across [0.3, 1.0] (not clustered at 0.0)
- [ ] Selected candidate > baseline in >50% of evolutionary steps
- [ ] Training loss trajectory comparable to non-evolutionary baseline
- [ ] All existing tests pass

---

### 1B: Single-Knob Complexity Tiers

**Agent**: `pact-backend-coder`
**Effort**: Small (~30 lines code + 3 YAML files per trainer)
**Risk**: Low

**Deliverables**:
1. Tier YAML files for SFT and KTO
2. `--tier` CLI argument in trainers
3. Config merge logic (tier overrides base)

**New files**:
- `Trainers/sft/configs/tiers/quick.yaml`
- `Trainers/sft/configs/tiers/standard.yaml`
- `Trainers/sft/configs/tiers/thorough.yaml`
- `Trainers/kto/configs/tiers/quick.yaml`
- `Trainers/kto/configs/tiers/standard.yaml`
- `Trainers/kto/configs/tiers/thorough.yaml`

**Files to modify**:
- `Trainers/sft/train_sft.py` — add `--tier` argument, merge tier config
- `Trainers/kto/train_kto.py` — same

**Tier definitions**:
| Tier | Rank | Alpha | LR | Epochs | Steps | Use Case |
|------|------|-------|----|--------|-------|----------|
| quick | 8 | 16 | 5e-4 | 1 | 200 | Exploratory, fast iteration |
| standard | 64 | 128 | 2e-4 | 1 | -1 | Production (current defaults) |
| thorough | 128 | 256 | 1e-4 | 3 | -1 | Maximum quality |

**Acceptance criteria**:
- [ ] `--tier quick` completes in <5 minutes
- [ ] `--tier standard` produces identical config to current defaults
- [ ] `--tier thorough` produces better eval scores on sufficient data
- [ ] Explicit flags override tier defaults (e.g., `--tier quick --learning-rate 1e-4`)

---

### 1C: Best-Checkpoint Selection via Eval

**Agent**: `pact-backend-coder`
**Effort**: Medium (~150 lines new code)
**Risk**: Low

**Deliverables**:
1. `CheckpointEvaluator` class in `shared/checkpoint_eval.py`
2. `--eval-checkpoints N` flag in trainers
3. Standalone CLI entry point
4. Results TSV output

**New files**:
- `shared/checkpoint_eval.py` — `CheckpointEvaluator` class
- `tests/test_checkpoint_eval.py` — unit tests

**Files to modify**:
- `Trainers/sft/train_sft.py` — add `--eval-checkpoints` flag, call after training
- `Trainers/kto/train_kto.py` — same

**New files** (additional):
- `shared/eval_backend.py` — `EvalBackend` protocol + `LocalEvalBackend` + `CloudEvalBackend`

**Core logic**:
```
1. Discover checkpoints in run_dir/checkpoints/checkpoint-{step}/
2. Read training log JSONL → extract per-checkpoint loss
3. Sort by loss ascending, take top N (cheapest pre-filter)
4. Always include final_model/ regardless of loss rank
5. Run Evaluator on each via EvalBackend → user's configured metric
6. Rank by eval score, copy best to run_dir/best_checkpoint/
7. Write checkpoint_eval_results.tsv (step, loss, eval_score, rank)
```

**Acceptance criteria**:
- [ ] Correctly discovers checkpoints and pre-filters by training loss
- [ ] Identifies cases where final_model is NOT the best
- [ ] Results TSV includes both training loss and eval score columns
- [ ] Best checkpoint copy loads successfully for inference
- [ ] Standalone CLI works independently of training
- [ ] Works with both `--eval-backend local` and `--eval-backend cloud`
- [ ] Local backend gates on GPU VRAM, errors with guidance if insufficient

---

## Stage 2: Core Features (Depends on Stage 1)

### 2: FitnessEvaluator as GRPO Reward Signal

**Agent**: `pact-backend-coder`
**Effort**: Small (~60 lines code + 1 YAML)
**Risk**: Low
**Dependencies**: None (can run parallel with Stage 1)

**Deliverables**:
1. Fitness reward function in `Trainers/grpo/src/rewards.py`
2. Reward config YAML
3. Integration into reward computation pipeline

**New files**:
- `Trainers/grpo/configs/rewards/fitness.yaml` — fitness reward rubric

**Files to modify**:
- `Trainers/grpo/src/rewards.py` — add `fitness_reward()`, wire into `compute_rewards()`

**How it works**:
```
total_reward = (
    args_match_weight × args_match_score +      # semantic (existing, weight 1.0)
    json_structure_weight × json_score +          # format (existing, weight 0.3)
    format_weight × format_score +                # format (existing, weight 0.2)
    fitness_weight × fitness_score                # structural (NEW, weight 0.3)
)
```

**Acceptance criteria**:
- [ ] `fitness_reward()` correctly calls `FitnessEvaluator` and returns 0.0-1.0
- [ ] Reward distribution is more informative with fitness component
- [ ] Tool call structural validity rate improves in GRPO-trained outputs
- [ ] Existing reward rubric tests still pass

---

### 3: Autonomous Experiment Loop

**Agent**: `pact-backend-coder`
**Effort**: Medium-Large (~400 lines new code)
**Risk**: Medium
**Dependencies**: 1A (evolutionary fixed), 1B (tier presets), 1C (checkpoint eval)

**Deliverables**:
1. `ExperimentLoop` class in `shared/flywheel/experiment_loop.py`
2. `ExperimentConfig` dataclass in `shared/flywheel/experiment_config.py`
3. Config YAML with default search space
4. CLI entry point in `tuner.py`
5. Results tracking (TSV + best config YAML)

**New files**:
- `shared/flywheel/experiment_loop.py` — core loop logic
- `shared/flywheel/experiment_config.py` — config dataclass
- `configs/flywheel/experiment_loop.yaml` — default search space
- `tests/flywheel/test_experiment_loop.py` — tests

**Files to modify**:
- `tuner.py` — add experiment loop menu entry

**Core loop**:
```
for i in range(max_experiments):
    1. Sample config from search space (random / grid)
    2. Write temp config with overrides + max_steps cap
    3. Run training subprocess (time-boxed)
    4. Run checkpoint eval (from 1C) → score
    5. Record in results.tsv
    6. If improved: save as new baseline
    7. If declined: discard
```

**Search space** (from configs):
```yaml
search_space:
  learning_rate: [5e-5, 1e-4, 2e-4, 5e-4]
  r: [8, 16, 32, 64]
  lora_alpha: [16, 32, 64, 128]
  num_train_epochs: [1, 2]
  warmup_ratio: [0.02, 0.05, 0.1]
  evolutionary.enabled: [true, false]      # ← tests evo vs non-evo (needs 1A)
  evolutionary.strategy.type: ["gradient_noise", "scale_variation"]
```

**Acceptance criteria**:
- [ ] Runs N experiments end-to-end without human intervention
- [ ] At least 1 config outperforms baseline in a 10-experiment run
- [ ] Results.tsv is complete and parseable
- [ ] Best config is reproducible (re-run produces similar metric)
- [ ] Time-boxing works (experiments don't exceed budget)

---

### 4: LoRA Weight Surgery

**Agent**: `pact-backend-coder`
**Effort**: Medium (~300 lines new code)
**Risk**: Medium
**Dependencies**: 1C (checkpoint eval provides best/final checkpoint pair for interpolation)

**Deliverables**:
1. `LoRASurgeon` class in `shared/evolutionary/lora_surgery.py`
2. Surgery operations (8 operations)
3. Surgery loop with eval feedback
4. Config YAML
5. CLI entry point

**New files**:
- `shared/evolutionary/lora_surgery.py` — `LoRASurgeon` class + all operations
- `configs/lora_surgery.yaml` — default surgery config
- `tests/test_lora_surgery.py` — tests

**Files to modify**:
- `tuner.py` — add surgery menu entry

**Operations** (in priority order):
1. **Alpha sweep** — modify `adapter_config.json` only, no weight changes (simplest)
2. **Layer-level scaling** — scale individual layers to find importance
3. **Module-type ablation** — zero all q_proj, or all v_proj, etc.
4. **Checkpoint interpolation** — blend best + final checkpoint (needs 1C)
5. **DARE** — drop & rescale for regularization
6. **Metrics-weighted averaging** — merge N checkpoints weighted by eval score
7. **SVD rank reduction** — compress LoRA via truncated SVD
8. **Attention vs MLP ablation** — which component matters more?

**Leverages existing tooling**: PEFT's `add_weighted_adapter()` for merging, `safetensors` for weight I/O.

**Acceptance criteria**:
- [ ] Layer importance scan identifies significant layers (>5% eval impact when zeroed)
- [ ] Alpha sweep finds a non-default alpha that matches or beats baseline
- [ ] Checkpoint interpolation finds a blend better than either endpoint
- [ ] Surgery loop produces an adapter that loads and runs inference correctly
- [ ] Results are reproducible (same operations → same scores)

---

## Agent Assignment & Parallelism Matrix

### PACT Orchestration Plan

```
Phase: PREPARE
  Agent: pact-preparer
  Task: Verify all file paths, confirm config schemas, check test infrastructure
  Duration: ~15 min

Phase: ARCHITECT
  Agent: pact-architect
  Task: Review plan for interface contracts between implementations,
        confirm DAG dependencies are correct, validate no circular deps
  Duration: ~15 min

Phase: CODE (parallelized)
  ┌─────────────────────────────────────────────────────────────────┐
  │ Wave 1 (all parallel — no dependencies between them)           │
  │                                                                 │
  │  Agent A: pact-backend-coder                                   │
  │    Task: Implementation 1A (Fix evolutionary training)          │
  │    Files: shared/evolutionary/{candidate_generator,             │
  │           strategies/gradient_noise, trainer_wrapper, config}.py│
  │           Trainers/sft/configs/config.yaml                     │
  │                                                                 │
  │  Agent B: pact-backend-coder                                   │
  │    Task: Implementation 1B (Complexity tiers)                   │
  │    Files: Trainers/{sft,kto}/configs/tiers/*.yaml              │
  │           Trainers/{sft,kto}/train_{sft,kto}.py                │
  │                                                                 │
  │  Agent C: pact-backend-coder                                   │
  │    Task: Implementation 1C (Checkpoint eval)                    │
  │    Files: shared/checkpoint_eval.py                             │
  │           Trainers/{sft,kto}/train_{sft,kto}.py (flag only)    │
  │                                                                 │
  │  Agent D: pact-backend-coder                                   │
  │    Task: Implementation 2 (GRPO reward wiring)                  │
  │    Files: Trainers/grpo/src/rewards.py                         │
  │           Trainers/grpo/configs/rewards/fitness.yaml            │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘
                              ↓
                    (Wave 1 complete)
                              ↓
  ┌─────────────────────────────────────────────────────────────────┐
  │ Wave 2 (parallel — both depend on Wave 1)                      │
  │                                                                 │
  │  Agent E: pact-backend-coder                                   │
  │    Task: Implementation 3 (Experiment loop)                     │
  │    Files: shared/flywheel/experiment_{loop,config}.py           │
  │           configs/flywheel/experiment_loop.yaml                │
  │           tuner.py                                              │
  │    Depends on: 1A (evo configs in search space),               │
  │               1B (tier presets), 1C (checkpoint eval reuse)     │
  │                                                                 │
  │  Agent F: pact-backend-coder                                   │
  │    Task: Implementation 4 (LoRA surgery)                        │
  │    Files: shared/evolutionary/lora_surgery.py                  │
  │           configs/lora_surgery.yaml                             │
  │           tuner.py                                              │
  │    Depends on: 1C (checkpoint eval for interpolation)           │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘

Phase: TEST
  Agent: pact-test-engineer
  Task: Write integration tests covering cross-implementation scenarios:
    - Experiment loop with evolutionary enabled/disabled
    - Surgery using checkpoints from checkpoint eval
    - GRPO training with fitness reward
  Duration: ~30 min
```

### File Conflict Matrix

No two agents should modify the same file simultaneously. Conflicts to manage:

| File | Modified By | Resolution |
|------|------------|------------|
| `Trainers/sft/train_sft.py` | 1A, 1B, 1C | Wave 1 agents touch **different sections**: 1A=evo config loading, 1B=tier arg parsing, 1C=eval-checkpoints flag. No overlap. |
| `Trainers/kto/train_kto.py` | 1B, 1C | Same: different sections. |
| `tuner.py` | 3, 4 | Wave 2: both add menu entries. Coordinate via separate handler functions. |

---

## Estimated Effort

| Implementation | Lines of Code | Agent Time | Human Review |
|---------------|---------------|------------|--------------|
| 1A: Fix Evo | ~50 modified, ~100 test | ~20 min | ~5 min |
| 1B: Tiers | ~30 code, ~90 YAML | ~15 min | ~5 min |
| 1C: Checkpoint Eval | ~150 new, ~80 test | ~25 min | ~5 min |
| 2: GRPO Reward | ~60 new, ~40 test | ~15 min | ~5 min |
| 3: Experiment Loop | ~400 new, ~150 test | ~40 min | ~10 min |
| 4: LoRA Surgery | ~300 new, ~120 test | ~35 min | ~10 min |
| **Total** | **~1,590 lines** | **~2.5 hours** | **~40 min** |

Wave 1 (4 parallel agents): ~25 min
Wave 2 (2 parallel agents): ~40 min
Testing: ~30 min
**Total wall-clock: ~1.5 hours** (with parallelism)

---

## Optimization Stack (When Complete)

```
┌────────────────────────────────────────────────────────────────┐
│ Layer 0: LoRA Surgery (Implementation 4)                       │
│   Post-training, per-weight. 1-3 min/experiment.              │
│   "Which weights matter? Can we improve without retraining?"  │
├────────────────────────────────────────────────────────────────┤
│ Layer 1: Evolutionary Training (Implementation 1A)             │
│   During training, per-step. N+1 forward passes/step.         │
│   "Which gradient update is best this step?"                  │
├────────────────────────────────────────────────────────────────┤
│ Layer 2: Experiment Loop (Implementation 3)                    │
│   Per training run, 10 min/experiment.                        │
│   "Which config produces the best model?"                     │
├────────────────────────────────────────────────────────────────┤
│ Layer 3: Flywheel (existing)                                   │
│   Per production cycle, days.                                 │
│   "What data should the next training run use?"               │
└────────────────────────────────────────────────────────────────┘

Support:
  ├─ Checkpoint Eval (1C) — feeds Layers 0, 2
  ├─ Complexity Tiers (1B) — simplifies Layer 2 search space
  └─ GRPO Reward Wiring (2) — enriches RL training signal
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Evolutionary training still unstable after gradient clipping | Low | Medium | Fall back to `scale_variation` strategy (no noise); add `warmup_steps: 500` |
| Experiment loop experiments take too long | Medium | Low | Reduce `max_steps` further; use `--tier quick` preset |
| LoRA surgery perturbs weights in ways that break inference | Low | High | Always validate loaded adapter before accepting; keep original untouched |
| Checkpoint eval is slow (N × full eval) | Medium | Low | Support `--quick-eval` with reduced scenario count |
| GRPO fitness reward dominates semantic rewards | Low | Medium | Start with low weight (0.1), tune up based on results |

---

## Success Metrics

| Metric | Baseline | Target | Measured By |
|--------|----------|--------|-------------|
| Evolutionary training functional | Disabled | Enabled, stable for 500+ steps | Gradient norm logs, fitness distribution |
| Best checkpoint != final model | Not checked | Detected in >30% of runs | Checkpoint eval results |
| Experiment loop finds improvement | N/A | >1 improvement per 10 experiments | results.tsv |
| LoRA surgery improves eval score | N/A | >2% improvement via alpha sweep or layer pruning | Surgery results |
| GRPO reward more informative | Existing rubrics only | Lower reward variance with fitness component | Reward distribution analysis |

---

## Resolved Decisions

### Decision 1: Eval Strategy — Full Eval, User-Configured Checkpoint Selection

**Decision**: Run full eval (whatever scenarios the user has configured) on each checkpoint. No quick-eval shortcut — this system must be flexible for any use case, not just our tool-calling testing.

**Checkpoint selection strategy**: The user configures how many checkpoints to evaluate. Instead of evaluating all checkpoints, select the **top N checkpoints by lowest training loss** from the saved checkpoints. This is a cheap pre-filter (just read the training log JSONL) that avoids wasting eval time on clearly-bad checkpoints.

**Config**:
```yaml
checkpoint_eval:
  eval_top_n: 5              # Evaluate the N checkpoints with lowest training loss
  eval_scenario: ""           # User provides their own scenario path
  include_final: true         # Always include final_model regardless of loss rank
```

**Implementation note**: Read `logs/training_latest.jsonl` to extract per-checkpoint loss, sort ascending, take top N, then run full eval on those. The eval scenario is user-provided — we don't hardcode `tool_prompts.yaml`. This keeps the system general-purpose.

**Impact on plan**:
- Implementation 1C (`CheckpointEvaluator`) adds loss-based pre-filtering
- Implementation 3 (Experiment Loop) uses the same pattern — short training run → read final loss → if loss is promising, run full eval
- Implementation 4 (Surgery) always runs full eval (weight perturbation, not training)

---

### Decision 2: GPU Strategy — Cloud-First, Local Hardware-Gated

**Decision**: Design cloud-first. Local execution should work but must be gated on actual hardware capabilities.

**Rationale**: Not everyone has an RTX 3090. The experiment loop and surgery need inference for eval — this maps naturally to cloud eval (cloud training already supported via `Trainers/cloud/` with HF Jobs, Modal, and RunPod). Local should be a convenience option, not the assumed default. The `CloudProvider` adapter pattern ensures we're not locked into any single provider.

**Architecture**:

```
ExperimentLoop / LoRASurgeon
    │
    ├── eval_backend: "cloud"  (default)
    │   └── Submit eval job via CloudEvalBackend
    │       └── Provider-agnostic: adapts to HF Jobs, Modal, RunPod, etc.
    │       └── Uses CloudProvider protocol (same as Trainers/cloud/)
    │       └── No local GPU needed for eval
    │       └── Parallel eval possible (multiple checkpoints at once)
    │
    └── eval_backend: "local"  (opt-in, hardware-gated)
        └── Check GPU capabilities at startup:
            ├── nvidia-smi → detect GPU model + VRAM
            ├── Minimum: 8 GB VRAM for inference-only eval
            ├── Recommended: 16+ GB for concurrent training + eval
            └── If insufficient: error with "use --eval-backend cloud"
```

**Provider-agnostic design** (critical — not HF-specific):

```python
# shared/eval_backend.py

class EvalBackend(Protocol):
    """Provider-agnostic eval interface."""
    async def run_eval(self, adapter_path: str, scenario: str) -> float: ...

class LocalEvalBackend:
    """Direct inference on local GPU. Hardware-gated."""
    def __init__(self, min_vram_gb: int = 8):
        self._validate_hardware(min_vram_gb)

    async def run_eval(self, adapter_path: str, scenario: str) -> float:
        # Load model via Unsloth, run Evaluator directly
        ...

class CloudEvalBackend:
    """Delegates to a CloudProvider. Provider is injected, not hardcoded."""
    def __init__(self, provider: CloudProvider):
        self.provider = provider  # HFJobsProvider, ModalProvider, RunPodProvider, etc.

    async def run_eval(self, adapter_path: str, scenario: str) -> float:
        job = await self.provider.submit_eval_job(adapter_path, scenario)
        result = await self.provider.wait_for_result(job)
        return result.eval_score

class CloudProvider(Protocol):
    """Adapter interface for cloud compute providers."""
    async def submit_eval_job(self, adapter_path: str, scenario: str) -> JobHandle: ...
    async def wait_for_result(self, job: JobHandle) -> JobResult: ...
    async def upload_adapter(self, local_path: str) -> str: ...  # Returns remote path
```

This mirrors the flywheel's existing `cloud_provider` config pattern (`"hf_jobs" | "runpod" | "modal"`). New providers = new adapter class, zero changes to experiment loop or surgery code.

**Hardware gating logic**:
```python
def check_local_eval_capability(model_size: str) -> bool:
    """Check if local GPU can handle inference for eval."""
    vram_gb = get_gpu_vram_gb()  # via nvidia-smi or torch.cuda
    if vram_gb is None:
        return False  # No GPU detected

    # Minimum VRAM for inference (4-bit quantized)
    min_vram = {
        "3b": 4,
        "7b": 8,
        "14b": 16,
        "20b": 24,
    }
    required = min_vram.get(model_size, 8)
    return vram_gb >= required
```

**Config**:
```yaml
experiment_loop:
  eval_backend: "cloud"          # "cloud" | "local"
  cloud_provider: "hf_jobs"     # "hf_jobs" | "modal" | "runpod"
  local_min_vram_gb: 8          # Gate for local eval

surgery:
  eval_backend: "cloud"
  cloud_provider: "modal"       # Can differ from experiment_loop
```

**Impact on plan**:
- Implementation 1C, 3, 4: All eval calls go through `EvalBackend` protocol
- `EvalBackend.run_eval(adapter_path, scenario) → float` — single interface
- `CloudEvalBackend` delegates to a `CloudProvider` adapter (injected, not hardcoded)
- `LocalEvalBackend` checks hardware before running, errors with guidance if insufficient
- Adding a new cloud provider = implement `CloudProvider` protocol, register in config

---

## Open Questions

1. ~~**Eval scenario for automated scoring**~~ → RESOLVED (Decision 1)
2. ~~**GPU contention**~~ → RESOLVED (Decision 2)
3. ~~**Experiment loop search strategy**~~ → RESOLVED (Decision 3)

### Decision 3: Search Strategy — LLM Reasoning + LightGBM Surrogate in Concert

**Decision**: Dual-strategy approach. An LLM reasons about past results and proposes next configs (like autoresearch). Simultaneously, a LightGBM surrogate model learns to predict eval scores from hyperparameter configs and identifies promising regions of the search space. The two work in concert.

**Why both?**
- **LLM reasoning** excels at: structural insights ("rank 64→128 helped, try 256"), noticing patterns ("warmup matters more at high LR"), proposing novel combinations the surrogate can't extrapolate to
- **LightGBM surrogate** excels at: quantitative prediction from tabular data, ranking 100s of candidate configs cheaply, identifying diminishing returns, feature importance (which hyperparams matter most)
- Together: LLM proposes candidates informed by surrogate predictions, surrogate validates LLM intuitions with data

**Existing infrastructure**: The `Trainers/ml/` pipeline already has LightGBM regression with feature engineering, sklearn Pipeline, and experiment tracking. We reuse it directly — no new ML code needed.

**Architecture**:

```
┌─────────────────────────────────────────────────────────────────┐
│ ExperimentLoop                                                  │
│                                                                  │
│  results.tsv ← accumulates after each experiment                │
│  ┌──────────────────────┐   ┌──────────────────────┐           │
│  │ LLM Advisor          │   │ LightGBM Surrogate   │           │
│  │                      │   │                      │           │
│  │ Input:               │   │ Input:               │           │
│  │  - results.tsv       │   │  - results.tsv       │           │
│  │  - search_space      │   │    (as training data) │           │
│  │  - program.md        │   │                      │           │
│  │                      │   │ Output:              │           │
│  │ Output:              │   │  - predicted scores   │           │
│  │  - proposed config   │   │    for all untried    │           │
│  │  - reasoning         │   │    configs            │           │
│  │                      │   │  - feature importance │           │
│  └──────────┬───────────┘   └──────────┬───────────┘           │
│             │                          │                        │
│             └──────────┬───────────────┘                        │
│                        ▼                                        │
│              ┌─────────────────┐                                │
│              │ Config Selector │                                │
│              │                 │                                │
│              │ Merge strategy: │                                │
│              │  Phase 1 (N<10):│                                │
│              │   LLM only     │                                │
│              │  Phase 2 (N≥10):│                                │
│              │   LLM proposes, │                                │
│              │   surrogate    │                                │
│              │   ranks/filters│                                │
│              └─────────────────┘                                │
│                        │                                        │
│                        ▼                                        │
│              Run experiment → append to results.tsv             │
│              Retrain surrogate every K experiments              │
└─────────────────────────────────────────────────────────────────┘
```

**Phase transitions**:
- **Phase 1 (experiments 1-10)**: Random sampling + LLM reasoning. Not enough data for the surrogate model. LLM reads results.tsv and proposes configs. Random fills gaps.
- **Phase 2 (experiments 10+)**: LLM + surrogate. Train LightGBM on results.tsv (hyperparams → eval_score). LLM proposes N candidate configs, surrogate predicts scores for each, top candidates are run. Surrogate retrained every 5 experiments.

**LLM advisor implementation** (autoresearch-style):
```python
class LLMAdvisor:
    def __init__(self, search_space: dict, program_md: str):
        self.search_space = search_space
        self.program_md = program_md  # Natural language instructions

    async def propose_config(self, results_history: pd.DataFrame) -> dict:
        """Ask LLM to propose next config based on results so far."""
        prompt = f"""You are optimizing hyperparameters for LoRA fine-tuning.

Search space:
{yaml.dump(self.search_space)}

Results so far (sorted by eval_score descending):
{results_history.to_markdown()}

Instructions:
{self.program_md}

Propose the next hyperparameter config to try. Return valid YAML.
Explain your reasoning briefly."""

        response = await llm_client.complete(prompt)
        return yaml.safe_load(extract_yaml(response))
```

**LightGBM surrogate implementation** (reuses existing ML pipeline):
```python
class SurrogateModel:
    def __init__(self, search_space: dict):
        self.search_space = search_space
        self.pipeline = None  # sklearn Pipeline with LightGBM

    def fit(self, results: pd.DataFrame):
        """Train surrogate on experiment history."""
        from Trainers.ml.pipeline_builder import build_pipeline
        from Trainers.ml.config import TrainingConfig

        config = TrainingConfig(**{
            "task": {"type": "regression", "target_column": "eval_score"},
            "features": {"numeric": {
                "columns": list(self.search_space.keys()),
                "scaler": "standard",
            }},
            "algorithm": {"name": "lightgbm", "params": {
                "n_estimators": 100,  # Small model, fast training
                "num_leaves": 15,
            }},
        })
        self.pipeline = build_pipeline(config)
        X = results[list(self.search_space.keys())]
        y = results["eval_score"]
        self.pipeline.fit(X, y)

    def predict_candidates(self, candidates: list[dict]) -> list[float]:
        """Predict eval scores for candidate configs."""
        df = pd.DataFrame(candidates)
        return self.pipeline.predict(df).tolist()

    def feature_importance(self) -> dict[str, float]:
        """Which hyperparameters matter most?"""
        lgbm = self.pipeline.named_steps["model"]
        return dict(zip(
            list(self.search_space.keys()),
            lgbm.feature_importances_,
        ))
```

**Concert mode** (LLM + surrogate together):
```python
async def select_next_config(self, results: pd.DataFrame) -> dict:
    """Dual-strategy config selection."""
    n = len(results)

    if n < 10:
        # Phase 1: LLM only (not enough data for surrogate)
        return await self.llm_advisor.propose_config(results)

    # Phase 2: LLM proposes, surrogate ranks
    # Retrain surrogate periodically
    if n % 5 == 0:
        self.surrogate.fit(results)
        importance = self.surrogate.feature_importance()
        logger.info(f"Feature importance: {importance}")

    # LLM proposes 5 candidates
    candidates = []
    for _ in range(5):
        candidate = await self.llm_advisor.propose_config(results)
        candidates.append(candidate)

    # Surrogate predicts scores for each
    predicted_scores = self.surrogate.predict_candidates(candidates)

    # Pick the candidate with highest predicted score
    best_idx = predicted_scores.index(max(predicted_scores))
    logger.info(f"Surrogate predicted scores: {predicted_scores}")
    logger.info(f"Selected candidate {best_idx} (predicted: {predicted_scores[best_idx]:.4f})")

    return candidates[best_idx]
```

**Impact on plan**:
- Implementation 3 (Experiment Loop) gets a `search_strategy: "llm_surrogate"` option
- Uses `shared/llm/` for LLM calls (already exists — OpenRouter, LMStudio, Ollama)
- Uses `Trainers/ml/pipeline_builder.py` for surrogate model (already exists)
- `program.md` file in experiment config dir guides LLM behavior (autoresearch pattern)
- Feature importance output reveals which hyperparams actually matter — feeds back into search space refinement
