# Research: Karpathy's nanochat & autoresearch

**Date**: 2026-03-21
**Purpose**: Identify techniques from Karpathy's recent projects that could improve our fine-tuning pipeline.

---

## Project Summaries

### nanochat (Oct 2025, ~49.7k stars)
- **Repo**: [github.com/karpathy/nanochat](https://github.com/karpathy/nanochat)
- **What**: Minimal full-stack LLM training + inference pipeline. "The best ChatGPT that $100 can buy."
- **Size**: ~8,000 lines of hand-written PyTorch. Successor to nanoGPT (now deprecated).
- **Pipeline**: Tokenizer -> Pretraining -> Mid-training -> SFT -> RL -> Inference + Chat UI

### autoresearch (Mar 2026, ~30.3k stars)
- **Repo**: [github.com/karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- **What**: Autonomous AI agent loop that modifies training code, runs 5-minute experiments, evaluates, keeps/discards, repeats.
- **Results**: 700 experiments in 2 days, 20 genuine optimizations, 11% speedup on larger model. Shopify CEO got 19% gain overnight.

---

## Techniques Worth Borrowing

### 1. Autonomous Experiment Loop (from autoresearch)

**What it does**: An LLM agent modifies `train.py`, runs a 5-minute training experiment, checks if `val_bpb` improved, keeps or reverts, repeats ~12x/hour.

**Architecture** (3 files only):
- `prepare.py` — immutable: data prep, tokenizer, eval harness
- `train.py` — ~630 lines, the ONLY file agents modify
- `program.md` — natural language instructions + constraints for the agent

**Key design choices**:
- Fixed 5-minute wall-clock budget per experiment (fair comparison)
- `val_bpb` (bits per byte) as the single metric — tokenizer-invariant
- Agent cannot modify data prep or install packages — constrained search space
- "You're not touching Python files — you're programming the `program.md`"

**Applicability to our pipeline**: HIGH. Our flywheel already has the pieces (inference logging, fitness evaluation, auto-retrain). We could add an "experiment agent" that:
- Tweaks hyperparameters in training configs (LR, LoRA rank, epochs)
- Runs short training experiments (e.g., 500 steps instead of full run)
- Compares eval scores via our Evaluator
- Keeps winning configs, discards losers
- This would be a natural extension of the flywheel's `FlywheelOrchestrator`

**Implementation sketch**:
```
configs/flywheel/experiment_loop.yaml:
  budget_minutes: 10
  metric: eval_score  # or val_loss
  max_experiments: 50
  search_space:
    learning_rate: [1e-6, 5e-6, 1e-5, 2e-5, 5e-5]
    lora_rank: [8, 16, 32, 64]
    lora_alpha: [16, 32, 64]
    epochs: [1, 2, 3]
```

---

### 2. The "Depth Dial" — Single-Knob Scaling (from nanochat)

**What it does**: A single `--depth` integer (number of transformer layers) automatically determines ALL other hyperparameters — width, heads, LR, weight decay, training horizon — producing compute-optimal models.

**Why it matters**: Eliminates hyperparameter tuning. Users don't need to understand transformer architecture. Just set depth and get an optimal model.

**Applicability to our pipeline**: MEDIUM. We could create preset "complexity tiers" for LoRA fine-tuning:
- `--tier small`: rank=8, alpha=16, lr=2e-5, epochs=1
- `--tier medium`: rank=16, alpha=32, lr=1e-5, epochs=2
- `--tier large`: rank=32, alpha=64, lr=5e-6, epochs=3

This would simplify the CLI UX significantly, especially for users who don't know what LoRA rank or learning rate to pick.

---

### 3. Mid-Training Stage (from nanochat)

**What it does**: Between pretraining and SFT, nanochat runs a "mid-training" stage that:
- Uses SmolTalk conversation data
- Includes 100K MMLU questions for knowledge
- Adds tool-use examples with `<|python_start|>...<|python_end|>` markers
- Adds GSM8K for math/calculator usage
- Algorithmically identical to pretraining but on conversational data

**Why it matters**: The model learns conversation structure and special tokens BEFORE fine-tuning. This means SFT can focus on quality/alignment rather than format learning.

**Applicability to our pipeline**: LOW-MEDIUM for current scope. We're doing LoRA fine-tuning on already-pretrained models, so mid-training is less relevant. However, if we ever do continued pretraining (e.g., domain adaptation), this staged approach is worth adopting.

---

### 4. SFT Format Matching (from nanochat)

**What it does**: Unlike pre/mid-training which concatenates examples into long rows for throughput, SFT "stretches out each example individually and pads them to exactly mimic the test-time format."

**Why it matters**: Training-inference format mismatch degrades quality. Matching formats exactly during SFT ensures the model sees the same token patterns at inference time.

**Applicability to our pipeline**: HIGH. We should verify our SFT training isn't packing/concatenating examples. Each conversation should be padded individually to match inference format. Check `Trainers/rtx3090_sft/train_sft.py` for packing behavior.

---

### 5. Simplified RL — GRPO/REINFORCE without PPO complexity (from nanochat)

**What it does**: nanochat's RL stage uses a dramatically simplified version of GRPO:
- No trust region
- No reference model
- No KL penalties
- On-policy updates (no PPO ratios/clip)
- Token-level normalization
- Mean-shift advantage

Practically equivalent to REINFORCE with group-relative advantage estimation.

**Why it matters**: Full RLHF/PPO is complex and resource-intensive. This shows you can get meaningful RL gains with a much simpler algorithm.

**Applicability to our pipeline**: HIGH. We already have GRPO in our training pipeline. Karpathy's experience validates the simpler approach. Key insight: for tool-calling models, RL with verifiable rewards (does the tool call parse correctly? does it produce the right result?) is the most impactful technique. Our `FitnessEvaluator` could serve as the reward signal.

---

### 6. Bits-per-Byte as Tokenizer-Invariant Metric (from nanochat/autoresearch)

**What it does**: Uses `val_bpb` (validation bits per byte) instead of standard cross-entropy loss. This metric is independent of vocabulary size and tokenization, so you can change the tokenizer between experiments and still compare fairly.

**Applicability to our pipeline**: LOW for LoRA fine-tuning (we don't change tokenizers). But useful if we ever compare models with different tokenizers.

---

### 7. Synthetic Data for Identity/Personality (from nanochat)

**What it does**: `dev/gen_synthetic_data.py` lets you specify desired behavior in words, then generates synthetic conversations from a larger LLM to impart arbitrary identity to your model.

**Applicability to our pipeline**: ALREADY DOING THIS. Our SynthChat system does exactly this. Validates our approach.

---

### 8. Custom Chat Template with Tool Boundaries (from nanochat)

**What it does**: Defines special tokens for tool invocation:
```
<|python_start|>...<|python_end|>   # tool call
<|output_start|>...<|output_end|>   # tool output
```

Clear boundaries make it trivial to parse tool calls during inference. The model learns to emit structured tool invocations within well-defined markers.

**Applicability to our pipeline**: MEDIUM. We already use XML-based tool formatting. But nanochat's approach of dedicated special tokens (rather than XML strings) is arguably more robust — special tokens can't appear in user content. Worth considering for future tokenizer/format decisions.

---

## Priority Recommendations

| Priority | Technique | Effort | Impact | Next Step |
|----------|-----------|--------|--------|-----------|
| **P0** | Autonomous experiment loop | Medium | High | Design `ExperimentAgent` for flywheel |
| **P1** | SFT format matching | Low | Medium | Audit current SFT packing behavior |
| **P1** | Simplified RL with verifiable rewards | Low | High | Wire `FitnessEvaluator` as GRPO reward |
| **P2** | Single-knob complexity tiers | Low | Medium | Add `--tier` presets to training CLI |
| **P3** | Mid-training stage | High | Low | Only if doing continued pretraining |
| **P3** | Special token tool boundaries | Medium | Low | Future tokenizer redesign |

---

## Integration Points in Our Codebase

### P0: Autonomous Experiment Loop → FlywheelOrchestrator

Karpathy's three design primitives map directly to our system:

| Primitive | autoresearch | Our Equivalent |
|-----------|-------------|----------------|
| **Editable asset** | `train.py` (single file) | `Trainers/sft/configs/config.yaml` or `Trainers/kto/configs/config.yaml` |
| **Scalar metric** | `val_bpb` | `overall_pass_rate` from `Evaluator/reporting.py` (0.0-1.0) |
| **Time-boxed cycle** | 5-min wall-clock | `max_steps` in trainer config (e.g., 500 steps) |

**Where it hooks in**:
- `shared/flywheel/orchestrator.py` — `FlywheelOrchestrator.run_cycle()` already runs CLEAN→TAG→STAGE→RETRAIN. An experiment loop wraps this with config mutation + eval comparison.
- `shared/experiment_tracking/adapters.py` — `eval_to_run_record()` extracts `overall_pass_rate` as `primary_metric`. This is our `val_bpb` equivalent.
- `shared/flywheel/readiness.py` — `ReadinessChecker.check()` gates whether enough data exists. Experiment loop bypasses this (uses fixed dataset).

**Config parameters to search over** (from actual config files):

| Trainer | Parameter | Current Default | Search Range |
|---------|-----------|-----------------|--------------|
| SFT | `learning_rate` | 2e-4 | [5e-5, 1e-4, 2e-4, 5e-4] |
| SFT | `r` (LoRA rank) | 64 | [8, 16, 32, 64, 128] |
| SFT | `lora_alpha` | 128 | [r, 2*r] |
| SFT | `num_train_epochs` | 1 | [1, 2, 3] |
| SFT | `packing` | false | [true, false] |
| SFT | `warmup_ratio` | 0.02 | [0.01, 0.02, 0.05, 0.1] |
| KTO | `learning_rate` | 1e-6 | [2e-7, 5e-7, 1e-6, 5e-6] |
| KTO | `beta` | 0.1 | [0.05, 0.1, 0.2, 0.5] |
| GRPO | `num_generations` | 4 | [2, 4, 8] |
| GRPO | `beta` | 0.1 | [0.01, 0.05, 0.1, 0.2] |
| GRPO | `temperature` | 1.0 | [0.7, 0.8, 1.0, 1.2] |

**Available scalar metrics for the feedback loop** (ranked by signal quality):
1. `overall_pass_rate` — from `Evaluator/reporting.py` (primary)
2. `avg_quality_score` — from `catalog.avg_score()` via `ReadinessReport`
3. `judge_pass_rate` — from `JudgeService` per-rubric scores
4. `final_loss` — from training run `RunRecord.primary_metric`
5. `score_distribution` histogram — from `CleaningResult`

---

### P1: SFT Format Matching — VERIFIED: Already Correct (Default)

Audited `Trainers/sft/train_sft.py` (lines 682-693):
- **Default**: `packing: false` in `config.yaml` — each example padded individually to `max_seq_length` (2048)
- **Optional**: `packing: true` concatenates examples for 2.5-5x faster training
- **`completion_only_loss: true`** — loss computed only on assistant tokens

**Finding**: Our default config matches nanochat's SFT approach (individual padding, no packing). The `packing: true` option exists for throughput but changes loss semantics. No action needed unless we're accidentally enabling packing.

**Caution**: If packing is enabled for speed, we lose the exact test-time format matching that Karpathy recommends. Document this tradeoff in trainer README.

---

### P1: FitnessEvaluator as GRPO Reward Signal — Already Partially Wired

The flywheel tagger (`shared/flywheel/tagger.py`) already routes logs to GRPO:
- `_is_grpo_eligible()`: checks `tools_requested AND has_tool_calls AND is_valid`
- `grpo_reward_scale` in `FlywheelConfig` scales `fitness_score → reward`
- `DatasetStager._write_grpo()` formats as `{"conversations": [...], "reward": score * scale}`

**What's missing**: The GRPO trainer (`Trainers/grpo/train_grpo.py`) uses its own reward rubrics (`Trainers/grpo/configs/rewards/*.yaml`) — field-level comparison against ground truth. The flywheel's `fitness_score` (schema validation pass/fail) is a coarser signal.

**Integration opportunity**: Combine both signals:
- `fitness_score` (0.0-1.0) from FitnessEvaluator → structural correctness
- Reward rubric scores from `Trainers/grpo/src/rewards.py` → semantic correctness
- Weighted sum = richer reward signal

**Concrete files**:
- `configs/flywheel/fitness_rules.yaml` — current: 3 JSON path validations (function.name, arguments exist, arguments valid JSON)
- `Trainers/grpo/configs/rewards/args_match.yaml` — weight 1.0, checks context fields + tool selection
- `Trainers/grpo/src/rewards.py` — reward computation with strategies: `equals`, `contains`, `key_overlap`

---

### P2: Single-Knob Complexity Tiers — Maps to Existing Config Structure

Our trainer configs already use YAML with all hyperparameters in one place. The "depth dial" concept translates to LoRA complexity tiers:

**Where it plugs in**: `Trainers/sft/configs/config.yaml` and `Trainers/kto/configs/config.yaml`

```yaml
# Proposed: tiers/ directory alongside config.yaml
# tiers/quick.yaml  — r=8, alpha=16, lr=5e-4, epochs=1, max_steps=200
# tiers/standard.yaml — r=64, alpha=128, lr=2e-4, epochs=1 (current default)
# tiers/thorough.yaml — r=128, alpha=256, lr=1e-4, epochs=3
```

**CLI integration**: `python train_sft.py --tier quick` loads tier preset, overrides apply on top.

**Note**: The evolutionary training system (`Trainers/sft/configs/config.yaml` lines 114-147) already exists but is disabled (`evolutionary.enabled: false`). This is complementary — evolutionary training mutates gradients per-step, while the experiment loop mutates configs per-run.

---

## Deep Dive: Evolutionary Training System (Existing)

### Architecture

Located in `shared/evolutionary/`, this is a **Gradient + Evolutionary Strategy hybrid** that intercepts SFT training steps:

```
shared/evolutionary/
├── config.py                    # EvolutionaryConfig dataclass
├── trainer_wrapper.py           # EvolutionaryTrainerWrapper (main, ~1154 lines)
├── candidate_generator.py       # Candidate orchestration + selection
└── strategies/
    ├── base.py                  # GradientCandidate dataclass
    ├── gradient_noise.py        # Gaussian noise on gradients
    ├── scale_variation.py       # Uniform gradient scaling [0.5x, 1.0x, 1.5x, 2.0x]
    └── combined.py              # Mix of both strategies
```

### How It Works (Per Training Step)

1. Standard forward + backward pass → extract base gradients
2. Generate N candidate gradient modifications (default: 4)
3. For each candidate: temporarily apply gradients → forward pass on eval batch → compute validation loss → restore weights
4. Select best candidate by fitness (`1/(1+val_loss)`)
5. Replace `param.grad` with selected candidate's gradients
6. Let optimizer apply the winning gradients

### How It Patches Unsloth — It Doesn't

The wrapper **does not patch Unsloth**. It replaces `SFTTrainer.training_step` at runtime:

```python
self._original_training_step = self.trainer.training_step
self.trainer.training_step = evolutionary_training_step
```

Unsloth's custom CUDA kernels still run normally through `trainer.compute_loss()`. The wrapper only uses standard PyTorch operations (`param.grad`, `param.data.sub_()`, `param.data.copy_()`). No generation calls, no interference with Flash Attention or fused ops.

**Compatibility chain**: Unsloth patches model → LoRA applied → SFTTrainer wraps model → Evolutionary wraps SFTTrainer's training_step method. Each layer is transparent to the one below.

### Current Status: Disabled

```yaml
evolutionary:
  enabled: false  # Disabled - high gradient norms, needs tuning
```

### Root Cause: High Gradient Norms

During candidate evaluation, the wrapper temporarily applies gradients as an SGD-like step:
```python
param.data.sub_(lr * candidate.gradients[name])
```

With SFT defaults (`lr=2e-4`, `noise_scale=0.1`):
- Early training gradient norms can be ~100+ for LoRA params
- Noise magnitude: `randn * 0.1 * 100 = ~10` per param
- Temporary weight shift pushes model far enough that val loss explodes
- All candidates produce near-infinite loss → fitness scores meaningless

### Proposed Fixes

| Fix | Effort | Expected Impact |
|-----|--------|-----------------|
| Set `warmup_steps: 200-500` | Config change | HIGH — lets gradients stabilize before evolutionary selection |
| Reduce `noise_scale: 0.01-0.05` | Config change | HIGH — directly reduces perturbation magnitude |
| Use `scale_variation` strategy | Config change | MEDIUM — no noise, just explores LR scaling |
| Gradient clipping before candidate generation | Small code change | MEDIUM — caps gradient magnitude before noise |
| Use `min_fitness_improvement: 0.01` | Config change | LOW — prevents selecting candidates only marginally better |

### Three Levels of Optimization (How They Stack)

| Level | Scope | What Mutates | Metric | Frequency |
|-------|-------|-------------|--------|-----------|
| **Evolutionary** (existing) | Per training step | Gradient updates | Validation loss | Every step |
| **Experiment Loop** (autoresearch-inspired) | Per training run | Config hyperparameters | Eval pass rate | Every ~10 min |
| **Flywheel** (existing) | Per production cycle | Training dataset | Fitness score | Every N days |

These are complementary, not competing. Evolutionary optimizes weight updates within a run, the experiment loop finds the best config for a run, and the flywheel optimizes what data future runs train on.

---

## Deep Dive: Muon Optimizer — Should We Integrate?

### What Is Muon?

**MomentUm Orthogonalized by Newton-Schulz** (by Keller Jordan, Oct 2024). Runs standard SGD with momentum, then orthogonalizes each 2D parameter's update to the nearest orthogonal matrix using 5 Newton-Schulz iterations. Now in PyTorch core as `torch.optim.Muon` (v2.9+).

Key properties:
- **Only for 2D (matrix) parameters** — embeddings, biases, scalars must use AdamW
- **Gradient normalization built-in** — Newton-Schulz orthogonalization inherently normalizes gradient magnitudes
- **50% less optimizer state memory** — 1 state (momentum) vs AdamW's 2 (m, v)
- **muP-scaled LR** — shouldn't need retuning when scaling model size

### Installation

```bash
# Native PyTorch (v2.9+)
import torch.optim
muon = torch.optim.Muon(params, lr=0.02)

# Or standalone
pip install git+https://github.com/KellerJordan/Muon
```

### Critical Finding: Muon Is NOT Recommended for LoRA Fine-Tuning

Multiple sources confirm:

1. **"Use Muon when building new capabilities, and standard optimizers when adjusting existing ones"** — Muon is designed for pretraining and full-parameter training at scale.

2. **LoRA params are already small** — Muon's overhead (Newton-Schulz iterations, all_gather for distributed) isn't worth it for LoRA's low-rank matrices.

3. **Optimizer mismatch degrades quality** — Fine-tuning a model pretrained with Muon using AdamW (or vice versa) "doesn't work very well."

4. **Muon is for 2D hidden layer params only** — LoRA's A and B matrices ARE 2D, but they're low-rank by design. Orthogonalizing a rank-16 matrix is semantically different from orthogonalizing a full-rank weight matrix.

### How Muon Could Help Our Gradient Norm Problem (Indirectly)

Even though Muon isn't right for LoRA fine-tuning, its core insight — **gradient orthogonalization normalizes magnitudes** — is directly relevant to the evolutionary system's gradient norm instability:

**Option A**: Add Newton-Schulz orthogonalization as a new evolutionary strategy:
```python
# strategies/orthogonal_noise.py
# 1. Compute base gradient
# 2. Orthogonalize via Newton-Schulz (normalizes magnitude)
# 3. Add noise to orthogonalized gradient
# 4. This prevents magnitude explosion while preserving direction
```

**Option B**: Add gradient norm clipping in the candidate generator before noise injection:
```python
# candidate_generator.py
max_norm = config.get('max_grad_norm', 1.0)
for name, grad in base_gradients.items():
    norm = grad.norm()
    if norm > max_norm:
        base_gradients[name] = grad * (max_norm / norm)
```

Option B is simpler and more likely to work. Option A is more principled but requires more implementation.

### Why Muon + LoRA Is Fundamentally Mismatched

Beyond practical concerns, there's a theoretical issue. LoRA decomposes weight updates into `A @ B` (two low-rank factor matrices). Muon applied to A and B separately is **not reparametrization-invariant** — orthogonalizing each factor independently skews the weight-space step and lets one factor dominate. Keller Jordan himself noted it's an "open question."

**Riemannion** ([arXiv:2507.12142](https://arxiv.org/abs/2507.12142)) solves this properly via Riemannian optimization on the fixed-rank manifold — but it's a research prototype, not production-ready for our pipeline.

Also: virtually all public checkpoints (Qwen, Llama, Mistral) are pretrained with AdamW. Fine-tuning with a different optimizer class degrades quality. PyTorch's `torch.optim.Muon` has an `adjust_lr_fn="match_rms_adamw"` mode that partially addresses this, but only for full fine-tuning.

### Recommendation

**Don't integrate Muon as the optimizer for LoRA fine-tuning.** Keep AdamW 8-bit.

**Do borrow the gradient normalization concept** for the evolutionary system — either via explicit gradient clipping (simple) or Newton-Schulz orthogonalization as a strategy (principled).

---

## Implementation Plan

### Top Recommendations (Ordered by Impact / Effort)

| # | Recommendation | Source Insight | Effort | Impact |
|---|---------------|----------------|--------|--------|
| 1 | Fix evolutionary training (gradient norm stabilization) | Muon's gradient normalization + autoresearch's constrained experiments | Small | High |
| 2 | Autonomous experiment loop for hyperparameter search | autoresearch's 3-file architecture | Medium | High |
| 3 | Wire FitnessEvaluator as GRPO reward signal | nanochat's simplified RL + RLVR trend | Small | High |
| 4 | Single-knob complexity tiers for training CLI | nanochat's `--depth` dial | Small | Medium |

---

### Implementation 1: Fix Evolutionary Training

**Goal**: Get the existing evolutionary system working by solving gradient norm instability.

**Files to modify**:
- `shared/evolutionary/candidate_generator.py` — add gradient clipping before candidate generation
- `shared/evolutionary/strategies/gradient_noise.py` — add per-parameter adaptive noise scaling
- `Trainers/sft/configs/config.yaml` — tune evolutionary config defaults
- `shared/evolutionary/trainer_wrapper.py` — add gradient norm logging for diagnostics

**Step 1: Add gradient norm clipping to CandidateGenerator**
```python
# candidate_generator.py — new method
@staticmethod
def clip_gradients(
    gradients: Dict[str, torch.Tensor],
    max_norm: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Clip per-parameter gradient norms before candidate generation."""
    clipped = {}
    for name, grad in gradients.items():
        norm = grad.norm()
        if norm > max_norm:
            clipped[name] = grad * (max_norm / norm)
        else:
            clipped[name] = grad
    return clipped
```

Call `clip_gradients()` on `base_gradients` in `trainer_wrapper.py` before passing to `generate()`.

**Step 2: Add adaptive noise scaling in GradientNoiseStrategy**
```python
# gradient_noise.py — modify generate_candidates()
# Instead of: noise = randn * noise_scale * grad.norm()
# Use: noise = randn * noise_scale * min(grad.norm(), max_grad_norm)
# This caps the noise magnitude regardless of gradient size
```

**Step 3: Update config defaults**
```yaml
evolutionary:
  enabled: true
  candidates: 4
  eval_batch_size: 2
  validation_config: "configs/fitness/tool_calling.yaml"
  strategy:
    type: "gradient_noise"
    params:
      noise_scale: 0.03          # Was 0.1 — reduced 3x
      max_grad_norm: 1.0         # NEW — clip before noise
  selection:
    method: "best"
    min_improvement: 0.01        # Was 0.0 — require meaningful improvement
  eval_frequency: 5              # Was 1 — reduce overhead, eval every 5 steps
  warmup_steps: 200              # Was 0 — let model stabilize first
```

**Step 4: Add gradient norm logging**
```python
# trainer_wrapper.py — in _evolutionary_step()
if self.config.log_candidates:
    grad_norms = {name: grad.norm().item() for name, grad in base_gradients.items()}
    avg_norm = sum(grad_norms.values()) / len(grad_norms)
    logger.info(f"Step {self.current_step}: avg_grad_norm={avg_norm:.4f}, "
                f"max_grad_norm={max(grad_norms.values()):.4f}")
```

**Step 5: Add unit tests**
```
tests/test_evolutionary/
├── test_gradient_clipping.py     # Verify clipping caps norms correctly
├── test_candidate_generation.py  # Verify noise magnitude stays bounded
├── test_fitness_evaluation.py    # Verify fitness scores are meaningful
└── test_trainer_wrapper.py       # End-to-end with mock trainer
```

**Validation**: Run a short SFT training (200 steps) with evolutionary enabled. Success criteria:
- Gradient norms stay below `max_grad_norm` after clipping
- Fitness scores distribute across [0.3, 1.0] (not all near 0.0)
- Selected candidate fitness > baseline fitness in >50% of evolutionary steps
- Training loss decreases comparably to non-evolutionary baseline

---

### Implementation 2: Autonomous Experiment Loop

**Goal**: Implement an autoresearch-inspired loop that searches over training hyperparameters by running short experiments and keeping winning configs.

**New files**:
- `shared/flywheel/experiment_loop.py` — ExperimentLoop class (core loop logic)
- `shared/flywheel/experiment_config.py` — ExperimentConfig dataclass
- `configs/flywheel/experiment_loop.yaml` — default search space and constraints
- `tests/flywheel/test_experiment_loop.py` — tests

**Modified files**:
- `shared/flywheel/orchestrator.py` — add `run_experiment_loop()` method
- `tuner.py` — add CLI entry point
- `Trainers/sft/configs/config.yaml` — ensure `max_steps` is overridable

**Architecture** (maps to autoresearch's 3 primitives):

```
┌──────────────────────────────────────────────────────────────┐
│ ExperimentLoop                                               │
│                                                              │
│  Editable Asset: configs/flywheel/experiment_loop.yaml       │
│    └─ search_space: {lr: [...], lora_rank: [...], ...}       │
│                                                              │
│  Scalar Metric: overall_pass_rate (from Evaluator)           │
│    └─ or: final_loss (from training run)                     │
│                                                              │
│  Time-Boxed Cycle: max_steps per experiment (e.g., 500)      │
│    └─ wall-clock budget: configurable (default: 10 min)      │
│                                                              │
│  Loop:                                                       │
│    1. Select next config (random / grid / Bayesian)          │
│    2. Write temp config.yaml with overrides                  │
│    3. Run training subprocess (max_steps=500)                │
│    4. Run eval subprocess (Evaluator/cli.py)                 │
│    5. Record result in results.tsv                           │
│    6. If metric improved: save config as new baseline        │
│    7. If metric declined: discard, revert to baseline        │
│    8. Repeat until budget exhausted                          │
│                                                              │
│  Output:                                                     │
│    results.tsv — all experiments with metrics                │
│    best_config.yaml — winning configuration                  │
│    experiment_log.jsonl — detailed logs for each run         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Step 1: ExperimentConfig dataclass**
```python
# shared/flywheel/experiment_config.py
@dataclass
class ExperimentConfig:
    # Budget
    max_experiments: int = 50
    budget_minutes_per_experiment: int = 10
    max_steps_per_experiment: int = 500

    # Search space (each key maps to a list of values to try)
    search_space: dict = field(default_factory=lambda: {
        "learning_rate": [5e-5, 1e-4, 2e-4, 5e-4],
        "r": [8, 16, 32, 64],
        "lora_alpha": [16, 32, 64, 128],
        "num_train_epochs": [1, 2],
        "warmup_ratio": [0.02, 0.05, 0.1],
    })

    # Strategy
    search_strategy: str = "random"  # "random", "grid", "bayesian"

    # Metric
    metric: str = "eval_pass_rate"  # "eval_pass_rate" | "final_loss"
    metric_direction: str = "maximize"  # "maximize" | "minimize"

    # Paths
    base_config_path: str = "Trainers/sft/configs/config.yaml"
    eval_scenario_path: str = "Evaluator/config/scenarios/tool_prompts.yaml"
    results_path: str = "experiments/results.tsv"
    dataset_path: str = ""  # Required — training dataset

    # Training
    trainer_type: str = "sft"  # "sft" | "kto" | "grpo"
```

**Step 2: ExperimentLoop core**
```python
# shared/flywheel/experiment_loop.py
class ExperimentLoop:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.baseline_metric = None
        self.best_config = None
        self.results: list[ExperimentResult] = []

    async def run(self) -> ExperimentSummary:
        """Main loop — run experiments until budget exhausted."""
        base_config = load_yaml(self.config.base_config_path)

        for i in range(self.config.max_experiments):
            # 1. Generate candidate config
            overrides = self._sample_config()

            # 2. Merge overrides into base config
            experiment_config = deep_merge(base_config, overrides)
            experiment_config["max_steps"] = self.config.max_steps_per_experiment

            # 3. Write temp config
            temp_config_path = f"experiments/run_{i:04d}/config.yaml"
            write_yaml(temp_config_path, experiment_config)

            # 4. Run training subprocess
            training_result = await self._run_training(temp_config_path)

            # 5. Run evaluation (if using eval_pass_rate)
            if self.config.metric == "eval_pass_rate":
                eval_result = await self._run_evaluation(training_result.model_path)
                metric_value = eval_result.overall_pass_rate
            else:
                metric_value = training_result.final_loss

            # 6. Record result
            result = ExperimentResult(
                run_id=i,
                overrides=overrides,
                metric=metric_value,
                duration_seconds=training_result.duration,
            )
            self.results.append(result)
            self._write_results_tsv(result)

            # 7. Keep or discard
            if self._is_improvement(metric_value):
                self.baseline_metric = metric_value
                self.best_config = experiment_config
                logger.info(f"Run {i}: IMPROVED to {metric_value:.4f}")
            else:
                logger.info(f"Run {i}: no improvement ({metric_value:.4f})")

        return ExperimentSummary(
            total_experiments=len(self.results),
            best_metric=self.baseline_metric,
            best_config=self.best_config,
            improvements_found=sum(1 for r in self.results if r.is_improvement),
        )
```

**Step 3: CLI integration**
```python
# tuner.py — add to menu
# [N] Experiment Loop (auto-tune hyperparameters)
#   └─ Runs N short training experiments, keeps best config
```

**Step 4: Results tracking**
```
experiments/
├── results.tsv           # Tab-separated: run_id, lr, rank, alpha, metric, duration, status
├── best_config.yaml      # Winning configuration
├── run_0001/
│   ├── config.yaml       # Config used for this run
│   ├── training.log      # Training output
│   └── eval_results.json # Evaluation results (if applicable)
├── run_0002/
│   └── ...
```

**Validation**: Run experiment loop with 10 experiments, 200 steps each, on a small dataset. Success criteria:
- At least 1 config outperforms baseline
- Results.tsv is complete and parseable
- Best config is reproducible (re-run produces similar metric)

---

### Implementation 3: FitnessEvaluator as GRPO Reward

**Goal**: Combine the flywheel's structural fitness scoring with GRPO's semantic reward rubrics for a richer reward signal.

**Files to modify**:
- `Trainers/grpo/src/rewards.py` — add FitnessEvaluator as a reward component
- `Trainers/grpo/configs/config.yaml` — add fitness reward weight
- `Trainers/grpo/configs/rewards/fitness.yaml` — NEW reward rubric wrapping FitnessEvaluator

**Step 1: Add fitness reward function**
```python
# Trainers/grpo/src/rewards.py — new function
def fitness_reward(
    model_output: str,
    ground_truth: dict,
    config_path: str = "configs/flywheel/fitness_rules.yaml",
) -> float:
    """Score model output using FitnessEvaluator (structural correctness).

    Complements semantic reward rubrics:
    - fitness_reward: Does the tool call parse? Is JSON valid? Required fields present?
    - args_match reward: Are the field VALUES correct? Right tool selected?
    """
    from shared.validation import FitnessEvaluator
    evaluator = FitnessEvaluator(config_path=config_path)
    result = evaluator.evaluate(model_output)
    return result.score  # 0.0-1.0
```

**Step 2: Register as reward component**
```yaml
# Trainers/grpo/configs/rewards/fitness.yaml
name: structural_fitness
weight: 0.3
type: fitness_evaluator
config_path: "configs/flywheel/fitness_rules.yaml"
description: "Structural correctness: tool call parses, JSON valid, required fields"
```

**Step 3: Wire into reward computation**
```python
# Trainers/grpo/src/rewards.py — modify compute_rewards()
# Add fitness_reward alongside existing reward rubrics:
#   total_reward = (
#       args_match_weight * args_match_score +    # semantic (existing)
#       json_structure_weight * json_score +       # format (existing)
#       fitness_weight * fitness_score             # structural (NEW)
#   )
```

**Validation**: Run GRPO training for 50 steps with and without fitness reward. Compare:
- Reward distribution (should be more informative with fitness component)
- Tool call structural validity rate in generated outputs
- Does adding fitness reward improve eval pass rate?

---

### Implementation 4: Single-Knob Complexity Tiers

**Goal**: Add `--tier` presets to training CLI so users don't need to understand LoRA hyperparameters.

**New files**:
- `Trainers/sft/configs/tiers/quick.yaml`
- `Trainers/sft/configs/tiers/standard.yaml`
- `Trainers/sft/configs/tiers/thorough.yaml`
- `Trainers/kto/configs/tiers/` — same structure

**Modified files**:
- `Trainers/sft/train_sft.py` — add `--tier` argument
- `Trainers/kto/train_kto.py` — add `--tier` argument

**Tier definitions**:

```yaml
# tiers/quick.yaml — Fast iteration, exploratory
r: 8
lora_alpha: 16
learning_rate: 5e-4
num_train_epochs: 1
max_steps: 200
warmup_ratio: 0.05
batch_size: 8
gradient_accumulation_steps: 2

# tiers/standard.yaml — Current defaults, production quality
r: 64
lora_alpha: 128
learning_rate: 2e-4
num_train_epochs: 1
warmup_ratio: 0.02
batch_size: 8
gradient_accumulation_steps: 4

# tiers/thorough.yaml — Maximum quality, slow
r: 128
lora_alpha: 256
learning_rate: 1e-4
num_train_epochs: 3
warmup_ratio: 0.1
batch_size: 4
gradient_accumulation_steps: 8
```

**Integration**:
```python
# train_sft.py — argument parsing
parser.add_argument("--tier", choices=["quick", "standard", "thorough"],
                    help="Preset complexity tier (overrides individual hyperparams)")

# Config loading
if args.tier:
    tier_config = load_yaml(f"configs/tiers/{args.tier}.yaml")
    config = deep_merge(config, tier_config)  # Tier overrides base
```

**Validation**: Run each tier on same small dataset, verify:
- `quick` completes in <5 minutes
- `standard` matches current defaults exactly
- `thorough` produces better eval scores than `standard` (on sufficient data)

---

### Implementation 4b: Best-Checkpoint Selection via Eval

**Goal**: After training completes, automatically evaluate the last N checkpoints and select the best one — not just the final checkpoint.

Training loss is a poor proxy for downstream task quality. A model with the lowest training loss may have overfit. The checkpoint that actually performs best on eval is often an earlier one. This is cheap to discover: just run the Evaluator on each saved checkpoint.

**Files to modify**:
- `Trainers/sft/train_sft.py` — add `--eval-checkpoints N` flag
- `Trainers/kto/train_kto.py` — same
- `tuner.py` — surface in CLI menu

**New files**:
- `shared/checkpoint_eval.py` — CheckpointEvaluator class

**How it works**:

```python
# shared/checkpoint_eval.py
class CheckpointEvaluator:
    def __init__(self, run_dir: str, eval_scenario: str):
        self.run_dir = Path(run_dir)
        self.eval_scenario = eval_scenario

    async def evaluate_checkpoints(
        self,
        last_n: int = 0,  # 0 = all checkpoints
    ) -> CheckpointReport:
        """Evaluate last N checkpoints + final_model, pick the best.

        1. Discover checkpoints: run_dir/checkpoints/checkpoint-{step}/
        2. Sort by step number
        3. Take last N (or all)
        4. Also include final_model/
        5. Run Evaluator on each → get overall_pass_rate
        6. Rank by eval score
        7. Copy best to run_dir/best_checkpoint/
        8. Write checkpoint_eval_results.tsv
        """
        checkpoints = self._discover_checkpoints()

        if last_n > 0:
            checkpoints = checkpoints[-last_n:]

        # Always include final_model
        final = self.run_dir / "final_model"
        if final.exists() and final not in checkpoints:
            checkpoints.append(final)

        results = []
        for ckpt in checkpoints:
            score = await self._run_eval(ckpt)
            results.append(CheckpointResult(
                path=ckpt,
                step=self._extract_step(ckpt),
                eval_score=score,
            ))

        # Sort by score, pick best
        results.sort(key=lambda r: r.eval_score, reverse=True)
        best = results[0]

        # Report
        logger.info(f"Best checkpoint: {best.path} (score={best.eval_score:.4f})")
        if best.path != final:
            logger.info(f"  ⚠ Final model scored {results[-1].eval_score:.4f} — "
                       f"earlier checkpoint is {best.eval_score - results[-1].eval_score:.4f} better")

        # Copy best to best_checkpoint/
        self._copy_best(best.path)

        # Write results
        self._write_tsv(results)

        return CheckpointReport(
            checkpoints_evaluated=len(results),
            best_checkpoint=best,
            final_model_rank=next(
                i+1 for i, r in enumerate(results) if r.path == final
            ),
            results=results,
        )

    async def _run_eval(self, checkpoint_path: Path) -> float:
        """Run Evaluator on a single checkpoint, return pass rate."""
        result = subprocess.run([
            "python", "-m", "Evaluator.cli",
            "--model", str(checkpoint_path),
            "--prompt-set", self.eval_scenario,
            "--output-format", "json",
        ], capture_output=True, text=True)
        eval_results = json.loads(result.stdout)
        return eval_results["results_summary"]["overall_pass_rate"]
```

**Integration with training**:

```python
# train_sft.py — after training completes
if args.eval_checkpoints:
    evaluator = CheckpointEvaluator(
        run_dir=output_dir,
        eval_scenario=args.eval_scenario or "Evaluator/config/scenarios/tool_prompts.yaml",
    )
    report = await evaluator.evaluate_checkpoints(last_n=args.eval_checkpoints)
    logger.info(f"Best: step {report.best_checkpoint.step} "
               f"(score={report.best_checkpoint.eval_score:.4f})")
```

**CLI usage**:
```bash
# Evaluate last 5 checkpoints after training
python train_sft.py --config config.yaml --eval-checkpoints 5

# Standalone: evaluate checkpoints from a previous run
python -m shared.checkpoint_eval \
    --run-dir sft_output_rtx3090/20260321_120000 \
    --last-n 5 \
    --eval-scenario Evaluator/config/scenarios/tool_prompts.yaml
```

**Output**:
```
experiments/checkpoint_eval/
├── checkpoint_eval_results.tsv   # step | path | eval_score | rank
└── best_checkpoint/              # Copy of the winning checkpoint
```

**Example output**:
```
Step    Path                              Score   Rank
500     checkpoints/checkpoint-500/       0.82    1     ← BEST
400     checkpoints/checkpoint-400/       0.79    2
600     final_model/                      0.76    3     ← Final was 3rd!
300     checkpoints/checkpoint-300/       0.74    4
200     checkpoints/checkpoint-200/       0.68    5
```

**Why this matters**: In practice, final checkpoint ≠ best checkpoint. Training loss typically continues decreasing while eval quality peaks and then degrades (overfitting). This is a ~30-minute investment (5 checkpoints × ~5 min eval each) that can catch 5-10% eval quality sitting in an earlier checkpoint.

**Feeds into surgery**: Once we know the best checkpoint AND the final checkpoint, checkpoint interpolation (Implementation 5, Operation 3) can search for an even better blend between them.

**Validation**: Run on an existing training run with 5+ saved checkpoints. Success criteria:
- Discovers that final_model is NOT always the best (expect this ~40% of the time)
- Results TSV is parseable and ordered correctly
- Best checkpoint copy is functional (can be loaded for inference)

---

### Implementation 5: LoRA Weight Surgery (Eval-Guided Post-Training Optimization)

**Goal**: Directly manipulate trained LoRA weights — scaling, pruning, interpolating, zeroing layers — and use the Evaluator as a fitness signal to find optimal weight configurations WITHOUT retraining.

This is like autoresearch but operating on trained weights instead of training configs. The loop is:
```
Load LoRA weights → perturb → eval → keep/discard → repeat
```

No gradient computation, no training step. Just weight manipulation + forward pass evaluation. Orders of magnitude faster than retraining.

**New files**:
- `shared/evolutionary/lora_surgery.py` — LoRASurgeon class (core manipulation + eval loop)
- `configs/lora_surgery.yaml` — default surgery configuration
- `tests/test_lora_surgery.py` — tests

**Modified files**:
- `tuner.py` — add CLI entry point

#### Background: LoRA Weight Structure

Each target module has two matrices per layer:
```
model.layers.{i}.self_attn.{q,k,v,o}_proj.lora_{A,B}.weight
model.layers.{i}.mlp.{gate,up,down}_proj.lora_{A,B}.weight
```

For Qwen2.5-7B with r=64: 32 layers × 7 modules × 2 matrices = 448 weight tensors.
The effective contribution of each module is: `(lora_alpha / r) × (B @ A) = 2.0 × (B @ A)`.

Stored in `adapter_model.safetensors` (~0.32 GB). Loaded via PEFT/Unsloth automatically.

#### Surgery Operations

**Operation 1: Layer-Level Scaling**
Scale individual layers' LoRA contribution by a factor. Reveals which layers matter most.
```python
def scale_layer(self, layer_idx: int, scale: float):
    """Scale all LoRA weights in a specific transformer layer.

    scale=0.0 → effectively removes LoRA from this layer
    scale=0.5 → halves this layer's LoRA contribution
    scale=1.5 → amplifies this layer's LoRA contribution
    """
    for module in ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]:
        key_a = f"model.layers.{layer_idx}.self_attn.{module}.lora_A.weight"
        key_b = f"model.layers.{layer_idx}.self_attn.{module}.lora_B.weight"
        # Also handle mlp keys
        if key_a in self.weights:
            self.weights[key_a] *= scale
```

**Operation 2: Module-Type Scaling**
Scale all instances of a specific module type (e.g., all q_proj across all layers).
```python
def scale_module_type(self, module_name: str, scale: float):
    """Scale all LoRA weights for a specific module type.

    scale_module_type("v_proj", 0.0) → removes LoRA from all v_proj
    Reveals whether attention vs MLP LoRA matters more.
    """
    for key in self.weights:
        if f".{module_name}.lora_" in key:
            self.weights[key] *= scale
```

**Operation 3: Checkpoint Interpolation (Model Soup)**
Linearly interpolate between two LoRA checkpoints (e.g., step 200 vs step 500).
```python
def interpolate(self, weights_a: dict, weights_b: dict, alpha: float) -> dict:
    """Interpolate between two LoRA weight sets.

    alpha=0.0 → pure weights_a
    alpha=1.0 → pure weights_b
    alpha=0.5 → average (classic model soup)

    Search over alpha to find optimal blend point.
    """
    result = {}
    for key in weights_a:
        result[key] = (1 - alpha) * weights_a[key] + alpha * weights_b[key]
    return result
```

**Operation 4: DARE (Drop And REscale)**
Randomly zero out a fraction of LoRA weights and rescale survivors. From the DARE paper — reduces interference when merging adapters, but also works as regularization on a single adapter.
```python
def dare(self, drop_rate: float = 0.1, seed: int = 42):
    """DARE: randomly zero weights and rescale survivors.

    If drop_rate=0.1: zero 10% of weights, multiply rest by 1/(1-0.1)
    Acts as post-hoc regularization. Can improve generalization.
    """
    rng = torch.Generator().manual_seed(seed)
    for key, tensor in self.weights.items():
        mask = torch.bernoulli(torch.ones_like(tensor) * (1 - drop_rate),
                              generator=rng)
        self.weights[key] = tensor * mask / (1 - drop_rate)
```

**Operation 5: SVD Rank Reduction**
Compress LoRA by reducing effective rank (keep top-k singular values).
```python
def reduce_rank(self, layer_idx: int, module: str, new_rank: int):
    """Reduce effective rank of a specific LoRA module via SVD.

    Given A (r × in) and B (out × r), compute:
    effective = B @ A  (out × in)
    U, S, V = SVD(effective)
    new_A = V[:new_rank, :]     (new_rank × in)
    new_B = U[:, :new_rank] * S[:new_rank]  (out × new_rank)

    Preserves the most important directions while compressing.
    NOTE: Changes adapter_config.json rank — requires save as new adapter.
    """
```

**Operation 6: Attention vs MLP Ablation**
Zero out all attention LoRA or all MLP LoRA to measure relative contribution.
```python
def ablate_attention(self):
    """Zero all attention LoRA weights (q,k,v,o_proj). Keep MLP only."""

def ablate_mlp(self):
    """Zero all MLP LoRA weights (gate,up,down_proj). Keep attention only."""
```

#### The Surgery Loop (Eval-Guided)

```python
class LoRASurgeon:
    def __init__(self, adapter_path: str, eval_config: str):
        self.base_weights = load_safetensors(adapter_path)
        self.adapter_config = load_json(adapter_path / "adapter_config.json")
        self.eval_config = eval_config
        self.baseline_score = None
        self.results: list[SurgeryResult] = []

    async def run_surgery_loop(self, operations: list[SurgeryOp]) -> SurgerySummary:
        """Main loop: try operations, eval each, keep improvements."""

        # 1. Establish baseline
        self.baseline_score = await self._evaluate(self.base_weights)
        logger.info(f"Baseline eval score: {self.baseline_score:.4f}")

        best_weights = self.base_weights.copy()
        best_score = self.baseline_score

        for op in operations:
            # 2. Apply operation to current best weights
            candidate_weights = op.apply(best_weights.copy())

            # 3. Save temporary adapter
            temp_path = self._save_temp_adapter(candidate_weights)

            # 4. Evaluate
            score = await self._evaluate(candidate_weights)

            # 5. Record result
            result = SurgeryResult(
                operation=op.name,
                params=op.params,
                score=score,
                delta=score - best_score,
                accepted=score > best_score,
            )
            self.results.append(result)

            # 6. Keep or discard
            if score > best_score:
                best_weights = candidate_weights
                best_score = score
                logger.info(f"  {op.name}: IMPROVED {score:.4f} (+{score-best_score:.4f})")
            else:
                logger.info(f"  {op.name}: no improvement ({score:.4f})")

            # 7. Cleanup temp adapter
            self._cleanup_temp(temp_path)

        # 8. Save winning weights
        self._save_adapter(best_weights, "surgically_optimized/")

        return SurgerySummary(
            baseline_score=self.baseline_score,
            final_score=best_score,
            improvement=best_score - self.baseline_score,
            operations_tried=len(self.results),
            operations_accepted=sum(1 for r in self.results if r.accepted),
        )

    async def _evaluate(self, weights: dict) -> float:
        """Save weights to temp adapter, run Evaluator, return pass rate."""
        temp_path = self._save_temp_adapter(weights)

        # Run evaluation subprocess
        result = subprocess.run([
            "python", "-m", "Evaluator.cli",
            "--model", str(temp_path),
            "--prompt-set", self.eval_config,
            "--output-format", "json",
        ], capture_output=True, text=True)

        # Parse overall_pass_rate from output
        eval_results = json.loads(result.stdout)
        return eval_results["results_summary"]["overall_pass_rate"]
```

#### Default Surgery Configuration

```yaml
# configs/lora_surgery.yaml
adapter_path: ""                  # Required — path to trained LoRA adapter
eval_scenario: "Evaluator/config/scenarios/tool_prompts.yaml"

# Operations to try (in order)
operations:
  # Phase 1: Layer importance discovery
  - type: scale_layer
    sweep:
      layers: "all"               # Try each layer independently
      scales: [0.0, 0.5, 1.5, 2.0]
    description: "Find which layers matter"

  # Phase 2: Module type ablation
  - type: scale_module_type
    sweep:
      modules: ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"]
      scales: [0.0]              # Zero each module type
    description: "Find which module types matter"

  # Phase 3: Checkpoint interpolation (if multiple checkpoints available)
  - type: interpolate
    sweep:
      checkpoint_a: "checkpoints/checkpoint-200"
      checkpoint_b: "final_model"
      alphas: [0.0, 0.25, 0.5, 0.75, 1.0]
    description: "Find optimal blend between checkpoints"

  # Phase 4: DARE regularization
  - type: dare
    sweep:
      drop_rates: [0.05, 0.1, 0.15, 0.2]
    description: "Post-hoc regularization via weight dropout"

# Output
output_dir: "experiments/surgery/"
results_file: "surgery_results.tsv"
```

#### CLI Integration

```bash
# Discover which layers matter
python tuner.py surgery --adapter final_model/ --operation layer-importance

# Interpolate between checkpoints
python tuner.py surgery --adapter final_model/ --operation interpolate \
    --checkpoint-a checkpoints/checkpoint-200 \
    --checkpoint-b final_model

# Full surgery loop (all operations)
python tuner.py surgery --adapter final_model/ --config configs/lora_surgery.yaml

# Output: experiments/surgery/
#   surgery_results.tsv    — all operations + scores
#   layer_importance.png   — heatmap of layer contributions (optional)
#   best_adapter/          — surgically optimized LoRA weights
```

**Operation 7: Alpha Sweep (Simplest Possible Surgery)**
The LoRA update is scaled by `lora_alpha / r`. Reducing alpha post-training attenuates the entire LoRA signal — equivalent to linear interpolation between base model and fine-tuned model. If training overfit (loss < 0.2), multiply alpha by 0.5 as a first fix.
```python
def sweep_alpha(self, alphas: list[float]):
    """Try different effective scaling factors.

    Default: lora_alpha=128, r=64 → scale=2.0
    alpha=64 → scale=1.0 (halved contribution)
    alpha=192 → scale=3.0 (amplified contribution)

    Just modifies adapter_config.json, no weight changes needed.
    """
    for alpha in alphas:
        config = self.adapter_config.copy()
        config["lora_alpha"] = alpha
        self._save_temp_adapter(self.base_weights, config)
        score = await self._evaluate(...)
```

**Operation 8: Metrics-Weighted Checkpoint Averaging (MWA)**
Save N checkpoints during training, evaluate each, merge using eval scores as interpolation weights. From [arXiv:2504.18580](https://arxiv.org/pdf/2504.18580) — outperformed best single checkpoint by 2.62%.
```python
def metrics_weighted_average(self, checkpoints: list[str]) -> dict:
    """Evaluate each checkpoint, merge weighted by eval score.

    1. Load each checkpoint's LoRA weights
    2. Evaluate each → get score_i
    3. Normalize scores: weight_i = score_i / sum(scores)
    4. Merged = sum(weight_i * weights_i)

    Can also use PEFT's add_weighted_adapter() directly.
    """
```

#### Research Backing

| Technique | Paper | Key Finding |
|-----------|-------|-------------|
| **Layer pruning** | [LoRA-drop (COLING 2025)](https://arxiv.org/abs/2402.07721) | Retaining only ~50% of LoRA modules matches full performance |
| **SVD compression** | [Post-hoc LoRA Compression](https://openreview.net/forum?id=Xg0u7lAIrs) | Many trained LoRAs have effective rank far below nominal |
| **DARE** | [Language Models are Super Mario](https://arxiv.org/abs/2311.03099) | Can drop 90-99% of delta params with rescaling |
| **TIES merging** | [TIES-Merging (Yadav et al.)](https://arxiv.org/abs/2306.01708) | Resolves parameter interference via trim + elect sign |
| **Checkpoint MWA** | [Metrics-Weighted Averaging](https://arxiv.org/pdf/2504.18580) | Eval-weighted checkpoint merge beats best single by 2.62% |
| **SLERP** | [Embedding Generalization](https://arxiv.org/html/2511.21703v1) | Preserves directional structure better than linear averaging |
| **Alpha scaling** | [Unsloth LoRA Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide) | Simplest post-training fix for overfitting |

**Existing tooling**: HuggingFace PEFT supports TIES, DARE, linear merging via `add_weighted_adapter()`. [mergekit](https://github.com/arcee-ai/mergekit) supports SLERP, TIES, DARE, task arithmetic. We can leverage these rather than reimplementing.

#### Key Advantages Over Retraining

| Aspect | Retraining | LoRA Surgery |
|--------|-----------|--------------|
| **Time per experiment** | 10-60 min | 1-3 min (eval only) |
| **GPU needed** | Yes (training) | Yes (inference only) |
| **VRAM** | 16-24 GB | 8-12 GB (inference) |
| **Reversibility** | New checkpoint | Instant (just reload original weights) |
| **Granularity** | Whole-model | Per-layer, per-module |
| **Search space** | Hyperparameters | Weight space directly |

#### What This Reveals

The layer importance scan is particularly valuable because it answers:
- **Which layers learned the most?** (zeroing them hurts eval the most)
- **Which layers are dead weight?** (zeroing them has no impact)
- **Is attention or MLP more important for tool-calling?** (ablation answers this)
- **Did training overfit certain layers?** (reducing their scale might improve generalization)

This data feeds back into Implementations 1-2: if we know layers 10-20 matter most, the evolutionary system can focus noise injection on those layers, and the experiment loop can test targeted LoRA configurations (e.g., higher rank for important layers).

**Validation**: Run layer importance scan on existing trained LoRA adapter. Success criteria:
- Identifies at least 3 layers where zeroing causes >5% eval drop
- Identifies at least 3 layers where zeroing has <1% impact
- Surgery loop finds a weight configuration that matches or beats baseline

---

### Execution Order

```
Phase 1 (Quick Wins — config changes + small code changes):
  └─ Implementation 1: Fix evolutionary training
  └─ Implementation 4: Complexity tiers
  └─ Implementation 4b: Best-checkpoint selection via eval

Phase 2 (Medium effort — new components):
  └─ Implementation 3: FitnessEvaluator as GRPO reward
  └─ Implementation 5: LoRA weight surgery

Phase 3 (Larger feature — new subsystem):
  └─ Implementation 2: Autonomous experiment loop
```

Phase 1 items can be done in parallel with no dependencies. 4b is the easiest win — just eval existing checkpoints. Phase 2 items are independent of each other. Phase 3 benefits from Phase 1 (the experiment loop can test evolutionary vs non-evolutionary training) and Phase 2 (surgery findings inform experiment search space).

4b naturally feeds into Implementation 5: once you know the best and final checkpoints, checkpoint interpolation can search for an even better blend between them.

**Full optimization stack when all implementations complete:**
```
Layer 0: LoRA Surgery (post-training, per-weight)
  "Which weights matter? Can we improve by pruning/scaling?"
  ↕ feeds insights into ↕
Layer 1: Evolutionary Training (during training, per-step)
  "Which gradient update is best this step?"
  ↕ operates within ↕
Layer 2: Experiment Loop (per-run, hyperparameter search)
  "Which training config produces the best model?"
  ↕ operates within ↕
Layer 3: Flywheel (per-cycle, dataset optimization)
  "What data should the next training run use?"
```

---

## Key Quotes

> "You're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the program.md Markdown files." — Karpathy on autoresearch

> "nanochat is not an LLM framework. There are no giant config objects, model factories, or conditional code monsters." — nanochat README

> "RLVR (RL with Verifiable Rewards) is the most consequential development of 2025, gobbling compute originally intended for pretraining." — Karpathy, 2025 Year in Review

---

## Sources

### Karpathy Projects
- [github.com/karpathy/nanochat](https://github.com/karpathy/nanochat) (~49.7k stars)
- [github.com/karpathy/autoresearch](https://github.com/karpathy/autoresearch) (~30.3k stars)
- [Karpathy's nanochat Discussion #1](https://github.com/karpathy/nanochat/discussions/1)
- [Karpathy's 2025 LLM Year in Review](https://karpathy.bearblog.dev/year-in-review-2025/)
- [nanochat launch tweet](https://x.com/karpathy/status/1977755427569111362)
- [autoresearch tweet](https://x.com/karpathy/status/2030371219518931079)

### Muon Optimizer
- [github.com/KellerJordan/Muon](https://github.com/KellerJordan/Muon) — Original implementation
- [torch.optim.Muon](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html) — PyTorch 2.9+ native
- [Keller Jordan's Muon blog post](https://kellerjordan.github.io/posts/muon/)
- [Deriving Muon — Jeremy Bernstein](https://jeremybernste.in/writing/deriving-muon)
- [HuggingFace Muon tutorial](https://discuss.huggingface.co/t/tutorial-understanding-and-implementing-the-muon-optimizer/167717)
- [PredNext: Muon for LLM Training](https://prednext.com/en/blog/optimizer-muon-2025/)
- [Riemannion: LoRA + Riemannian Muon (arXiv:2507.12142)](https://arxiv.org/abs/2507.12142)
- [MuonAll: Muon for all params including 1D (arXiv:2511.06086)](https://arxiv.org/abs/2511.06086)
