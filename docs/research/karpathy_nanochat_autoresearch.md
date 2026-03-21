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

### Recommendation

**Don't integrate Muon as the optimizer for LoRA fine-tuning.** Keep AdamW 8-bit.

**Do borrow the gradient normalization concept** for the evolutionary system — either via explicit gradient clipping (simple) or Newton-Schulz orthogonalization as a strategy (principled).

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
