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

## Key Quotes

> "You're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the program.md Markdown files." — Karpathy on autoresearch

> "nanochat is not an LLM framework. There are no giant config objects, model factories, or conditional code monsters." — nanochat README

> "RLVR (RL with Verifiable Rewards) is the most consequential development of 2025, gobbling compute originally intended for pretraining." — Karpathy, 2025 Year in Review

---

## Sources

- [github.com/karpathy/nanochat](https://github.com/karpathy/nanochat) (~49.7k stars)
- [github.com/karpathy/autoresearch](https://github.com/karpathy/autoresearch) (~30.3k stars)
- [Karpathy's nanochat Discussion #1](https://github.com/karpathy/nanochat/discussions/1)
- [Karpathy's 2025 LLM Year in Review](https://karpathy.bearblog.dev/year-in-review-2025/)
- [nanochat launch tweet](https://x.com/karpathy/status/1977755427569111362)
- [autoresearch tweet](https://x.com/karpathy/status/2030371219518931079)
