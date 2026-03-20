# Per-Example Loss Tracking for Unsloth SFT

## TL;DR Recommendation

**Use post-hoc inference.** After training completes, run the trained model over each example individually (batch_size=1, no shuffle, `torch.no_grad()`) and compute the cross-entropy loss per example. This is the simplest, most reliable approach for our setup (RTX 3090, 2K-5K examples, Unsloth SFT with LoRA). It avoids all complications from shuffling, gradient accumulation, and packing. For 5K examples at ~2048 tokens each, expect ~15-30 minutes of inference time on an RTX 3090 with a 7B model in 4-bit. The result is a clean `(jsonl_line_index, loss)` mapping with zero training loop modifications.

---

## 1. Unsloth Shuffling and Packing Behavior

### Current Codebase Configuration (`config.yaml`)

```yaml
packing: false
completion_only_loss: true
seed: 42
group_by_length: false
```

### Data Shuffling

HF Trainer (which SFTTrainer and Unsloth wrap) uses **`RandomSampler` by default**, which shuffles the dataset each epoch. Key details:

- **Seed-controlled**: The sampler uses `seed` (or `data_seed` if set) plus the epoch number to deterministically shuffle. With `seed: 42`, the shuffle order is reproducible across identical runs.
- **`use_seedable_sampler: True`** (default since transformers ~4.36): Ensures fully deterministic sampling across processes.
- **Per-epoch re-shuffle**: The sampler sets its generator seed to `seed + epoch` before each epoch, so data order changes between epochs but is deterministic within the same seed.

**Implication**: Step N does NOT correspond to JSONL line N. The mapping is `step_N → shuffled_indices[N * batch_size : (N+1) * batch_size]`, and with gradient accumulation, the logged loss at step N is the average over `batch_size * gradient_accumulation_steps` examples.

### Sequence Packing

Our config has `packing: false`. When packing IS enabled:

- Unsloth supports packing strategies: `bfd` (best-fit decreasing, truncates overflow), `bfd-requeue` (re-queues overflow tokens), and `wrapped` (aggressive, cuts mid-sequence).
- Packing merges multiple examples into a single sequence, making per-example loss tracking significantly harder since loss is computed over the combined sequence with `padding_free` collation.
- **With packing disabled**, each batch element corresponds to exactly one dataset example (padded to batch max length). This is the key enabler for any per-example tracking approach.

### Reproducibility

With `packing: false` and a fixed `seed: 42`:
- The shuffle order is **fully deterministic** and reproducible.
- You CAN reconstruct which examples appeared in which batch by replaying the sampler with the same seed.
- However, this is fragile — any change to batch size, dataset size, or number of GPUs changes the mapping.

**Source**: [HF Trainer sampler](https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py), [Unsloth packing docs](https://unsloth.ai/docs/new/3x-faster-training-packing), [GitHub Issue #617](https://github.com/unslothai/unsloth/issues/617)

---

## 2. HF Trainer Hook Points for Online Loss Capture

### Available Hooks

| Hook | When It Fires | What's Available | Per-Example? |
|------|---------------|------------------|-------------|
| `compute_loss(model, inputs, return_outputs)` | Every forward pass | Model, tokenized batch, can return per-sample loss | **Yes** (with override) |
| `compute_loss_func` (config param) | Every forward pass | Raw outputs, labels, batch count | **Yes** (with custom func) |
| `on_log(args, state, control, logs)` | Every `logging_steps` | Aggregated loss (average over accumulated steps) | **No** |
| `on_step_end(args, state, control)` | Every optimizer step | State with `log_history` | **No** |
| `training_step(model, inputs)` | Every micro-batch | Model, inputs, returns loss | **Possible** (with override) |

### Online Approach: Subclass SFTTrainer and Override `compute_loss`

```python
class LossTrackingSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.per_example_losses = []  # [(global_step, [example_indices], [losses])]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Get the standard loss (scalar, averaged)
        outputs = model(**inputs)
        logits = outputs.logits

        labels = inputs["labels"]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Per-token loss with reduction='none'
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        # Reshape to (batch_size, seq_len) and average per example
        per_token_loss = per_token_loss.view(shift_labels.size())
        mask = (shift_labels != -100).float()
        per_example_loss = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # Log per-example losses
        # NOTE: We need dataset indices, which are NOT in `inputs` by default
        self.per_example_losses.append(per_example_loss.detach().cpu().tolist())

        # Return averaged loss for backprop
        avg_loss = per_example_loss.mean()
        return (avg_loss, outputs) if return_outputs else avg_loss
```

### Critical Problem: Dataset Index Tracking

The fundamental issue with the online approach is that **HF Trainer does not pass dataset indices to `compute_loss`**. The `inputs` dict contains tokenized tensors (`input_ids`, `labels`, `attention_mask`) but NOT which dataset row they came from.

**Workarounds**:
1. **Add an `index` column to the dataset**: Modify the dataset to include a `row_index` field, and modify the data collator to preserve it. This is invasive and may conflict with SFTTrainer's internal preprocessing.
2. **Replay the sampler**: Reconstruct the index mapping by replaying `RandomSampler` with the same seed after training. Fragile but possible.
3. **Disable shuffling**: Set `dataloader_shuffle=False` in TrainingArguments (not a standard parameter; would require subclassing the Trainer's `_get_train_sampler`). Then step order = dataset order.

### Verdict on Online Approach

**Feasible but invasive.** Requires:
- Subclassing SFTTrainer (which wraps HF Trainer)
- Custom loss computation that may interact poorly with Unsloth's optimized kernels
- Either modifying the dataset/collator to carry indices or replaying the sampler
- Careful handling of gradient accumulation (micro-batch losses must be paired with correct indices)
- Risk of breaking Unsloth's memory optimizations or attention kernel fusion

**Not recommended for our use case** given the simpler post-hoc alternative.

---

## 3. Post-Hoc Inference Approach (RECOMMENDED)

### How It Works

After training completes:

1. Load the trained LoRA model (already in memory, or reload from checkpoint)
2. Set `model.eval()` and use `torch.no_grad()`
3. Iterate over each JSONL example sequentially (no shuffle, batch_size=1)
4. Apply the same chat template and tokenization used during training
5. Forward pass → get logits → compute cross-entropy loss (only on completion tokens, matching `completion_only_loss: true`)
6. Record `(jsonl_line_index, loss_value)` pairs

### Implementation Sketch

```python
import torch
import json
from pathlib import Path

def compute_per_example_losses(model, tokenizer, dataset_path, max_seq_length=2048):
    """Compute per-example loss for each JSONL line using the trained model.

    Args:
        model: Trained model (already loaded, with LoRA merged or applied)
        tokenizer: Tokenizer with chat template applied
        dataset_path: Path to the original JSONL training file
        max_seq_length: Max sequence length (must match training)

    Returns:
        List of dicts: [{"index": 0, "loss": 1.234, "num_tokens": 512}, ...]
    """
    model.eval()
    results = []
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)

    with open(dataset_path) as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        example = json.loads(line)
        messages = example.get("messages") or example.get("conversations")

        # Apply same preprocessing as training
        text = tokenizer.apply_chat_template(
            sanitize_conversations(messages),
            tokenize=False,
            add_generation_prompt=False,
        )

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_length,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()

        # Apply completion-only masking (same as training)
        # For completion_only_loss, mask prompt tokens to -100
        # This requires identifying the assistant response boundary
        # (implementation depends on chat template)

        per_token_loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        # Average over non-masked tokens
        mask = (shift_labels != -100).float().view(-1)
        avg_loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1)

        results.append({
            "index": idx,
            "loss": avg_loss.item(),
            "num_completion_tokens": int(mask.sum().item()),
            "num_total_tokens": inputs["input_ids"].shape[1],
        })

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(lines)} examples")

    return results
```

### Performance Estimate (RTX 3090)

| Dataset Size | Model Size | Quantization | Est. Time | Throughput |
|-------------|-----------|-------------|-----------|-----------|
| 2,000 examples | 7B | 4-bit | ~8-12 min | ~3-4 examples/sec |
| 5,000 examples | 7B | 4-bit | ~20-30 min | ~3-4 examples/sec |
| 2,000 examples | 3B | 4-bit | ~4-6 min | ~6-8 examples/sec |
| 5,000 examples | 1.2B | none | ~5-8 min | ~10-15 examples/sec |

These estimates assume ~2048 tokens per example, batch_size=1, and forward pass only (no backward). Actual throughput depends on sequence length distribution.

**Optimization**: Can batch to `batch_size=4-8` for ~2-3x speedup, but requires careful padding and index tracking. For 2K-5K examples, batch_size=1 is fast enough and simpler.

### Pros

- **Zero training loop changes**: No subclassing, no custom loss, no risk of breaking Unsloth optimizations
- **Exact index mapping**: Sequential iteration over JSONL means `results[i]` = JSONL line `i`
- **Reproducible**: Same model + same data = same losses, always
- **Completion-only masking**: Can exactly replicate the training loss computation (mask prompt tokens)
- **Multi-model comparison**: Run the same script with each of the 3 trained models to get 3 loss columns per example
- **Works with any training config**: Packing, no packing, any batch size, any shuffle — doesn't matter
- **Separable from training**: Can be run as a standalone step, even on a different machine

### Cons

- **Post-training only**: Can't observe loss trajectory during training (but this isn't needed for the data quality scoring use case)
- **Not exactly "training loss"**: The model sees each example in isolation rather than in the context of gradient updates during training. For a well-converged model, this difference is negligible.
- **Extra GPU time**: Adds ~15-30 min per model. With 3 models, that's ~45-90 min total. Acceptable for a one-time scoring pipeline.

### Key Consideration: Training Loss vs. Inference Loss

The post-hoc loss is the model's loss on each example **after training is complete**. This differs slightly from the loss observed during training (which changes as the model updates). For data quality scoring with LightGBM, the post-training loss is actually **more useful** because:

1. It reflects the model's final capability on each example
2. Low post-training loss = the model learned this example well
3. High post-training loss = the model struggled with this example (potentially noisy, mislabeled, or too complex)
4. Comparing across 3 models: examples where ALL models have high loss are likely bad data; examples where only one model has high loss reveal model-specific weaknesses

---

## 4. TracIn / Captum Influence Functions

### What They Do

- **TracIn** (Training Run Influence): Estimates how much each training example contributed to a model's prediction on a test example, using gradient dot products across checkpoints.
- **Captum**: PyTorch library that includes TracIn and other influence function implementations.

### Feasibility Assessment for Our Setup

| Factor | Assessment |
|--------|-----------|
| **Dataset size (2K-5K)** | Manageable — influence functions scale with N^2 in the naive case, but 5K^2 = 25M is tractable |
| **LoRA parameters** | Reduces gradient dimensionality significantly (only ~1-10M params vs full 7B), making gradient dot products faster |
| **RTX 3090 (24GB)** | Sufficient for per-checkpoint gradient computation with 4-bit base + LoRA |
| **Implementation complexity** | HIGH — requires saving gradients at multiple checkpoints, computing pairwise gradient dot products |
| **Unsloth compatibility** | UNKNOWN — Captum may not work with Unsloth's custom kernels and 4-bit quantization |
| **Value for our use case** | LOW — we want per-example loss for data quality scoring, not influence attribution |

### Verdict

**Overkill for our use case.** TracIn answers "which training examples influenced this prediction?" — we're asking the simpler question "how well did the model learn each example?" Post-hoc loss answers our question directly.

TracIn would be valuable if we wanted to:
- Identify which training examples cause a specific failure mode
- Understand training data redundancy
- Debug specific model behaviors

None of these are our current goal. **Skip TracIn/Captum.**

**Source**: [Lacuna Inc. - LoRA-Enhanced Influence-Based Unlearning (2025)](https://arxiv.org/html/2506.04044v1)

---

## 5. Community Patterns and Libraries

### Established Patterns

1. **`DataCollatorForCompletionOnlyLM`** (TRL): Used for completion-only loss masking. Already effectively used in our codebase via `completion_only_loss: true` in SFTConfig. Relevant because any post-hoc loss computation must replicate this masking.

2. **`assistant_only_loss`** (TRL ≥0.29): Newer TRL feature that uses `{% generation %}` / `{% endgeneration %}` tags in Jinja chat templates to identify assistant tokens. Our codebase uses `completion_only_loss: true` instead, which is the older but equivalent approach for our dataset format.

3. **OpenAI's data quality scoring**: OpenAI's fine-tuning API reports per-example metrics. No open-source equivalent exists, but the post-hoc inference approach replicates this.

4. **RapidFire AI**: TRL integration for rapid SFT experimentation. Supports parallel config sweeps but doesn't offer per-example loss tracking.

5. **Weights & Biases**: Can log custom per-step metrics. Could be used to log per-example losses if we went with the online approach, but doesn't help with the core index-tracking problem.

### No Established Per-Example Loss Library

There is **no widely-adopted library** for per-example loss tracking in HF Trainer / TRL / Unsloth workflows. The community consensus (from HF forums and GitHub issues) is:
- Subclass Trainer and override `compute_loss` for online tracking
- Or run a separate inference pass after training (post-hoc approach)

The post-hoc approach is more common because it's simpler and doesn't risk breaking training.

---

## 6. Practical Recommendation

### Recommended Approach: Post-Hoc Inference Pipeline

```
Training Flow (unchanged):
  JSONL → SFTTrainer (Unsloth) → trained LoRA model → checkpoint

Per-Example Loss Scoring (new step):
  checkpoint + JSONL → sequential inference → per_example_losses.jsonl

LightGBM Data Quality Scoring (downstream):
  per_example_losses.jsonl (3 models) → feature matrix → LightGBM → quality scores
```

### Infrastructure Changes Needed

1. **New module**: `shared/experiment_tracking/per_example_loss.py`
   - Function: `compute_per_example_losses(model, tokenizer, dataset_path, max_seq_length) → List[Dict]`
   - Handles: chat template application, completion-only masking, sequential inference
   - Output: JSONL file with `{"index": N, "loss": X, "num_tokens": Y}` per line

2. **Integration point**: Post-training hook in `train_sft.py`
   - After `trainer.train()` completes and model is saved
   - Call `compute_per_example_losses()` and save alongside training artifacts
   - Store at `{run_dir}/per_example_losses.jsonl`

3. **RunRecord extension** (optional): Add `per_example_losses_path` field to track where the loss file is stored for each run

4. **Multi-model comparison script**: `Tools/compare_example_losses.py`
   - Load per_example_losses.jsonl from N runs
   - Join on `index` column
   - Output feature matrix for LightGBM: `[index, loss_model_A, loss_model_B, loss_model_C, num_tokens, ...]`

### Completion-Only Masking in Post-Hoc Inference

The most important implementation detail: the post-hoc inference MUST replicate the same masking used during training. Since our config uses `completion_only_loss: true`:

- During training, TRL's `DataCollatorForLanguageModeling` sets prompt token labels to `-100`
- In post-hoc inference, we must identify the prompt/completion boundary and apply the same mask
- The boundary is determined by the chat template — everything before the last assistant turn start marker is "prompt"
- Using `tokenizer.apply_chat_template()` with the same template ensures consistent tokenization

### Output Schema

```jsonl
{"index": 0, "loss": 1.2345, "num_completion_tokens": 312, "num_total_tokens": 1024, "jsonl_hash": "a1b2c3"}
{"index": 1, "loss": 0.8901, "num_completion_tokens": 156, "num_total_tokens": 512, "jsonl_hash": "d4e5f6"}
```

The `jsonl_hash` field (first 6 chars of the line's SHA-256) provides a data integrity check — ensures the loss file corresponds to the exact dataset version used for training.

---

## Summary Comparison

| Approach | Complexity | Accuracy | Unsloth Compat | Index Tracking | Recommendation |
|----------|-----------|----------|----------------|----------------|----------------|
| Online (subclass compute_loss) | HIGH | High (during training) | RISKY | Hard (no built-in) | Not recommended |
| Sampler replay | MEDIUM | Good (if seed matches) | OK | Fragile | Not recommended |
| Post-hoc inference | LOW | High (final model) | SAFE | Trivial (sequential) | **RECOMMENDED** |
| TracIn/Captum | VERY HIGH | Different metric | Unknown | N/A | Overkill |

### Key Decision Points

1. **Why not online?** Too invasive — risks breaking Unsloth's optimized kernels, requires solving the index-tracking problem, and the "during-training" loss isn't more useful than post-training loss for data quality scoring.

2. **Why post-hoc?** Zero training changes, trivial index mapping, safe with any Unsloth version, produces exactly the metric we need (final model loss per example).

3. **Is ~30 min per model acceptable?** Yes — this is a one-time step per experiment, and the 3-model pipeline will take hours for training anyway. Adding ~90 min of inference is negligible.

4. **What about packing?** Post-hoc inference is packing-agnostic. Even if we enable packing in the future, the post-hoc approach still works unchanged because it processes examples individually.

---

## 7. Data Quality Interpretation: What Per-Example Loss Actually Signals

### The Ambiguity Problem

Raw per-example loss after training is **ambiguous**. A high loss on a given example could mean any of:

| Interpretation | Signal | Quality Implication |
|---------------|--------|-------------------|
| **Noisy/mislabeled data** | The response is wrong or incoherent | BAD data — should be removed |
| **Conflicting data** | This example contradicts other training examples | BAD data — inconsistency in dataset |
| **Genuinely difficult** | Complex reasoning, rare pattern, long-tail knowledge | GOOD data — the model needs more capacity or exposure |
| **Formatting outlier** | Unusual structure that tokenizes differently | NEUTRAL — may need normalization, not removal |
| **Insufficient training** | Model hasn't converged on this example yet | NEUTRAL — more epochs might help |

**Key insight from the literature**: Raw loss alone cannot distinguish noise from difficulty. The research community has developed several approaches to disambiguate these signals.

### 7.1 IFD Score (Instruction Following Difficulty)

**Paper**: Li et al., "From Quantity to Quality" (NAACL 2024)
**GitHub**: [tianyi-lab/Cherry_LLM](https://github.com/tianyi-lab/Cherry_LLM)

The IFD score isolates the instructional difficulty from the inherent generation difficulty of the response:

```
IFD(Q, A) = L(A|Q) / L(A)
```

Where:
- `L(A|Q)` = cross-entropy loss on the response **given** the instruction (conditioned)
- `L(A)` = cross-entropy loss on the response **alone** (unconditioned)

**Interpretation**:
- **IFD < 1**: The instruction helps — the model finds it easier to generate A when given Q. This is "easy" instruction-following data.
- **IFD ≈ 1**: The instruction provides minimal guidance — the response is equally likely with or without the instruction.
- **IFD > 1**: The instruction actually hinders generation — the model finds it HARDER to generate A when given Q. This indicates a genuinely challenging instruction that requires real learning.

**For data selection**: Li et al. showed that selecting examples with moderate-to-high IFD (but filtering out IFD > threshold where misalignment is too extreme) and training on only 5-10% of data matched or exceeded full-dataset performance.

**Relevance to our pipeline**: IFD gives us a quality signal that raw loss cannot. An example with high raw loss but IFD ≈ 1 is probably noise (the response is just hard to generate regardless). An example with high raw loss AND high IFD is probably genuinely difficult and valuable.

**Implementation cost**: Requires one additional inference pass per example WITHOUT the instruction prefix (just the response alone). For 5K examples, this adds ~15-30 min on RTX 3090.

### 7.2 Loss Delta (Base Model vs. Trained Model)

Compare each example's loss under the base (pre-training) model vs. the fine-tuned model:

```
loss_delta = L_base(A|Q) - L_trained(A|Q)
```

**Interpretation**:
- **Large positive delta** (base loss >> trained loss): The model learned this example well. High "learning signal" — the fine-tuning was effective here.
- **Small positive delta**: Minimal learning happened. Either the base model already knew this, or training barely affected it.
- **Near-zero or negative delta**: The fine-tuning didn't help (or hurt) on this example. Potential red flag — may conflict with other data or be too noisy.

**Research support**: The "Delta Learning Hypothesis" (2025) demonstrates that what matters is the *quality gap* between signals, not absolute quality. This extends to per-example analysis: the delta between base and trained loss is a stronger quality indicator than either loss alone.

**Relevance to our pipeline**: Running the base model (before any training) over each example gives us a "pre-training baseline" for each example. The delta tells us HOW MUCH each example contributed to the model's learning, which is directly what LightGBM should predict.

**Implementation cost**: Requires inference with the base model over all examples — one additional ~15-30 min pass per base model. Since we're comparing 3 trained models, we only need 1 base model pass per unique base model.

### 7.3 Literature Survey: How Leading Papers Interpret Loss

| Paper | Year | Method | What They Measure | Key Finding |
|-------|------|--------|-------------------|-------------|
| **LIMA** | 2023 | Manual curation | N/A (human selection) | 1,000 high-quality examples > 52K mediocre ones. Perplexity does NOT correlate with response quality. |
| **AlpaGasus** | 2023 | ChatGPT scoring (1-5) | Accuracy, helpfulness, relevance | 9K filtered > 52K unfiltered. Low-quality data actively damages performance. |
| **IFD/Cherry** | 2024 | Loss ratio (conditioned/unconditioned) | Instruction-following difficulty | 5-10% of data matches full performance. IFD disambiguates difficulty from noise. |
| **DEITA** | 2024 | Complexity x Quality scorer | Evol-complexity, evol-quality, diversity | 6K selected > 300K unfiltered. Quality and complexity must BOTH be high. |
| **D3** | 2025 | Diversity + Difficulty + Dependability | UPD (loss x entropy), teacher eval, embedding distance | Multiplicative scoring — all 3 dimensions must be high. High loss + high entropy = low difficulty (multiple valid answers, not model failure). |
| **DELIFT** | 2024 | Pairwise utility (ICL-based) | How useful is example X for predicting example Y? | 70% data reduction without performance loss. Dynamic scoring adapts to model state. |
| **DSIR** | 2023 | Importance resampling | Distribution match to target domain | Works for pretraining data selection. Less relevant for instruction tuning. |

**Synthesis**: The field has converged on the idea that **raw loss is a necessary but insufficient signal**. The strongest data selection methods combine:
1. A loss-based signal (how hard is this for the model?)
2. A quality/dependability signal (is the response actually good?)
3. A diversity signal (does this add new information?)

### 7.4 Disambiguating Difficulty from Noise

The D3 paper (IJCAI 2025) provides the clearest framework for this problem. Their key insight:

> High loss can come from two sources: (a) the model genuinely struggles with the instruction (valuable difficulty), or (b) the response has multiple valid completions with high entropy (generation diversity, not real difficulty).

Their **Uncertainty-based Prediction Difficulty (UPD)** metric:

```
UPD(i) = sigmoid(Loss(i)) * max(1 - Entropy(i) / log(vocab_size)^beta, 0)
```

- High loss + low entropy = HIGH difficulty (model is confidently wrong → valuable)
- High loss + high entropy = LOW difficulty (model is uncertain among many valid answers → not real difficulty)
- Low loss (any entropy) = LOW difficulty (model handles it fine)

**Practical takeaway for our pipeline**: When computing per-example loss, also compute per-example entropy. The combination of loss and entropy is a much better quality signal than loss alone.

### 7.5 Feature Engineering Recommendations for LightGBM

Based on the literature survey, here are the concrete features to compute and feed to LightGBM:

#### Primary Features (per example, per model)

| Feature | How to Compute | What It Signals |
|---------|---------------|----------------|
| `loss_trained` | Post-hoc inference with trained model | Final model's difficulty with this example |
| `loss_base` | Post-hoc inference with base model (pre-training) | Inherent difficulty for the architecture |
| `loss_delta` | `loss_base - loss_trained` | How much the model LEARNED from this example |
| `loss_ratio` | `loss_trained / loss_base` | Relative improvement (normalized for difficulty) |
| `ifd_score` | `L(A\|Q) / L(A)` on the trained model | Instruction-following difficulty vs response difficulty |
| `entropy_trained` | Mean entropy of token predictions (trained model) | Prediction certainty — disambiguates difficulty from noise |
| `num_completion_tokens` | Count of non-masked tokens | Longer responses may have naturally higher loss |

#### Cross-Model Features (computed across all 3 trained models)

| Feature | How to Compute | What It Signals |
|---------|---------------|----------------|
| `loss_mean` | Mean of `loss_trained` across 3 models | Overall difficulty consensus |
| `loss_std` | Std dev of `loss_trained` across 3 models | Model disagreement — high std = example is architecture-sensitive |
| `loss_min` | Min of `loss_trained` across 3 models | At least one model learned it well |
| `loss_max` | Max of `loss_trained` across 3 models | Worst-case difficulty |
| `delta_mean` | Mean of `loss_delta` across 3 models | Consensus learning signal |
| `all_high_loss` | Bool: all 3 models have loss > threshold | Strong noise indicator (no model could learn it) |
| `disagreement` | Max loss - min loss across models | Architecture sensitivity |

#### Metadata Features (from the JSONL itself, no inference needed)

| Feature | Source | What It Signals |
|---------|--------|----------------|
| `num_turns` | Count of conversation turns | Multi-turn complexity |
| `num_tool_calls` | Count of `<tool_call>` blocks | Tool-calling complexity |
| `response_length` | Character count of assistant response | Response verbosity |
| `instruction_length` | Character count of user message | Instruction complexity |
| `has_code` | Whether response contains code blocks | Code generation difficulty |

### 7.6 What Should the LightGBM Target Variable Be?

This is the critical design question. Options:

#### Option A: Human/LLM-Judged Quality Score (RECOMMENDED)

Score a sample of examples (e.g., 200-500) using an LLM judge (GPT-4, Claude) on dimensions like:
- Response correctness (0-5)
- Response completeness (0-5)
- Instruction following (0-5)
- Tool-calling accuracy (0-5)

Train LightGBM to predict this composite score from the loss-based features. Then apply the trained LightGBM to score ALL examples.

**Why this is best**: The loss features are *predictive inputs*, not the target. The target should be actual quality (as judged by a strong model or human). This follows the AlpaGasus/DEITA pattern and avoids circular reasoning.

#### Option B: Cross-Model Agreement Signal

Target = 1 if all 3 models achieve low loss on the example, 0 if any model has high loss. This is fully automated (no LLM judge needed) but may conflate difficulty with noise.

#### Option C: Leave-One-Out Training Impact

For each example, train with and without it, measure downstream eval impact. This is the gold standard but computationally prohibitive (5K x 3 models x training runs).

#### Recommendation

**Start with Option A** (LLM-judged quality on a sample of 200-500 examples), using the feature set from Section 7.5. This gives us:
1. Interpretable quality scores
2. Efficient labeling (200-500 examples scored by LLM, not 5K)
3. Rich feature set from multi-model loss comparison
4. The LightGBM model generalizes the LLM judge's judgment to all 5K examples using the loss-based features as proxies

If Option A results are unsatisfactory, fall back to Option B (cross-model agreement) as a fully automated alternative.

### 7.7 Summary: What Loss Tells Us (and Doesn't)

**Loss DOES signal**:
- How well the model can reproduce this example's response
- Relative difficulty compared to other examples
- When combined across models: architecture-sensitive vs universally hard examples

**Loss DOES NOT signal**:
- Whether the response is actually correct
- Whether the instruction is clear
- Whether the example is duplicated or redundant
- Whether the example contributes to downstream task performance

**Loss BECOMES a quality signal when**:
- Combined with IFD score (disambiguates instruction difficulty from response difficulty)
- Combined with entropy (disambiguates genuine difficulty from noise)
- Compared across models (consensus = real signal, disagreement = architecture-dependent)
- Compared to base model (learning delta = how much the model actually learned)
- Used as a FEATURE for a quality predictor, not as the quality metric itself
