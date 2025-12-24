# Latent Space Reasoning for Tool Calls: Design Analysis

## Executive Summary

This document analyzes how **latent space reasoning** techniques could be applied to tool-call fine-tuning in Synaptic-Tuner. The core idea: instead of generating explicit `<thinking>` blocks as text, the model could reason in its hidden state space, producing more efficient and potentially higher-quality tool selection.

---

## Background: Current Approach

### Your Current Training Format

```json
{
  "conversations": [
    {"role": "system", "content": "<session_context>..."},
    {"role": "user", "content": "Run MoodDJ over Playlist Ideas..."},
    {"role": "assistant", "content": "<thinking>\n{\"goal\": \"...\", \"memory\": \"...\", \"requirements\": [...], \"assessment\": {...}, \"confidence\": 0.9, \"plan\": [...]}\n</thinking>",
     "tool_calls": [{"function": {"name": "agentManager_executePrompt", "arguments": "..."}}]}
  ]
}
```

**Current reasoning structure (explicit tokens):**
- `goal` → What the model is trying to accomplish
- `memory` → Relevant context from prior conversation
- `requirements` → Preconditions to check
- `assessment` → Complexity/risk evaluation
- `confidence` → Model's certainty score
- `plan` → Step-by-step execution plan

---

## Latent Space Reasoning Approaches

### 1. Coconut (Meta AI) - Continuous Thought

**Paper:** [Training LLMs to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769)
**Repo:** [facebookresearch/coconut](https://github.com/facebookresearch/coconut)

**How it works:**
```
[Question] → <bot> → [latent₁] → [latent₂] → ... → <eot> → [Answer/Tool Call]
```

Instead of generating "Let me think..." as text tokens, the model:
1. Outputs a special `<bot>` (begin-of-thought) token
2. Uses its **last hidden state** as the next input embedding
3. Iterates N times in latent space (no token generation)
4. Outputs `<eot>` (end-of-thought) and generates the answer

**Training curriculum:**
- Stage 0: Normal CoT with all reasoning explicit
- Stage 1: Replace first reasoning step with latent thought
- Stage 2: Replace second reasoning step with latent thought
- Stage N: All reasoning in latent space

**Results:** 99.9% accuracy on logical reasoning (vs 98.8% for CoT), with **fewer output tokens**.

---

### 2. Evolutionary Latent Optimization (dl1683)

**Repo:** [dl1683/Latent-Space-Reasoning](https://github.com/dl1683/Latent-Space-Reasoning)

**How it works:**
```
[Query] → Encode → [latent₀]
           ↓
    ┌──────────────────────────────┐
    │  Evolutionary Loop:          │
    │  1. Mutate latents (noise)   │
    │  2. Score with trained judge │
    │  3. Select best              │
    │  4. Crossover parents        │
    │  Repeat N generations        │
    └──────────────────────────────┘
           ↓
[best_latent] → Decode → [Response/Tool Call]
```

**Key innovation:** A trained **latent scorer** neural network predicts response quality from hidden states without decoding. Evolutionary search finds better latent representations.

**Results:** Produces more specific, actionable outputs (e.g., concrete implementation options vs. generic advice).

---

### 3. Latent-SFT with Vocabulary-Space Superposition

**Paper:** [Latent Reasoning in LLMs as a Vocabulary-Space Superposition](https://arxiv.org/abs/2510.15522)

**Key insight:** The degradation in latent reasoning comes from **unstructured latent space**. Solution: constrain latent space to the **column space of the LLM vocabulary** (treat latent reasoning as a superposition over vocabulary probabilities).

**Results:** Matches explicit SFT performance while cutting reasoning chains by **4x**.

---

## Mapping to Tool-Call Training

### Conceptual Translation

| Current (Explicit) | Latent Space Equivalent |
|-------------------|------------------------|
| `<thinking>` block | `<bot>...<eot>` continuous thought |
| `"goal": "..."` | Encoded in latent dimensions |
| `"requirements": [...]` | Latent attention patterns |
| `"confidence": 0.9` | Learned scoring mechanism |
| `"plan": [...]` | Latent multi-step reasoning |
| Tool selection logic | Latent→tool decoder |

### Potential Benefits for Tool Calls

1. **Faster inference**: No need to generate 200+ tokens of `<thinking>` JSON
2. **Better tool selection**: Latent space can encode more nuanced decision criteria
3. **Privacy**: Reasoning process isn't exposed to end users
4. **Multi-path exploration**: Coconut enables BFS-like exploration of tool choices
5. **Efficiency**: Latent thoughts compress reasoning significantly

### Potential Challenges

1. **Interpretability loss**: Can't inspect why the model chose a specific tool
2. **Debugging difficulty**: Harder to understand failures
3. **Training complexity**: Multi-stage curriculum required
4. **Judge quality**: Evolutionary approach needs trained latent scorer

---

## Implementation Options

### Option A: Hybrid Coconut (Recommended Starting Point)

**Approach:** Keep some structured reasoning explicit, move complex parts to latent space.

```
[User query]
<bot>[latent planning: 2 cycles]<eot>
<tool_decision>
  tool: agentManager_executePrompt
  confidence: high
</tool_decision>
<tool_call>{"name": "...", "arguments": {...}}</tool_call>
```

**Implementation steps:**
1. Fork facebookresearch/coconut
2. Adapt for ChatML format
3. Define tool-call specific special tokens
4. Create curriculum data:
   - Stage 0: Your current data (full `<thinking>` blocks)
   - Stage 1: Replace `goal` + `memory` with latent
   - Stage 2: Replace `requirements` + `assessment` with latent
   - Stage 3: Replace `plan` with latent
   - Stage 4: Full latent reasoning

**Changes to `train_sft.py`:**
```python
# New config options
config.latent_reasoning = {
    "enabled": True,
    "method": "coconut",
    "c_thought": 2,  # continuous thoughts per stage
    "epochs_per_stage": 1,
    "max_latent_stage": 4,
}

# Add special tokens to tokenizer
special_tokens = ["<bot>", "<eot>", "<tool_decision>"]
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
```

**Training config (YAML):**
```yaml
method: coconut
c_thought: 2
epochs_per_stage: 1
max_latent_stage: 4
model_id: "unsloth/Qwen2.5-7B-bnb-4bit"
```

**Pros:**
- Maintains some interpretability
- Gradual transition
- Meta AI validated approach
- Supports breadth-first tool exploration

**Cons:**
- Complex multi-stage training
- Requires curriculum data generation

---

### Option B: Evolutionary Latent Scorer

**Approach:** Train a separate judge model to score tool-call latents, then use evolutionary optimization at inference.

**Implementation steps:**
1. Generate tool-call attempts (good and bad examples)
2. Extract hidden states from your SFT model
3. Train scorer: `latent → quality_score`
4. At inference: evolve latents, decode best one to tool call

**Scorer architecture:**
```python
class LatentToolScorer(nn.Module):
    def __init__(self, hidden_dim=4096):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, latent):
        return self.layers(latent)
```

**Pros:**
- Can improve already-trained models
- Inference-time optimization
- No retraining base model

**Cons:**
- Added inference latency (evolutionary loop)
- Requires training separate scorer
- Quality limited by scorer accuracy

---

### Option C: Latent-SFT with Vocabulary Superposition

**Approach:** Train model where latent thoughts are constrained to vocabulary space.

**Key modification:** Instead of free-form latent vectors, latent thoughts are softmax distributions over vocabulary.

**Pros:**
- Better structured latent space
- Maintains connection to token semantics
- Proven compression ratio (4x)

**Cons:**
- Most research-stage implementation
- Requires custom training loop modifications

---

## Recommended Phased Approach

### Phase 1: Baseline + Evaluation Framework (1-2 weeks)

1. Create tool-call evaluation benchmark
2. Measure current model's:
   - Tool selection accuracy
   - Token generation count
   - Inference latency
   - Reasoning quality (human eval)

### Phase 2: Coconut Integration (2-3 weeks)

1. Fork coconut repo, adapt for your architecture
2. Generate curriculum data from existing datasets
3. Add CLI option: `--latent-reasoning coconut`
4. Train and compare against baseline

### Phase 3: Hybrid Optimization (1-2 weeks)

1. Keep explicit tool name/confidence
2. Move planning to latent space
3. A/B test interpretability vs efficiency

### Phase 4: Advanced (Optional)

1. Train latent scorer for evolutionary optimization
2. Implement inference-time latent search
3. Explore vocabulary-superposition approach

---

## Data Format Changes

### Current Format
```json
{
  "conversations": [
    {"role": "assistant", "content": "<thinking>{...}</thinking>", "tool_calls": [...]}
  ]
}
```

### Coconut Curriculum Format (Stage 0)
```json
{
  "question": "Run MoodDJ over Playlist Ideas...",
  "answer": "agentManager_executePrompt({...})",
  "steps": [
    "Identify goal: generate chillwave directions",
    "Check requirements: MoodDJ agent exists",
    "Assess complexity: low risk, familiar pattern",
    "Plan: invoke agent with append action",
    "Select tool: agentManager_executePrompt"
  ]
}
```

### Coconut Training Format (Final Stage)
```json
{
  "question": "Run MoodDJ over Playlist Ideas...",
  "answer": "agentManager_executePrompt({...})",
  "steps": []  // Empty - all reasoning is latent
}
```

---

## Configuration Options (Proposed)

```yaml
# training_config.yaml

latent_reasoning:
  enabled: false  # Toggle feature
  method: "coconut"  # coconut | evolutionary | vocabulary_superposition

  # Coconut-specific
  coconut:
    c_thought: 2  # Continuous thoughts per reasoning step
    epochs_per_stage: 1
    max_latent_stage: 4
    pad_latent_to_max: true

  # Evolutionary-specific
  evolutionary:
    chains: 8  # Parallel lineages
    generations: 10
    temperature_init: 0.5
    temperature_decay: 0.9
    scorer_checkpoint: "checkpoints/latent_scorer/best.pt"

  # Special tokens
  special_tokens:
    begin_thought: "<bot>"
    end_thought: "<eot>"
    tool_decision: "<tool_decision>"
```

---

## Evaluation Metrics

| Metric | Current | Target with Latent |
|--------|---------|-------------------|
| Tool selection accuracy | TBD | ≥ current |
| Tokens per tool call | ~200-300 | ~50-100 |
| Inference latency | TBD | ≤ current |
| Reasoning quality (human) | TBD | ≥ current |
| Edge case handling | TBD | ≥ current |

---

## Open Questions

1. **How much interpretability do you need?** If debugging tool selection is critical, pure latent may not be suitable.

2. **What's your inference budget?** Evolutionary approach adds latency; Coconut is faster.

3. **Do you want to retrain or enhance existing models?** Evolutionary scorer can enhance without retraining.

4. **What base model?** Coconut research used GPT-2; your Qwen/Llama models may behave differently.

---

## References

- [Training LLMs to Reason in Continuous Latent Space (Coconut)](https://arxiv.org/abs/2412.06769)
- [facebookresearch/coconut](https://github.com/facebookresearch/coconut)
- [dl1683/Latent-Space-Reasoning](https://github.com/dl1683/Latent-Space-Reasoning)
- [Latent Reasoning as Vocabulary-Space Superposition](https://arxiv.org/abs/2510.15522)
- [Latent Thinking Optimization](https://arxiv.org/abs/2509.26314)

---

## Next Steps

1. **Decide on approach**: Hybrid Coconut vs Evolutionary vs Full latent
2. **Create evaluation benchmark**: Tool selection test set with ground truth
3. **Generate curriculum data**: Convert existing datasets to staged format
4. **Prototype integration**: Fork coconut, adapt to your training pipeline
5. **A/B testing**: Compare against current thinking-block approach
