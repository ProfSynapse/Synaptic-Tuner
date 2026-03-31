# Mechanistic Interpretability Integration Research

**Date:** 2026-03-31
**Status:** Research / Proposal
**Context:** Identifying how open-source mechanistic interpretability tools could integrate into our fine-tuning → evaluation → improvement cycle.

---

## Executive Summary

Mechanistic interpretability (mech interp) has matured rapidly — MIT Technology Review named it a "2026 Breakthrough Technology." The field now offers production-grade open-source tools that can provide **deeper signal about what fine-tuning actually changes inside a model**, beyond surface-level eval metrics like loss and pass rates.

This document maps the open-source landscape to our pipeline and proposes concrete integration points.

---

## 1. Tool Landscape

### Tier 1: High-Priority Tools (Direct Pipeline Integration)

#### SAELens
- **Repo:** [decoderesearch/SAELens](https://github.com/decoderesearch/SAELens)
- **What:** Train and analyze sparse autoencoders on any PyTorch model. Decompose dense activations into interpretable features.
- **Model support:** Any HuggingFace model (works via `encode()`/`decode()` on extracted activations)
- **Status:** Active (v6 refactor, 2025). Large ecosystem: SAE-Vis, SAEBench, Neuronpedia integration.
- **Integration potential:** **HIGH** — Train SAEs on base model, then compare feature activations before/after LoRA fine-tuning to see exactly which features changed.

#### TransformerLens
- **Repo:** [TransformerLensOrg/TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
- **What:** Load 50+ open-source models, cache/edit any internal activation. v3 beta (Mar 2026) supports large models.
- **Model support:** GPT-2, Llama, Gemma, Qwen, Mistral, and more. Best for models ≤9B (v3 extends to larger).
- **Status:** Active (v3.0.0b3, March 2026). Created by Neel Nanda (DeepMind).
- **Integration potential:** **HIGH** — Foundation for activation analysis, logit attribution, and circuit discovery during eval.

#### Anthropic's Circuit Tracer
- **Repo:** [safety-research/circuit-tracer](https://github.com/safety-research/circuit-tracer)
- **What:** Trace computational circuits in models using cross-layer transcoders. Produces attribution graphs showing how features connect.
- **Model support:** Gemma-2-2B, Llama-3.1-1B, Qwen3-4B (expanding).
- **Status:** Active (2025). Open-sourced by Anthropic Fellows.
- **Integration potential:** **MEDIUM-HIGH** — Compare circuits before/after fine-tuning to see if tool-calling circuits are forming correctly.

#### pyvene
- **Repo:** [stanfordnlp/pyvene](https://github.com/stanfordnlp/pyvene)
- **What:** Declarative intervention library for any PyTorch model. Interventions are serializable, shareable via HuggingFace.
- **Model support:** Any PyTorch model (RNNs, transformers, Mamba, etc.)
- **Status:** Active (NAACL 2024 paper, Stanford NLP).
- **Integration potential:** **HIGH** — Run causal interventions to test whether fine-tuned models have learned the right internal representations (not just producing correct outputs by shortcut).

### Tier 2: Supporting Tools

#### NNsight + NNterp
- **Repo:** ndif-team/nnsight (730+ stars, ICLR 2025)
- **What:** "Write once, run anywhere" interpretability. Same code works on GPT-2 locally or Llama-405B remotely via NDIF.
- **v0.6 (Feb 2026):** First-class AI agent support, NNterp for standardized interface across 50+ model variants.
- **Integration potential:** **MEDIUM** — Useful for running interp on models too large for our RTX 3090.

#### Goodfire Ember (SAEs)
- **What:** Production mech interp API + open-sourced SAE weights for Llama 3.3 70B.
- **Status:** Series B ($150M, Feb 2026). SAE weights open-sourced, but API deprecated for general use.
- **Integration potential:** **LOW-MEDIUM** — Pre-trained SAEs could bootstrap our analysis, but model-specific.

#### OpenMOSS/Language-Model-SAEs
- **Repo:** [OpenMOSS/Language-Model-SAEs](https://github.com/OpenMOSS/Language-Model-SAEs) (168 stars)
- **What:** Fully-distributed SAE training framework. Supports CrossCoders, CLTs, MoLT, Lorsa.
- **Status:** Active (v2.0.0b5). Research-grade.
- **Integration potential:** **MEDIUM** — Alternative to SAELens for distributed SAE training at scale.

#### Gemma Scope 2 (Google DeepMind)
- **HuggingFace:** [google/gemma-scope-2](https://huggingface.co/google/gemma-scope-2)
- **What:** Pre-trained SAEs + transcoders for all Gemma 3 models (270M–27B). 110PB of activation data, 1T+ SAE parameters.
- **Integration potential:** **HIGH if training Gemma models** — Free, pre-computed interpretability infrastructure.

#### EleutherAI Delphi + Auto-Interp Pipeline
- **What:** Automated feature explanation pipeline. Generate natural language descriptions of SAE features using LLMs. Cost: ~$1,300 for 1.5M features (vs $200K for prior methods).
- **Integration potential:** **MEDIUM** — Auto-label features to make SAE analysis human-readable.

#### Baukit
- **Repo:** [davidbau/baukit](https://github.com/davidbau/baukit)
- **What:** Lightweight tracing/editing toolkit. Spiritual predecessor to NNsight.
- **Integration potential:** **LOW** — Superseded by NNsight for most use cases.

### Tier 3: Research Techniques (No Standalone Tool)

#### Representation Engineering / Activation Steering
- **What:** Find linear directions in activation space that correspond to concepts (honesty, tool-use, refusal). Steer by adding vectors at inference.
- **Key advances (2026):** CAST (conditional steering, ICLR 2025), SAE-guided steering, Steering Vector Fields.
- **Integration potential:** **HIGH for steering** — Could steer fine-tuned models at inference time without retraining.

#### Activation Patching / Causal Tracing
- **What:** Replace activations from one forward pass with another to identify causally important components.
- **Tools:** Built into TransformerLens, pyvene, NNsight.
- **Integration potential:** **HIGH** — Compare which components are causally important for tool-calling in base vs. fine-tuned model.

---

## 2. Integration Points with Our Pipeline

### 2.1 Post-Training Diagnostic (NEW STAGE)

**Where:** After SFT/KTO training completes, before evaluation.

**What it does:**
1. Load base model and fine-tuned model (LoRA adapter)
2. Run a diagnostic prompt set through both
3. Compare internal activations at key layers using SAELens or TransformerLens
4. Report: which features activated/deactivated, which circuits changed

**Why:** Current pipeline only knows training loss went down. This tells you *what the model actually learned* — did it learn the right tool-calling circuits, or did it learn a shortcut?

**Key research backing:**
- LoRA creates **distributed circuits**, not isolated neurons (Lee, 2025)
- Fine-tuning exhibits **"delayed specialization"** — early layers stable, late layers restructured (Ma et al., 2025)
- SAE features from base models **retain effectiveness after instruction-tuning** (ICLR 2026 code generation paper)

**Proposed implementation:**
```
ExperimentSpec.yaml:
  interpretability:
    enabled: true
    method: "sae_diff"          # or "activation_patch", "circuit_trace"
    layers: [15, 16, 17, 18]    # Focus layers (late transformer blocks)
    probe_prompts: "interp_probes.yaml"  # Diagnostic prompts
    sae_model: "pretrained"     # or "train_fresh"
```

### 2.2 Evaluation Enhancement (EXTEND EXISTING)

**Where:** Extend `Evaluator/runner.py` with interpretability-aware validators.

**What it does:**
1. For each eval prompt, capture intermediate activations
2. Check if the model is using the "right" internal features (e.g., tool-selection features active when tool-calling is expected)
3. Add interp-based metrics alongside schema/behavior/judge validation

**Why:** A model can produce correct outputs for wrong reasons (shortcut learning). Activation analysis catches this.

**New validator:** `InterpretabilityValidator`
- Checks: Are tool-calling features active? Is the reasoning circuit engaged?
- Output: `interp_score` (0.0–1.0) based on feature activation alignment

### 2.3 Flywheel Feature Drift Detection (EXTEND EXISTING)

**Where:** Extend `shared/flywheel/cleaner.py` or add `shared/flywheel/interp_monitor.py`.

**What it does:**
1. Periodically run SAE analysis on inference logs
2. Track feature activation distributions over time
3. Alert when features drift (model forgetting tool-calling capabilities, developing shortcuts)

**Why:** The flywheel hot-swaps LoRA adapters. Feature drift detection catches degradation that loss metrics miss.

### 2.4 Guided LoRA Placement (NEW CAPABILITY)

**Where:** Extend training config to support mechanistically-guided LoRA rank/layer selection.

**What it does:**
1. Before training, run activation patching on eval failures
2. Identify which layers/heads are most causally important for the failing behavior
3. Concentrate LoRA rank on those layers (higher rank where it matters, lower elsewhere)

**Why:** Research shows SAE analysis can pinpoint the exact layers where LoRA should be applied (Feb 2026 medical VLM paper). Instead of uniform LoRA across all layers, focus compute where it matters.

**Research backing:** "Mechanistically Guided LoRA" (arXiv:2603.00148) — used SAEs to find that a key feature at layer 17 mediated the target behavior, then applied LoRA to layers 15–19 with consistency loss.

### 2.5 Activation Steering for Inference-Time Fixes (NEW CAPABILITY)

**Where:** New module `shared/steering/` or integrate with vLLM proxy.

**What it does:**
1. When eval identifies a specific failure mode (e.g., model refuses tool calls, hallucinates parameters)
2. Extract a steering vector that corrects the behavior
3. Apply at inference time via the vLLM proxy, without retraining

**Why:** Faster iteration than retraining. Conditional Activation Steering (CAST, ICLR 2025) only steers when the problematic pattern is detected.

**Practical example:** If the fine-tuned model occasionally fails to call tools when it should, extract a "tool-calling" direction from successful vs. failed examples and inject it at inference time.

### 2.6 SynthChat Quality Signal (EXTEND EXISTING)

**Where:** Extend `SynthChat/services/core/validation_service.py`.

**What it does:**
1. For synthetic training examples, check if the teacher model's response activates the expected features in a reference model
2. Flag synthetic examples where the internal representation doesn't match expectations (even if the output looks correct)

**Why:** Current SynthChat validation is output-level (schema, rubric, judge). Interp adds representation-level quality signal.

---

## 3. Recommended Implementation Roadmap

### Phase 1: Foundation (Low effort, high signal)
1. **Add SAELens + TransformerLens to requirements** (new `requirements-interp.txt`)
2. **Build `shared/interpretability/` module** with:
   - `activation_extractor.py` — Extract activations from base and fine-tuned models
   - `feature_diff.py` — Compare SAE feature activations before/after training
   - `report.py` — Generate human-readable interp reports
3. **Create diagnostic prompt set** (`configs/interp/diagnostic_prompts.yaml`) covering tool-calling, reasoning, and refusal scenarios
4. **Wire into experiment handler** as optional post-training stage

### Phase 2: Evaluation Integration (Medium effort)
5. **Add `InterpretabilityValidator`** to Evaluator
6. **Add feature drift monitoring** to flywheel cleaner
7. **Add interp metrics** to experiment tracking (RunRecord)

### Phase 3: Advanced (Higher effort, higher payoff)
8. **Mechanistically-guided LoRA placement**
9. **Activation steering via proxy**
10. **SynthChat representation-level validation**

---

## 4. Key Considerations

### Compute Requirements
- SAE training: Significant GPU time (but pre-trained SAEs available for Gemma, Llama, GPT-2)
- Activation extraction: ~2x inference cost per prompt (one forward pass with hooks)
- Circuit tracing: GPU-intensive but one-time per analysis

### Model Compatibility
Our pipeline primarily trains Mistral/Llama/Qwen models with Unsloth. Compatibility:
- **TransformerLens v3:** Supports Llama, Qwen, Mistral, Gemma ✓
- **SAELens:** Works with any HuggingFace model ✓
- **pyvene:** Works with any PyTorch model ✓
- **Circuit Tracer:** Currently Gemma-2, Llama-3.1, Qwen3 only ⚠️

### Goodhart's Law Warning
Research (Oct 2025 toxicity probe paper) warns: **when interpretability metrics become training targets, they may cease to be reliable.** We should use interp signals for *diagnostics and monitoring*, not as direct training loss terms. Keep interp as an independent observer, not an optimizer.

### Practical Starting Point
The lowest-friction entry is:
1. Install SAELens
2. Use pre-trained SAEs (Gemma Scope 2 or SAELens pretrained)
3. Run `encode()` on activations from eval prompts, before and after fine-tuning
4. Diff the feature activation vectors
5. Report which features changed most

This gives immediate insight with minimal engineering.

---

## 5. Key References

- **Open Problems in Mechanistic Interpretability** (Jan 2025) — 29 researchers, 18 orgs. Field roadmap.
- **Anthropic Circuit Tracing** (Mar 2025) — Cross-layer transcoders, attribution graphs.
- **Gemma Scope 2** (Dec 2025) — Largest open-source interp infrastructure.
- **InterLoRA** (ICML 2025) — Mechanistic interpretability guiding LoRA architecture.
- **Mechanistically Guided LoRA for Medical VLMs** (Feb 2026) — SAE-identified features guide LoRA placement.
- **ICLR 2026 Code Generation Interp** — Base model SAE features predict fine-tuned model behavior.
- **"Why Steering Works"** (Feb 2026) — Unifies weight fine-tuning, LoRA, and activation steering.
- **CAST** (ICLR 2025) — Conditional activation steering without classifiers.
- **NNsight 0.6** (Feb 2026) — AI agent-native interpretability with NDIF remote execution.
