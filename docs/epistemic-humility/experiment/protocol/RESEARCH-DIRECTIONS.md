---
title: "Research directions (supersedes PROTOCOL.md v0.1 framing)"
date: 2026-06-09
status: active-direction
decided_by: user
tags: [research-directions, epistemic-humility, alignment, coherent-humility]
---

# Research directions — user steer of 2026-06-09

## Process decision

**Draft the full meta-analysis BEFORE locking the experiment.** PROTOCOL.md
v0.1 (SFT-vs-KTO abstention) remains a candidate design, not the chosen one.
The meta-analysis findings — now reorganized around the Formalization Stack
(essays/INTEGRATION.md) plus the two additions below — pick the hypothesis.

## Two scope corrections

1. **GRPO must be in the design space.** Synaptic Tuner natively supports
   SFT/KTO/GRPO; the RL-for-abstention line (RLCR 2507.16806, RLKF 2403.18349,
   Abstain-R1, FiSCoRe 2510.24020, Reinforced Hesitation 2511.11500) is the
   GRPO-adjacent literature and is as underexplored for humility as KTO.
   A GRPO arm with a calibration-aware reward (RLCR-style Brier reward is the
   template) is the natural third method arm and uses repo-native tooling.

2. **The experiment's real target is COHERENT humility, not just behavioral
   humility.** User formulation: *"can we engineer a model to reliably express
   intellectual humility that aligns coherently across the tokens, the hidden
   space, and its stated confidence — and is that verifiable, i.e., through
   some mech interp?"*

## The coherent-humility frame, operationalized

Three measurement layers on the same model, same questions:

| Layer | Signal | Measurement |
|---|---|---|
| L-token | token-probability confidence | logprob-based ECE/AUROC (answer tokens) |
| L-hidden | internal knowledge state | linear probes on hidden states for "will answer correctly" (P(IK)-style) — trainable because we control the weights (LoRA, open 3B/7B) |
| L-stated | verbalized confidence / abstention behavior | truthful rate, over-refusal, verbalized-confidence ECE |

**Core research question (RQ-coherence):** does humility fine-tuning (SFT /
KTO / GRPO) produce *agreement* across the three layers, or only surface
behavior at L-stated while L-token and L-hidden are unchanged (or degraded)?
This is the essay's "performed awareness vs possessed awareness" question
(Socrates' final challenge) and Gani's estimator-fragility finding turned into
a falsifiable training experiment.

**Verifiability (RQ-verify):** can a probe trained on the BASE model's hidden
states still read the fine-tuned model's knowledge state? Three observable
outcomes, all informative:
- behavioral change + hidden-state change, layers agree -> humility was
  "internalized" (coherence increased)
- behavioral change, hidden states unchanged, probe transfers -> humility is a
  surface policy on top of an unchanged epistemic state ("performed")
- probe stops transferring -> fine-tuning moved the representation; localize
  with per-layer probes

Concrete mech-interp toolkit (all standard, all feasible on LoRA'd small
models): per-layer linear probes for correctness, probe-transfer analysis
base->tuned, logit-lens on abstention tokens, activation patching of the
"refusal direction" as stretch goal. Literature to integrate (search running):
P(IK) probes (2207.05221), Azaria & Mitchell internal-state lie detection,
CCS (Burns), ITI (Li et al.), representation engineering (Zou et al.),
refusal-direction work (Arditi et al.), plus 2026 faithful-calibration work
(2606.03969).

**Alignment framing (why this matters beyond benchmarks):** a model whose
stated uncertainty provably tracks its internal state is auditable; one whose
humility is performed is a polite liar with better manners. The
coherence-delta between training methods is the alignment-relevant quantity.

## Updated work order

1. Mech-interp/probing literature search -> raw-reports/06 (running)
2. Full meta-analysis draft (paper/draft-v0.md), organized by Formalization
   Stack, including the coherence/faithfulness axis as the L1.5 gap
3. THEN: hypothesis selection with user; merge PROTOCOL v0.1 + GRPO arm +
   coherence measurement into PROTOCOL v0.3 for sign-off
4. Experiment execution per HANDOFF.md §4
