---
title: "Essay 1 integration memo"
date: 2026-06-09
source_essay: 01-artificial-intellectual-humility.md
status: proposals-awaiting-user-signoff
tags: [integration, epistemic-humility, formalization-stack]
---

# What "Artificial Intellectual Humility" changes for the research program

The essay contributes a four-level **Formalization Stack** that is a better
organizing taxonomy than our flat five claim families, plus 13 papers from the
2025–2026 cluster our search agents missed (now in the library manifest,
flagged for ID verification on enrich).

## 1. Mapping the stack onto the meta-analysis

| Essay level | What it formalizes | Our claim families | Coverage in our corpus |
|---|---|---|---|
| L1: Confidence scores | "how sure am I?" | C1 (calibration vs alignment), parts of C5 | Strong (12+ papers) |
| L2: Structured ignorance (SIC, gap-naming) | "what am I missing?" | none — new axis | Thin: only abstention-as-refusal; gap-naming absent |
| L3: Distributional failure signatures | "what shape is my failure?" (third-person humility) | none — new axis | Absent |
| L4: Objective/reward uncertainty | "what should I even optimize?" | C4/C5 sycophancy mechanism | Indirect only |

**Proposal for the meta-analysis paper:** adopt the stack as the organizing
framework for the synthesis (with credit/coordination — it originates in the
user's essay series). The quantitative claim families become measurements
*within* levels, and the paper's discussion section gains a principled
account of why the literature clusters at L1: that's where the metrics are.
This also differentiates the meta-analysis from the existing honesty/abstention
surveys (2409.18786, 2407.18418), which are method-taxonomies, not
depth-taxonomies.

## 2. New evidence relevant to existing claim families

- **C1 sharpened:** three independent 2025–26 groups (2606.03969, 2508.15050,
  2506.18183) show *reasoning* (not just RLHF) fails to improve — or impairs —
  calibration; consistent with AbstentionBench's "reasoning FT degrades
  abstention 24%". Add rows to effects.csv once numbers are verified.
- **C1 caveat (estimator fragility):** token-prob, hidden-state, and
  consistency estimators of "internal confidence" diverge (2606.03969). Our
  experiment's ECE metric measures *one* estimator. The protocol should say so
  explicitly rather than claiming to measure "the model's confidence."
- **C4/C5 mechanism:** sycophancy is representationally encoded (2604.03147)
  and reward-uncertainty prevents the collapse that enables it (2606.03962) —
  upgrade the sycophancy section of the meta-analysis from phenomenology to
  mechanism.

## 3. Concrete protocol implications (PROTOCOL.md v0.2 candidates)

1. **Response-format factor (recommended, cheap):** our abstention target
   currently says "I don't know." The essay's L2 suggests a second format:
   *structured* abstention naming the gap ("I don't know X because it requires
   the intersection of A and B"). A 2×2 (method: SFT/KTO × format:
   plain-IDK/gap-naming) at 3B would test whether KTO trains *structured*
   humility as well as flat refusal — connecting the experiment to L2 and the
   essay series. Costs: 2 extra 3B runs + SynthChat templates for gap-naming
   targets (the Unknown-Unknown dataset construction recipe in 2606.08571 —
   domain-intersection fusion — is reproducible with SynthChat if the original
   data isn't released).
2. **Diversity-collapse metric (recommended, free):** 2606.08543 warns RL-style
   training can collapse output diversity — a trained model that always emits
   the same refusal template has learned the form of ignorance without the
   substance. Add distinct-n / self-BLEU over abstention responses to the eval
   suite. Zero training cost, one more analysis script.
3. **Steerable-Hard framing for OOD analysis (discussion-level):** when our
   trained models wrongly refuse a known question (over-refusal), L3 predicts
   some of those are "steerable" — recoverable with rephrasing. Sampling k
   rephrasings for refused-known questions quantifies how much over-refusal is
   shallow vs deep. Optional analysis, no extra training.
4. **NOT adopting:** L4 (reward uncertainty) is out of scope for this
   experiment — different machinery (reward ensembles), different paper.

## 4. Follow-ups queued

- Verify the 13 cited arXiv IDs on enrich (several are 2026 preprints).
- Check whether 2606.08571 released the Unknown-Unknown dataset (7,347
  cross-domain questions) — if public, it's a candidate OOD-far eval set; if
  not, SynthChat can reproduce the construction recipe.
- UaIT (aclanthology 2024.emnlp-main.1205) — add to manifest when ACL
  anthology is reachable (not arXiv-ID'd in the essay).
