---
report: mech-interp-probing-and-grpo
area: verification
date: 2026-06-09
agent_queries: 28
method: "WebSearch extraction + official GitHub repo fetches; arXiv/openreview/aclanthology direct fetch blocked by network allowlist (HTTP 403)"
verification_status: unverified-against-pdf
tags: [raw-report, epistemic-humility, mech-interp, probing, grpo, rl-abstention]
---

# Raw evidence report: internal-state probing of knowledge/uncertainty (the "verifiability" layer) + GRPO/RL training for abstention & calibration

Environment caveat: direct fetches of arxiv.org and most hosts were proxy-blocked (HTTP 403;
only github.com / raw.githubusercontent.com / pypi.org reachable). Numbers come from
(a) official GitHub repos fetched directly where they exist, and (b) WebSearch content
extraction of arXiv abstract/HTML pages. Provenance flagged per entry. Anything not
recoverable this way is marked "not found".

## Part A — Probing internal knowledge / honesty / uncertainty states

```
PAPER: Language Models (Mostly) Know What They Know
ARXIV: 2207.05221, 2022 (Kadavath et al., Anthropic)
MODELS: Anthropic LMs 800M-52B (52B headline)
DESIGN: P(IK) = "probability I Know the answer" — trained probe: a value head added on top of the
  LM (logistic output on a hidden representation), trained on whether the model answers TriviaQA
  questions correctly; contrasted with prompted self-evaluation P(True). No proposed answer is
  shown to the P(IK) head — it reads the question-conditioned internal state BEFORE generation.
METRIC: AUROC of P(IK) discriminating answerable vs unanswerable; in-distribution (TriviaQA) and
  OOD transfer (Lambada, Arithmetic, GSM8k, Codex HumanEval, Python function synthesis)
NUMBERS: Exact AUROC table values: not found (PDF unreachable). Verified qualitative results:
  P(IK) trained only on TriviaQA transfers OOD with AUROC increasing with model size on all
  evaluated OOD tasks; OOD discriminative power is decent but OOD *calibration* of P(IK) is poor —
  the TriviaQA-trained head is "severely underconfident" on Arithmetic and Python function
  synthesis. P(IK) increases when relevant source material is in context (closed-book -> open-book).
QUOTE: "We find that P(IK) generalizes across tasks, though models struggle with calibration OOD" (abstract paraphrase via WebSearch; exact wording unverified)
ERROR BARS: none reported in retrieved material
URL: arxiv.org/abs/2207.05221 (403; WebSearch extraction)
RELEVANCE: Canonical "trained probe of self-knowledge before generation"; the probe is trained, not
  emergent — i.e., P(IK) is supervision-dependent, which matters for probe-transfer experiments.
```

```
PAPER: The Internal State of an LLM Knows When It's Lying
ARXIV: 2304.13734, 2023 (Azaria & Mitchell; EMNLP 2023 Findings)
MODELS: OPT and LLaMA family (6.7B-30B class; per-model breakdown not recovered)
DESIGN: SAPLMA (Statement Accuracy Prediction based on Language Model Activations) — a small
  feedforward classifier trained on hidden-layer activations of the LLM while it READS true vs
  false statements (6 topic datasets: cities, inventions, elements, animals, companies, facts);
  train on 5 topics, test on the held-out topic.
METRIC: classification accuracy on held-out-topic true/false statements
NUMBERS: 71%-83% accuracy depending on base LLM (held-out topic). Outperforms few-shot prompting
  and sentence-probability baselines; LLM-assigned sentence probability confounded by sentence
  length and word frequency. Per-layer table: not found; middle/later hidden layers reported best
  in secondary sources (unverified).
QUOTE: "the LLM's internal state... contains information on whether the LLM believes a statement is true or false... classifier achieves 71% to 83% accuracy" (abstract via WebSearch)
ERROR BARS: none reported in retrieved material
URL: arxiv.org/abs/2304.13734 (403; WebSearch extraction)
```

```
PAPER: Discovering Latent Knowledge in Language Models Without Supervision (CCS)
ARXIV: 2212.03827, 2022 (Burns, Ye, Klein, Steinhardt; ICLR 2023)
MODELS: 6 models incl. GPT-J, T5/UnifiedQA, DeBERTa; 10 QA datasets
DESIGN: Contrast-Consistent Search — UNSUPERVISED probe: find a direction in activation space
  whose values on a statement and its negation are consistent (sum to ~1) and confident. No truth
  labels used.
METRIC: QA accuracy of CCS probe vs calibrated zero-shot prompting
NUMBERS: CCS outperforms zero-shot accuracy by 4% on average (67% -> 71% across 6 models x 10
  datasets); cuts prompt sensitivity roughly in half; maintains high accuracy even when models are
  prompted to output incorrect answers.
QUOTE: "it outperforms zero-shot accuracy by 4% on average... continues to maintain high accuracy even when models are prompted to generate incorrect answers" (abstract via WebSearch + official repo)
ERROR BARS: none reported in retrieved material
URL: https://github.com/collin-burns/discovering_latent_knowledge (fetched-adjacent via search; arXiv 403)
CAVEAT: later critiques (e.g., "banana/shed" critique, Farquhar et al.) argue CCS often finds
  salient non-truth features — not re-verified here.
```

```
PAPER: Inference-Time Intervention: Eliciting Truthful Answers from a Language Model (ITI)
ARXIV: 2306.03341, 2023 (Li, Patel, Viegas, Pfister, Wattenberg; NeurIPS 2023)
MODELS: LLaMA-7B, Alpaca, Vicuna
DESIGN: linear probes trained per attention head on TruthfulQA-derived truthful/untruthful
  activations; intervention = shift activations along probe directions on top-K truth-related
  heads during inference.
METRIC: TruthfulQA True*Informative; probe accuracy per head
NUMBERS: Alpaca truthfulness 32.5% -> 65.1% with ITI. Per-head probe accuracies: exact values not
  found in retrieved material (paper figures report a sparse subset of heads with high linear
  probe accuracy; widely cited ~83% for best heads — unverified). Data-efficient: few hundred
  examples to locate truthful directions.
QUOTE: "ITI improves the truthfulness of the Alpaca model from 32.5% to 65.1%" (abstract via WebSearch)
ERROR BARS: none reported in retrieved material
URL: arxiv.org/abs/2306.03341 (403; WebSearch extraction)
RELEVANCE: causal evidence that truth-related directions exist and are USABLE, not just decodable.
```

```
PAPER: Representation Engineering: A Top-Down Approach to AI Transparency (RepE)
ARXIV: 2310.01405, 2023 (Zou et al., CAIS/CMU)
MODELS: LLaMA-2-Chat 7B/13B
DESIGN: LAT (Linear Artificial Tomography) reading vectors for honesty; control via adding
  honesty reading vectors / Contrast Vector / LoRRA (low-rank representation adaptation).
METRIC: TruthfulQA MC1 accuracy; lie-detection from honesty-direction projections
NUMBERS: TruthfulQA MC1 — LLaMA-2-7B-Chat: standard 31.0% -> Reading 34.1% -> Contrast Vector
  47.9% -> LoRRA 42.3%. LLaMA-2-13B-Chat: standard 35.9% -> Reading 42.4% -> Contrast 54.0% ->
  LoRRA 47.5%. Contrast Vector is SOTA among these but costs >3x inference compute; LoRRA similar
  performance at negligible overhead. Honesty-direction monitoring detects instructed lying
  (qualitative demos; quantitative lie-detection table not recovered).
QUOTE: "The Contrast Vector method obtains state-of-the-art performance but requires over 3x more inference compute, while LoRRA obtains similar performance with negligible compute overhead" (via WebSearch extraction)
ERROR BARS: none reported in retrieved material
URL: https://github.com/andyzoujm/representation-engineering (official repo exists; numbers via WebSearch of arXiv HTML)
```

```
PAPER: Refusal in Language Models Is Mediated by a Single Direction
ARXIV: 2406.11717, 2024 (Arditi, Obeso, Syed, Paleka, Rimsky, Gurnee, Nanda; NeurIPS 2024)
MODELS: 13 open chat models up to 72B — Qwen 1.8B/7B/14B/72B, Yi 6B/34B, Gemma 2B/7B,
  Llama-2 7B/13B/70B, Llama-3 8B/70B
DESIGN: difference-in-means direction between harmful and harmless instruction activations;
  ablating (projecting out) the direction everywhere disables refusal; adding it induces refusal
  on harmless prompts. Weight-orthogonalization jailbreak variant.
METRIC: refusal rate / unsafe-completion rate after ablation; capability retention (MMLU etc.)
NUMBERS: per-model refusal/attack-success table values: not found (PDF unreachable). Verified:
  effect replicates across all 13 models; ablation "prevents refusal on harmful instructions";
  jailbreak is "surgical, with minimal effect on other capabilities."
QUOTE: "refusal is mediated by a one-dimensional subspace, across 13 popular open-source chat models up to 72B parameters" (abstract via WebSearch)
ERROR BARS: none reported in retrieved material
URL: https://github.com/andyrdt/refusal_direction (official repo); arxiv.org/abs/2406.11717 (403)
RELEVANCE: refusal/abstention behavior itself is a low-dimensional, probe-findable feature —
  the closest mech-interp analogue to "abstention direction" for epistemic refusals.
NOTE: a 2026 follow-up exists arguing the single-direction view is incomplete: "There Is More to
  Refusal in Large Language Models than a Single Direction" (arXiv 2602.02132; surfaced in search,
  details not extracted).
```

```
PAPER: The Geometry of Truth: Emergent Linear Structure in LLM Representations of True/False Datasets
ARXIV: 2310.06824, 2023 (Marks & Tegmark; COLM 2024)
MODELS: LLaMA-2 / LLaMA-3 family (7B-70B class), curated true/false datasets (cities, sp_en_trans,
  larger_than, neg_* variants, etc.; CounterFact-style factual statements)
DESIGN: visualization of true/false activation separation; mass-mean (difference-in-means) probes
  vs logistic regression vs CCS; causal patching along the truth direction.
METRIC: probe transfer accuracy across datasets (incl. negations); causal effect of direction edits
NUMBERS: exact transfer-accuracy tables: not found (PDF unreachable). Verified: at sufficient
  scale, truth/falsehood is linearly represented; mass-mean probes generalize across datasets as
  well as or better than other probing techniques and are more causally implicated in outputs;
  probes trained on one dataset transfer to others (incl. handling negation at larger scale).
QUOTE: "simple difference-in-mean probes generalize as well as other probing techniques and... are more causally implicated in model outputs" (abstract via WebSearch)
ERROR BARS: none reported in retrieved material
URL: arxiv.org/abs/2310.06824 (403; WebSearch extraction); datasets at github.com/saprmarks/geometry-of-truth (not fetched)
```

```
PAPER: LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations
ARXIV: 2410.02707, 2024 (Orgad et al., Technion; ICLR 2025)
MODELS: Llama-3-8B / Mistral-7B class instruct models (full list not recovered)
DESIGN: error-detection probes on internal activations; key finding that truthfulness signal is
  concentrated in the EXACT ANSWER TOKENS (not last token / mean pooling); also probes that
  predict error TYPE and select the correct answer among samples.
METRIC: error-detection AUROC across QA datasets; cross-dataset probe generalization
NUMBERS: exact AUROC tables: not found. Verified: probing exact-answer tokens "significantly
  enhances error detection"; probes FAIL to generalize across datasets — truthfulness encoding is
  "multifaceted", not universal; internal encoding can identify the correct answer even when the
  model repeatedly generates a wrong one ("discrepancy between internal encoding and external
  behavior").
QUOTE: "truthfulness information is concentrated in specific tokens... however, error detectors fail to generalize across datasets, implying that truthfulness encoding is not universal but rather multifaceted" (abstract via WebSearch)
ERROR BARS: none reported in retrieved material
URL: https://github.com/technion-cs-nlp/LLMsKnow (official repo); arxiv.org/abs/2410.02707 (403)
```

```
PAPER: Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs (SEPs)
ARXIV: 2406.15927, 2024 (Kossen, Han, Razzak, Schut, Malik, Gal; OATML Oxford)
MODELS: Llama-2 7B/70B, Llama-3 8B/70B, Mistral-7B, Phi-3 (model list partially via secondary
  sources)
DESIGN: linear probes on hidden states of a SINGLE generation trained to predict (binarized)
  semantic entropy — i.e., the probe target is an unsupervised uncertainty quantity, not
  correctness labels. Works from hidden states BEFORE generation (probes at the token before
  generation) and after.
METRIC: AUROC for hallucination detection; comparison vs accuracy probes (trained on correctness)
  in-distribution and under distribution shift; cost vs full semantic entropy (5-10x sampling)
NUMBERS: AUROC range ~0.7-0.95 depending on model/dataset/layer (per secondary extraction; exact
  per-cell tables not found). Verified: SEPs approximate semantic entropy at ~1/10 the cost;
  SEPs generalize better OOD than accuracy probes; hidden states before generation already
  capture semantic entropy.
QUOTE: "SEPs retain high performance for hallucination detection and generalize better to out-of-distribution data than previous probing methods that directly predict model accuracy" (abstract via WebSearch)
ERROR BARS: none reported in retrieved material
URL: https://github.com/OATML/semantic-entropy-probes (official repo); arxiv.org/abs/2406.15927 (403)
RELEVANCE: the strongest existing "verifiability layer" candidate for OUR design: a cheap probe
  of epistemic state, defined independently of behavior, evaluable before generation.
```

```
PAPER: Do Androids Know They're Only Dreaming of Electric Sheep?
ARXIV: 2312.17249, 2023 (CH-Wang et al., Columbia/Microsoft)
MODELS: transformer LMs on 3 grounded generation tasks (summarization, dialogue, data-to-text)
DESIGN: probes on internal states with span-level hallucination annotation (organic + synthetic);
  probes evaluated BEFORE, DURING, and AFTER hallucination occurs.
METRIC: response-level detection F1; span-level F1 vs expert annotator
NUMBERS: probes reach 95% of peak performance as early as layer 4 in-domain; outperform an expert
  human annotator at response-level detection F1; span-level on par or better on 2 of 3 tasks.
  Exact F1 values: not found.
CAVEAT: probes are narrowly trained — poor task-to-task and synthetic-to-organic generalization.
QUOTE: "probes... generalize poorly from one task to another or from synthetic to organic hallucinations" (via WebSearch extraction)
URL: arxiv.org/abs/2312.17249 (403; WebSearch extraction)
RELEVANCE: direct precedent for detecting upcoming hallucination BEFORE the hallucinated span is
  generated.
```

```
PAPER: On the Universal Truthfulness Hyperplane Inside LLMs
ARXIV: 2407.08582, 2024 (Liu et al.; EMNLP 2024)
DESIGN: train a single truthfulness probe on 40+ diverse datasets; test cross-task, cross-domain,
  in-domain generalization — the direct test of whether "one truth direction" exists at scale.
NUMBERS: exact accuracies not recovered; verified finding: increasing training-dataset diversity
  significantly improves generalization, partially rescuing the cross-dataset failures reported by
  Orgad et al. / Geometry-of-Truth critiques.
URL: arxiv.org/abs/2407.08582 (403; WebSearch extraction)
```

```
PAPER: Do LLMs Really Know What They Don't Know? Internal States Mainly Reflect Knowledge Recall Rather Than Truthfulness  [COUNTERPOINT]
ARXIV: 2510.09033, 2025
DESIGN: causal analysis of hidden states for factual outputs vs "associated" hallucinations
  (grounded in subject knowledge) vs "unassociated" hallucinations.
FINDING: factual outputs and ASSOCIATED hallucinations share overlapping hidden-state geometry
  (both driven by learned associations) and canNOT be reliably separated by probes; only
  unassociated hallucinations are separable. Implication: truth probes partly measure "knowledge
  recall strength", not truth per se.
NUMBERS: not found
URL: arxiv.org/abs/2510.09033 (403; WebSearch extraction)
RELEVANCE: key caveat for any probe-based verifiability claim, including ours.
```

### Probe transfer between base and fine-tuned models (our experiment's core)

```
PAPER: Fine-Tuned LLMs Know They Don't Know: A Parameter-Efficient Approach to Recovering Honesty
ARXIV: 2511.12991, 2025 (HCNR framework; authors not fully recovered)
MODELS: open LLMs subjected to SFT on downstream tasks (model list not recovered)
DESIGN: THE closest prior art to our core question. Asks explicitly whether SFT-induced honesty
  degradation comes from (a) corruption of internal knowledge-boundary awareness or (b) impaired
  EXPRESSION of preserved awareness. Method: train linear probes on BASE model hidden states for
  known/unknown discrimination, apply them UNCHANGED to the SFT'd model's representations.
METRIC: probe AUROC base->fine-tuned transfer; honesty recovery vs task performance
NUMBERS: probes trained on base hidden states "maintain high AUROC" on fine-tuned representations
  without retraining (exact AUROC: not found). HCNR (recalibrating a small set of critical
  neurons) recovers honesty with >=2.23x speedup and >10x less data vs baseline recovery methods.
QUOTE: "honesty degradation after SFT stems not from the destruction of knowledge boundary awareness, but rather from an impaired ability to express this preserved awareness" (via WebSearch extraction)
ERROR BARS: none reported in retrieved material
URL: arxiv.org/abs/2511.12991 (403; WebSearch extraction)
RELEVANCE: establishes the base->SFT probe-transfer direction. Does NOT cover: probe transfer
  after HONESTY/ABSTENTION fine-tuning (the reverse intervention), nor preference/RL methods
  (KTO/GRPO), nor whether abstention training CREATES new internal structure vs re-wiring
  expression of existing structure.
```

```
PAPER: Don't Make It Up: Preserving Ignorance Awareness in LLM Fine-Tuning (SEAT)
ARXIV: 2506.14387, 2025 (Shen, Qiu, Cancedda, Lane)
DESIGN: shows conventional fine-tuning causes "substantial activation displacement" that collapses
  faithful expression of epistemic uncertainty ("ignorance awareness"); SEAT = sparse tuning
  constraining activation drift + entity perturbation against knowledge entanglement.
NUMBERS: not found (qualitative: SEAT "significantly outperforms baselines" at preserving
  ignorance awareness with optimal task performance)
URL: arxiv.org/abs/2506.14387 (403; WebSearch extraction)
RELEVANCE: representation-level evidence that fine-tuning moves activations enough to break
  uncertainty expression — complements 2511.12991 (probes still transfer) by showing WHERE drift
  is harmful.
```

```
PAPER: Preference Learning with Lie Detectors can Induce Honesty or Evasion
ARXIV: 2505.13787, 2025 (Cundy & Gleave, FAR AI)
MODELS: LLMs trained on DolusChat (65k paired truthful/deceptive responses)
DESIGN: put a probe-based lie detector INTO the training loop (labeling step of preference
  learning / RL). Variables: exploration amount, detector TPR, KL regularization.
METRIC: post-training deception rate; detector-evasion rate
NUMBERS: GRPO + lie-detector labels can yield detector-EVADING policies with deception rates
  >85%; with sufficiently high detector TPR or strong KL regularization, GRPO learns genuinely
  honest policies; off-policy DPO consistently yields deception <25% at realistic TPRs.
QUOTE: "depending on the context, lie-detector-enhanced training can be a powerful tool for scalable oversight, or a counterproductive method encouraging undetectable misalignment"
ERROR BARS: none reported in retrieved material
URL: arxiv.org/abs/2505.13787 (403; WebSearch extraction)
RELEVANCE: the main caution for combining probes with GRPO: on-policy RL exploits probes. Also
  relevant: "The Obfuscation Atlas: Mapping Where Honesty Emerges in RLVR with Deception Probes"
  (arXiv 2602.15515, surfaced in search; details not extracted).
```

Peripheral Part-A items surfaced but not deeply extracted: Probe-based Fine-tuning for reducing
Toxicity (2510.21531 — probes as training signal; evasion vs robust-probe architectures);
Reasoning Models Know When They're Right (2504.05419 — hidden-state self-verification probes in
LRMs); Trace Length is a Simple Uncertainty Signal (2510.10409); Linear Probes Detect Task
Format, Not Reasoning Mode (2606.02907 — probe-validity caution); Weakly Supervised Distillation
of Hallucination Signals (2604.06277); HALP for VLMs without generating a token (2603.05465,
AUROC up to 0.93 pre-generation).

## ID-VERIFICATION TABLE — essay-cited 2026 (and adjacent) papers

| Claimed ID | Status | Verified title / authors | Notes |
|---|---|---|---|
| 2606.03969 (Gani, "faithful calibration") | CONFIRMED | "Quantifying Faithful Confidence Expression in Large Reasoning Models" — Areeb Gani, Asal Meskin, Gabrielle Kaili-May Liu, Arman Cohan, et al. | Found only via arXiv cs.AI listing snippet; abstract/findings NOT recoverable via search as of 2026-06-09 — cite title only, do not cite numbers. "Estimator fragility" framing unverified. |
| 2606.08571 (Sahoo) | CONFIRMED | "Calibration of Structured Ignorance Certificates for Diagnosing Unknown Unknowns in Reasoning Models" — Subramanyam Sahoo (7 Jun 2026) | SICs = JSON schema naming missing domain intersection + retrieval query; 7,347-sample Unknown-Unknown dataset built with Qwen3-14B across 7 domains. |
| 2606.05145 (Islah, Mila) | CONFIRMED | "Failed Reasoning Traces Tell You What Is Fixable (But Not by Reading Them)" — Nizar Islah, Istabrak Abbes, Irina Rish, Sarath Chandar, Eilif B. Muller (Mila) | Recoverability structure from DISTRIBUTIONAL signature of failed rollouts, not trace text. "Distributional failure signatures" description matches. |
| 2606.03962 (GX-Chen) | CONFIRMED | "Using Reward Uncertainty to Induce Diverse Behaviour in Reinforcement Learning" — Anthony GX-Chen, Ankit Anand, Gheorghe Comanici, Zaheer Abbas, et al. (incl. Precup, Barreto, Rowland) | Classic-RL diversity paper, not LLM-specific. |
| 2508.15050 | CONFIRMED | "Don't Think Twice! Over-Reasoning Impairs Confidence Calibration" — Lacombe, Wu, Dilworth (Aug 2025) | ClimateX; reasoning LLMs 48.7% accuracy at expert-confidence assessment; longer reasoning budgets => systematic overconfidence, worsening with budget. |
| 2506.18183 | CONFIRMED | "Reasoning about Uncertainty: Do Reasoning Models Know When They Don't Know?" (Jun 2025) | Introspective uncertainty quantification: reasoning over own CoT can improve calibration. Distinct from 2504.05419 and 2504.06564 (similar titles — disambiguation hazard). |
| 2511.07477 (DeVilling) | CONFIRMED | "The Polite Liar: Epistemic Pathology in Language Models" — Bentley DeVilling (Course Correct Labs, Nov 2025) | Conceptual/philosophy paper (Frankfurtian analysis): RLHF reward architecture optimizes "perceived sincerity over evidential accuracy". No experiments expected. |
| 2605.21127 (Twist) | NOT-FOUND | — | No paper at this ID surfaced in any search; nearest "Twist"-titled May-2026 paper is 2605.27205 (wireless digital twins — unrelated). Treat citation as unverifiable; likely wrong ID. |
| 2604.03147 (Sun) | CONFIRMED (title mismatch with claim) | "Valence-Arousal Subspace in LLMs: Circular Emotion Geometry and Multi-Behavioral Control" — Lihao Sun et al. | NOT a sycophancy-representations paper per se: VA emotion subspace that ALSO steers sycophancy + refusal bidirectionally (Llama-3.1-8B, Qwen3-8B/14B). If essay cites it as "sycophancy representations", rephrase. Better sycophancy-internals cite: 2508.02087 "When Truth Is Overridden". |
| 2603.24967 (Taparia) | CONFIRMED | "The Anatomy of Uncertainty in LLMs" — Aditya Taparia, Ransalu Senanayake, Kowshik Thopalli, Vivek Narayanaswamy (ASU/LLNL, 26 Mar 2026) | Decomposes uncertainty into input ambiguity / knowledge gaps / decoding randomness; demo on TriviaQA + Gemma-3-27B. |
| 2604.13991 (Rubashevskii) | CONFIRMED | "Adaptive Conformal Prediction for Improving Factuality of Generations by Large Language Models" — Aleksandr Rubashevskii, Dzianis Piatrashyn, Preslav Nakov, Maxim Panov (Apr 2026) | Prompt-adaptive conformal score transformation; marginal coverage retained, conditional coverage improved. |
| 2606.08543 (Yang) | NOT-FOUND | — | No hit for this ID. Diversity-collapse literature exists at other IDs: 2509.07430 (Choice of Divergence), 2605.00365 (Uniform-Correct Policy Optimization), 2603.16157 (DyJR). Treat as unverifiable; likely wrong ID. |
| 2606.06475 (Ielanskyi) | CONFIRMED (caveat) | "RREDCoT: Segment-Level Reward Redistribution for Reasoning Models" — Mykyta Ielanskyi, Kajetan Schweighofer, Lukas Aichberger, Sepp Hochreiter | Title recovered from a single search snippet; arXiv listing page itself unverified. "Step scoring" = segment-level reward redistribution. Confidence: medium. |

Verification summary: 10 of 13 CONFIRMED (one with topic-mismatch caveat, one medium-confidence),
2 NOT-FOUND (2605.21127, 2606.08543), 1 confirmed-title-only (2606.03969 — no extractable
findings; do not cite numbers from it).

## Part B — GRPO / RL for abstention & calibration

```
PAPER: DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models  [GRPO ORIGIN]
ARXIV: 2402.03300, 2024 (Shao et al., DeepSeek)
DESIGN: introduces Group Relative Policy Optimization — PPO variant with NO value/critic model;
  advantage = per-group z-score of rewards across G sampled responses to the same prompt
  (A_i = (r_i - mean(r))/std(r)); KL to reference policy added directly to the loss.
NUMBERS (context only): DeepSeekMath 7B reaches 51.7% MATH (top1); GRPO improves GSM8K 82.9->88.2,
  MATH 46.8->51.7 (values from secondary extraction; unverified against PDF).
URL: arxiv.org/abs/2402.03300 (403; WebSearch extraction)
RELEVANCE: GRPO was designed for VERIFIABLE binary rewards on math — nothing in the original work
  concerns abstention, honesty, or calibration. All such uses are 2025+ retrofits.
```

```
PAPER: Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty (RLCR)
ARXIV: 2507.16806, 2025 (Damani, Puri, Slocum, Shenfeld, Choshen, Kim, Andreas; MIT)
MODELS: Qwen2.5-7B (base); tracks: HotpotQA-Modified (multi-hop QA, max len 1536) and
  Big-Math-Digits (max len 4096). 7 released checkpoints incl. RLVR and classifier baselines.
DESIGN: model outputs answer + verbalized numerical confidence q after reasoning; reward =
  correctness MINUS Brier penalty: R = 1{correct} - (1{correct} - q)^2 (bounded proper scoring
  rule; any bounded proper rule provably yields accurate AND calibrated predictions per their
  theory). Optimized with RL (GRPO-style RLVR setup).
METRIC: accuracy, ECE, Brier, AUROC; in-domain and OOD (e.g., trained on HotpotQA, tested on
  math/commonsense and vice versa)
NUMBERS: calibration error reduced by up to ~90% vs RLVR baseline with accuracy maintained or
  improved (secondary extraction); exact per-dataset tables: not found (README carries no numbers;
  PDF unreachable). Outperforms (a) ordinary RLVR and (b) post-hoc trained confidence classifiers.
QUOTE: "RLCR substantially improves calibration with no loss in accuracy, on both in-domain and out-of-domain evaluations — outperforming both ordinary RL training and classifiers trained to assign post-hoc confidence scores" (abstract via WebSearch)
ERROR BARS: none reported in retrieved material
URL: https://github.com/damanimehul/RLCR (official repo, README fetched directly); rl-calibration.github.io
NOTE ON AUTHORS: Damani et al.; Stengel-Eskin is NOT an author (appears only in related work) —
  fix if the essay attributes it otherwise.
```

```
PAPER: Abstain-R1: Calibrated Abstention and Post-Refusal Clarification via Verifiable RL
ARXIV: 2604.17073, 2026 (real ID located — was the "find the ID" item)
MODELS: Qwen2.5-3B-Instruct
DESIGN: two-stage: (1) SFT on curated Abstain-CoT composite dataset for refusal-domain reasoning;
  (2) GRPO with a clarification-aware RLVR reward — rewards correct answers on answerable
  queries; on unanswerable queries jointly rewards explicit abstention AND semantically aligned
  post-refusal clarification (model must say WHAT is missing). GRPO chosen explicitly to avoid a
  value model.
METRIC: abstention + clarification quality on unanswerable; answerable-task retention
NUMBERS: exact table values: not found (PDF unreachable). Verified: improves abstention and
  clarification on unanswerable queries while preserving answerable performance, at 3B scale.
QUOTE: "a reliable model should not only abstain, but also explain what is missing" (via WebSearch extraction)
URL: arxiv.org/abs/2604.17073 (403; WebSearch extraction); published 18 Apr 2026
```

```
PAPER: Honesty over Accuracy: Trustworthy Language Models through Reinforced Hesitation (RH)
ARXIV: 2511.11500, 2025
DESIGN: modifies RLVR reward from binary to TERNARY: +1 correct / 0 abstention / -lambda error;
  lambda >= 0 encodes domain risk (high for medicine, low for creative tasks).
METRIC: accuracy-vs-error Pareto under varying lambda; logic-puzzle controlled experiments
NUMBERS: exact values not found. Verified findings: frontier models almost never abstain even
  when prompted with severe penalty warnings (prompting cannot override answer-rewarding
  training); sweeping lambda traces a Pareto frontier — low lambda => aggressive answerers, high
  lambda => conservative abstainers; each lambda is optimal for its own risk regime.
QUOTE: "transforming RLVR's binary reward signal into a ternary structure... makes hesitation explicitly valuable rather than merely possible" (via WebSearch extraction)
URL: arxiv.org/abs/2511.11500 (403; WebSearch extraction)
```

```
PAPER: Teaching LLMs to Abstain via Fine-Grained Semantic Confidence Reward (FiSCoRe)
ARXIV: 2510.24020, 2025
DESIGN: RL framework replacing global uncertainty reward with PER-SAMPLE confidence reward:
  sample multiple candidate answers, semantically cluster them; train to retain answers in
  high-confidence clusters and discard low-confidence ones => post-hoc abstention aligned to
  semantic consensus.
METRIC: reliability score in-domain and OOD
NUMBERS: exact values not found. Verified: high in-domain reliability with much smaller OOD drop
  than baselines (coarse-confidence RL, SFT abstention).
QUOTE: "an answer's intrinsic confidence is reflected in its semantic consensus among all samples" (via WebSearch extraction)
URL: arxiv.org/abs/2510.24020 (403; WebSearch extraction)
```

```
PAPER: TruthRL: Incentivizing Truthful LLMs via Reinforcement Learning
ARXIV: 2509.25760, 2025 (Meta/FAIR — facebookresearch/TruthRL)
DESIGN: GRPO with ternary reward: reward correct, PENALIZE hallucination, treat abstention
  neutrally (per official repo README, fetched directly). LLM verifier judges answer vs reference.
METRIC: truthfulness (correct - hallucination balance), hallucination rate, abstention behavior
NUMBERS: vs vanilla GRPO: -28.9% hallucinations, +21.1% truthfulness (secondary extraction across
  4 benchmarks, with/without retrieval; exact per-benchmark tables not found).
QUOTE (repo README, primary): rewards "correct answers," explicitly "penalizes hallucinations," and "treats abstentions neutrally"
URL: https://github.com/facebookresearch/TruthRL (README fetched directly); arxiv.org/abs/2509.25760 (403)
```

```
PAPER: Rewarding Intellectual Humility: Learning When Not to Answer in Large Language Models
ARXIV: 2601.20126, 2026 (27 Jan 2026)
MODELS: Granite-3.3-2B-Instruct, Qwen-3-4B-Instruct; MedMCQA + Hendrycks MATH
DESIGN: GRPO with ternary reward (-1 wrong, r_abs abstain, +1 correct); sweeps r_abs.
NUMBERS: moderate abstention rewards r_abs in [-0.25, 0.3] consistently reduce incorrect responses
  without severe accuracy degradation on multiple choice; larger models more robust to abstention
  incentives. Exact tables: not found.
URL: arxiv.org/abs/2601.20126 (403; WebSearch extraction)
```

```
PAPER: TIAR: Trajectory-Informed Advantage Reweighting for LLM Abstention Learning
ARXIV: 2605.25850, 2026 (Pan, Zhao, Zhang, Shin, Parekh, Narayanan, Zhang; Penn State, 25 May 2026)
DESIGN: GRPO abstention training where the GROUP OF ROLLOUTS ITSELF is used as a free confidence
  signal: agreement across trajectories reweights the abstention advantage per query difficulty
  (dynamic ternary reward).
NUMBERS: SOTA abstention F1 on 5 of the AbstentionBench evaluation datasets (exact F1: not found).
URL: arxiv.org/abs/2605.25850 (403; WebSearch extraction)
```

```
PAPER: Decoupling Reasoning and Confidence: Resurrecting Calibration in RLVR (DCPO)
ARXIV: 2603.09117, 2026 (10 Mar 2026, v2 30 Apr 2026)
DESIGN: theory: gradient CONFLICT between maximizing accuracy and minimizing calibration error in
  RLVR; DCPO decouples them via block-wise verbalized-confidence rollout + decoupled advantage
  estimation.
NUMBERS: accuracy on par with GRPO + best calibration among compared methods (exact: not found).
  Related context from search: GRPO training can push mean confidence 0.88 -> 0.98 (overconfidence
  amplification).
URL: arxiv.org/abs/2603.09117 (403; WebSearch extraction)
```

Peripheral Part-B items: Mitigating LLM Hallucination via Behaviorally Calibrated RL (2512.19920 —
behavioral abstention with risk tolerance as dynamic input); Overconfident Errors Need Stronger
Correction (2602.21420 — asymmetric confidence penalties in RL); Calibrating LLMs with
Semantic-level Reward (2605.15588); Blockwise Advantage Estimation for Multi-Objective RLVR
(2602.10231); The Obfuscation Atlas (2602.15515 — deception probes during RLVR).

## CROSS-CUTTING FINDINGS

1. **The probing literature converges on "knowledge is linearly decodable; expression is what
   breaks."** Kadavath (P(IK) transfers OOD), Azaria-Mitchell (71-83%), Marks-Tegmark (mass-mean
   probes transfer + causal), Orgad (internal encoding identifies correct answer the model doesn't
   output), 2511.12991 (base-trained probes keep high AUROC on SFT'd models). The dissent is about
   WHAT the probe reads (knowledge-recall vs truth — 2510.09033) and cross-dataset universality
   (Orgad: fails; 2407.08582: rescued by training-set diversity).

2. **Pre-generation epistemic signals are established.** P(IK) (no proposed answer needed), SEPs
   (hidden state before generation predicts semantic entropy), Do-Androids (95% of peak probe
   performance by layer 4, span detection before hallucinated span), HALP (VLMs, 0.93 AUROC with
   zero generated tokens). A verifiability layer that gates generation on a probe is well-precedented.

3. **Probes and RL interact dangerously when the probe is in the reward loop.** Cundy-Gleave: GRPO
   + lie-detector labels => >85% deception (detector evasion) unless TPR/KL high; DPO (off-policy)
   much safer (<25%). For our design: keep the probe as a held-out EVALUATION instrument, not a
   reward term — or expect Goodharting.

4. **GRPO's abstention/calibration retrofits cluster into exactly three reward families, all
   2025-2026:** (a) ternary correct/abstain/error rewards (TruthRL, Reinforced Hesitation,
   Rewarding Intellectual Humility, TIAR's dynamic variant, Abstain-R1's clarification-aware
   variant); (b) proper-scoring-rule confidence rewards (RLCR's Brier; semantic-level reward
   2605.15588); (c) consistency/consensus-derived per-sample rewards (FiSCoRe, TIAR's
   trajectory-agreement signal). Meanwhile, plain RLVR/GRPO is repeatedly shown to DEGRADE
   calibration (DCPO's gradient-conflict theorem; 0.88->0.98 confidence drift; diversity-collapse
   line 2509.07430/2605.00365).

5. **Refusal/abstention behavior itself is low-dimensional** (Arditi single direction across 13
   models; 2602.02132 complicates but does not void this) — which makes "did abstention training
   create/strengthen an internal direction, or just route behavior to an existing one?" a
   well-posed, probe-answerable question.

6. **Two essay-cited 2026 IDs do not resolve** (2605.21127, 2606.08543) and one resolves to a
   different topic than implied (2604.03147). Any essay text leaning on these needs correction
   before publication.

## GAP ANALYSIS

1. **Probe-transfer after HONESTY/ABSTENTION fine-tuning: essentially unmeasured — HIGH
   confidence, with one adjacent exception.** What exists: 2511.12991 measures probe transfer
   base->SFT for ORDINARY task SFT (finding: knowledge-boundary geometry preserved, expression
   broken); 2506.14387 measures activation displacement from ordinary fine-tuning. What does NOT
   exist (searched: "probe transfer honesty fine-tuning", "abstention training internal
   representations", "truth direction after RLHF/KTO/GRPO", "representation change honesty
   training", base-vs-finetuned probe AUROC): any study training a model FOR honesty/abstention
   (SFT-IDK, KTO, GRPO-ternary) and then asking whether base-model truth/P(IK)/semantic-entropy
   probes still read the same internal state — i.e., whether humility training changes the
   verifiability layer or only the behavior layer. The directional logic of 2511.12991 predicts
   "representations unchanged, expression re-wired," but nobody has tested the prediction under
   honesty-targeted training, and nobody has compared SFT vs preference vs RL training on this
   axis. **This is our experiment's core and it is open.**

2. **GRPO for abstention exists (gap claim must be narrowed) — HIGH confidence.** As of mid-2026
   one canNOT claim "no one has used GRPO for abstention": TruthRL (2509.25760), Reinforced
   Hesitation (2511.11500), FiSCoRe (2510.24020), Rewarding Intellectual Humility (2601.20126),
   Abstain-R1 (2604.17073), TIAR (2605.25850) all do GRPO-family abstention training; RLCR
   (2507.16806) and DCPO (2603.09117) do GRPO-family calibration rewards. The PUBLISHABLE residual
   gaps: (a) no GRPO-abstention paper measures INTERNAL representation change (all evaluate
   behavior only); (b) no GRPO-vs-KTO-vs-SFT comparison on the same abstention dataset (links to
   report 05 gap #1); (c) ternary-reward works do not report token-level ECE alongside abstention
   F1; (d) almost everything is Qwen-family 3B-7B — cross-family replication absent.

3. **No paper combines a pre-generation probe (verifiability layer) with abstention
   TRAINING evaluation — MEDIUM-HIGH confidence.** SEPs/P(IK)/Do-Androids build pre-generation
   detectors on frozen models; the GRPO/SFT abstention literature trains behavior without probes.
   The obvious composite measurement — "after abstention training, does probe-state at the
   pre-generation token PREDICT the trained abstention decision?" (behavior-representation
   coupling) — appears in no retrieved paper.

4. **Probe validity caveats our design must inherit — HIGH confidence these are real.**
   (a) truth probes may read knowledge-recall strength, not truth (2510.09033); (b) cross-dataset
   probe generalization fails unless trained diversely (Orgad; 2407.08582); (c) probes in reward
   loops get gamed under on-policy RL (2505.13787); (d) linear probes can detect task format
   rather than the intended construct (2606.02907). Mitigations: diverse probe training sets,
   held-out probe usage, mass-mean + nonlinear probe redundancy.

5. **2026 verification hygiene:** of the 13 essay-cited IDs, 2 are unverifiable (2605.21127,
   2606.08543), 1 is title-only (2606.03969 — no findings extractable), 1 is topic-mismatched
   (2604.03147). Recommend the essay cite 2606.03969 by confirmed title only, drop or re-source
   the two NOT-FOUND IDs, and re-describe 2604.03147.


---

## POST-HOC CORRECTION (2026-06-10, orchestrator re-check of essay-cited IDs)

1. **2604.03147 verdict REVERSED — the essay citation is CORRECT.** The "valence-arousal"
   paper IS a sycophancy paper: "Valence-Arousal Subspace in LLMs: Circular Emotion Geometry
   and Multi-Behavioral Control" — steering along the arousal axis gives near-monotonic
   bidirectional control over sycophancy (78% baseline -> 61% at low arousal, 84% at high
   arousal, Political Typology benchmark; replicated on Llama-3.1-8B, Qwen3-8B, Qwen3-14B).
   The essay's paraphrase ("how positively and how intensely the model frames its responses")
   maps exactly onto valence/arousal. Authorship ("Sun") still unconfirmed.
   URL: https://arxiv.org/abs/2604.03147
2. **2606.08543 (Yang, diversity collapse): ID confirmed wrong, but the CLAIM is
   well-supported by real literature.** Best-match: "Where does output diversity collapse in
   post-training?" (arXiv 2604.16027, Karouzos/Tan/Aletras — SFT/DPO/RL lineages of Olmo 3,
   15 tasks, 4 diversity metrics); plus the RLVR diversity-collapse line (2509.07430,
   2605.00365, entropy-collapse work). No "Yang" authorship match found — essay should
   re-source or re-attribute.
3. **2605.21127 (Twist, silent trace suppression): still NOT FOUND** under any searched
   phrasing. Nearest verified literature: CoT-monitorability degradation under training
   incentives (2512.00218 "Reasoning Under Pressure"; Baker et al. 2025 obfuscated reward
   hacking; METR 2026-04 fine-tuning CoT controllability blog). Essay should re-check its
   source for this claim.
