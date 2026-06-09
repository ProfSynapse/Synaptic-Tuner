---
report: generalization-sft-vs-preference-gap-analysis
area: methods
date: 2026-06-09
agent_queries: 24
method: "WebSearch extraction; WebFetch blocked on all paper hosts (HTTP 403)"
verification_status: unverified-against-pdf
tags: [raw-report, epistemic-humility, generalization, gap-analysis]
---

# Raw evidence report: cross-domain generalization of trained humility + SFT vs preference-method comparisons + gap analysis

## Part A — Cross-domain generalization of trained abstention/calibration

```
PAPER: R-Tuning: Instructing Large Language Models to Say 'I Don't Know'
ARXIV: 2311.09677, 2023 (NAACL 2024 Outstanding Paper)
MODELS: LLaMA/OpenLLaMA family, 3B/7B/13B
COMPARISON: Refusal-aware SFT vs vanilla instruction tuning; in-domain vs OOD (single-task + multi-task over ParaRel, HotpotQA, SelfAware, HaluEval, FalseQA, NEC, MMLU, WiCE, FEVER)
METRIC + NUMBERS: AP score; exact table values not found. R-Tuning 13B "performs significantly better than Vanilla on out-of-domain ParaRel and in-domain MMLU"; higher AP on unseen HaluEval-QA.
QUOTE: "the refusal ability was found to be a meta-skill that could be generalized to other tasks"
URL: https://arxiv.org/abs/2311.09677
```

```
PAPER: Large Language Models Must Be Taught to Know What They Don't Know
ARXIV: 2406.08391, 2024 (NeurIPS 2024; Kapoor, Gruver, ..., Wilson)
MODELS: multiple open LLMs (LoRA calibration-tuning; ~20,000 graded generations per base model)
COMPARISON: LoRA calibration tuning vs zero-shot/prompt baselines; in-domain vs new-domain transfer
METRIC + NUMBERS: "A thousand graded examples are sufficient to outperform baseline methods"; exact ECE/AUROC not found.
QUOTE: "fine-tuning on a small dataset of correct and incorrect answers can create an uncertainty estimate with good generalization" ... "transfers to new domains outside of the calibration-tuning train set"
URL: https://arxiv.org/abs/2406.08391
```

```
PAPER: Teaching Models to Express Their Uncertainty in Words
ARXIV: 2205.14334, 2022
MODELS: GPT-3 175B, SFT on CalibratedMath
COMPARISON: verbalized probability (SFT) vs logit-based; calibration under distribution shift
METRIC + NUMBERS: see report 01 (repo README MSE/MAD tables).
QUOTE: "The model remains moderately calibrated under distribution shift"
URL: https://arxiv.org/abs/2205.14334
```

```
PAPER: Improving Metacognition and Uncertainty Communication in Language Models
ARXIV: 2510.05126, 2025 (Steyvers, Belem, Smyth)
MODELS: multiple LLMs; SFT on general knowledge + math + open-ended trivia
COMPARISON: SFT for confidence estimation vs pairwise confidence comparison; in-domain vs new domains/formats
METRIC + NUMBERS: not found (calibration + discrimination improved both in-domain and cross-domain).
QUOTE: "Fine-tuning reliably improved calibration and discrimination... not only in the training domain but also when models were evaluated on new domains"; BUT "enhancing the model's ability to assess certainty about a single answer does not translate into a better ability to compare confidence across answers" — metacognitive skills are learned as distinct routines.
URL: https://arxiv.org/abs/2510.05126
```

```
PAPER: Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty (RLCR)
ARXIV: 2507.16806, 2025 (Damani et al.)
MODELS: open reasoning LMs (repo: github.com/damanimehul/RLCR)
COMPARISON: RL binary-correctness (RLVR) vs RL Brier-augmented (RLCR) vs post-hoc classifiers; ID vs OOD
METRIC + NUMBERS: exact values not found.
QUOTE: "RLVR degrades calibration in OOD tasks, while RLCR significantly improves it"; "RLCR substantially improves calibration with no loss in accuracy, on both in-domain and out-of-distribution evaluations"
URL: https://arxiv.org/abs/2507.16806
```

```
PAPER: Annotation-Efficient Universal Honesty Alignment (EliCal / HonestyBench)
ARXIV: 2510.17509, 2025
MODELS: 3 open LLMs
COMPARISON: calibration-only vs elicitation-then-calibration; ID (38k, 10 QA datasets) vs OOD (33k incl. MMLU)
METRIC + NUMBERS: "near-optimal alignment with only 1k correctness annotations (~0.18% of full supervision)"; exact scores not found.
URL: https://arxiv.org/abs/2510.17509
```

```
PAPER: AbstentionBench (see report 03) — reasoning fine-tuning degrades abstention by 24% on average; scaling does not help.
ARXIV: 2506.09038, 2025
```

```
PAPER: Fine-Tuning LLMs to Appropriately Abstain with Semantic Entropy
ARXIV: 2410.17234, 2024
COMPARISON: SFT and DPO variants of semantic-entropy abstention vs R-Tuning(-U); TriviaQA, SQuAD — one of very few papers using BOTH SFT and a preference method for the same abstention objective, label-free
METRIC + NUMBERS: see report 02 (AED values).
URL: https://arxiv.org/abs/2410.17234
```

## Part B — SFT vs preference-based methods for calibration/honesty/abstention

```
PAPER: GPT-4 Technical Report
ARXIV: 2303.08774 — ECE 0.007 (pretrained) -> 0.074 (post-RLHF) on MMLU subset (see report 01).
```

```
PAPER: Taming Overconfidence in LLMs: Reward Calibration in RLHF
ARXIV: 2410.09724, 2024 — see report 01 (PPO-M: -6.44 ECE, +2.73 acc on GSM8K vs vanilla PPO).
```

```
PAPER: Restoring Calibration for Aligned Large Language Models
ARXIV: 2505.01997, 2025 — see report 01 (CFT out-domain conf-ECE 0.0225 vs LS 0.0499 vs TS 0.1160).
QUOTE: "the preference collapse issue in alignment undesirably generalizes to the calibration scenario, causing LLMs to exhibit overconfidence and poor calibration"
```

```
PAPER: Fine-tuning Language Models for Factuality (FactTune)
ARXIV: 2311.08401, 2023 (Tian, Mitchell, Yao, Manning, Finn)
MODELS: Llama-2-7B (vs Llama-2-chat)
COMPARISON: DPO over auto-generated factuality preference pairs (FactTune-FS: FactScore-referenced; FactTune-MC: model-confidence) vs SFT, RLHF, factuality decoding; held-out topics (biographies, medical QA)
METRIC + NUMBERS: FactTune-FS "up to 17.06 correct facts with only 2.00 errors" on biographies; "at least 23% [error reduction] on biographies and 12% on medical QA"; "at 7B scale, compared to Llama-2-chat, a 40% reduction in factual error rate" on medical questions. Per-condition FactScore table: not found.
QUOTE: "learning from automatically generated factuality preference rankings significantly improves the factuality of Llama-2 on held-out topics compared with RLHF or decoding strategies"
URL: https://arxiv.org/abs/2311.08401
```

```
PAPER: LACIE (see report 02) — DPO listener-aware calibration: +20.7 AUROC avg, -7.8 calibration error, +18% absolute precision.
ARXIV: 2405.21028, 2024
```

```
PAPER: Insights into Alignment: Evaluating DPO and its Variants Across Multiple Tasks
ARXIV: 2404.14723, 2024
MODELS: Mistral-family open LLMs, 13 benchmarks
COMPARISON: SFT vs DPO vs IPO vs KTO vs CPO — closest existing KTO-vs-SFT truthfulness comparison
METRIC + NUMBERS: TruthfulQA: "KTO and IPO outperform SFT by 17.5%" (from SFT'd base); "KTO, based on a pre-trained model, outperforms SFT by 9.5%". "KTO outperforms other methods across all tasks except multi-task understanding."
QUOTE: "KTO outperforms other methods across all tasks except for multi-task understanding"
URL: https://arxiv.org/abs/2404.14723
```

```
PAPER: KTO: Model Alignment as Prospect Theoretic Optimization
ARXIV: 2402.01306, 2024 (Ethayarajh et al.)
MODELS: Pythia/Llama 1B-30B
COMPARISON: SFT vs SFT+DPO vs SFT+KTO vs KTO-alone; unpaired binary feedback, asymmetric loss-aversion weighting
METRIC + NUMBERS: "SFT+KTO is competitive with SFT+DPO at scales 1B-30B"; "KTO alone is better than DPO alone for Llama-{7B,13B,30B}". NO calibration/abstention metrics anywhere in the paper.
QUOTE: "Without doing supervised fine-tuning first, DPO-aligned models tend to ramble and hallucinate entire conversations, while KTO does not suffer from this issue."
URL: https://arxiv.org/abs/2402.01306
```

```
PAPER: FLAME (see report 03) — standard SFT+DPO alignment encourages false facts; factuality-aware variants fix.
ARXIV: 2405.01525, 2024
```

```
PAPER: Fine-Tuning Methods for LLMs in Clinical Medicine by SFT and DPO
ARXIV: n/a — JMIR 2025;27:e76048
MODELS: Llama3 7B [as printed], Mistral 7B v2
COMPARISON: SFT vs DPO on 4 clinical tasks
METRIC + NUMBERS: Clinical reasoning accuracy +8% DPO over SFT for Llama3 (p=0.003), +7% for Mistral (p=0.004) — RARE example of reported p-values in this literature.
URL: https://www.jmir.org/2025/1/e76048
```

```
PAPER: Know Your Limits survey (2407.18418) — flags "abstention as a meta-capability that transcends specific tasks or domains" as open.
PAPER: Honesty survey (2409.18786) — open problems: honesty definitions, known/unknown distinction; flags "great potential of reinforcement learning to improve self-knowledge" as underexplored.
```

Peripheral: An Embarrassingly Simple Defense Against LLM Abliteration Attacks (2505.19056) — only dose-response-style study found, but for SAFETY refusals: refusal-data fractions 2/5/10/30% of SFT mix; at 2%: 6% over-refusal, 24% attack success; over-refusal "predictably increases" with fraction. Also: CATTO (2601.23096, per-token calibration loss added to DPO), FiSCoRe (2510.24020, RL fine-grained semantic-confidence reward), CRaFT (2410.06913).

## GAP ANALYSIS (prioritized for a 3B-7B SFT-vs-KTO experiment on open QA)

1. **No paper applies KTO to abstention/honesty/calibration training — HIGH confidence.** Searched: "KTO abstention/I don't know/refuse unknown", "KTO + calibration/confidence + ECE", "honesty/knowledge-boundary alignment + KTO 2025-2026", KTO paper's own application list. Zero hits as of 2026-06. KTO's properties fit abstention naturally: (a) unpaired binary feedback maps directly onto known/unknown splits without preference pairs; (b) asymmetric loss aversion mirrors asymmetric cost of hallucination vs abstention. Nearest indirect evidence: KTO beats SFT by 9.5-17.5% on TruthfulQA (2404.14723); KTO-without-SFT hallucinates less than DPO-without-SFT (2402.01306). **Direct SFT vs KTO on IDK-training with calibration/AP metrics is unclaimed territory.**

2. **No SFT vs DPO vs KTO three-way on the same abstention dataset — HIGH confidence.** Only 2410.17234 uses both SFT and DPO for the same abstention objective; it omits KTO. 2404.14723 compares SFT/DPO/KTO but on generic benchmarks, not abstention training. (Note: Cheng 2401.13275 compares SFT/DPO/PPO/BoN/HIR on Idk data — closest prior art; still no KTO.)

3. **No dose-response study of IDK-example fraction for epistemic abstention — MEDIUM-HIGH confidence.** Only fraction-sweep found (2505.19056) concerns safety refusals. R-Tuning successors vary WHICH questions become IDK, not the fraction, and report no abstention-precision vs over-refusal Pareto curve.

4. **Cross-domain humility transfer thinly and inconsistently evidenced — HIGH confidence under-tested.** Positive: R-Tuning meta-skill claim; Kapoor 2406.08391; Steyvers 2510.05126; RLCR OOD. But: (a) R-Tuning OOD evidence is SFT-only; (b) abstention survey lists meta-capability as open; (c) AbstentionBench shows training can DEGRADE abstention 24%; (d) abstention instruction-tuning reported to "struggle to generalize across domains and LLMs." **No paper measures whether KTO- or DPO-trained abstention transfers OOD.** A "train on TriviaQA-split -> test on domain-shifted QA" matrix with SFT vs KTO would be novel.

5. **Mechanism asymmetry unreconciled — MEDIUM confidence.** Preference methods damage token-level calibration (GPT-4 TR, 2505.01997, 2410.09724) yet are the best tool for factuality/abstention quality (FactTune, LACIE, Cheng DPO>SFT). No paper measures token-level ECE AND abstention quality AND factuality after the same preference-training run. KTO's status: completely unmeasured.

6. **Small-model (3B) coverage weak — MEDIUM confidence.** Nearly all evidence is 7B+. KTO paper covers 1B-30B but not for humility. Whether abstention-as-meta-skill holds at 3B is untested (exception: Abstain-R1 fine-tunes Qwen2.5-3B with RL, not KTO/SFT comparison).

7. **No standardized OOD abstention metric set — MEDIUM confidence.** AP (R-Tuning), prudence/over-conservativeness (2312.07000), truthful rate/quadrants (2401.13275), ECE/AUROC, listener-acceptance (LACIE), abstention recall (AbstentionBench), AED (2410.17234). A new experiment should report truthful-rate + AP + ECE + over-refusal jointly.
