---
report: abstention-idk-finetuning
area: abstention-finetuning
date: 2026-06-09
agent_queries: 16
method: "WebSearch extraction + official GitHub repo fetches; arXiv/ACL/OpenReview direct fetch blocked (HTTP 403)"
verification_status: unverified-against-pdf
tags: [raw-report, epistemic-humility, abstention]
---

# Raw evidence report: abstention / "I don't know" fine-tuning methods

Caveat: sandbox network allowlist blocked direct fetches of arXiv/ACL/OpenReview full texts
(only raw.githubusercontent.com and pypi.org reachable). Numbers extracted via search-snippet
retrieval against arXiv HTML/PDF pages and official GitHub repos. Unverifiable table values
are marked "not found"; second-hand values flagged.

```
PAPER: R-Tuning: Instructing Large Language Models to Say 'I Don't Know'
ARXIV: 2311.09677, 2023 (NAACL 2024, Outstanding Paper Award)
MODELS: OpenLLaMA-3B, LLaMA-7B, LLaMA-13B (base models)
TRAINING METHOD: SFT (refusal-aware instruction tuning via LMFlow, full fine-tuning; no preference optimization). Variants: R-Tuning-R (refusal suffix) and R-Tuning-U (unsure/uncertainty-sorted, unsupervised)
TRAIN DATA: Unknowns identified by comparing base model's pre-finetuning predictions against gold labels (wrong -> "I am unsure"/refusal suffix). 5 training datasets incl. ParaRel and MMLU; eval also on HaluEval, FalseQA, NEC. Data via Google Drive (no license file)
METRIC: Accuracy on willingly-answered questions; Average Precision (AP) over refusal ranking
NUMBERS: ParaRel + OpenLLaMA-3B: accuracy on answered questions 93.23% (R-Tuning) vs 92.89% (vanilla SFT) in-domain; 69.41% vs 68.42% out-of-domain [second-hand — verify against Table 1]. Exact AP values per dataset: not found
GENERALIZATION: Yes — multi-task experiments show refusal is a "meta-skill" generalizing OOD; R-Tuning-U gives better calibration than uncertainty-based test-time filtering
QUOTE: "the refusal ability was found to be a meta-skill that could be generalized to other tasks."
URL: https://github.com/shizhediao/R-Tuning ; https://arxiv.org/abs/2311.09677
```

```
PAPER: Alignment for Honesty
ARXIV: 2312.07000, 2023 (NeurIPS 2024)
MODELS: LLaMA2-Chat-13B (main); released "Confucius" 13B models
TRAINING METHOD: SFT (full-parameter, CoLLiE). Variants: ABSOLUTE, CONFIDENCE-NUM, CONFIDENCE-VERB, MULTISAMPLE — all supervised; no DPO/KTO
TRAIN DATA: TriviaQA-based; unknowns via "expected accuracy" (k=10 sampled answers; threshold on fraction correct); unknown answers replaced with refusals. Train size: not found
METRIC: Prudence score (refuses unknowns), over-conservativeness score (wrongly refuses knowns), composite honesty score; accuracy
NUMBERS: Never-refusing model scores honesty 50.00% (metric calibration). Non-AmbigQA (OOD): MULTISAMPLE honesty 70.18 / prudence 64.73; CONFIDENCE-VERB honesty 68.74 / prudence 51.11 [via search snippet]. Full TriviaQA Table 3: not found
GENERALIZATION: Yes — trained on TriviaQA, eval OOD on Non-AmbigQA, MMLU, PUQA (1,000 questions about 2023 scientific literature, guaranteed-unknown), PKQA (1,000 likely-known). CONFIDENCE-VERB "consistently outperforms baselines on all three datasets"; models become "slightly over-conservative"
QUOTE: "significant room for improving, particularly in areas such as calibration and generalization across various families of LLMs." (README)
URL: https://github.com/GAIR-NLP/alignment-for-honesty ; https://arxiv.org/abs/2312.07000
```

```
PAPER: Can AI Assistants Know What They Don't Know?
ARXIV: 2401.13275, 2024 (ICML 2024; Cheng et al., OpenMOSS/Fudan)
MODELS: Llama-2-7b-chat (main), Llama-2-13b/70b-chat, Baichuan2-7B-Chat, Mistral-7B-Instruct-v0.1
TRAINING METHOD: THE key SFT-vs-preference comparison for abstention: Idk-Prompting, Idk-SFT, Idk-BoN (reward model + best-of-10), Idk-DPO, Idk-PPO, Idk-HIR. Full fine-tuning (llama-recipes FSDP, lr 2e-5, 10 epochs)
TRAIN DATA: Model-specific "Idk" dataset from TriviaQA: unknown if accuracy over 10 sampled answers < "Ik threshold" (0.1-1.0); unknown answers replaced with refusals. Released 4 model-specific Idk datasets + preference data (threshold 1.0); license not stated
METRIC: TRUTHFUL rate = Ik-Ik (known answered correctly) + Ik-Idk (unknown refused); knowledge quadrants
NUMBERS: TriviaQA Idk test, Llama-2-7b-chat (Ik=1.0): Idk-Prompting 66.93%, Idk-SFT 74.75%, Idk-DPO 77.89%, Idk-BoN 78.96% (best), Idk-PPO 76.47%, Idk-HIR 75.91% truthful. Quadrants for best: Ik-Ik 38.37% + Ik-Idk 40.59%; residual Idk-Ik 11.53%, Idk-Idk 9.51%. Idk-SFT threshold sweep: ~70% truthful at Ik=0.9 down to ~20% at Ik=0.1. Llama-2-70b-chat Idk-SFT ~ +5.8% over 7b
GENERALIZATION: Yes — OOD on Natural Questions (and ALCUNA); on NQ, Idk-HIR achieves highest truthful rate (exact OOD numbers not found). "Refusing to answer unknown questions can be generalized to OOD data"
QUOTE: "Idk-SFT allows the model to refuse to answer more questions it does not know, but it also tends to make the model more conservative, leading to incorrect refusals to answer some questions that it actually knows... preference-aware optimization, like DPO, can alleviate the model's excessive conservatism."
URL: https://github.com/OpenMOSS/Say-I-Dont-Know ; https://arxiv.org/abs/2401.13275
```

```
PAPER: SaySelf: Teaching LLMs to Express Confidence with Self-Reflective Rationales
ARXIV: 2405.20974, 2024 (EMNLP 2024)
MODELS: Mistral-7B (LoRA/PEFT)
TRAINING METHOD: (1) SFT on self-reflective rationales + verbalized confidence (clustering sampled reasoning chains, GPT-4 summarization); (2) RL (PPO-style, reward penalizing miscalibrated confidence). Baselines include R-Tuning
TRAIN DATA: HotpotQA-derived; stage-1 SFT file sft_reason_conf.jsonl = 8,603 examples (counted from repo); repo MIT
METRIC: ECE, AUROC, accuracy, rationale faithfulness
NUMBERS: not found (exact tables unretrievable); "significantly outperforms all baseline approaches in reducing ECE and improving AUROC", in-distribution and OOD
GENERALIZATION: Yes — HotpotQA in-distribution + multiple OOD QA datasets
QUOTE: "SaySelf significantly outperforms all baseline approaches in reducing the calibration error (ECE)... in both in-distribution (HotpotQA) and out-of-distribution datasets."
URL: https://github.com/xu1868/SaySelf ; https://arxiv.org/abs/2405.20974
```

```
PAPER: Linguistic Calibration of Long-Form Generations
ARXIV: 2404.00474, 2024 (ICML 2024)
MODELS: Llama 2 7B
TRAINING METHOD: SFT ("summary distillation") + decision-based RL (PPO) with surrogate-reader reward (ExtractAnswers + ForecastProbs trained on Claude 2.0 outputs)
TRAIN DATA: TriviaQA-derived long-form QA; HF tatsu-lab/linguistic_calibration, CC BY-NC 4.0 (data; contains API-LLM generations); code Apache 2.0
METRIC: Forecast ECE, accuracy, human + simulated-reader evaluation
NUMBERS: exact values not found. "LC RL pareto-dominates Factuality RL and SFT, with significantly better forecast ECE while matching or exceeding their accuracy"; "forecast ECE comparable to GPT-4 baselines"
GENERALIZATION: Yes — OOD on SciQ, Jeopardy, BioASQ; transfers to held-out person-biography generation
QUOTE: "significantly more calibrated than strong finetuned factuality baselines with comparable accuracy... under significant domain shifts to scientific and biomedical questions."
URL: https://github.com/tatsu-lab/linguistic_calibration ; https://arxiv.org/abs/2404.00474
```

```
PAPER: Calibration-Tuning / Large Language Models Must Be Taught to Know What They Don't Know
ARXIV: 2402.08819 (UncertaiNLP 2024); 2406.08391 (NeurIPS 2024), Kapoor, Gruver et al.
MODELS: Llama-2 7B / 7B-Chat / 13B / 13B-Chat, Mistral-7B-v0.1, Mistral-7B-Instruct-v0.2
TRAINING METHOD: LoRA (and LoRA + language prompt "Is the proposed answer correct?") vs frozen-feature probe; supervised calibration objective on graded own-generations. Tunes an uncertainty estimate, not answers; no DPO/KTO
TRAIN DATA: ~20,000 generations per base model, auto-graded; released on HF (calibration-tuning/Llama-2-7b-hf-20k-oe etc.); dataset license not stated; code Apache 2.0
METRIC: ECE (after temperature scaling), AUROC, selective prediction on MC and open-ended MMLU
NUMBERS: exact ECE/AUROC not found (Figure 3). Sample efficiency: "1000 points is almost as valuable as 20,000"
GENERALIZATION: Yes — LoRA + Prompt generalizes MC -> open-ended and OOD better than probing
QUOTE: "a thousand graded examples being sufficient to outperform baseline methods."
URL: https://github.com/activatedgeek/calibration-tuning ; https://arxiv.org/abs/2406.08391
```

```
PAPER: LACIE: Listener-Aware Finetuning for Confidence Calibration
ARXIV: 2405.21028, 2024 (NeurIPS 2024)
MODELS: Mistral-7B, Llama-3-8B, Llama-3-70B
TRAINING METHOD: DPO only (no SFT-vs-DPO ablation): preference pairs from two-agent speaker/simulated-listener game
TRAIN DATA: 10,000 TriviaQA questions x 10 sampled responses, labeled by simulated listener acceptance + gold correctness; repo Apache 2.0
METRIC: Listener acceptance precision/recall; truthfulness (TruthfulQA)
NUMBERS: "47% fewer incorrect answers being accepted [by human listeners] while maintaining the same level of acceptance for correct answers". Per-model tables: not found
GENERALIZATION: Yes — trained TriviaQA, "large increase in truthfulness on TruthfulQA"
QUOTE: "driven by a 47% reduction in false positives... better able to express low confidence when its answer was wrong."
URL: https://arxiv.org/abs/2405.21028 ; https://github.com/esteng/pragmatic_calibration
```

```
PAPER: Rejection Improves Reliability: Training LLMs to Refuse Unknown Questions Using RL from Knowledge Feedback (RLKF)
ARXIV: 2403.18349, 2024
MODELS: Llama-2-Chat family
TRAINING METHOD: RL (PPO) against "reliable" reward model trained on auto-generated model-specific preference pairs from knowledge feedback
TRAIN DATA: synthesized via sampling consistency vs gold answers; size/license not found
METRIC: reliability (accuracy on known + rejection of unknown)
NUMBERS: not found
GENERALIZATION: not found
URL: https://arxiv.org/abs/2403.18349
```

```
PAPER: The Art of Saying No: Contextual Noncompliance in Language Models (CoCoNot)
ARXIV: 2407.12043, 2024 (NeurIPS 2024 D&B)
MODELS: Llama-2-7B (SFT with Tulu2Mix), Tulu-2 7B/13B (continued SFT, LoRA, DPO)
TRAINING METHOD: Direct SFT vs continued SFT vs LoRA-SFT vs DPO (on contrast set vs over-refusal) — one of few explicit SFT-vs-LoRA-vs-DPO comparisons for refusal; includes epistemic-humility categories
TRAIN DATA: CoCoNot taxonomy, 5 categories; HF allenai/coconot; code repo MIT
METRIC: compliance/(over-)refusal rates + general capability benchmarks
NUMBERS: "GPT-4 incorrectly complies with as many as 30% of requests" in understudied categories. Post-training tables: not found
GENERALIZATION: Trade-off focus: direct SFT -> over-refusal + capability decline; LoRA balances; DPO on contrast set improves compliance while maintaining other metrics
URL: https://arxiv.org/abs/2407.12043 ; https://github.com/allenai/noncompliance
```

```
PAPER: Fine-Tuning LLMs to Appropriately Abstain with Semantic Entropy
ARXIV: 2410.17234, 2024
MODELS: SE (Llama) and SE (DeBERTa) entailment variants; base model details not found
TRAINING METHOD: SFT, label-free: semantic entropy over 10 samples; high-entropy -> relabel "I don't know", low-entropy -> keep own answer. Compared against R-Tuning and R-Tuning-U
TRAIN DATA: self-generated from QA training sets; size/license not found
METRIC: Accuracy-Engagement Distance (AED; lower better)
NUMBERS: Long-QA in-distribution AED: SE (Llama) 0.364, SE (DeBERTa) 0.411 vs R-Tuning 0.399, R-Tuning-U 0.469, Original 0.380
GENERALIZATION: Yes — ID and OOD, Long-QA and Short-QA: SE typically equal or lower AED than R-Tuning variants
URL: https://arxiv.org/abs/2410.17234
```

Additional (lighter coverage): ConfTuner (2508.18847, verbalized-confidence training), Abstain-R1 (verifiable-RL abstention on Qwen2.5-3B-Instruct — small-model abstention), Fine-Grained Semantic Confidence Reward (2510.24020, RL), Honesty over Accuracy: Reinforced Hesitation (2511.11500), AbstentionBench (reasoning-trained models ~24% lower abstention recall; scale 8B->405B negligible).

## Open-source training datasets from this literature

| Dataset | Where | License |
|---|---|---|
| CoCoNot | HF `allenai/coconot` | not confirmed (code MIT) |
| Linguistic Calibration data | HF `tatsu-lab/linguistic_calibration` | CC BY-NC 4.0 (data); Apache 2.0 (code) |
| Calibration-tuning graded gens (~20k/model) | HF `calibration-tuning/*` | not stated; code Apache 2.0 |
| Idk datasets (4 model-specific) + preference data | GitHub `OpenMOSS/Say-I-Dont-Know` (Google Drive links) | no license file |
| R-Tuning refusal-aware datasets | GitHub `shizhediao/R-Tuning` (Google Drive) | no license file |
| SaySelf SFT set (8,603 in-repo) | GitHub `xu1868/SaySelf` | MIT |
| LACIE preference pairs (10k TriviaQA) | GitHub `esteng/pragmatic_calibration` | Apache 2.0 |
| Confucius honesty models + PUQA/PKQA | GitHub `GAIR-NLP/alignment-for-honesty` | Llama2 (models); data not stated |

## Explicitly stated limitations / gaps

1. **Over-refusal trade-off is universal**: Idk-SFT "more conservative, incorrect refusals" [2401.13275]; "slightly over-conservative" [2312.07000]; direct SFT -> "over-refusal and decline in general capabilities" [CoCoNot].
2. **Calibration/generalization across model families** flagged open by Alignment for Honesty README.
3. **Model-specificity of IDK data**: Idk and calibration-tuning datasets constructed per base model; labels don't transfer.
4. **Abstention as meta-capability open** [Know Your Limits survey].
5. **Label cost**: graded own-generations needed (~1,000 suffice per Kapoor).
6. **KEY GAP VERIFIED**: Cheng 2401.13275 is the closest SFT-vs-preference comparison (SFT 74.75 vs DPO 77.89 / PPO 76.47 / BoN 78.96 truthful); CoCoNot compares SFT/LoRA/DPO for refusal; LACIE is DPO-only. **No paper uses KTO for abstention/IDK training; no systematic SFT-vs-DPO-vs-KTO abstention comparison exists; small-model (1B-3B) coverage nearly absent** as of June 2026.
