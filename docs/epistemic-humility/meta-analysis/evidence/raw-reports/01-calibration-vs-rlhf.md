---
report: calibration-vs-rlhf
area: calibration
date: 2026-06-09
agent_queries: 15
method: "WebSearch extraction + official GitHub repo fetches; arXiv/ACL/openreview direct fetch blocked by network allowlist (HTTP 403)"
verification_status: unverified-against-pdf
tags: [raw-report, epistemic-humility, calibration]
---

# Raw evidence report: calibration of LLM confidence vs pretraining / instruction-tuning / RLHF

Environment caveat: direct fetches of arxiv.org, aclanthology.org, openreview.net, and
openai.com were proxy-blocked (HTTP 403; only github.com/raw.githubusercontent.com/pypi.org
reachable), so numbers come from (a) official paper GitHub repos fetched directly (primary),
and (b) WebSearch content extraction of arXiv/ACL pages (secondary). Provenance flagged per
entry.

```
PAPER: Language Models (Mostly) Know What They Know
ARXIV: 2207.05221, 2022 (Kadavath et al., Anthropic)
MODELS: Anthropic LMs at 800M, 3B, 12B, 52B parameters (52B is the headline model); also RLHF policies derived from them
INTERVENTION: pretrained-only models vs RLHF policies; self-evaluation prompting (P(True)), trained P(IK) classifier
METRIC: ECE (10 equal-mass bins), RMS calibration error, Brier score (chance = 0.25 on binary tasks), AUROC of P(IK); datasets: BIG-Bench + MMLU + TriviaQA, Lambada, GSM8k, Codex HumanEval, arithmetic
NUMBERS: Mostly not found as exact table values (full PDF unreachable). Verified findings: RLHF policies "naively appear very miscalibrated," and a single temperature adjustment T = 2.5 applied across all evaluations largely restores calibration; 52B model "very well-calibrated except near the tails, where it is overconfident"; for 52B models, answers labeled P(True) > 50% are far more likely to be correct; P(IK) AUROC increases with model size on all three OOD evals when trained only on TriviaQA. Exact AUROC/Brier table values: not found.
QUOTE: "RLHF policies naively appear very miscalibrated, since RL finetuning tends to collapse language model predictions towards behaviors that receive the most reward. However, a simple temperature adjustment (with the same temperature T = 2.5 for all evaluations) largely fixes calibration issues"
ERROR BARS: none reported in retrieved material
URL: arxiv.org/abs/2207.05221 (403; content via WebSearch extraction of arxiv.org/pdf/2207.05221 and ar5iv mirror snippets)
```

```
PAPER: GPT-4 Technical Report
ARXIV: 2303.08774, 2023 (OpenAI)
MODELS: GPT-4 (size undisclosed): pre-trained base vs post-trained (RLHF/PPO) version
INTERVENTION: pretrained-only vs RLHF post-training
METRIC: ECE on a subset of MMLU (multiple-choice A/B/C/D logprob confidence vs accuracy, Figure 8); subset size not reported
NUMBERS: ECE pretrained GPT-4 = 0.007 -> post-RLHF GPT-4 = 0.074 on the MMLU subset (~10x degradation)
QUOTE: "Left: Calibration plot of the pre-trained GPT-4 model on a subset of the MMLU dataset... The model's confidence in its prediction closely matches the probability of being correct... Right: Calibration plot of the post-trained GPT-4 model on the same subset... post-training hurts calibration significantly." ECE 0.007 (pre-trained) and 0.074 (post-trained) are printed on the Figure 8 panels.
ERROR BARS: none reported
URL: cdn.openai.com/papers/gpt-4.pdf (403; values confirmed via targeted WebSearch quoting Figure 8)
```

```
PAPER: Teaching Models to Express Their Uncertainty in Words
ARXIV: 2205.14334, 2022 (Lin, Hilton, Evans; TMLR)
MODELS: GPT-3 175B (davinci), supervised-finetuned to verbalize confidence
INTERVENTION: pretrained + supervised finetuning to emit verbalized probability; compared against logit-based confidence
METRIC: MSE and MAD of confidence vs empirical accuracy on CalibratedMath (21 arithmetic task types), evaluated under distribution shift (add-subtract -> multiply-divide and multi-answer)
NUMBERS (from official repo README, primary source): Multi-answer shift — verbalized numbers (finetune) MSE 22.0 / MAD 16.4; answer logit (zero-shot) MSE 37.4 / MAD 33.7; indirect logit (finetune) MSE 33.7 / MAD 38.4; constant baseline MSE 34.1 / MAD 31.1. Multiply-divide shift — verbalized MSE 15.5 / MAD 19.0; answer logit MSE 10.4 / MAD 9.4; indirect logit MSE 11.7 / MAD 7.1; constant baseline MSE 15.3 / MAD 8.5.
QUOTE: "CalibratedMath is a test suite of simple arithmetic tasks. Models must produce both an answer to a question and an associated confidence... 21 tasks in total" (repo README)
ERROR BARS: none reported in README tables
URL: https://raw.githubusercontent.com/sylinrl/CalibratedMath/main/README.md (fetched directly — primary source)
```

```
PAPER: Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback
ARXIV: 2305.14975, 2023 (Tian, Mitchell, Zhou, Sharma, Rafailov, Yao, Finn, Manning; EMNLP 2023)
MODELS: ChatGPT (gpt-3.5-turbo), GPT-4, Claude (RLHF-LMs)
INTERVENTION: RLHF models; prompting methods — conditional token probabilities ("Label prob.", "Is True prob.") vs verbalized confidence (Verb. 1S, Verb. 2S, Ling. 1S)
METRIC: ECE, temperature-scaled ECE-t, Brier score BS-t, AUC; datasets TriviaQA, SciQ, TruthfulQA; temperature fit via 5-fold splits
NUMBERS: Verbalized confidences reduce ECE by ~50% relative vs the model's conditional probabilities, on average across models/datasets. Per-cell table values: not found (PDF unreachable). Side finding: GPT-3.5-turbo is systematically UNDER-confident on TruthfulQA.
QUOTE: "verbalized confidences emitted as output tokens are typically better-calibrated than the model's conditional probabilities on the TriviaQA, SciQ, and TruthfulQA benchmarks, often reducing the expected calibration error by a relative 50%" (abstract)
ERROR BARS: none reported in retrieved material (5-fold protocol used for temperature fitting only)
URL: arxiv.org/abs/2305.14975 / aclanthology.org/2023.emnlp-main.330 (403; abstract verified verbatim via WebSearch)
```

```
PAPER: Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs
ARXIV: 2306.13063, 2023 (ICLR 2024; Xiong et al.)
MODELS: GPT-3 175B, GPT-3.5-turbo, GPT-4, Vicuna-13B, LLaMA-2(-Chat) 70B
INTERVENTION: RLHF/instruction-tuned models; prompting (vanilla/CoT/self-probing/top-k/multi-step), sampling + aggregation for consistency-based confidence
METRIC: ECE and AUROC (plus AUPRC) on confidence calibration and failure prediction; 5 dataset types: commonsense (SportsUnderstanding, StrategyQA), math (GSM8K, SVAMP), symbolic (DateUnderstanding, ObjectCounting), law (ProfessionalLaw), ethics (BusinessEthics)
NUMBERS: Verbalized confidence is overconfident — values cluster in 80-100%, typically multiples of 5 (e.g., 85%, 90%); paper treats ECE > 0.25 and AUPRC-Neg < 0.6 as significant deviation from ideal, which vanilla verbalized confidence frequently hits; white-box (token-prob) access improves failure-prediction AUROC only from 0.522 to 0.605 vs black-box methods. Per-model ECE table values: not found.
QUOTE: "LLMs, when verbalizing their confidence, tend to be overconfident, potentially imitating human patterns of expressing confidence" (repo README); most verbalized confidences fall in 80-100% in multiples of 5.
ERROR BARS: none reported in retrieved material
URL: https://raw.githubusercontent.com/MiaoXiong2320/llm-uncertainty/main/README.md (fetched directly — primary source)
```

```
PAPER: On the Calibration of Large Language Models and Alignment
ARXIV: 2311.13240, 2023 (Zhu, Xu, Wang, Zhang, Mao; Findings of EMNLP 2023)
MODELS: open LLMs across pretraining checkpoints and aligned variants (instruction-tuned on Alpaca vs OpenAssistant data, RLHF)
INTERVENTION: full pipeline study — pretraining scale/dynamics vs instruction tuning vs RLHF
METRIC: ECE on generation, factuality, and understanding tasks
NUMBERS: not found (exact table values unreachable). Verified directional findings: larger parameter scales and longer pretraining improve calibration; instruction tuning generally weakens it; synthetic instruction data (Alpaca/self-instruct style) harms calibration more than diverse human-labeled data (OpenAssistant Conversations).
QUOTE: "larger parameter scales and longer training dynamics improve calibration during pretraining, while instruction tuning generally weakens it, with synthetic data exacerbating this effect"
ERROR BARS: unknown
URL: arxiv.org/abs/2311.13240 / aclanthology.org/2023.findings-emnlp.654 (403; via WebSearch extraction)
```

```
PAPER: Llamas Know What GPTs Don't Show: Surrogate Models for Confidence Estimation
ARXIV: 2311.08877, 2023 (Shrivastava, Liang, Kumar; Stanford)
MODELS: GPT-4, GPT-3.5, Claude-v1.3 (closed, RLHF) scored by Llama-2-70B (open base) as surrogate
INTERVENTION: RLHF models; verbalized (linguistic) confidence vs surrogate-model token probabilities
METRIC: AUC (selective classification), ECE; 12 QA datasets
NUMBERS: GPT-4 linguistic confidence AUC = 80.5% (avg over 12 datasets, ~7 pts above random baseline); using Llama-2-70B surrogate probabilities: GPT-4 80.5% -> 82.1%, Claude-v1.3 73.5% -> 76.3%, GPT-3.5 59.0% -> 72.1%; composing linguistic + surrogate = 84.6% avg AUC on GPT-4 (SOTA on all 12 datasets). Also notes base versions generally outperform chat versions for both linguistic confidences and model probabilities.
QUOTE: "using Llama 2 70B probabilities as a surrogate improves AUC from 80.5% to 82.1% for GPT-4, 73.5% to 76.3% for Claude-v1.3, and 59.0% to 72.1% for GPT-3.5"
ERROR BARS: none reported in retrieved material
URL: arxiv.org/abs/2311.08877 (403; via WebSearch extraction)
```

```
PAPER: Benchmarking LLMs via Uncertainty Quantification (conformal prediction)
ARXIV: 2401.12794, 2024 (Ye et al.; NeurIPS 2024 D&B)
MODELS: 8 series (Llama-2, Mistral, Falcon, MPT, Yi, Qwen, DeepSeek, InternLM), 1.8B-72B; base vs chat/instruct variants
INTERVENTION: pretrained base vs instruction-finetuned, measured via conformal prediction set size (+ ECE comparisons of CP vs entropy/perplexity)
METRIC: Accuracy, conformal prediction Set Size (uncertainty), Coverage Rate, UAcc, ECE; 5 tasks x 10,000 instances each (MMLU-based QA, CosmosQA, HellaSwag, HaluEval dialogue + summarization)
NUMBERS: Llama-2 base: 70B = 65.86% acc / 2.62 set size; 13B = 52.52% / 3.06; 7B = 45.60% / 3.20. Chat variants show systematically larger prediction sets (higher uncertainty) than base counterparts even when accuracy improves for smaller models. Conformal prediction achieved lowest average ECE vs entropy/max-prob baselines (e.g., on InternLM-7B).
QUOTE: "instruction-finetuning tends to increase the uncertainty of LLMs" (repo README)
ERROR BARS: none reported in README; n = 10,000 per task is large
URL: https://raw.githubusercontent.com/smartyfh/LLM-Uncertainty-Bench/main/README.md (fetched directly — primary source)
```

```
PAPER: Taming Overconfidence in LLMs: Reward Calibration in RLHF
ARXIV: 2410.09724, 2024 (Leng et al.; ICLR 2025)
MODELS: Llama3-8B, Mistral-7B (PPO pipelines; PPO-M / PPO-C variants vs vanilla PPO)
INTERVENTION: RLHF (PPO) and calibrated-reward RLHF variants
METRIC: ECE, AUC, accuracy across six datasets incl. GSM8K
NUMBERS: Reward models are biased toward high-confidence responses regardless of quality; vanilla RLHF induces verbalized overconfidence. PPO-M on Llama3-8B reduces ECE by 6.44 points and raises accuracy by 2.73 points on GSM8K vs vanilla PPO; PPO-M and PPO-C lower ECE on all six datasets while maintaining or improving accuracy. Absolute baseline ECE values: not found.
QUOTE: "PPO-M on Llama3-8B reduces ECE by 6.44 points and increases accuracy by 2.73 points on GSM8K"
ERROR BARS: unknown
URL: arxiv.org/abs/2410.09724 + github.com/SeanLeng1/Reward-Calibration (via WebSearch extraction)
```

```
PAPER: Investigating Uncertainty Calibration of Aligned Language Models under the Multiple-Choice Setting
ARXIV: 2310.11732, 2023
MODELS: Llama family 7B-70B (pretrained) vs Vicuna and Llama-2-Chat (aligned)
INTERVENTION: pretrained vs SFT/RLHF-aligned, logit-based confidence, multiple-choice (MMLU-style)
METRIC: ECE on multiple-choice tasks
NUMBERS: not found (exact ECE values unreachable). Verified findings: aligned LMs are overconfident vs pretrained counterparts in both zero-shot and in-context settings; decomposes uncertainty into answer-decision vs format-preference components — conflation of the two explains aligned-model overconfidence; in-context examples calibrate pretrained LMs (by fixing format preference) but have only marginal calibration effect on aligned LMs.
QUOTE: "aligned language models tend to be overconfident in output answers compared to the corresponding pre-trained LMs"
ERROR BARS: unknown
URL: arxiv.org/abs/2310.11732 (403; via WebSearch extraction)
```

```
PAPER: Causal Understanding by LLMs: The Role of Uncertainty
ARXIV: 2509.20088, 2025
MODELS: Pythia-7B (base) vs Dolly-v2-7B (instruction-tuned from the same Pythia base)
INTERVENTION: pretrained-only vs instruction-tuned (controlled same-base comparison)
METRIC: ECE and predictive entropy on causal-understanding QA
NUMBERS: Pythia-7B ECE = 0.13 (entropy 1.32) -> Dolly-v2-7B ECE = 0.36 (entropy 0.92): instruction tuning nearly triples calibration error while reducing entropy by ~30%
QUOTE: "instruction tuning does not even half the entropy while almost tripling the calibration error — comparing pythia-7b (ECE=0.13, entropy=1.32) with dolly-v2-7b (ECE=0.36, entropy=0.92)"
ERROR BARS: unknown
URL: arxiv.org/abs/2509.20088 (403; via WebSearch extraction)
```

```
PAPER: Restoring Calibration for Aligned Large Language Models: A Calibration-Aware Fine-Tuning Approach
ARXIV: 2505.01997, 2025 (ICML 2025)
MODELS: Vicuna-7B-v1.5 (RLHF-aligned), Mistral-7B, Olmo-2-7B, Llama-3.1-Tulu-8B (DPO-aligned)
INTERVENTION: post-hoc repair of RLHF/DPO-induced miscalibration via calibration-aware fine-tuning (CFT) and EM-based ECE regularization
METRIC: conf-ECE, in-domain and out-of-domain
NUMBERS: CFT achieves out-domain conf-ECE = 0.0225 vs Label Smoothing 0.0499 and Temperature Scaling 0.1160. Identifies "calibratable" vs "non-calibratable" regimes; preference-aligned models typically remain calibratable.
QUOTE: "while pre-trained models are typically well-calibrated, LLMs tend to become poorly calibrated after alignment with human preferences"; CFT "reduce[s] out-domain conf-ECE to 0.0225 (vs. Label Smoothing's 0.0499 and Temperature Scaling's 0.1160)"
ERROR BARS: unknown
URL: arxiv.org/abs/2505.01997 + github.com/ZhanliangAaronWang/RestoreLLMCalibration (via WebSearch extraction)
```

## Cross-cutting findings

1. **Pretrained token-probability calibration is near-perfect at scale; RLHF/post-training degrades it roughly an order of magnitude.** [GPT-4 TR: ECE 0.007 -> 0.074; Kadavath: RLHF policies "naively very miscalibrated"; Tian 2305.14975; Restoring Calibration 2505.01997]
2. **The degradation is largely a distribution-sharpening artifact, partially reversible post-hoc.** Single temperature T = 2.5 restores RLHF policy calibration [Kadavath]; temperature scaling and calibration-aware fine-tuning recover much of the loss [Tian ECE-t; 2505.01997; 2410.09724 fixes it inside the RLHF loop via calibrated rewards].
3. **Verbalized confidence partially recovers calibration for RLHF models** (~50% relative ECE reduction vs conditional probabilities) [Tian 2305.14975], **but is itself systematically overconfident**, clustering at 80-100% in multiples of 5 [Xiong 2306.13063; Leng 2410.09724].
4. **Instruction tuning alone (no RL) also damages calibration**, and synthetic/homogeneous instruction data (Alpaca-style) hurts more than diverse human data (OpenAssistant) [Zhu 2311.13240; Pythia->Dolly ECE 0.13->0.36 in 2509.20088; conformal set sizes grow base->chat in Ye 2401.12794; 2310.11732].
5. **Scale improves calibration and self-knowledge for base models** (P(IK)/P(True) AUROC rises with size) but only weakly fixes verbalized overconfidence in aligned models [Kadavath; Zhu 2311.13240; Xiong 2306.13063].
6. **Open base-model probabilities transfer as confidence estimates for closed RLHF models**, beating the closed models' own verbalized confidence (GPT-3.5 AUC 59.0 -> 72.1 with Llama-2-70B surrogate) — evidence the calibration signal survives in pretrained weights but is masked by alignment [Shrivastava 2311.08877].
7. **Methodological note:** error bars/variance are essentially never reported (none found in any of the 12 papers' retrieved material); sample sizes often unreported (GPT-4 "subset of MMLU" of unspecified n). Ye 2401.12794 (10,000 instances/task) and Tian's 5-fold temperature protocol are the closest to variance-aware designs. Metric heterogeneity (ECE bins, ECE-t, RMS-CE, MSE/MAD, AUC) requires normalization before pooling.
