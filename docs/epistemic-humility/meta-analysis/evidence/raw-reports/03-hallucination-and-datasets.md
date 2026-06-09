---
report: finetuning-hallucination-and-dataset-inventory
area: hallucination
date: 2026-06-09
agent_queries: 15
method: "WebSearch extraction + official GitHub repo fetches; arXiv/ACL/HF direct fetch blocked (HTTP 403)"
verification_status: unverified-against-pdf
tags: [raw-report, epistemic-humility, hallucination, datasets]
---

# Raw evidence report: fine-tuning's causal effect on hallucination + knowledge-boundary benchmark/dataset inventory

## A. Causal / mechanistic papers

```
PAPER: Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?
ARXIV: 2405.05904, 2024 (EMNLP 2024 main; Gekhman, Yona, Aharoni, Eyal, Feder, Reichart, Herzig)
MODELS: PaLM 2-M
DESIGN: Controlled closed-book QA fine-tuning on EntityQuestions (Wikidata triplets). Manipulated: (a) proportion of Unknown examples, (b) training duration (early-stop vs convergence). Knowledge categorized via SliCK using P_correct(q,a;M,T) from few-shot sampling at T=0 and T>0.
METRIC: Exact-match test accuracy; fitting speed; SliCK category shifts.
NUMBERS:
  - SliCK thresholds: HighlyKnown P_correct(T>0) >= 0.85; MaybeKnown 0.30 <= P < 0.85; WeaklyKnown 0 < P < 0.30; Unknown P = 0.
  - Unknown examples fitted "substantially slower" than Known; best dev performance when most Known but "only a few" Unknown are fitted.
  - Once fitted, Unknown examples "linearly increase the model's tendency to hallucinate".
  - HighlyKnown facts exceed 95% post-fine-tuning accuracy; MaybeKnown-only training beats HighlyKnown-only (both early-stop and convergence).
  - Higher %-Unknown mixes underperform on OOD test relations (7 held-out EntityQuestions relations).
  - Exact per-condition Figure 1-2 accuracies: not found (PDF inaccessible).
QUOTE: "fine-tuning examples that introduce new knowledge are learned significantly slower... as the examples with new knowledge are eventually learned, they linearly increase the model's tendency to hallucinate." "Early-stopping can minimize the hallucination risk."
URL: https://arxiv.org/abs/2405.05904 ; https://aclanthology.org/2024.emnlp-main.444/
```

```
PAPER: Unfamiliar Finetuning Examples Control How Language Models Hallucinate
ARXIV: 2403.05612, 2024 (NAACL 2025; Kang, Wallace, Tomlin, Kumar, Levine)
MODELS: Llama2-7B-class open models
DESIGN: Manipulated supervision attached to "unfamiliar" finetuning examples: ground-truth labels vs hedged/IDK labels. SFT, RL, and reward-model finetuning on TriviaQA and MMLU; plus RL factuality finetuning with conservative reward model for long-form biography/plot generation.
METRIC: Hallucination form on unfamiliar test queries; FActScore; reward-model overestimation.
NUMBERS: Exact per-condition values: not found. Directional confirmed: hallucinated predictions "mirror the responses associated with unfamiliar finetuning examples"; conservative reward models improve RL factuality finetuning.
QUOTE: "As inputs become more unfamiliar, LLM outputs default towards a 'hedged' prediction, whose form is determined by how the unfamiliar examples in the finetuning data are supervised... models can be influenced to respond to unfamiliar queries with 'I don't know'."
URL: https://arxiv.org/abs/2403.05612
```

```
PAPER: FLAME: Factuality-Aware Alignment for Large Language Models
ARXIV: 2405.01525, 2024 (NeurIPS 2024)
MODELS: Llama-2-70B-chat-class alignment pipeline (SFT + DPO)
DESIGN: Manipulated whether alignment data is novel-to-the-LLM (human-written) vs elicited from the model's own knowledge; factuality-aware SFT + DPO with factuality reward.
METRIC: FActScore; instruction-following win rate.
NUMBERS: Conventional alignment "often leads to more false facts". FLAME: +5.6 FActScore over standard alignment at 51.2% instruction-following win rate (no helpfulness loss). Per-stage breakdown: not found.
QUOTE: "training the LLM on new knowledge or unfamiliar texts can encourage hallucination... making SFT less factual as it trains on human labeled data that may be novel to the LLM."
URL: https://arxiv.org/abs/2405.01525
```

```
PAPER: Why Fine-Tuning Encourages Hallucinations and How to Fix It
ARXIV: 2604.15574, 2026 (Kaplan, Gekhman, et al.)
MODELS: not found
DESIGN: Mechanism follow-up to Gekhman 2024: output-distribution drift during SFT on new facts. Interventions: self-distillation SFT regularizing drift; freezing "factual-plasticity" parameter groups.
NUMBERS: not found (April 2026; tables not retrievable).
QUOTE: "a self-distillation-based SFT method... minimizing hallucinations with respect to pre-existing knowledge by regularizing output-distribution drift."
URL: https://arxiv.org/abs/2604.15574
```

```
PAPER: Why Language Models Hallucinate
ARXIV: 2509.04664, 2025 (Kalai, Nachum, Vempala, Zhang)
DESIGN: Theory: reduction of generation errors to binary classification (Is-It-Valid); binary-graded benchmarks reward guessing over abstention.
NUMBERS: theoretical bounds (generative error rate >= 2x IIV misclassification rate); empirical values not found.
QUOTE: "language models hallucinate because the training and evaluation procedures reward guessing over acknowledging uncertainty... models are optimized to be good test-takers."
URL: https://arxiv.org/abs/2509.04664
```

```
PAPER: John Schulman — RLHF: Progress and Challenges (Berkeley talk, 2023-04-19; not arXiv)
DESIGN: Hypothesis: behavior cloning (SFT) causes hallucination via labeler-model knowledge mismatch; RLHF can teach reliance on internal knowledge + calibrated abstention.
NUMBERS: qualitative only.
QUOTE: "if we succeed in training the model to generalize in these cases, then we essentially teach the model to make stuff up!" (Goldberg gist summary)
URL: https://news.berkeley.edu/2023/04/24/berkeley-talks-transcript-chatgpt-developer-john-schulman/ ; https://gist.github.com/yoavg/6bff0fecd65950898eba1bb321cfbd81
```

## B. Knowledge-boundary / self-knowledge benchmarks

```
PAPER: TruthfulQA: Measuring How Models Mimic Human Falsehoods
ARXIV: 2109.07958, 2021/2022 (ACL 2022)
MODELS: GPT-3 (125M-175B), GPT-Neo/J (125M-6B), GPT-2, UnifiedQA
DESIGN: 817 adversarial questions, 38 categories, imitative falsehoods; inverse scaling test across 4 families.
METRIC: % truthful (human eval), % truthful+informative, GPT-judge.
NUMBERS: Best model (GPT-3 175B helpful prompt): 58% truthful vs human 94%; false+informative 42% vs human 6%. GPT-3-175B QA prompt: 20.4% truthful (scalar), 58.1% true at threshold, 21.4% truthful-and-informative. Inverse scaling: GPT-J 6B 17% less truthful than GPT-J 125M.
QUOTE: "The best model was truthful on 58% of questions, while human performance was 94%... the largest models were generally the least truthful."
URL: https://arxiv.org/abs/2109.07958 ; https://github.com/sylinrl/TruthfulQA
```

```
PAPER: Do Large Language Models Know What They Don't Know? (SelfAware)
ARXIV: 2305.18153, 2023 (ACL Findings 2023; Yin et al.)
MODELS: 20 LLMs — GPT-3 series, InstructGPT series, GPT-3.5-turbo, GPT-4, LLaMA 7B/13B/65B, Alpaca-7B, Vicuna
DESIGN: 1,032 unanswerable + 2,337 answerable; Direct vs Instruction vs ICL input forms; base vs instruction-tuned.
METRIC: F1 on detecting unanswerable questions.
NUMBERS: GPT-4 (instruction) F1 = 75.47 (best); LLaMA-65B = 46.89; human = 84.93. Self-knowledge increases with size, instruction tuning, ICL. Full per-model table: not found.
URL: https://arxiv.org/abs/2305.18153 ; https://github.com/yinzhangyue/SelfAware (data CC-BY-SA-4.0, code Apache-2.0)
```

```
PAPER: Knowledge of Knowledge (KUQ)
ARXIV: 2305.13712, 2023 (ACL Findings 2024; Amayuelas et al.)
MODELS: LLaMA-family fine-tunes; GPT-3.5/4 evaluation
DESIGN: KUQ = 6,884 questions: known-unknowns categorized by uncertainty source (future, unsolved, controversial, ambiguous, counterfactual) + known questions from SQuAD/TriviaQA/HotpotQA. Fine-tuning on KUQ vs not.
METRIC: F1 known-vs-unknown classification; known-question accuracy.
NUMBERS: "considerable increase in F1 relative to pre-fine-tuning"; exact values not found. Trade-off: known-question accuracy "saw a slight decline" after fine-tuning.
URL: https://arxiv.org/abs/2305.13712 ; https://github.com/amayuelas/knowledge-of-knowledge (MIT) ; HF: amayuelas/KUQ
```

```
PAPER: Do LLMs Know When to NOT Answer? (AbstainQA)
ARXIV: 2407.16221, 2024
MODELS: GPT-4, Mixtral 8x22B, others
DESIGN: answerable/unanswerable, well- vs under-represented domains, fact vs reasoning; Answerable-Unanswerable Confusion Matrix (AUCM); prompting interventions.
NUMBERS: not found. "even powerful models like GPT-4 and Mixtral 8x22b encounter difficulties with abstention."
URL: https://arxiv.org/abs/2407.16221
```

```
PAPER: Benchmarking Hallucination based on Unanswerable Math Word Problems (UMWP)
ARXIV: 2403.03558, 2024 (LREC-COLING 2024)
MODELS: 31 LLMs
DESIGN: 5,200 MWP (2,600 answerable + 2,600 unanswerable, 5 categories).
METRIC: F1 unanswerability recognition.
NUMBERS: per-model F1 not found. "ICL and RLHF training significantly enhance the model's ability to avoid hallucination"; scales with size.
URL: https://arxiv.org/abs/2403.03558
```

```
PAPER: Don't Hallucinate, Abstain: Identifying LLM Knowledge Gaps via Multi-LLM Collaboration
ARXIV: 2402.00367, 2024 (ACL 2024; Feng et al.)
MODELS: Mistral-7B, LLaMA2-70B, ChatGPT
DESIGN: 13+ abstention mechanisms compared vs proposed Cooperate/Compete multi-LLM methods, 4 QA datasets.
METRIC: Reliable Accuracy (R-Acc), Abstain Accuracy (A-Acc), Effective Reliability, abstain F1.
NUMBERS: up to 19.3% improvement in abstain accuracy over strongest baseline. Tables: not found.
URL: https://arxiv.org/abs/2402.00367
```

```
PAPER: AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions
ARXIV: 2506.09038, 2025 (FAIR/Meta)
MODELS: 20 frontier LLMs (GPT-4o, o1-class, DeepSeek-R1-distill, Llama 3.1 8B/70B/405B, Qwen2.5-32B, Mistral-7B, OLMo-7B...)
DESIGN: 20 datasets / 6 abstention scenarios. Manipulated: scale; reasoning fine-tuning vs base instruct.
METRIC: abstention recall (+ accuracy).
NUMBERS: reasoning fine-tuning degrades abstention by 24% on average (verify against PDF); scale 8B->405B "almost no effect". Per-model recall: not found.
QUOTE: "abstention remains a key problem even for frontier LLMs, with model scale having almost no effect... reasoning fine-tuning hurts abstention."
URL: https://arxiv.org/abs/2506.09038 ; https://github.com/facebookresearch/AbstentionBench (Apache-2.0) ; HF: facebook/AbstentionBench
```

```
PAPER: R-Tuning (see report 02 for full entry)
ARXIV: 2311.09677 — also belongs in this inventory for its training/eval dataset releases.
```

```
PAPER: Can AI Assistants Know What They Don't Know? (see report 02 for full entry)
ARXIV: 2401.13275 — Idk datasets inventory entry.
```

```
PAPER: FActScore: Fine-grained Atomic Evaluation of Factual Precision
ARXIV: 2305.14251, 2023 (EMNLP 2023; Min et al.)
DESIGN: biography generation; atomic-fact verification against Wikipedia. Standard long-form metric used by FLAME/Kang.
NUMBERS: InstructGPT 42.5%, ChatGPT 58.3%, PerplexityAI 71.5%. Automated estimator <2% error vs human.
URL: https://arxiv.org/abs/2305.14251 ; https://github.com/shmsw25/FActScore
```

```
PAPER: When Not to Trust Language Models (PopQA)
ARXIV: 2212.10511, 2022/2023 (ACL 2023; Mallen et al.)
DESIGN: 14k entity-centric questions with Wikipedia page-view popularity; manipulated popularity + retrieval.
NUMBERS: accuracy strongly correlated with page views; scaling does not fix the tail. Exact accuracies: not found.
URL: https://arxiv.org/abs/2212.10511 ; HF: akariasai/PopQA
```

## DATASET INVENTORY

| Dataset | HF hub / GitHub | Size | License | Use | Model-specific split needed? |
|---|---|---|---|---|---|
| SelfAware | GitHub yinzhangyue/SelfAware | 3,369 Q | Data CC-BY-SA-4.0, code Apache-2.0 | Eval | No |
| KUQ | `amayuelas/KUQ` (HF) | 6,884 Q | MIT | Both | No |
| Idk datasets | GitHub OpenMOSS/Say-I-Dont-Know | TriviaQA-derived per-model SFT + pref | unconfirmed | Train + eval | **Yes — model-specific** |
| R-Tuning data | Google Drive via GitHub | ParaRel/MMLU-derived | not found | Train + eval | **Yes** |
| CoCoNot | `allenai/coconot` | 11,477 SFT + 1,001 eval; contrast 927+379 | MIT repo; dataset terms verify | Both | No |
| AbstentionBench | `facebook/AbstentionBench` | 20 sub-datasets | Apache-2.0 (code) | Eval | No |
| UMWP | GitHub Yuki-Asuuna/UMWP | 5,200 | not found | Eval | No |
| TruthfulQA | `truthfulqa/truthful_qa` | 817 | Apache-2.0 | Eval | No |
| TriviaQA | `mandarjoshi/trivia_qa` | ~95k QA | Apache-2.0 (web portion caveats) | Train+eval substrate | Split construction required |
| Natural Questions | `google-research-datasets/natural_questions` | ~307k/7.8k | CC BY-SA 3.0 | Train+eval substrate | Same |
| MMLU | `cais/mmlu` | ~14k test | MIT | Eval | Model-specific correctness split for abstention |
| PopQA | `akariasai/PopQA` | 14k | MIT | Eval; popularity = model-agnostic "likely unknown" proxy | Optional |
| EntityQuestions | GitHub princeton-nlp/EntityQuestions | ~176k | MIT | Gekhman substrate; SliCK splits | **Yes** |
| SQuAD 2.0 | `rajpurkar/squad_v2` | ~150k | CC BY-SA 4.0 | Context-grounded abstention | No |
| FalseQA | GitHub thunlp/FalseQA | 2,365 | not found | False-premise rebuttal | No |
| BIG-Bench Known Unknowns | via AbstentionBench | tens of items | Apache-2.0 | Eval | No |
| FreshQA | GitHub freshllms/freshqa | ~600 | Apache-2.0 | Outdated-knowledge abstention | No (time-sensitive) |
| HaluEval | GitHub RUCAIBox/HaluEval | 35k | MIT | Hallucination recognition | No |

## Cross-cutting findings

1. **SFT on facts the model doesn't know causally drives hallucination** — linear in fitted Unknown examples; mechanism = output-distribution drift. [2405.05904; 2604.15574; 2405.01525; 2403.05612]
2. **Training duration mediates the harm; early stopping is a cheap mitigation.** [2405.05904]
3. **The remedy is supervision-side: relabel the model's own unknowns with abstention targets** — requires model-specific known/unknown splits. [2403.05612; 2311.09677; 2401.13275; 2305.13712]
4. **Pure SFT abstention overshoots into over-refusal; preference optimization corrects it.** [2401.13275; 2407.12043; 2305.13712]
5. **Behavior-cloning hypothesis (labeler-model knowledge mismatch) unifies the mechanism; RL-based objectives are the principled fix** — but the wrong RL backfires: reasoning-RL degrades abstention 24%. [Schulman talk; 2403.03558; 2405.01525; 2506.09038; 2509.04664]
6. **Scaling alone does not buy epistemic humility** — TruthfulQA inverse scaling; SelfAware plateau (GPT-4 75.47 vs human 84.93); PopQA long tail; AbstentionBench flat 8B->405B. [2109.07958; 2305.18153; 2212.10511; 2506.09038]
7. **Measurement converging on:** atomic-fact precision for long-form (FActScore) + abstention-aware confusion matrices for QA (AUCM, knowledge quadrants, R-Acc/A-Acc). [2305.14251; 2407.16221; 2401.13275; 2402.00367]
