---
name: say-i-dont-know-method-outputs
source: https://github.com/OpenMOSS/Say-I-Dont-Know (outputs/)
paper: "2401.13275"
paper_title: "Can AI Assistants Know What They Don't Know?"
license: "unstated — analysis use only, do not redistribute as training data"
files: [triviaqa_test_llama2_7b_chat_idk_sft.json, triviaqa_test_llama2_7b_chat_idk_dpo.json, triviaqa_test_llama2_7b_chat_idk_ppo.json, triviaqa_test_llama2_7b_chat_idk_bon.json, triviaqa_test_llama2_7b_chat_idk_hir.json]
size: "~48 MB, 5 method-output files on the TriviaQA Idk test set"
role: meta-analysis-reanalysis
fetched: 2026-06-09
tags: [dataset, epistemic-humility, abstention, reanalysis]
---

## What it is

Per-method test outputs from Cheng et al. (ICML 2024): Llama-2-7b-chat aligned
with Idk-SFT / DPO / PPO / BoN / HIR, evaluated on the TriviaQA Idk test set.

## Why it matters for the meta-analysis

arXiv full texts are network-blocked here, but these raw outputs let us
INDEPENDENTLY recompute the paper's truthful rates and knowledge quadrants
(Ik-Ik / Ik-Idk / Idk-Ik / Idk-Idk) — primary-source verification of the
SFT-vs-preference-optimization comparison, and effect-size inputs with known n.
