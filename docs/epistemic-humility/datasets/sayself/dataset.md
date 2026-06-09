---
name: sayself-training-data
source: https://github.com/xu1868/SaySelf
paper: "2405.20974"
paper_title: "SaySelf: Teaching LLMs to Express Confidence with Self-Reflective Rationales"
license: MIT
files: [sft_reason_conf.jsonl, truthful_qa.json, strategyqa.json, pararel_data.json]
size: "8,603 SFT examples + 3 small eval sets"
role: reference-train
fetched: 2026-06-09
tags: [dataset, epistemic-humility, confidence-training, train]
---

## What it is

SaySelf stage-1 SFT data: HotpotQA-derived examples with self-reflective
rationales + verbalized confidence targets. Reference for response-style
design of our own confidence/abstention targets.
