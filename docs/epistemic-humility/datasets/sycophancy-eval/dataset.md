---
name: sycophancy-eval
source: https://github.com/meg-tong/sycophancy-eval
paper: "2310.13548"
paper_title: "Towards Understanding Sycophancy in Language Models"
license: "MIT per HF mirror; no LICENSE file in source repo — verify before redistribution"
files: [answer.jsonl, are_you_sure.jsonl, feedback.jsonl]
size: "~34 MB across 3 task files"
role: eval
fetched: 2026-06-09
tags: [dataset, epistemic-humility, sycophancy, eval]
---

## What it is

Sharma et al. (Anthropic) free-form sycophancy evaluation: answer sycophancy,
"are you sure?" capitulation, and feedback sycophancy tasks.

## Role in our experiment

Optional third eval axis: does abstention/humility training reduce capitulation
under "are you sure?" pressure (the inverted-humility failure mode)?
