---
name: SelfAware
source: https://github.com/yinzhangyue/SelfAware
paper: "2305.18153"
paper_title: "Do Large Language Models Know What They Don't Know?"
license: Apache-2.0
files: [SelfAware.json]
size: "3,369 questions (2,337 answerable + 1,032 unanswerable)"
role: eval
fetched: 2026-06-09
fetched_from: "raw repo clone @ main"
tags: [dataset, epistemic-humility, knowledge-boundary, abstention-eval]
---

## What it is

Benchmark for knowledge-boundary self-awareness. Unanswerable questions are
sourced from Quora/HowStuffWorks (no definitive answer exists); answerable
questions come from SQuAD/HotpotQA/TriviaQA. Standard metric: F1 over
unanswerable-question detection.

## Role in our experiment

Held-out **evaluation** set for abstention quality (never used in training).
Note: "unanswerable to anyone" is a different construct from "unknown to this
model" — keep the distinction explicit in analysis.
