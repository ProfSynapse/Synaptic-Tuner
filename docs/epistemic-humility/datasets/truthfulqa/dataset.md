---
name: TruthfulQA
source: https://github.com/sylinrl/TruthfulQA
paper: "2109.07958"
paper_title: "TruthfulQA: Measuring How Models Mimic Human Falsehoods"
license: Apache-2.0
files: [TruthfulQA.csv]
size: "817 questions, 38 categories"
role: eval
fetched: 2026-06-09
fetched_from: "raw repo clone @ main"
tags: [dataset, epistemic-humility, truthfulness, hallucination-eval]
---

## What it is

817 questions crafted so that humans often answer falsely due to
misconceptions; measures whether models reproduce imitative falsehoods.
Includes best/correct/incorrect reference answers per question (MC1/MC2
multiple-choice variants are derived from these columns).

## Role in our experiment

Secondary **evaluation** axis: checks whether humility training shifts
truthfulness, and guards against the failure mode where abstention training
just suppresses all answers (over-refusal would show as degraded informative
truthfulness here).
