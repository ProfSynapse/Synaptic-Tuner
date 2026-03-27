---
name: case-studies
description: End-to-end case studies showing how to implement the full training pipeline for different skill types. Covers three complete worked examples — tool-calling training, essay-style training, and agentic search (RAG agent) training — demonstrating dataset design, synthetic generation, validation, fine-tuning, evaluation, and iteration. Use when onboarding to the project, understanding how all components fit together, explaining the pipeline to others, or planning a new training capability. This skill is about UNDERSTANDING the system holistically — reference the other skills for specific CLI commands.
allowed-tools: Read, Bash, Write, Grep, Glob
---

# Case Studies: Implementing the Training Pipeline

Three end-to-end worked examples showing how to take a capability from concept to trained model.

## Why Case Studies?

The other skills teach you how to use individual tools:
- **synthetic-data-generation** — how to run SynthChat
- **fine-tuning** — how to run trainers
- **evaluation** — how to run evals
- **upload-deployment** — how to ship models

This skill shows you **how they all connect** — the decisions, the iteration, and the order of operations that turn an idea into a trained capability.

## The Three Case Studies

| Case Study | What It Teaches | Reference |
|-----------|----------------|-----------|
| **Tool Calling** | Structured output training — teaching a model to call APIs with correct syntax, context objects, and parameters | `reference/tool-calling-pipeline.md` |
| **Essay Style** | Creative output training — teaching a model to transform messy brainstorms into structured outlines with voice and personality | `reference/essay-style-pipeline.md` |
| **Agentic Search** | RAG agent training — teaching a model to search a corpus, select relevant documents, and answer grounded in sources | `reference/agentic-search-pipeline.md` |

## The Universal Pipeline

All three case studies follow the same high-level pipeline, but diverge in dataset design and validation:

```
┌──────────────────────────────────────────────────────────┐
│  1. DEFINE THE CAPABILITY                                 │
│     What should the model do? What does good look like?   │
│     → Rubrics, schemas, specifications                    │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│  2. CREATE TRAINING DATA                                  │
│     How do we generate enough high-quality examples?      │
│     → SynthChat scenarios, handcrafted seeds, self-play   │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│  3. VALIDATE & IMPROVE                                    │
│     How do we ensure quality before training?             │
│     → Schema validation, rubric scoring, manual review    │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│  4. TRAIN                                                 │
│     SFT first (learn the format), then KTO (learn         │
│     preferences), optionally GRPO (optimize rewards)      │
│     → Trainers with YAML config                           │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│  5. EVALUATE                                              │
│     Does the model do what we trained it to do?           │
│     → Evaluator with YAML scenarios                       │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│  6. ITERATE                                               │
│     What failed? Generate more data targeting weaknesses. │
│     → Failure analysis → targeted generation → retrain    │
└──────────────────────────────────────────────────────────┘
```

## Key Design Principles

### 1. Schema-First, Not Example-First
Define what "correct" looks like before writing any training data. For tools, this means JSON schemas. For essays, this means rubrics. The schema is the source of truth — everything validates against it.

### 2. SFT Teaches Format, KTO Teaches Judgment
SFT (Supervised Fine-Tuning) teaches the model WHAT to do — tool call syntax, output structure, response format. KTO (Kahneman-Tversky Optimization) teaches the model WHICH responses are better — preferring clarification over reckless action, preferring concise outlines over bloated ones. Never try to teach both at once.

### 3. Paired Contrastive Examples
For KTO, every good example needs a realistic bad counterpart using the SAME user request. The bad example should be a plausible mistake, not garbage — wrong tool selected, missing context fields, overly verbose outline. This is what teaches the model judgment.

### 4. Validate Before You Train
Training on bad data is worse than not training at all. Every dataset goes through structural validation (schema checks) and quality validation (rubric scoring) before it touches a trainer.

### 5. Iterate on Failures
After evaluation, the failure analysis tells you exactly what to generate next. If the model keeps producing empty `memory` fields, make more examples that demonstrate rich session memory. If outlines are too long, add negative examples of bloated outlines.

## Progressive Reference

| Reference | When to Load |
|-----------|-------------|
| **Tool Calling Pipeline** | Understanding the full tools training journey — from schema to trained model | `reference/tool-calling-pipeline.md` |
| **Essay Style Pipeline** | Understanding the full essay training journey — from brainstorm to outline model | `reference/essay-style-pipeline.md` |
| **Agentic Search Pipeline** | Understanding the full RAG agent training journey — from corpus to grounded-answer model | `reference/agentic-search-pipeline.md` |
| **Pipeline Comparison** | Side-by-side comparison of how the pipelines differ at each stage | `reference/pipeline-comparison.md` |

## Cross-References to Other Skills

At each stage of the pipeline, you'll use tools documented in the other skills:

| Pipeline Stage | Skill to Reference |
|---------------|-------------------|
| Generate data | `synethetic-data-generation` |
| Validate data | `synethetic-data-generation` (rubrics, validate command) |
| SFT / KTO / GRPO training | `fine-tuning` |
| Evaluate model | `evaluation` |
| Upload & deploy | `upload-deployment` |

## Tips

- Read the tool-calling case study first — it's the simpler, more mechanical pipeline
- The essay case study shows how to adapt the pipeline for creative/subjective outputs
- The agentic search case study shows how to train multi-step reasoning where tools are means to an end
- All three pipelines use the same trainers, evaluator, and upload tools — only the data differs
- When planning a new capability, map it to whichever case study is closer, then adapt
