# Embedding & Reranker Pipeline: Retrieval-Model Training Design

**Document Version:** 1.0
**Created:** 2026-06-14
**Updated:** 2026-06-14
**Status:** Proposal (awaiting review) — no code written yet
**Purpose:** Blueprint for adding retrieval-model training to Synaptic-Tuner — bi-encoder embeddings, LoRA/frozen-head adapters over a base, and rerankers — covering the full path from synthetic data through fine-tuning to evaluation

**Scope of v1:** Bi-encoder embedding training on `sentence-transformers` with a plug-and-play model registry
**Training Focus:** Retrieval models (encoder + decoder-as-embedder), reusing the existing method-dispatch, LoRA, and evaluation seams

---

## Table of Contents

1. [Goal](#1-goal)
2. [Current State](#2-current-state)
3. [Plug-and-Play Model Registry](#3-plug-and-play-model-registry)
4. [Pillar 1: Synthetic Data Generation](#4-pillar-1-synthetic-data-generation)
5. [Pillar 2: Bi-Encoder Training (v1 Core)](#5-pillar-2-bi-encoder-training-v1-core)
6. [Pillar 3: Retrieval Evaluation](#6-pillar-3-retrieval-evaluation)
7. [Pillar 4: Adapters and Rerankers Over a Frozen Base](#7-pillar-4-adapters-and-rerankers-over-a-frozen-base)
8. [Dependencies and Isolation](#8-dependencies-and-isolation)
9. [Documentation and Skill Updates](#9-documentation-and-skill-updates)
10. [Phased Rollout](#10-phased-rollout)
11. [Open Items for Sign-Off](#11-open-items-for-sign-off)

---

## 1. Goal

Add a third training family to Synaptic-Tuner — **retrieval models** — alongside the existing
causal-LM families (SFT / KTO / GRPO):

1. **Bi-encoder embeddings** — generate synthetic retrieval data, fine-tune an embedding model
   (full, LoRA, or frozen-head), and evaluate it with retrieval metrics. *(v1)*
2. **Adapters over an existing base** — train a small swappable layer (LoRA on the encoder, or a
   projection/MLP head over a frozen base) instead of a full fine-tune. *(v1 supports `lora` and
   `frozen_head` modes; serving registry in v2)*
3. **Reranker** — a second-stage cross-encoder (or frozen-head) that re-scores the bi-encoder's
   top-k, testable in complete isolation from the embedder. *(v2)*

**Design principle (researcher plug-and-play):** every architecture-specific detail
(family, pooling, normalization, query/passage prompt prefixes, LoRA target modules, Matryoshka
dims, `trust_remote_code`) lives in a **model registry YAML**. Supporting a new base model is one
registry entry, not a Python change.

---

## 2. Current State

What we reuse versus what we build.

### Reusable as-is
- **Method dispatch by convention.** `shared/utilities/paths.py:11` —
  `TRAINING_METHODS = ("sft", "kto", "grpo", "dpo")`. A new method is the string plus a
  `Trainers/<method>/` dir with `train_<method>.py` + `configs/config.yaml`. CLI routing
  (`tuner/cli/router.py`), the RTX/Mac backends, recipes (`Trainers/recipes/`), and the HF Jobs
  backend then pick it up automatically.
- **LoRA + merge + upload + cloud.** `peft` is already a dependency. `shared/model_loading/merge.py`,
  `shared/upload/strategies/{lora,merged_16bit}.py`, and
  `tuner/backends/training/cloud/hf_jobs_backend.py` are method-agnostic.
- **SynthChat is fully config-driven.** New data types are new scenario/rubric YAML plus a
  generation path; the judge-improve engine, `StreamingResultWriter`, label framework, and the
  `shared/llm` client pool all carry over.
- **Evaluator pluggable verifier seam.** `shared/verifiers/builtins/` (next to `assertion_verifier`,
  `llm_judge`) plus the `shared/experiment_tracking/` registry (`run_type="evaluation"`).

### New work (and one correction)
- ⚠️ **Embedding/reranker models are encoder (BERT-family) or decoder-as-embedder architectures
  with pooling + contrastive/triplet/ranking losses. The Unsloth `FastLanguageModel` path used by
  SFT/KTO does NOT apply.** v1 builds a genuinely separate trainer on `sentence-transformers`
  (native PEFT/LoRA, native losses, native evaluators). PEFT works on encoders, so the *LoRA story*
  survives — just not the Unsloth wrapper.
- **No retrieval metrics** anywhere (recall@k, MRR, nDCG, MAP). New module + verifier required.
- **New deps** (`sentence-transformers`, `faiss-cpu`, optional `mteb`) must be **method-local**, not
  global — the Unsloth/transformers pins are tightly version-locked (see SFT `requirements.txt`).

---

## 3. Plug-and-Play Model Registry

This is the core of the design. New file:
**`Trainers/embedding/configs/model_registry.yaml`**.

Each entry fully describes how to load, prompt, and adapt a base model so the trainer code stays
architecture-agnostic. `sentence-transformers` already understands most of this natively (it loads a
folder with `modules.json` = Transformer + Pooling + Normalize, and supports per-input `prompts`);
the registry maps a friendly name to ST config + sensible LoRA targets + prompt templates.

```yaml
# Trainers/embedding/configs/model_registry.yaml
models:
  bge-base-en:
    hf_id: BAAI/bge-base-en-v1.5
    family: bert                 # bert | xlm-roberta | decoder
    embedding_type: bi_encoder   # bi_encoder | cross_encoder
    pooling: cls                 # mean | cls | last_token | weighted_mean
    normalize: true
    max_seq_length: 512
    default_dim: 768
    matryoshka_dims: []          # optional truncation dims
    query_prompt: "Represent this sentence for searching relevant passages: "
    passage_prompt: ""
    lora_target_modules: [query, key, value, dense]
    trust_remote_code: false

  e5-base:
    hf_id: intfloat/e5-base-v2
    family: bert
    embedding_type: bi_encoder
    pooling: mean
    normalize: true
    max_seq_length: 512
    default_dim: 768
    query_prompt: "query: "      # E5 requires query:/passage: prefixes
    passage_prompt: "passage: "
    lora_target_modules: [query, key, value, dense]

  gte-base:
    hf_id: thenlper/gte-base
    family: bert
    pooling: mean
    normalize: true
    max_seq_length: 512
    lora_target_modules: [query, key, value, dense]

  qwen3-embedding-0.6b:
    hf_id: Qwen/Qwen3-Embedding-0.6B
    family: decoder              # LLM2Vec-style: last-token pooling, instruct prompt
    pooling: last_token
    normalize: true
    max_seq_length: 8192
    default_dim: 1024
    query_prompt: "Instruct: Given a query, retrieve relevant passages\nQuery: "
    passage_prompt: ""
    lora_target_modules: [q_proj, k_proj, v_proj, o_proj]
    trust_remote_code: true
```

New file: **`Trainers/embedding/src/registry.py`** — loader + `EmbeddingModelSpec` dataclass,
mirroring `tuner/discovery/base_models.py`. Validates entries, resolves defaults, and exposes
`list_models()` / `get_spec(name)`. The trainer, evaluator, and serving layer all read specs from
here so "what does this model need" is defined once.

Researcher workflow to add a model: drop a new block in `model_registry.yaml`. No Python change.

---

## 4. Pillar 1: Synthetic Data Generation

Add **non-chat data types** to SynthChat, reusing the judge-improve engine.

| File | Change |
|------|--------|
| `SynthChat/scenarios/embedding_triplets.yaml` | **New.** `type: triplet` scenarios with a 3-stage generation path: `query → positive → hard_negative(s)`. Optional `type: graded_pairs` emitting relevance 0–3 for nDCG. |
| `SynthChat/rubrics/triplet_quality.yaml` | **New.** Judge query clarity, positive relevance, and that negatives are *plausible-but-wrong* (hard), not trivially unrelated. |
| `SynthChat/rubrics/negative_diversity.yaml` | **New.** Penalize near-duplicate negatives; reward a mix of hard/medium/easy. |
| `SynthChat/generator.py` | **Extend.** Add `_generate_triplet_example()` dispatched on `scenario.type == "triplet"`. Reuses `PromptRenderer`, the LLM client pool, and the judge-improve loop. |
| `SynthChat/result_writer.py` | **Extend.** Allow non-`conversations` row schemas (write `{query, positive, negatives, metadata}`). |
| `SynthChat/labeling.py` | **Extend.** Triplet difficulty / negative-type labels into `metadata.labels.flat`. |

**Output schema** (`Datasets/embedding/*.jsonl`):
```json
{"query":"...","positive":"...","negatives":["...","..."],
 "metadata":{"type":"triplet","scenario_key":"...","difficulty":"hard","iterations":2,"success":true}}
```

**Hard-negative mining (quality multiplier).** New script
**`tools/embedding/mine_hard_negatives.py`**: load a base embedder (via the registry) + build a FAISS
index over the generated corpus, then attach near-miss passages as additional negatives. This is the
difference between a toy dataset and a useful one. Config-driven (top-k, similarity floor/ceiling to
avoid false negatives).

---

## 5. Pillar 2: Bi-Encoder Training (v1 Core)

### New trainer directory: `Trainers/embedding/`

```
Trainers/embedding/
├── train_embedding.py          # entry point (mirrors train_sft.py shape)
├── requirements.txt            # sentence-transformers, faiss-cpu, datasets (METHOD-LOCAL)
├── configs/
│   ├── config.yaml             # default embedding run config
│   └── model_registry.yaml     # plug-and-play base models (Section 3)
└── src/
    ├── registry.py             # EmbeddingModelSpec loader
    ├── model_loader.py         # ST load + adapter mode (full | lora | frozen_head)
    ├── data_loader.py          # triplet/pairs JSONL → ST dataset + prompt prefixing
    ├── losses.py               # config → ST loss mapping
    ├── evaluation.py           # in-training ST IR evaluator (recall@k/MRR/nDCG on a dev split)
    └── callbacks.py            # adapt Trainers/shared/callbacks to the ST trainer
```

### Adapter modes (the "don't fully fine-tune" axis)

In `src/model_loader.py`:
- `full` — fine-tune the whole encoder.
- `lora` — `peft.LoraConfig` applied via ST's PEFT integration using `spec.lora_target_modules`.
  Output is a small adapter; reuses existing merge/upload strategies.
- `frozen_head` — freeze the base, train only an appended Dense projection/MLP. The literal "simple
  thing you put over it." Smallest, cheapest, CPU-trainable.

### Config-selectable losses

In `src/losses.py`:
- `MultipleNegativesRankingLoss` (in-batch negatives — the workhorse; pairs or triplets).
- `TripletLoss` (explicit margins).
- `CoSENTLoss` / `CosineSimilarityLoss` (graded relevance data).
- Optional `MatryoshkaLoss` wrapper when `matryoshka_dims` set.

### Default config shape

```yaml
# Trainers/embedding/configs/config.yaml
model:
  registry_name: bge-base-en     # resolved against model_registry.yaml
  adapter_mode: lora             # full | lora | frozen_head
training:
  loss: multiple_negatives_ranking
  batch_size: 64                 # large batch matters for in-batch negatives
  epochs: 1
  learning_rate: 2.0e-5
  warmup_ratio: 0.1
lora:
  r: 16
  alpha: 32
  dropout: 0.05
dataset:
  local_file: Datasets/embedding/my_triplets.jsonl
  eval_split: 0.05
evaluation:
  metrics: [recall@10, mrr@10, ndcg@10]
```

### Wiring into existing surfaces

| File | Change |
|------|--------|
| `shared/utilities/paths.py:11` | Add `"embedding"` to `TRAINING_METHODS`. |
| `tuner/backends/training/rtx_backend.py` | `get_available_methods()` returns `[..., "embedding"]`; config-filename mapping if needed. |
| `tuner/backends/training/mac_backend.py` | Same (MPS works for ST). |
| `Trainers/recipes/embedding_bge_base_smoke.yaml` | **New** recipe (`method: embedding`, `target: local|both`) matching the existing recipe schema, with a dedicated `job.image` / `setup.pip` for the ST stack. |
| `tuner/backends/training/cloud/` | New image profile / requirements overlay for the ST stack, isolated from the Unsloth trainer runtime. |

Output layout reuses the canonical
`embedding_output/YYYYMMDD_HHMMSS/{final_model,checkpoints,logs,training_lineage.json}`.

---

## 6. Pillar 3: Retrieval Evaluation

| File | Change |
|------|--------|
| `shared/ml/retrieval_metrics.py` | **New.** Pure functions: `recall_at_k`, `mrr`, `ndcg_at_k`, `map_score`. No heavy deps. |
| `shared/verifiers/builtins/retrieval_verifier.py` | **New.** Verifier registered as `retrieval_metrics`, parallel to `llm_judge`. Given a corpus + `{query → relevant_ids}` qrels, embeds/retrieves (FAISS), scores. Returns a `RetrievalValidationResult`. |
| `Evaluator/runner.py` | **Extend.** Optional `retrieval` field on `EvaluationRecord`; route scenarios with a `retrieval_config` block to the new verifier. |
| `Evaluator/config/scenarios/embedding_retrieval.yaml` | **New.** Config-first scenario declaring corpus path, queries, qrels, metrics. |
| `Evaluator/recipes/embedding_retrieval_eval.yaml` | **New** eval recipe. |
| `shared/experiment_tracking/adapters.py` | **Extend.** Map retrieval results → `RunRecord` (`primary_metric: ndcg@10`) so embedding runs sit beside SFT/KTO in the registry. |

Scenario shape:
```yaml
tests:
  - id: retrieval_smoke
    retrieval_config:
      corpus: Datasets/embedding/corpus.jsonl
      queries: Datasets/embedding/queries.jsonl
      qrels: Datasets/embedding/qrels.jsonl
      metrics: [recall@10, mrr@10, ndcg@10]
      model: { registry_name: bge-base-en }   # or a trained run/adapter path
```

**Later:** a thin **MTEB/BEIR** adapter for standardized benchmarking
(`tools/embedding/run_mteb.py`).

---

## 7. Pillar 4: Adapters and Rerankers Over a Frozen Base

Designed now, built in v2. Make "an adapter/reranker over an existing model" a first-class,
composable *inference* concept:

- **Adapter/reranker registry** mirroring the model registry: `{base_model, adapter_or_head, type}`.
- **Two-stage pipeline:** base bi-encoder retrieves top-k → reranker (cross-encoder or frozen-head)
  re-scores → final ranking. Swap rerankers over the *same frozen base* and A/B them with no base
  retraining.
- **`Trainers/reranker/`** trainer (cross-encoder pairwise/listwise) reusing the same registry,
  data, and eval seams. Reranker quality is measured as **nDCG lift over base retrieval** in the
  retrieval verifier.

This is intentionally deferred so v1 ships the foundation (a trained embedder + retrieval eval) that
rerankers rank on top of.

---

## 8. Dependencies and Isolation

- `Trainers/embedding/requirements.txt`: `sentence-transformers`, `faiss-cpu` (or `faiss-gpu`),
  `datasets`. **Method-local** — do not touch the global Unsloth/transformers pins.
- `Evaluator/requirements.txt`: add `faiss-cpu` (+ optional `mteb`).
- Cloud: a dedicated ST image profile / pip overlay, isolated from the Unsloth trainer runtime
  (same discipline as the existing bucket-sync overlay split).

---

## 9. Documentation and Skill Updates

So the workflow is canonical, not ad hoc:

- `.skills/fine-tuning/reference/dataset-formats.md` — add the triplet/graded-pairs schemas.
- `.skills/fine-tuning/SKILL.md` — add the `embedding` method, recipe, and CLI examples.
- `.skills/evaluation/` — document the retrieval verifier + scenario schema.
- `.skills/synethetic-data-generation/` — document triplet scenarios + hard-negative mining.
- Sync mirrors: `python3 .skills/scripts/sync_skill_trees.py`.

---

## 10. Phased Rollout

| Phase | Deliverable |
|-------|-------------|
| **1 (v1)** | Model registry + `Trainers/embedding/` trainer (full/LoRA/frozen-head) + `retrieval_metrics` + retrieval verifier/eval + experiment-tracking wiring + smoke recipe. Train & evaluate a bge/e5 model end-to-end on a hand-made tiny triplet set. |
| **2** | SynthChat triplet generation + hard-negative mining; generate a real dataset and retrain. |
| **3** | `Trainers/reranker/` cross-encoder + adapter/reranker serving registry + two-stage retrieve→rerank pipeline + nDCG-lift eval. |
| **4** | Cloud HF Jobs image profile for the ST stack; MTEB/BEIR benchmarking adapter. |

---

## 11. Open Items for Sign-Off

1. **FAISS flavor:** `faiss-cpu` default (portable, fine for dev/eval corpora) vs `faiss-gpu` for
   large-corpus hard-negative mining. Recommend `faiss-cpu` for v1.
2. **Initial registry seed:** which exact base models to ship entries for first (recommend
   `bge-base-en`, `e5-base`, `gte-base` for the BERT path + `qwen3-embedding-0.6b` for the decoder
   path, to exercise both families).
3. **Reranker family for v2:** cross-encoder (recommended) vs late-interaction (ColBERT) vs
   LLM-as-reranker.
