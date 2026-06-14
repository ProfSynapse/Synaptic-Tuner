# Embedding & Reranker Pipeline: Retrieval-Model Training Design

**Document Version:** 1.1
**Created:** 2026-06-14
**Updated:** 2026-06-14
**Status:** Proposal (awaiting review) — no code written yet
**Purpose:** Blueprint for adding retrieval-model training to Synaptic-Tuner — bi-encoder embeddings, LoRA/frozen-head adapters over a base, and rerankers — covering the full path from synthetic data through fine-tuning to evaluation

**Scope of v1:** Bi-encoder embedding training on `sentence-transformers` (accelerated by Unsloth's `FastSentenceTransformer`) with a plug-and-play model registry
**Training Focus:** Retrieval models (encoder + decoder-as-embedder), reusing the existing method-dispatch, LoRA, merge/upload, and evaluation seams
**v1.1 change:** Incorporates the finding that Unsloth shipped a native embedding/reranker training path in its 2026 release (Section 2) — the trainer rides the existing Unsloth stack instead of a separate one. Adds a runnable skill + example fixtures with a test-and-refine loop (Section 9).

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
9. [Skills, Examples, and the Test-Refine Loop](#9-skills-examples-and-the-test-refine-loop)
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

### New work
- **Embedding/reranker models are encoder (BERT-family) or decoder-as-embedder architectures with
  pooling + contrastive/triplet/ranking losses.** This is a different training shape from the
  causal-LM `FastLanguageModel` path used by SFT/KTO — it needs its own trainer, losses, and
  evaluators. Crucially, Unsloth now provides that path directly (see below), so we build on
  `sentence-transformers` accelerated by Unsloth rather than a fully separate stack.
- **No retrieval metrics** anywhere (recall@k, MRR, nDCG, MAP). New module + verifier required.
- **New deps** (`sentence-transformers`, `faiss-cpu`, optional `mteb`) added **method-local**, not
  global — the trainer-runtime pins are tightly version-locked (see SFT `requirements.txt`).

### Unsloth has a native embedding/reranker path (2026 research finding)

Research (June 2026) confirms Unsloth's first 2026 release added embedding training, built with
Hugging Face, and this materially simplifies the design because the repo already runs on Unsloth
images and drives Unsloth's save/merge API.

- **`FastSentenceTransformer`** — a new class that mirrors the `sentence-transformers` API
  (`from_pretrained(...)`, `for_inference=True` at inference) and integrates with the standard
  `SentenceTransformerTrainer`. Supports training **embedding, classifier, BERT, and reranker**
  models ~1.8–3.3x faster, with ~20% less VRAM and 2x context vs FA2, no accuracy loss.
- **LoRA supported**; **QLoRA is WIP** (bitsandbytes 4-bit incompatibility with recent
  `transformers`). EmbeddingGemma-300M trains with LoRA on ~6 GB VRAM (QLoRA ~3 GB once stable).
- **Named support:** `BAAI/bge-large-en`, EmbeddingGemma, and **Matryoshka** embedding models.
- **Save/export reuses what the repo already does:** `save_pretrained()` (adapters),
  `save_pretrained_merged()`, `push_to_hub()`, `push_to_hub_merged()`, `push_to_hub_gguf()`. The
  repo's `shared/model_loading/merge.py` already calls `save_pretrained_merged` for Unsloth models,
  so the embedding merge/upload path drops straight into the existing strategies.
- **Loss in examples:** `CachedMultipleNegativesRankingLoss` (memory-efficient in-batch negatives).
- **Version floor:** `transformers==5.1.0`, `trl==0.27.1`, PyTorch 2.9+. The repo's modern Unsloth
  image (`unsloth/unsloth:2026.1.2-pt2.9.0-cu12.8`, already referenced in the fine-tuning skill)
  satisfies this; the legacy SFT pin (`transformers 4.45.2`) does not, so embedding runs use the
  modern image profile, not the legacy one.
- **Deploy targets:** transformers, sentence-transformers, LangChain, Weaviate, TEI, vLLM,
  llama.cpp — i.e. trained adapters/models slot into the existing vLLM/serving story.
- **Caveat:** Unsloth warns that custom heads / nonstandard pooling need verification — relevant to
  the `frozen_head` adapter mode (Section 5) and cross-encoder rerankers (Section 7), which should be
  validated against a plain `SentenceTransformer` baseline.

**Design consequence:** the trainer targets the `sentence-transformers` API and loads via
`FastSentenceTransformer` when available (fast path on the Unsloth image), falling back to a plain
`SentenceTransformer` otherwise (e.g. Mac/MPS, CPU dev). Same training code, two loaders.

**Sources:**
[Unsloth 2026 update](https://unslothai.substack.com/p/unsloth-2026-update-faster-moe) ·
[Unsloth discussion #4020 (embedding support)](https://github.com/unslothai/unsloth/discussions/4020) ·
[Unsloth embedding fine-tuning docs](https://unsloth.ai/docs/basics/embedding-finetuning) ·
[sentence-transformers × Unsloth integration](https://sbert.net/examples/sentence_transformer/training/unsloth/README.html)

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
    ├── model_loader.py         # dual loader: FastSentenceTransformer (fast) | SentenceTransformer (fallback) + adapter mode
    ├── data_loader.py          # triplet/pairs JSONL → ST dataset + prompt prefixing
    ├── losses.py               # config → ST loss mapping
    ├── evaluation.py           # in-training ST IR evaluator (recall@k/MRR/nDCG on a dev split)
    └── callbacks.py            # adapt Trainers/shared/callbacks to the ST trainer
```

The trainer targets the standard `sentence-transformers` `SentenceTransformerTrainer` +
`SentenceTransformerTrainingArguments`, so the training loop is identical regardless of loader.

### Dual loader (fast path + fallback)

In `src/model_loader.py`:
- **Fast path:** `from unsloth import FastSentenceTransformer; FastSentenceTransformer.from_pretrained(spec.hf_id, ...)`
  on the modern Unsloth image — 1.8–3.3x faster, ~20% less VRAM. Exposes `save_pretrained_merged` /
  `push_to_hub*`, so it plugs straight into `shared/model_loading/merge.py` and the existing upload
  strategies.
- **Fallback:** plain `SentenceTransformer(spec.hf_id, ...)` for Mac/MPS and CPU dev, where Unsloth's
  CUDA path isn't available. Selected automatically by capability probe.

### Adapter modes (the "don't fully fine-tune" axis)

- `full` — fine-tune the whole encoder.
- `lora` — `peft.LoraConfig` using `spec.lora_target_modules` (Unsloth LoRA on the fast path; PEFT on
  the fallback). Output is a small adapter; reuses existing merge/upload strategies. *(QLoRA deferred
  until Unsloth's 4-bit embedding path stabilizes — see Section 2.)*
- `frozen_head` — freeze the base, train only an appended Dense projection/MLP. The literal "simple
  thing you put over it." Smallest, cheapest, CPU-trainable. Validate output embeddings against a
  plain-`SentenceTransformer` baseline (Unsloth's nonstandard-pooling caveat).

### Config-selectable losses

In `src/losses.py`:
- `CachedMultipleNegativesRankingLoss` / `MultipleNegativesRankingLoss` (in-batch negatives — the
  workhorse; pairs or triplets; the cached variant is what Unsloth's example uses for memory).
- `TripletLoss` (explicit margins).
- `CoSENTLoss` / `CosineSimilarityLoss` (graded relevance data).
- Optional `MatryoshkaLoss` wrapper when `matryoshka_dims` set (Unsloth supports Matryoshka models).

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
| `Trainers/recipes/embedding_bge_base_smoke.yaml` | **New** recipe (`method: embedding`, `target: local|both`) matching the existing recipe schema, pinned to the modern Unsloth image with `setup.pip` adding `sentence-transformers` + `faiss-cpu`. |
| `tuner/backends/training/cloud/` | Reuse the modern Unsloth image profile (`unsloth/unsloth:2026.1.2-pt2.9.0-cu12.8` class) with an ST/faiss pip overlay — no separate base image needed. |

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
  data, and eval seams. Unsloth's 2026 path already lists **reranker** as a supported model type, so
  this rides the same `FastSentenceTransformer`/ST stack as the embedding trainer. Reranker quality
  is measured as **nDCG lift over base retrieval** in the retrieval verifier.

This is intentionally deferred so v1 ships the foundation (a trained embedder + retrieval eval) that
rerankers rank on top of.

---

## 8. Dependencies and Isolation

- `Trainers/embedding/requirements.txt`: `sentence-transformers`, `faiss-cpu` (or `faiss-gpu`),
  `datasets`. **Method-local** — do not touch the legacy SFT pins.
- `Evaluator/requirements.txt`: add `faiss-cpu` (+ optional `mteb`).
- **Runtime/image:** ride the **modern Unsloth image** (`transformers>=5.1.0`, `trl>=0.27.1`,
  torch 2.9+) that `FastSentenceTransformer` requires — the same image class the repo's current
  Qwen3.5 recipes already use. Add `sentence-transformers` + `faiss-cpu` as a pip overlay; do not
  perturb the legacy `transformers 4.45.2` SFT pin. Keep Buckets-only Hub packages off the trainer
  `PYTHONPATH`, same discipline as the existing bucket-sync overlay split.

---

## 9. Skills, Examples, and the Test-Refine Loop

The workflow must be runnable and iterable by a researcher from day one — canonical skills plus tiny
checked-in fixtures so we can train/evaluate/refine on examples before committing to a real dataset.
`.skills/` is the canonical source; mirrors are regenerated, never hand-edited.

### New canonical skill: `.skills/embedding-training/SKILL.md`
A single end-to-end skill (in the spirit of `case-studies`) that ties the three subsystems together
and is the entry point for "how do I train and test an embedding model here":
- Generate triplets (SynthChat) → train (`embedding` method, pick base from the registry, pick
  adapter mode) → evaluate retrieval → read metrics → refine (mine harder negatives, change loss,
  swap base) → re-run. Each step a copy-pasteable CLI command, no ad hoc Python.
- Documents the registry (how to add a base model), the adapter modes, and the dual loader
  (Unsloth fast path vs ST fallback).
- A **progressive reference** subtree mirroring the fine-tuning skill: `reference/embedding-training.md`,
  `reference/retrieval-eval.md`, `reference/triplet-data.md`.

### Targeted updates to existing skills
- `.skills/fine-tuning/SKILL.md` + `reference/dataset-formats.md` — add the `embedding` method,
  recipe, CLI examples, and the triplet/graded-pairs schemas.
- `.skills/evaluation/` — document the `retrieval_metrics` verifier + scenario schema.
- `.skills/synethetic-data-generation/` — document triplet scenarios + hard-negative mining.
- After edits: `python3 .skills/scripts/sync_skill_trees.py` to refresh `.agents/skills` and
  `.claude/skills` mirrors.

### Checked-in example fixtures (the "test and refine on examples" loop)
Tiny, fast, committed so the whole pipeline runs in minutes on CPU/MPS and serves as both smoke test
and the worked example in the skill:
- `Datasets/embedding/examples/triplets_smoke.jsonl` — ~20 hand-written `{query, positive, negatives}`.
- `Datasets/embedding/examples/{corpus,queries,qrels}.jsonl` — a tiny labeled retrieval set for eval.
- `Trainers/recipes/embedding_bge_base_smoke.yaml` — `frozen_head` or small-`r` `lora`, a few steps.
- `Evaluator/config/scenarios/embedding_retrieval_smoke.yaml` + `Evaluator/recipes/embedding_retrieval_eval.yaml`.
- `tests/` smoke test asserting the trainer produces an adapter and the retrieval verifier emits
  finite recall@k/MRR/nDCG — wired so it runs in CI like the existing trainer smoke tests.

### The refine loop the skill teaches
1. Train on `triplets_smoke.jsonl` (fast path or fallback).
2. Evaluate against the smoke retrieval set → baseline recall@k / nDCG.
3. Inspect failures (queries with low rank-of-positive).
4. Refine: mine harder negatives, switch loss (e.g. MNRL → cached MNRL), bump `r`, or swap the
   registry base — one knob at a time.
5. Re-run; the experiment-tracking registry (Section 6) makes runs comparable.

---

## 10. Phased Rollout

| Phase | Deliverable |
|-------|-------------|
| **1 (v1)** | Model registry + `Trainers/embedding/` trainer (dual loader; full/LoRA/frozen-head) + `retrieval_metrics` + retrieval verifier/eval + experiment-tracking wiring + **`.skills/embedding-training/` skill + checked-in example fixtures + CI smoke test** (Section 9). Train & evaluate a bge/e5 model end-to-end on the smoke triplet set, then refine. |
| **2** | SynthChat triplet generation + hard-negative mining; generate a real dataset and retrain. |
| **3** | `Trainers/reranker/` cross-encoder (Unsloth supports reranker training) + adapter/reranker serving registry + two-stage retrieve→rerank pipeline + nDCG-lift eval. |
| **4** | Cloud HF Jobs runs on the modern Unsloth image + ST/faiss overlay; MTEB/BEIR benchmarking adapter. |

---

## 11. Open Items for Sign-Off

1. **FAISS flavor:** `faiss-cpu` default (portable, fine for dev/eval corpora) vs `faiss-gpu` for
   large-corpus hard-negative mining. Recommend `faiss-cpu` for v1.
2. **Initial registry seed:** which exact base models to ship entries for first (recommend
   `bge-base-en`, `e5-base`, `gte-base` for the BERT path + `qwen3-embedding-0.6b` for the decoder
   path, to exercise both families).
3. **Reranker family for v2:** cross-encoder (recommended) vs late-interaction (ColBERT) vs
   LLM-as-reranker.
