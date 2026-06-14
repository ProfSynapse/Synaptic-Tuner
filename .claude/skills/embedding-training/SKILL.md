---
name: embedding-training
description: Complete reference for the embedding fine-tuning pipeline (SentenceTransformers bi-encoders) — the `embedding` training method, model registry, dual loader (Unsloth fast path + ST fallback), adapter modes (full/lora/frozen_head), triplet/pairs dataset format, retrieval evaluation (recall@k/MRR/nDCG/MAP via the `retrieval` verifier), and the local-Docker smoke recipe. Use when training a retrieval embedder, picking a base model + adapter mode, preparing triplet data, or evaluating a fine-tuned embedder against a labeled corpus. This skill is about USING the embedding pipeline via CLI and YAML — never modifying source code.
allowed-tools: Read, Bash, Write, Grep, Glob
---

# Embedding Training Pipeline

Fine-tune SentenceTransformer bi-encoders for retrieval with the `embedding`
training method. This skill covers the end-to-end loop: obtain triplets → train
(pick a base + adapter mode) → evaluate retrieval against a labeled corpus →
read metrics → refine. Every step is a copy-pasteable CLI command or a YAML
edit; there is no ad hoc Python.

The `embedding` method is a sibling to `sft`/`kto`/`grpo`/`dpo` — it is wired
into the same method registry (`shared/utilities/paths.py` `TRAINING_METHODS`),
the same recipe/local-run path (`tuner.py local-run`), and the same output
layout (`embedding_output/<timestamp>/...`). What differs is the model family
(encoder, not causal-LM), the loss (contrastive, not next-token), and the
evaluation axis (corpus-level retrieval metrics, not per-completion correctness).

## Quick Reference

| Task | Command |
|------|---------|
| Local Docker embedding smoke | `python tuner.py local-run --job-config Trainers/recipes/embedding_bge_base_smoke.yaml --yes` |
| List registered models | inspect `Trainers/embedding/configs/model_registry.yaml` |
| Validate a triplet dataset shape | run the smoke recipe `--dry-run` (loader validates the JSONL) |
| Retrieval eval (untrained base) | `python -m Evaluator.cli --scenario embedding_retrieval_smoke.yaml` (see `reference/retrieval-eval.md`) |
| Retrieval eval (trained adapter) | swap `model.registry_name` for `model.path` in the scenario |

> The exact `local-run` / eval flags follow the same surface as the SFT path —
> verify against `tuner/cli/parser.py` or `--help` before scripting; do not
> guess flags from memory.

## Method at a Glance

| Aspect | Embedding method |
|--------|------------------|
| **Model family** | encoder bi-encoder (BERT / XLM-RoBERTa / decoder-as-embedder) |
| **Loss** | `multiple_negatives_ranking` (MNRL), optionally Matryoshka-wrapped |
| **Dataset** | triplets `{query, positive, negatives}` or pairs `{query, positive}` |
| **Adapter modes** | `full` · `lora` · `frozen_head` (no `qlora` in v1) |
| **Evaluation** | corpus-level retrieval: recall@k / MRR / nDCG@k / MAP against qrels |
| **Output** | `embedding_output/<timestamp>/` (adapter or merged encoder) |

## The Registry (SSOT)

Every base model the pipeline can train or load is a block in
`Trainers/embedding/configs/model_registry.yaml`. Adding a model is **one YAML
block — no Python change** (the config-driven rule). Each block describes how to
LOAD, PROMPT, and ADAPT the model. The four seed models:

| `registry_name` | HF id | Family | Pooling | Prompt required? | Notes |
|-----------------|-------|--------|---------|------------------|-------|
| `bge-base-en` | `BAAI/bge-base-en-v1.5` | bert | cls | no | query prompt optional |
| `e5-base` | `intfloat/e5-base-v2` | bert | mean | **yes** | requires `query:`/`passage:` prefixes |
| `gte-base` | `thenlper/gte-base` | bert | mean | no | no prompts |
| `qwen3-embedding-0.6b` | `Qwen/Qwen3-Embedding-0.6B` | decoder | last_token | no | native 32k, MRL dims `[1024..128]` |

See `reference/embedding-training.md` for the full spec schema (pooling,
`lora_target_modules`, `matryoshka_dims`, `prompt_required`) and how the dual
loader picks the fast vs fallback path.

## Adapter Modes

| Mode | What trains | LoRA? | When |
|------|-------------|-------|------|
| `full` | the whole encoder | no | best quality, most VRAM |
| `lora` | a small adapter on the encoder | yes | default; reuses merge/upload |
| `frozen_head` | only an appended Dense/MLP head | no | smallest, CPU-trainable smoke |

`frozen_head` carries a **compare-to-ST-baseline gate (R6)**: before training the
head, the frozen-base embeddings are asserted to match a plain
`SentenceTransformer` baseline within tolerance (guards against Unsloth
nonstandard-pooling drift). The smoke test enforces this.

There is **no `qlora` mode in v1** — the enum is exactly `{full, lora,
frozen_head}`. A `qlora` value in config raises `ValueError` ("deferred to a
later phase").

## Dataset Format (triplets / pairs)

Training data is JSONL. The loader (`Trainers/embedding/src/data_loader.py`)
accepts field aliases and explodes multi-negative records into one row per
negative:

```jsonl
{"query": "How do I reset my password?", "positive": "Open Settings → Security → Reset Password.", "negatives": ["Our refund policy allows returns within 30 days.", "Dark mode is under Appearance."]}
{"query": "What payment methods are accepted?", "positive": "We accept Visa, Mastercard, PayPal, and Apple Pay."}
```

- Anchor aliases: `query` / `anchor` / `question`. Positive aliases: `positive` /
  `pos`. Negative aliases: `negative` / `negatives` / `neg` (scalar or list).
- A `negatives` list explodes into one `(anchor, positive, negative)` row per
  negative (standard ST hard-negative shape).
- Pair records (no negatives) yield `(anchor, positive)` — but do NOT mix pair
  and triplet records in one file; the loader drops pair rows when any record
  has negatives.
- The spec's `query_prompt` is prepended to anchors and `passage_prompt` to
  positives/negatives. E5-style models **require** these prefixes.

Checked-in canonical fixtures live at `Datasets/embedding/examples/`:
`triplets_smoke.jsonl` (training) and `corpus.jsonl` / `queries.jsonl` /
`qrels.jsonl` (retrieval eval). See `reference/triplet-data.md`.

## Retrieval Evaluation

Embedding models are scored **corpus-level**, not per-completion. The `retrieval`
verifier embeds a document corpus and a query set, runs FAISS top-k
nearest-neighbour retrieval, and aggregates recall@k / MRR / nDCG@k / MAP against
qrels, applying a pass/warn/fail threshold ladder. Thresholds live in scenario
YAML (config-driven), never in Python:

```yaml
retrieval_config:
  corpus: Datasets/embedding/examples/corpus.jsonl
  queries: Datasets/embedding/examples/queries.jsonl
  qrels: Datasets/embedding/examples/qrels.jsonl
  metrics: [recall@10, mrr@10, ndcg@10]
  model: { registry_name: bge-base-en }   # or model.path to a trained adapter
  thresholds:
    min: { ndcg@10: 0.30, recall@10: 0.40 }   # below min → fail
    warn_margin: 0.05                          # within [min, min+margin) → warn
  primary_metric: ndcg@10
```

See `reference/retrieval-eval.md` for the metric-spec grammar (`<metric>@<k>`),
the threshold ladder, and how to point the eval at a trained run.

## The Smoke Recipe

`Trainers/recipes/embedding_bge_base_smoke.yaml` trains a small bge-base-en LoRA
adapter for a few steps on the local-Docker path. It rides the modern Unsloth
image (which ships `transformers 4.57.1`) and layers the embedding deps
(`sentence-transformers`, `faiss-cpu`, `datasets`) via a `setup.pip` overlay —
**without** re-pinning transformers and without touching the legacy SFT
`transformers 4.45.2` island.

```bash
python tuner.py local-run \
  --job-config Trainers/recipes/embedding_bge_base_smoke.yaml \
  --yes
```

> **Overlay pins are determined by the cloud smoke, not invented.** The recipe's
> `setup.pip` entries are intentionally unpinned (`sentence-transformers`,
> `faiss-cpu`, `datasets` with `TBD-PENDING-SMOKE-TEST` comments). The repo has a
> documented history of image/numpy/transformers mismatches, so the exact pins
> must be captured empirically from the FIRST WORKING cloud run on the modern
> Unsloth image and only then written back as `pkg==X.Y.Z`. Do not hardcode
> speculative versions. See `reference/embedding-training.md` → "Overlay pins".

## Progressive Reference

| Reference | When to Load | Path |
|-----------|-------------|------|
| **Embedding Training** | Registry schema, dual loader, adapter modes, training config, overlay pins | `reference/embedding-training.md` |
| **Retrieval Eval** | Metric grammar, threshold ladder, evaluating a trained adapter | `reference/retrieval-eval.md` |
| **Triplet Data** | Authoring triplets/pairs + retrieval fixtures, field aliases, validation | `reference/triplet-data.md` |

## Cross-References

| Need | Skill |
|------|-------|
| SFT/KTO/GRPO training, local-run + cloud mechanics | `fine-tuning` |
| Scenario authoring, eval CLI, results | `evaluation` |
| Generating synthetic triplets (Phase 2 forward ref) | `synethetic-data-generation` |
| Merging/uploading a trained embedder | `upload-deployment` |

## Tips

- Start with `bge-base-en` + `lora` — the cheapest path that reuses
  merge/upload. Switch to `frozen_head` for a CPU-only smoke.
- `e5-base` REQUIRES query/passage prompts; the loader applies them from the
  registry, so use the registry name, never a bare HF id, for E5.
- Heavy GPU training is NOT a CI gate; the CPU fallback path is what CI runs.
- Keep the smoke fast: a handful of `max_steps` is enough to prove the method
  wires end-to-end. Bump for a real run.
- Always `--dry-run` first to validate the dataset shape and config without
  training.
