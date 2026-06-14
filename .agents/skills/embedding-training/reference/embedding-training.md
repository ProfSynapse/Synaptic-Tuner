# Embedding Training Reference

Deep-dive on the embedding model registry, the dual loader, adapter modes,
training config, and the overlay-pin discipline. Companion to `SKILL.md`.

---

## Model Registry (the SSOT)

`Trainers/embedding/configs/model_registry.yaml` is the single source of truth
for every base model. Each `models.<name>` block is validated into a frozen
`EmbeddingModelSpec` at load time (`Trainers/embedding/src/registry.py`); a typo
or out-of-domain value fails loudly with a `ValueError` naming the offending key.

### Spec schema (one YAML block → one `EmbeddingModelSpec`)

| Key | Type | Domain / default | Meaning |
|-----|------|------------------|---------|
| `hf_id` | str (required) | — | canonical HF model id |
| `fast_path_hf_id` | str / null | null | optional `unsloth/`-prefixed mirror for the fast path; null → use `hf_id` |
| `family` | str (required) | `bert` \| `xlm-roberta` \| `decoder` | drives loader + LoRA targets |
| `embedding_type` | str | `bi_encoder` \| `cross_encoder` (default `bi_encoder`) | cross-encoder reserved for Phase 3 |
| `pooling` | str | `mean` \| `cls` \| `last_token` \| `weighted_mean` | sentence pooling |
| `normalize` | bool | true | L2-normalize embeddings |
| `max_seq_length` | int | 512 | tokenizer truncation length |
| `default_dim` | int / null | null | output embedding dim |
| `matryoshka_dims` | list[int] | `[]` | MRL truncation dims; must be ≤ `default_dim` and sorted **descending** |
| `query_prompt` | str | `""` | prefix prepended to queries/anchors |
| `passage_prompt` | str | `""` | prefix prepended to passages/positives/negatives |
| `prompt_required` | bool | false | if true, BOTH prompts must be non-empty |
| `lora_target_modules` | list[str] | `[]` | modules the LoRA adapter targets |
| `lora_task_type` | str / null | null | `FEATURE_EXTRACTION` for decoder; null for encoder |
| `trust_remote_code` | bool | false | pass to the loader |

Validation rules enforced by `registry.py`:

- **Unknown keys fail** — catches typos (config-driven discipline).
- `family` and `pooling` must be in their enum domains.
- `matryoshka_dims` entries must each be ≤ `default_dim` and listed in
  descending order.
- `prompt_required: true` requires both `query_prompt` and `passage_prompt`
  non-empty (E5 needs `query:` / `passage:`).
- `family: decoder` requires `lora_task_type: FEATURE_EXTRACTION` — if omitted,
  the loader warns and defaults it on; a conflicting value raises.

### Adding a model

Add one block; no Python change:

```yaml
models:
  my-new-embedder:
    hf_id: org/my-embedder
    family: bert
    pooling: mean
    normalize: true
    max_seq_length: 512
    default_dim: 768
    query_prompt: ""
    passage_prompt: ""
    prompt_required: false
    lora_target_modules: [query, key, value, dense]
    lora_task_type: null
    trust_remote_code: false
```

The loader surface (the only thing callers import):

```python
from registry import load_registry, get_spec, list_models
specs = load_registry()            # {name: EmbeddingModelSpec}
spec  = get_spec("bge-base-en")    # one validated spec
names = list_models()              # sorted registry keys
```

---

## The Dual Loader (R1, R6)

`Trainers/embedding/src/model_loader.py` resolves a spec into a loaded model via
a **two-path** strategy:

- **Fast path** — Unsloth `FastModel`/`FastSentenceTransformer` when CUDA + a
  compatible Unsloth runtime are present (uses `resolved_fast_path_id()`).
- **Fallback path** — plain `sentence_transformers.SentenceTransformer` on CPU/MPS
  or when the fast path is unavailable. This is what CI exercises.

The capability probe that chooses between them **never raises** — every failure
mode (no CUDA, no unsloth, import error) degrades cleanly to the fallback. The
loader applies the spec's pooling, normalization, and prompts uniformly across
both paths, so a model trains/evaluates identically (within tolerance) regardless
of which path loaded it.

---

## Adapter Modes (R6, R8)

Set `model.adapter_mode` in the training config:

| Mode | Trains | LoRA | R6 gate |
|------|--------|------|---------|
| `full` | whole encoder (`full_finetuning=True` on the fast path) | no | — |
| `lora` | a small adapter (`target_modules = spec.lora_target_modules`) | yes | — |
| `frozen_head` | only an appended Dense/MLP head | no | **compare-to-ST-baseline** |

**`frozen_head` R6 gate:** before training the head, the pipeline asserts the
frozen-base embeddings match a plain `SentenceTransformer` baseline within
tolerance. This guards against Unsloth nonstandard-pooling drift. The smoke test
enforces it.

**No `qlora` in v1 (R8).** The enum is exactly `{full, lora, frozen_head}`. A
`qlora` value raises `ValueError` ("deferred to a later phase"). QLoRA for
embedding is a later-phase item.

---

## Training Config (§6.2)

`Trainers/embedding/configs/config.yaml` is the default run config; recipe YAML
or CLI flags override individual values.

```yaml
model:
  registry_name: bge-base-en       # key into model_registry.yaml
  adapter_mode: lora               # full | lora | frozen_head (NO qlora)

training:
  loss: multiple_negatives_ranking
  batch_size: 64
  epochs: 1
  learning_rate: 2.0e-5
  warmup_ratio: 0.1

lora:
  r: 16
  alpha: 32
  dropout: 0.05

dataset:
  local_file: Datasets/embedding/examples/triplets_smoke.jsonl
  eval_split: 0.05

evaluation:
  metrics: [recall@10, mrr@10, ndcg@10]
```

- `loss: multiple_negatives_ranking` (MNRL) is the contrastive default; with a
  registry that declares `matryoshka_dims`, the trainer wraps it in a Matryoshka
  loss so the model trains all truncation dims at once.
- `dataset.eval_split` carves an in-training eval set; `0.0` disables it.
- `evaluation.metrics` uses the canonical `<metric>@<k>` grammar (see
  `retrieval-eval.md`).

---

## Output Layout

The `embedding` method is registered in `TRAINING_METHODS`
(`shared/utilities/paths.py`), which auto-derives the canonical output dir:

```
embedding_output/YYYYMMDD_HHMMSS/
├── checkpoints/
├── logs/
├── final_model/          # adapter (lora/frozen_head) or merged encoder (full)
└── training_lineage.json
```

A `lora` adapter reuses the existing merge/upload path (the merge seam is
family-parametrized — `merge.py` threads `family="embedding"` for embedder
merges; the causal-LM default path is unchanged).

---

## Overlay Pins (the TBD discipline)

The embedding stack (`sentence-transformers`, `faiss-cpu`, `datasets`) is layered
on top of the modern Unsloth image via the recipe's `setup.pip` overlay. The
image ships `transformers 4.57.1`, so the overlay does **not** re-pin
transformers and never touches the legacy SFT `transformers 4.45.2` island.

**Exact pins are TBD-pending the cloud smoke test — never invented.** The repo
has a documented history of image/numpy/transformers mismatches, so:

1. Run the smoke recipe once on the modern Unsloth image (cloud / GPU).
2. Capture the FIRST WORKING version set from that run.
3. Write the pins back as `sentence-transformers==X.Y.Z`, `faiss-cpu==X.Y.Z`,
   `datasets==X.Y.Z` in:
   - `Trainers/recipes/embedding_bge_base_smoke.yaml` (`setup.pip`)
   - `Trainers/embedding/requirements.txt`
   - `Evaluator/requirements.txt` (for `faiss-cpu`)

Until then the entries stay unpinned with `TBD-PENDING-SMOKE-TEST` comments. Do
not hardcode speculative versions; an unverified pin against the wrong
transformers/numpy is worse than an unpinned floor.

> **Local CPU note:** the fallback path runs on `sentence-transformers 5.2.0` +
> `faiss-cpu` + `datasets` with `numpy 1.26.x` (verified in-worktree). That
> validates the package SET and the fixtures, but it is NOT the cloud overlay
> pin — the cloud image's transformers/numpy floor differs and must be captured
> from a real cloud run.
