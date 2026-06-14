# Phase 1 Contracts — Embedding & Reranker Pipeline

Binding interface spec. CODE implements against these signatures. Where this
conflicts with the blueprint, the PREPARE corrections (already folded in here)
win.

Conventions: Python signatures are illustrative of the **contract** (names,
arg/return types, semantics), not final implementation. YAML blocks are the
**schema** to honor.

---

## 1. Model Registry + `EmbeddingModelSpec` (R7)

**Files (new):**
- `Trainers/embedding/configs/model_registry.yaml` — the seed data.
- `Trainers/embedding/src/registry.py` — loader + dataclass.

### 1.1 `model_registry.yaml` schema

```yaml
# Trainers/embedding/configs/model_registry.yaml
# Each entry fully describes how to load, prompt, and adapt a base model.
# Adding a model is ONE block — no Python change (config-driven rule).
models:
  bge-base-en:
    hf_id: BAAI/bge-base-en-v1.5
    fast_path_hf_id: null         # optional unsloth/-prefixed mirror for the fast path (R7/§3.2); null → use hf_id
    family: bert                  # bert | xlm-roberta | decoder
    embedding_type: bi_encoder    # bi_encoder | cross_encoder  (cross_encoder reserved for Phase 3)
    pooling: cls                  # mean | cls | last_token | weighted_mean
    normalize: true
    max_seq_length: 512
    default_dim: 768
    matryoshka_dims: []           # optional truncation dims (empty = disabled)
    query_prompt: "Represent this sentence for searching relevant passages: "
    passage_prompt: ""
    prompt_required: false        # R7: bge query prompt is OPTIONAL (no prompt → slight degradation only)
    lora_target_modules: [query, key, value, dense]   # R7: verify via named_modules() in CODE smoke
    lora_task_type: null          # null for encoder (PEFT infers); FEATURE_EXTRACTION for decoder
    trust_remote_code: false

  e5-base:
    hf_id: intfloat/e5-base-v2
    family: bert
    embedding_type: bi_encoder
    pooling: mean
    normalize: true
    max_seq_length: 512
    default_dim: 768
    query_prompt: "query: "       # E5 REQUIRES query:/passage: prefixes
    passage_prompt: "passage: "
    prompt_required: true
    lora_target_modules: [query, key, value, dense]
    trust_remote_code: false

  gte-base:
    hf_id: thenlper/gte-base
    family: bert
    embedding_type: bi_encoder
    pooling: mean
    normalize: true
    max_seq_length: 512
    default_dim: 768
    query_prompt: ""
    passage_prompt: ""
    prompt_required: false
    lora_target_modules: [query, key, value, dense]
    trust_remote_code: false

  qwen3-embedding-0.6b:
    hf_id: Qwen/Qwen3-Embedding-0.6B
    family: decoder               # last-token pooling, instruct prompt
    embedding_type: bi_encoder
    pooling: last_token
    normalize: true
    max_seq_length: 32768         # R7: native 32k, NOT 8192
    default_dim: 1024             # R7: user-definable 32–1024 (MRL)
    matryoshka_dims: [1024, 768, 512, 256, 128]   # R7: MRL/Matryoshka is native here
    query_prompt: "Instruct: Given a query, retrieve relevant passages\nQuery: "
    passage_prompt: ""
    prompt_required: false
    # R7: full decoder target set incl. MLP-proj (materially affects adapter capacity)
    lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
    lora_task_type: FEATURE_EXTRACTION    # R7: required for decoder-as-embedder
    trust_remote_code: false      # R7: doc claimed true; HF card shows standard usage. CODE verifies; default false.
```

### 1.2 `EmbeddingModelSpec` dataclass contract

`Trainers/embedding/src/registry.py` — mirrors the spirit of
`tuner/discovery/base_models.py`. Frozen dataclass; loader validates and
resolves defaults.

```python
@dataclass(frozen=True)
class EmbeddingModelSpec:
    name: str                              # registry key (e.g. "bge-base-en")
    hf_id: str
    family: str                            # "bert" | "xlm-roberta" | "decoder"
    embedding_type: str = "bi_encoder"     # "bi_encoder" | "cross_encoder"
    pooling: str = "mean"                  # "mean" | "cls" | "last_token" | "weighted_mean"
    normalize: bool = True
    max_seq_length: int = 512
    default_dim: int | None = None
    matryoshka_dims: tuple[int, ...] = ()
    query_prompt: str = ""
    passage_prompt: str = ""
    prompt_required: bool = False
    lora_target_modules: tuple[str, ...] = ()
    lora_task_type: str | None = None      # "FEATURE_EXTRACTION" for decoder; None otherwise
    trust_remote_code: bool = False
    fast_path_hf_id: str | None = None     # optional unsloth/-prefixed mirror id

    def resolved_fast_path_id(self) -> str:
        """hf_id to use on the Unsloth fast path (mirror override or canonical)."""
        return self.fast_path_hf_id or self.hf_id

# Module-level API (the only surface trainer/evaluator/serving import):
def load_registry(path: Path | None = None) -> dict[str, EmbeddingModelSpec]: ...
def get_spec(name: str, path: Path | None = None) -> EmbeddingModelSpec: ...
def list_models(path: Path | None = None) -> list[str]: ...
```

**Validation rules (loader enforces, raising `ValueError` with the offending key):**
- `family ∈ {bert, xlm-roberta, decoder}`; `pooling ∈ {mean, cls, last_token, weighted_mean}`; `embedding_type ∈ {bi_encoder, cross_encoder}`.
- `decoder` family → `lora_task_type` must be `FEATURE_EXTRACTION` (warn-and-set if omitted).
- `prompt_required: true` → both `query_prompt` and `passage_prompt` non-empty.
- `matryoshka_dims` entries all ≤ `default_dim`, sorted desc.
- Unknown keys in a YAML block → `ValueError` (catch typos; config-driven discipline).

**CODE-time verification (flag, do not block design):** the BERT
`lora_target_modules` names (`query/key/value/dense`) must be confirmed to
resolve against the ST `Transformer` wrapper via a `named_modules()` dump before
trusting them (R7). The registry is the SSOT; if names differ, fix the YAML, not
code.

---

## 2. Dual-Loader Contract (R1, R6)

**File (new):** `Trainers/embedding/src/model_loader.py`

### 2.1 Design stance: fallback-primary

Per PREPARE §3.4 and lead priority: **plain `SentenceTransformer` is the
correctness baseline.** `FastSentenceTransformer` is an *optional* accelerator
selected by a capability probe. The trainer code is identical regardless of
which loader returns — both yield a uniform object.

### 2.2 Capability probe semantics

```python
@dataclass(frozen=True)
class LoaderCapabilities:
    fast_path_available: bool      # unsloth importable AND CUDA AND opt-in not disabled
    cuda: bool
    reason: str                    # human-readable why fast/fallback was chosen

def probe_capabilities(*, allow_fast_path: bool = True) -> LoaderCapabilities:
    """Decide fast vs fallback. NEVER raises — probe failure → fallback.
    Order:
      1. allow_fast_path is False                     → fallback (reason="disabled by config")
      2. torch.cuda.is_available() is False           → fallback (reason="no CUDA (Mac/MPS/CPU)")
      3. `import unsloth; FastSentenceTransformer`     → fast (reason="unsloth available")
         ImportError                                   → fallback (reason="unsloth not importable")
    """
```

The probe is **import-guarded and total** — any failure degrades to fallback.
This makes the runtime-unverified status of `FastSentenceTransformer` (R1) a
non-blocking optimization rather than a load-bearing assumption.

### 2.3 Uniform return object

```python
@dataclass
class LoadedEmbeddingModel:
    model: Any                     # SentenceTransformer-compatible (fast or fallback)
    spec: EmbeddingModelSpec
    loader_path: str               # "fast" | "fallback"
    capabilities: LoaderCapabilities

def load_embedding_model(
    spec: EmbeddingModelSpec,
    adapter_mode: str,             # "full" | "lora" | "frozen_head"
    *,
    lora_config: Mapping[str, Any] | None = None,
    allow_fast_path: bool = True,
) -> LoadedEmbeddingModel:
    """
    Fast path:     from unsloth import FastSentenceTransformer
                   FastSentenceTransformer.from_pretrained(
                       spec.resolved_fast_path_id(), max_seq_length=spec.max_seq_length,
                       full_finetuning=(adapter_mode == "full"))
    Fallback:      SentenceTransformer(spec.hf_id, trust_remote_code=spec.trust_remote_code, ...)
    Both honor spec.pooling / spec.normalize via ST modules.json semantics.
    Returns a uniform LoadedEmbeddingModel so downstream code is loader-agnostic.
    """
```

**Loader method-surface (corrected at CODE-time by WU-A runtime evidence):**
- BOTH loaders expose `save_pretrained` (adapter dir) and `push_to_hub` — so
  **LoRA upload** (the copytree path) works from either loader.
- `save_pretrained_merged` is an **Unsloth-patched method present ONLY on
  `FastSentenceTransformer`** (fast path). Plain `SentenceTransformer` (the
  fallback) does **NOT** expose it.
- **Consequence (the merge/training axis split):** *training/loading* is
  fallback-primary (plain ST is the baseline, §2.1), but the **merge** step (§3)
  is Unsloth-only for both families — it loads the saved adapter via
  `FastSentenceTransformer.from_pretrained` and calls *that* object's
  `save_pretrained_merged`. A model trained on the plain-ST fallback (CPU/MPS
  dev) therefore needs an Unsloth/CUDA box to produce a merged-16bit artifact.
  This only bites the **merged-16bit** path; **LoRA-only upload needs no merge**
  (copytree), so fallback-trained LoRA adapters ship without ever touching the
  Unsloth merge path. Merge already requires GPU+Unsloth today for causal-LM, so
  this is a consistent runtime assumption, not a new constraint.

### 2.4 Adapter-mode axis (R6, R8)

| Mode | Meaning | LoRA? | R6 gate |
|------|---------|-------|---------|
| `full` | fine-tune whole encoder (`full_finetuning=True` on fast path) | no | — |
| `lora` | `peft.LoraConfig(target_modules=spec.lora_target_modules, task_type=spec.lora_task_type)` (Unsloth LoRA on fast, PEFT on fallback). Output = small adapter → reuses merge/upload. | yes | — |
| `frozen_head` | freeze base, train only appended Dense/MLP head. Smallest, CPU-trainable. | no | **Compare-to-ST-baseline gate (R6):** before training the head, assert frozen-base embeddings match a plain-`SentenceTransformer` baseline within tolerance (Unsloth nonstandard-pooling caveat). Smoke test enforces this. |

**R6 disposition (v1) — cloud-smoke-deferred (lead-accepted A′).** The R6
compare-to-ST-baseline gate is a *fidelity check on the frozen-base
representation*; it only has meaning when the real fast-path encoder runs on
GPU, which is the same Docker/GPU blocker that defers the overlay pins (§7).
Therefore in v1 R6 is **NOT** a runtime assertion in `Trainers/embedding/`
source — it rides the deferred cloud-smoke checklist (§7.4) as a **named,
blocking pre-production gate**. What this does and does NOT defer:
- The `frozen_head` **MODE** (loadable, produces `dim == default_dim`
  embeddings) **IS** already covered by the loader/probe unit tests — it is not
  deferred.
- Only the compare-to-baseline **GATE** (the fidelity assertion) defers to the
  cloud smoke.

A runtime assertion was considered and rejected for v1: it could not execute its
meaningful path in any environment available to this workflow (it would only
*fire* on the same cloud run the smoke already governs), so it adds unverifiable
source for zero earlier enforcement.

**Phase-2 future-trigger:** if `frozen_head` graduates from "smallest /
experimental mode" to a **PRIMARY supported mode**, R6 becomes a runtime
assertion in `evaluation`/training (compare trained frozen_head metrics vs the
untrained-ST baseline, warn/fail) and this §2.4 disposition is revisited. Not a
v1 change.

**No `qlora` mode in v1 (R8).** The enum is exactly `{full, lora, frozen_head}`.
A `qlora` value in config → `ValueError` with a "deferred to a later phase"
message.

---

## 3. Family-Parametrized Merge Seam (R2)

**File (modified):** `shared/model_loading/merge.py`

**Lead-confirmed decision:** parametrize IN PLACE by `family`. Alter existing
`FastLanguageModel` callers directly (no backward-compat shim, no parallel
embedding-only module). The causal-LM path MUST remain behavior-identical
(test-engineer verifies SFT/KTO merge still works).

### 3.1 The seam

`merge_lora_checkpoint()` (currently merge.py:73-118) hardcodes `from unsloth
import FastLanguageModel` at L91 and `FastLanguageModel.from_pretrained(...)` at
L100. The family must thread through.

```python
# New: a small family→loader dispatch (module-level, explicit, testable).
def _merge_loader_for_family(family: str):
    """Return the from_pretrained callable for a model family.
    "causal_lm" (default) → unsloth.FastLanguageModel.from_pretrained
    "embedding"           → the ST/FastSentenceTransformer merge path
    Unknown family        → ValueError.
    """

def merge_lora_checkpoint(
    lora_path: Path,
    output_path: Path,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
    family: str = "causal_lm",        # NEW PARAM, default preserves existing behavior
) -> Path:
    ...
```

### 3.2 Behavior-preservation requirement (the load-bearing constraint)

- The **default `family="causal_lm"`** path must be byte-for-behavior identical
  to today: same `FastLanguageModel.from_pretrained(... load_in_4bit ...)` then
  `save_pretrained_merged(..., save_method="merged_16bit")`.
- Existing callers (`find_or_create_merged` → `resolve_model_path`, GRPO trainer,
  `merge_handler`, upload workflows) get the default and are unaffected in
  behavior. Where a caller now *knows* the family (embedding runs), it threads
  `family="embedding"`.
- The `embedding` branch loads via the ST/`FastSentenceTransformer` path and
  calls its `save_pretrained_merged`. `load_in_4bit` is **ignored** for the
  embedding family in v1 (QLoRA deferred, R8) — the dispatch must not pass a
  4-bit flag into an ST loader that doesn't support it.

### 3.3 Family resolution: who supplies it

The family is resolved from the **registry spec** at the call site that initiates
an embedding merge (the embedding trainer / upload path). Causal-LM callers omit
it (default). `merge.py` itself does NOT read the registry — it receives `family`
as a parameter (keeps `shared/` import-light and free of `Trainers/` deps).

**LoRA upload is unaffected** (`shared/upload/strategies/lora.py` is a pure
copytree, `requires_gpu→False`) — no change. Only the **merge** half splits.

---

## 4. Retrieval Metrics, Verifier, Evaluator, Tracking (R4, R5, R9)

### 4.1 `shared/ml/retrieval_metrics.py` — pure functions (new file)

A **sibling** to `shared/ml/metrics.py` (which is classification/regression-only,
sklearn-backed). Retrieval metrics operate on **ranked id lists + qrels**, a
different shape — do NOT extend `compute_metrics`.

```python
# Pure functions, no heavy deps (numpy at most). All take ranked retrieved ids
# and the relevant-id ground truth; all return float in [0, 1].

def recall_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float: ...
def mrr(retrieved: Sequence[str], relevant: Set[str], k: int | None = None) -> float: ...
def ndcg_at_k(retrieved: Sequence[str], relevance: Mapping[str, int], k: int) -> float:
    """Graded nDCG. relevance maps id→grade (0..3); supports graded_pairs data."""
def map_score(retrieved: Sequence[str], relevant: Set[str], k: int | None = None) -> float: ...

# Convenience aggregator over a query set (mean across queries):
def aggregate_retrieval_metrics(
    results: Sequence[QueryResult],          # per-query (retrieved ids, relevant/relevance)
    metric_specs: Sequence[str],             # e.g. ["recall@10", "mrr@10", "ndcg@10"]
) -> dict[str, float]:
    """Parse 'metric@k' specs, dispatch to the pure fns, return {spec: mean_value}."""
```

Metric-spec string grammar: `"<metric>@<k>"` where metric ∈
`{recall, mrr, ndcg, map}`, `k` a positive int. This is the canonical spec
format used by the scenario YAML, the verifier, and the lineage.

### 4.2 `retrieval_verifier` — registration + the corpus-level impedance resolution (R9)

**File (new):** `shared/verifiers/builtins/retrieval_verifier.py`
**Registry type key:** `retrieval` (the contract). **Filename:**
`retrieval_verifier.py` (sibling `*_verifier.py` convention). R9 resolved: type
key ≠ filename, and that is fine — the registry key is what scenarios reference.

**Impedance mismatch (must be designed around):** the existing `VerifierInput`
(contract.py) is **per-completion** — one `completion_text` → scalar
`score`/`passed`. Retrieval verification is **corpus-level** — embed a corpus,
run a query set against qrels, aggregate. These are different input axes.

**Resolution:** the retrieval verifier registers in the same registry (so it is
discoverable + buildable via `build_verifier`), but it is invoked through a
**dedicated corpus-level entry point**, not the per-completion `verify(sample)`
loop. It does not consume `VerifierInput`; it consumes a retrieval config.

```python
@dataclass(frozen=True)
class RetrievalValidationResult:
    metrics: dict[str, float]          # {"recall@10": .., "ndcg@10": ..}
    passed: bool                       # all thresholds met
    warned: bool                       # met min but below warn_margin on any metric
    primary_metric_name: str           # e.g. "ndcg@10" (first in scenario, or explicit)
    primary_metric: float
    detail: dict[str, Any]             # per-query diagnostics (low-rank-of-positive etc.)

@register("retrieval")
def _build_retrieval_verifier(spec: Mapping) -> "RetrievalVerifier": ...

class RetrievalVerifier:
    name = "retrieval"
    def evaluate_retrieval(self, cfg: RetrievalConfig) -> RetrievalValidationResult:
        """Load corpus+queries+qrels, embed via registry spec (or a trained
        run/adapter path), FAISS top-k retrieve, aggregate_retrieval_metrics,
        apply thresholds → passed/warned."""
```

`RetrievalConfig` carries: `corpus`, `queries`, `qrels` (JSONL paths),
`metrics` (spec list), `model` (`{registry_name}` or a run/adapter path),
plus the R4 threshold block (§4.3).

> `shared/verifiers` MUST NOT import `Evaluator/` or `Trainers/` (existing
> boundary rule, see `tool_sequence.py` docstring). The verifier reads the
> registry via `Trainers/embedding/src/registry.py`? — NO. To honor the
> boundary, the **embedding/FAISS retrieval mechanics live in the verifier's own
> module under `shared/`**, and the registry spec needed for eval-time embedding
> is passed in via `RetrievalConfig` (resolved by the Evaluator caller, which
> may import `Trainers/`). The verifier stays `shared/`-pure.

### 4.3 `Evaluator/runner.py` — `EvaluationRecord.retrieval` field + status branch (R4)

**File (modified):** `Evaluator/runner.py`

Add an optional field + a status branch. Continuous metrics → pass/warn/fail via
scenario thresholds.

```python
@dataclass
class EvaluationRecord:
    ...
    retrieval: Optional["RetrievalValidationResult"] = None   # NEW

    @property
    def status(self) -> str:
        if self.error is not None:
            return "fail"
        # NEW retrieval branch — placed BEFORE the correctness branch so a
        # retrieval scenario is evaluated on its own ladder:
        if self.retrieval is not None:
            if not self.retrieval.passed:
                return "fail"
            return "warn" if self.retrieval.warned else "pass"
        # ... existing correctness/environment/judge ladder unchanged ...
```

**Threshold convention (R4) — scenario YAML:**

```yaml
tests:
  - id: retrieval_smoke
    retrieval_config:
      corpus: Datasets/embedding/examples/corpus.jsonl
      queries: Datasets/embedding/examples/queries.jsonl
      qrels: Datasets/embedding/examples/qrels.jsonl
      metrics: [recall@10, mrr@10, ndcg@10]
      model: { registry_name: bge-base-en }     # or a trained run/adapter path
      thresholds:
        min:  { ndcg@10: 0.30, recall@10: 0.40 }   # below min → fail
        warn_margin: 0.05                           # within [min, min+margin) on any → warn
      primary_metric: ndcg@10                       # → RunRecord.primary_metric_name
```

`passed = all(metric >= min[metric] for metric in min)`;
`warned = passed and any(min[m] <= value < min[m] + warn_margin)`.
Metrics without a `min` entry are reported but not gating.

**Corpus-level routing branch in `runner.py` (CODE amendment — required for the
feature to be live, not just smoke-callable):** the field + status branch above
are inert unless `runner.py` actually *invokes* `evaluate_retrieval`. So
`runner.py` ALSO gets a routing branch: when a scenario declares a
`retrieval_config` block, resolve it into a `RetrievalConfig` and call
`RetrievalVerifier.evaluate_retrieval(cfg)` **ONCE per scenario, as a SIBLING to
the per-completion loop — NOT inside it**, then attach the
`RetrievalValidationResult` to the `EvaluationRecord.retrieval` field. This
preserves the §4.2 fence (retrieval never flows through the per-completion
`verify(VerifierInput)` path) while making retrieval eval actually run in
production (the end-to-end Phase 1 goal), not only in the CI smoke. The
resolution of `RetrievalConfig` (which may read a registry spec or a trained
run/adapter path) happens in `runner.py` / the Evaluator caller — which is
allowed to import `Trainers/` — keeping the `shared/` verifier itself pure.

### 4.4 Retrieval lineage + experiment-tracking adapter (R5)

**Files:** new lineage written by the eval run; new adapter in
`shared/experiment_tracking/adapters.py`; `run_type` enum extension in
`schema.py`.

**`retrieval_eval_lineage.json` shape** (parallel to `evaluation_lineage.json`):

```json
{
  "timestamp": "2026-06-14T...Z",
  "model": { "base_model": "bge-base-en", "run_or_adapter_path": "..." },
  "dataset": { "corpus": "...", "queries": "...", "qrels": "..." },
  "results": { "metrics": { "ndcg@10": 0.41, "recall@10": 0.55, "mrr@10": 0.38 },
               "primary_metric_name": "ndcg@10", "primary_metric": 0.41 },
  "hardware": { "device": "cpu" }
}
```

**Adapter** (mirrors `_training_lineage_to_run_record`):

```python
def retrieval_eval_to_run_record(
    lineage: dict[str, Any], run_dir: str, *,
    run_id: str | None = None, cloud: bool = False,
) -> RunRecord:
    results = lineage.get("results", {})
    return RunRecord(
        run_id=run_id or str(uuid.uuid4()),
        run_type=f"cloud_embedding" if cloud else "embedding",   # NEW enum value
        name=f"EMBEDDING eval {lineage.get('timestamp','')}".strip(),
        timestamp=lineage.get("timestamp", ...),
        status="completed",
        output_dir=run_dir,
        tags={"method": "embedding", "provider": "cloud" if cloud else "local"},
        model_name=lineage.get("model", {}).get("base_model"),
        dataset_source=lineage.get("dataset", {}).get("queries"),
        primary_metric=results.get("primary_metric"),            # e.g. ndcg@10 value
        primary_metric_name=results.get("primary_metric_name"),  # "ndcg@10"
    )
```

**`schema.py:49`** `run_type` enum comment extends to include `"embedding"` and
`"cloud_embedding"` (no migration — `from_dict` drops unknowns; this just
documents the new valid values; `primary_metric`/`primary_metric_name` already
exist).

---

## 5. Method-Wiring (R3)

**Lead-confirmed split:** surgical add-string for train-time surfaces;
**exclude** `embedding` from the 3 eval-backend discovery tuples.

### 5.1 Train-time surfaces — ADD `embedding`

> **PREPARE under-counted the train-time gate set.** R3 named `base_cloud` +
> `rtx`/`mac`. A `tests/` registration-sweep test
> (`tests/trainers/dpo/test_dpo_method_registration.py`) reveals the project's
> *full* enumeration surface and pins it. Two additional train-time gates exist
> that R3 missed and that embedding MUST register at: `tuner/cli/parser.py:222`
> (`--method choices`) and `tuner/backends/training/cloud/hf_jobs_backend.py:125`.
> Without `parser.py`, `--method embedding` is rejected at the CLI; without
> `hf_jobs_backend`, cloud HF Jobs embedding runs can't dispatch.

| File:line | Change | Rationale |
|-----------|--------|-----------|
| `shared/utilities/paths.py:11` | `TRAINING_METHODS = ("sft", "kto", "grpo", "dpo", "embedding")` | Asserts `embedding_output/` convention (wanted). Adds a dead `rtx3090_embedding` legacy-dir map entry — harmless, flag in HANDOFF. |
| `tuner/backends/training/cloud/base_cloud.py:110` | `SUPPORTED_METHODS = (..., "embedding")` | Cloud `validate_training_method` must accept embedding for HF Jobs runs. |
| `tuner/backends/training/rtx_backend.py:106` | `get_available_methods()` returns `[..., "embedding"]` | RTX/CUDA backend runs the fast path. |
| `tuner/cli/parser.py:222` | `--method choices=[..., "embedding"]` | **(R3 gap)** Embedding must be CLI-selectable for cloud-pipeline. |
| `tuner/backends/training/cloud/hf_jobs_backend.py:125` | return list `[..., "embedding"]` | **(R3 gap)** Cloud HF Jobs method list — embedding cloud runs (recipe + image overlay) are in scope. |

### 5.1.1 Required test-pin updates (SAME commit as the wiring edits)

`tests/trainers/dpo/test_dpo_method_registration.py` is a registration-sweep pin
that asserts **exact literals** at the named-gate sites. The exclusion-aware
embedding wiring changes some of those literals, so the test's needle strings
must update in the same commit:

| Pin (test fn / line) | Pin type | Effect of embedding wiring | Required action |
|----------------------|----------|----------------------------|-----------------|
| `test_named_gate_sites_source_contains_dpo` (L42-44) | exact-substring needle on `parser.py`, `hf_jobs_backend.py`, `rtx_backend.py` | the literal `["sft","kto","grpo","dpo"]` becomes `[...,"embedding"]` — old needle no longer matches | update each needle to the new 5-element literal **in the same commit** as the WU-C edit |
| `test_lifecycle_iteration_sites_include_dpo` (L53-65) | presence-marker (`"dpo"` ∈ source) on the 3 eval-backends + discovery/handlers | embedding is **excluded** from the 3 eval-backends → those tuples stay `(...,"dpo")` → `"dpo"` still present | **no change** — pin stays green (this is the design-intended exclusion) |
| `test_no_three_method_tuple_left_unregistered` (L70-89) | AST-ish grep for stale **3-method** tuples | embedding makes 4-tuples into 5-tuples; the scan only catches 3-tuples | **no change** — unaffected |
| `test_paths_training_methods_includes_dpo` (L18-27) | membership + derived-map | additive; dpo assertions unaffected | **no change** |
| `test_base_cloud_supported_methods_includes_dpo` (L32-37) | membership | additive | **no change** |

**New sibling test (WU-D, expected by convention):**
`tests/trainers/embedding/test_embedding_method_registration.py`, mirroring the
dpo registration-sweep, asserting `embedding` IS registered at the 5 train-time
gates AND — explicitly — is **NOT** required at the 3 eval-backend tuples
(documenting the exclusion as intentional, not an omission). This is the
mechanism that makes the R3 exclusion decision durable against a future
"consistency fix."

### 5.1.2 Functional method-allowlist gate (CODE amendment — 6th gate, owned by WU-B)

`shared/experiment_tracking/experiment_spec.py:211` — `ExperimentSpec.validate()`
contains a **functional** method-allowlist:
`if self.method not in {"sft", "kto", "grpo", "dpo"}: issues.append(...)`. This is
NOT a test pin and NOT one of the CLI/backend named gates — it is a runtime
validation that would **reject an embedding experiment spec** at `validate()`
time. The §5.1.1 registration-sweep (grep of `tests/` + named gates) did not
surface it because it lives in `experiment_tracking`.

**Decision (lead-ruled):** add `"embedding"` to this allowlist set in v1.
**Owner: WU-B**, because `shared/experiment_tracking/` is WU-B's single-owned
surface (alongside the tracking adapter §4.4 and `schema.py` run_type). This
keeps the experiment-tracking edits in one work-unit.

The 3 `iter_training_output_dirs` discovery sites
(`tuner/discovery/base_models.py`, `doctor_handler.py`, `merge_handler.py`) stay
**deferred** to the §5.4 future-dedup task — they are off the v1 smoke path and
share the same eval-backend-discovery rationale as §5.3 (embedding is not
discovered/served through those quantization/merge loops).

### 5.2 Mac backend — DO NOT add embedding to v1 (PREPARE §2.3)

`mac_backend.get_available_methods()` returns `['sft']` only (MLX/SFT-only
today). The ST fallback loader is NOT an MLX path. **Decision: Mac/MPS embedding
is out of v1 backend scope.** The fallback loader still works for local Mac dev
*directly* (running `train_embedding.py`), but it does not slot into the Mac
*backend*'s method list in v1. Flag as a Phase-2+ consideration.

### 5.3 Eval-backend discovery tuples — EXCLUDE embedding (correctness call)

`tuner/backends/evaluation/{llamacpp:131, mlc:69, unsloth:63}_backend.py` each
run `for method in ("sft","kto","grpo","dpo"):` to discover **trained model
output dirs for quantization/serving** (`iter_training_output_dirs`). Embedding
models are NOT served/quantized through llamacpp/mlc/unsloth backends — embedding
eval flows through the new `retrieval` verifier + `embedding_retrieval` scenario.
Adding `embedding` here would make these backends scan a phantom
`embedding_output/` for serving they can't do.

**Action:** do NOT edit these three tuples. Add a one-line comment at each
(or one shared note) explaining the deliberate exclusion, so a future reader
doesn't "fix" the apparent inconsistency.

### 5.4 Broader dedup — FUTURE cleanup (flag, do not do in v1)

The duplication (5 hardcoded method literals not importing `TRAINING_METHODS`)
is real tech debt. Refactoring all of them to import the tuple is a sweeping
backend change that would touch behavior across methods and balloon the PR.
**Decision: do not refactor in this PR.** Flag in the HANDOFF as a standalone
future-cleanup task. Only the 3 train-time surfaces above are touched.

---

## 6. `Trainers/embedding/` Module Layout

**Directory (new):**

```
Trainers/embedding/
├── train_embedding.py          # entry point — mirrors train_sft.py bootstrap shape
├── requirements.txt            # METHOD-LOCAL pinned island (see §7)
├── configs/
│   ├── config.yaml             # default embedding run config (blueprint §5 shape)
│   └── model_registry.yaml     # §1 seed data
└── src/
    ├── registry.py             # §1 EmbeddingModelSpec loader
    ├── model_loader.py         # §2 dual loader + capability probe
    ├── data_loader.py          # triplet/pairs JSONL → ST dataset + prompt prefixing
    ├── losses.py               # config → ST loss mapping (§6.1)
    ├── evaluation.py           # in-training ST IR evaluator (recall@k/MRR/nDCG on dev split)
    └── callbacks.py            # adapt Trainers/shared/callbacks to the ST trainer
```

`train_embedding.py` bootstrap mirrors `Trainers/sft/train_sft.py`:
`sys.path.insert` for `src` + repo root, `init_trainer_env()` before
torch/unsloth import, then load registry spec → `load_embedding_model` →
`SentenceTransformerTrainer` + `SentenceTransformerTrainingArguments`. The
training loop is loader-agnostic.

### 6.1 `losses.py` config → ST loss map

| config `loss` | ST loss | Notes |
|---------------|---------|-------|
| `multiple_negatives_ranking` | `MultipleNegativesRankingLoss` | in-batch negatives workhorse |
| `cached_multiple_negatives_ranking` | `CachedMultipleNegativesRankingLoss` | memory-efficient (Unsloth example default) |
| `triplet` | `TripletLoss` | explicit margins |
| `cosent` / `cosine_similarity` | `CoSENTLoss` / `CosineSimilarityLoss` | graded-relevance data |
| (wrapper) `matryoshka` | `MatryoshkaLoss` over a base loss | applied when `spec.matryoshka_dims` non-empty |

Use `BatchSamplers.NO_DUPLICATES` for in-batch-negative losses (PREPARE §3.1 —
matters for MNRL correctness).

### 6.2 `config.yaml` default shape (blueprint §5, unchanged)

```yaml
model:
  registry_name: bge-base-en
  adapter_mode: lora             # full | lora | frozen_head   (NO qlora — R8)
training:
  loss: multiple_negatives_ranking
  batch_size: 64
  epochs: 1
  learning_rate: 2.0e-5
  warmup_ratio: 0.1
lora: { r: 16, alpha: 32, dropout: 0.05 }
dataset:
  local_file: Datasets/embedding/examples/triplets_smoke.jsonl
  eval_split: 0.05
evaluation:
  metrics: [recall@10, mrr@10, ndcg@10]
```

Output layout reuses canonical
`embedding_output/YYYYMMDD_HHMMSS/{final_model,checkpoints,logs,training_lineage.json}`
(auto-asserted by adding `embedding` to `TRAINING_METHODS`).

---

## 7. Dependencies, Recipe, Image Overlay (Image-floor correction)

### 7.1 `Trainers/embedding/requirements.txt` — method-local island

New pinned island. **Never touch the legacy SFT `transformers 4.45.2` pins.**
Contents: `sentence-transformers`, `faiss-cpu`, `datasets`. **Exact pins are
TBD-pending-CODE-smoke-test** (lead-approved) — the repo has a documented history
of image/numpy mismatches (PREPARE §3.2, `cloud-training.md:208-214`), so CODE
must smoke-test the overlay and pin empirically. Do NOT hardcode speculative
version numbers in the design.

### 7.2 Recipe (new): `Trainers/recipes/embedding_bge_base_smoke.yaml`

Matches the confirmed recipe schema (`qwen35_2b_sft_smoke.yaml`). Rides the
modern Unsloth image with a `setup.pip` overlay adding `sentence-transformers` +
`faiss-cpu` (pins TBD by smoke test). `method: embedding`, `target: local|both`,
`run.trainer: Trainers/embedding/train_embedding.py`.

**Image-floor correction (PREPARE §3.2):** the blueprint's claim that the image
"satisfies transformers>=5.1.0" is WRONG — the image ships `transformers 4.57.1`.
The recipe's `setup.pip` overlay pins the ST/transformers stack on top, exactly
like the SFT smoke recipe overrides `transformers==5.5.0`. The overlay is the
mechanism; its exact pins are a CODE smoke-test output.

### 7.3 `Evaluator/requirements.txt`

Add `faiss-cpu` (+ optional `mteb`, deferred). Method-local discipline applies.

### 7.4 Deferred Cloud-Smoke Checklist (v1 blocking pre-production gates)

The v1 workflow runs without Docker/GPU, so a small set of guarantees cannot be
*demonstrated* here. They are not gaps — each is a **named, blocking gate** that
MUST pass on a real GPU/cloud smoke run before `embedding` training is relied on
in production. This checklist is the single, loud home for those deferrals.

| Gate | What must pass on the cloud smoke | Until then |
|------|-----------------------------------|------------|
| **Overlay pins (§7.1, §7.2)** | The `setup.pip` overlay — the three embedding-specific packages `sentence-transformers`, `faiss-cpu`, `datasets` — resolves cleanly on the modern Unsloth image. `transformers` is NOT added/re-pinned by the overlay: the image already ships `transformers 4.57.1`, and the CONSTRAINT is that the overlay must resolve against that shipped version with no numpy/transformers mismatch. Exact pins for the three packages captured empirically. | Pins stay `TBD-pending-CODE-smoke-test`; do NOT hardcode speculative versions, and do NOT add a `transformers` pin to the overlay. |
| **R6 frozen-base fidelity** | The `frozen_head` compare-to-ST-baseline gate (§2.4): frozen-base embeddings match a plain-`SentenceTransformer` baseline within tolerance on the real fast-path encoder. | **`frozen_head` must not be relied on for a production run until the compare-to-ST-baseline smoke passes on GPU.** The MODE is unit-tested (loadable, correct dim); only the fidelity GATE defers. |
| **FAISS E2E must RUN (not skip)** | The retrieval E2E test uses `pytest.importorskip("faiss")`, so where `faiss-cpu` is absent it SILENTLY SKIPS rather than fails. The cloud-smoke gate MUST assert the FAISS E2E test actually EXECUTED — i.e. `faiss-cpu` is present on the GPU smoke runner and the test reports pass, not skip. | A skipped retrieval E2E MUST NOT be counted as a passing gate. The smoke runner must verify `faiss-cpu` is installed (it is the third overlay package above) so the importorskip branch is never taken; treat a skip as a gate FAILURE. |

The R6 WARN-log honesty backstop (a one-line "fidelity gate cloud-deferred"
notice in the `frozen_head` training path) is itself **deferred to Phase 2** —
it too only surfaces on the cloud run this checklist already governs, so the
checklist entry above is the v1 honesty backstop.

---

## 8. Skill, Fixtures, CI Smoke (blueprint §9)

### 8.1 `.skills/embedding-training/` (new canonical skill)

`SKILL.md` + a `reference/` subtree
(`embedding-training.md`, `retrieval-eval.md`, `triplet-data.md`), in the spirit
of `case-studies`. End-to-end: generate/obtain triplets → train (`embedding`
method, pick base + adapter mode) → evaluate retrieval → read metrics → refine.
Each step a copy-pasteable CLI command, no ad hoc Python. Documents the registry,
adapter modes, and the dual loader.

**`.skills/` is canonical** — after authoring, run
`python3 .skills/scripts/sync_skill_trees.py` to refresh `.agents/skills` and
`.claude/skills` mirrors. The CI smoke (§8.3) should include
`sync_skill_trees.py --check`.

Targeted edits to existing skills (`fine-tuning`, `evaluation`,
`synethetic-data-generation`) per blueprint §9 — add the `embedding` method,
recipe, retrieval verifier, and triplet schemas.

### 8.2 Checked-in fixtures (the test-and-refine loop)

Tiny, fast, CPU/MPS-runnable, committed:
- `Datasets/embedding/examples/triplets_smoke.jsonl` — ~20 hand-written `{query, positive, negatives}`.
- `Datasets/embedding/examples/{corpus,queries,qrels}.jsonl` — tiny labeled retrieval set.
- `Trainers/recipes/embedding_bge_base_smoke.yaml` — `frozen_head` or small-`r` `lora`, few steps.
- `Evaluator/config/scenarios/embedding_retrieval_smoke.yaml` + `Evaluator/recipes/embedding_retrieval_eval.yaml`.

### 8.3 CI smoke test shape

`tests/` smoke (wired like existing trainer smokes) asserting:
1. registry loads + all 4 seed specs validate;
2. retrieval_metrics pure fns return finite values in `[0,1]` on a fixture;
3. the retrieval verifier emits finite recall@k/MRR/nDCG on the smoke set;
4. (if CPU-feasible in CI) `frozen_head` train on `triplets_smoke.jsonl`
   produces an adapter/head artifact — otherwise gate behind a marker;
5. `sync_skill_trees.py --check` passes (mirrors in sync).

Heavy GPU training is NOT a CI gate — the fast path is exercised only where CUDA
is available. The **fallback path on CPU** is what CI runs.
