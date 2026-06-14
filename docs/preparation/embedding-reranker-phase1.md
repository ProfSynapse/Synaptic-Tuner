# PREPARE: Embedding & Reranker Pipeline — Phase 1 (v1 core)

**Phase:** PACT Prepare
**Author:** preparer (team pact-3c41533e)
**Date:** 2026-06-14
**Verifies:** `docs/EMBEDDING_RERANKER_PIPELINE_PLAN.md` v1.1 (status: *Proposal — awaiting review*)
**Evidence bar:** static file/line reading (Read/Grep) over worktree
`.worktrees/embedding-reranker-pipeline-v1` + HF model pages + Unsloth official
notebooks (Context7). **No runtime execution** — image/import verification is
explicitly deferred to the architect/code phase (lead-confirmed scope).

---

## Executive Summary

The design doc's **overall direction is sound and most reuse seams are real**,
but several Section-2 "reusable as-is" claims are **overstated** in ways the
architect must design around. The single most important correction: the doc
says the embedding merge/upload path "drops straight into the existing
strategies" — this is **only half true**. The LoRA-adapter upload strategy is
genuinely model-agnostic and reuses cleanly; the **merge / merged-16bit path is
hardcoded to causal-LM `FastLanguageModel`** and will NOT load an
`FastSentenceTransformer`/ST embedding model without a loader change.

The Unsloth `FastSentenceTransformer` path is **confirmed to exist upstream**
(official Unsloth notebooks, including a Qwen3-Embedding-0.6B notebook — one of
our seed models) but is **NOT runtime-verified against the repo's pinned
image**, and the doc's specific claim that
`unsloth/unsloth:2026.1.2-pt2.9.0-cu12.8` "satisfies transformers>=5.1.0" is
**contradicted by repo evidence** (the repo's own recipes pin
`transformers==5.5.0` *on top of* the image because the shipped image carries
`transformers 4.57.1`). The dual-loader capability probe is therefore a
**genuine unknown** the architect must treat as such (lead-flagged as the
priority to de-risk).

All four seed models exist on HF with attributes close to the registry, with a
few concrete corrections (see §4).

**Verdict for the architect:** proceed with the design, but anchor on the
corrected claims in §2 (especially the merge-loader split and the method-string
duplication) and resolve the open risks in §6 before committing to interfaces.

---

## 1. Scope Verified

Per Task #8, I verified four fronts with file/line evidence:
1. The Section-2 "Reusable as-is" claims against real code.
2. The Unsloth `FastSentenceTransformer` path (existence, image floor, ST/faiss presence).
3. The four seed models on HF and their registry-relevant attributes.
4. Integration risks/unknowns for the architect.

Worktree branch: `embedding-reranker-pipeline-v1`. Target dir
`docs/preparation/` exists (4 sibling research docs).

> **CLAUDE.md note (HANDOFF flag):** `CLAUDE.md` is present in the worktree and
> is gitignored there — I did **not** edit or create it. No CLAUDE.md changes
> are proposed by this prep doc. If the architect adds the `embedding` method,
> the project-structure section of the *root* `CLAUDE.md` (Trainers/ list) would
> benefit from an `embedding/` entry, but that is an orchestrator action outside
> the worktree.

---

## 2. Reusability Claims — Confirmed / Corrected (file:line evidence)

Legend: ✅ confirmed as-is · ⚠️ partially true / needs work · ❌ wrong as stated.

### 2.1 ✅ Method dispatch by convention — CONFIRMED, with a derivation caveat
- **Claim:** `shared/utilities/paths.py:11` holds `TRAINING_METHODS`; adding a
  string + `Trainers/<method>/` dir is the extension point.
- **Evidence:** `shared/utilities/paths.py:11` is **exactly**
  `TRAINING_METHODS = ("sft", "kto", "grpo", "dpo")`. ✅
- **Caveat the doc misses:** lines 13–17 *derive* dicts from this tuple —
  `CANONICAL_TRAINER_DIRS`, `LEGACY_TRAINER_DIRS` (`rtx3090_<method>`),
  `CANONICAL_OUTPUT_DIRS` (`<method>_output`), `LEGACY_OUTPUT_DIRS`. Adding
  `"embedding"` auto-asserts an `embedding_output/` convention (which the doc
  *does* want, line 302–303) **and** a phantom `rtx3090_embedding` legacy dir
  that will never exist. Harmless but the architect should know the legacy-dir
  map will contain a dead `embedding` entry.

### 2.2 ❌ "Everything picks it up automatically" — OVERSTATED: method string is duplicated in ≥4 places
- **Claim (Section 2):** adding the method to `paths.py` means CLI routing,
  RTX/Mac backends, recipes, and HF Jobs backend "pick it up automatically."
- **Evidence — the tuple is hardcoded, not imported, in multiple files:**
  - `tuner/backends/evaluation/llamacpp_backend.py:131` — `for method in ("sft", "kto", "grpo", "dpo"):`
  - `tuner/backends/evaluation/mlc_backend.py:69` — same literal tuple
  - `tuner/backends/evaluation/unsloth_backend.py:63` — same literal tuple
  - `tuner/backends/training/cloud/base_cloud.py:113` — `validate_training_method(method, backend_name)`
  - `tuner/backends/training/rtx_backend.py` — `get_available_methods()` returns `['sft', 'kto', 'grpo', 'dpo']` (hardcoded list, not derived from `TRAINING_METHODS`)
- **Correction:** "automatic" is false. Adding `embedding` requires touching
  **each** hardcoded tuple/list (or refactoring them to import
  `TRAINING_METHODS`). The doc's Section-5 wiring table only lists
  `rtx_backend` + `mac_backend`; it **omits** the three evaluation backends and
  `base_cloud.validate_training_method`. **Architect action:** enumerate every
  hardcoded method tuple and decide add-string vs refactor-to-import.

### 2.3 ⚠️ Mac backend "same (MPS works for ST)" — MISLEADING
- **Claim (Section 5):** `mac_backend.py` gets `get_available_methods() → [..., "embedding"]`, "MPS works for ST."
- **Evidence:** `mac_backend.get_available_methods()` currently returns
  **`['sft']` only** — not even kto/grpo. Its docstring: "Supervised
  fine-tuning using MLX framework optimized for Metal GPU" and loads
  `Trainers/mlx_sft_mac/config/config.yaml`.
- **Correction:** the Mac backend is **MLX-only, SFT-only** today. The ST
  fallback loader (`SentenceTransformer` on MPS) is *not* an MLX path, so
  "embedding on Mac" does not slot into the existing Mac backend shape — it's a
  new integration, not a one-line method add. **Architect must decide** whether
  the ST-fallback loader rides the Mac backend at all in v1, or whether Mac/MPS
  embedding is out of v1 scope.

### 2.4 ⚠️ LoRA + merge + upload "method-agnostic" — SPLIT VERDICT
- **LoRA upload — ✅ CONFIRMED model-agnostic.** `shared/upload/strategies/lora.py`
  `LoRASaveStrategy._execute_save` is a pure `shutil.copytree` of adapter files,
  `requires_gpu() → False`, no model class touched. An ST/Unsloth LoRA adapter
  dir copies through unchanged. Clean reuse. ✅
- **Merge / merged-16bit — ❌ causal-LM-coupled.** `shared/model_loading/merge.py:91`
  hardcodes `from unsloth import FastLanguageModel`; line 100
  `FastLanguageModel.from_pretrained(...)`; line 108
  `model.save_pretrained_merged(..., save_method="merged_16bit")`.
  `shared/upload/strategies/merged_16bit.py` (`Merged16BitStrategy`) drives that
  merge path. An embedding model loaded via `FastSentenceTransformer` is **not**
  a `FastLanguageModel`; re-loading a saved embedding adapter through
  `FastLanguageModel.from_pretrained` will not work.
- **Correction:** the doc's "the repo's `merge.py` already calls
  `save_pretrained_merged` for Unsloth models, so the embedding merge/upload
  path drops straight into the existing strategies" (lines 94–96) is **wrong for
  the merge half**. The *method name* `save_pretrained_merged` matches, but the
  *loader* is causal-LM-specific. **Architect action:** parametrize the merge
  loader (FastLanguageModel vs FastSentenceTransformer) or give embeddings their
  own merge entry point. LoRA-only upload needs no change.
- `peft` dependency presence: the doc claims "peft is already a dependency" —
  confirmed via `Trainers/sft/requirements.txt` (`peft==0.7.1`). ✅ (Note: that
  pin is the *legacy* stack; see §3.)

### 2.5 ✅ Pluggable verifier seam — CONFIRMED clean
- **Claim:** new verifier registers next to `assertion_verifier`, `llm_judge`.
- **Evidence:** `shared/verifiers/builtins/__init__.py` registers via
  import-side-effect (`@register(...)` decorators); current members
  `substring, structure, llm_judge, args_match, assertion_verifier,
  tool_sequence`. Adding `retrieval_verifier` = new module + one import line. ✅
- **Caveat:** the doc names the type string `retrieval_metrics`
  (Section 6) but the file `retrieval_verifier.py`; the registry key is what
  matters — architect should fix the type-string vs filename naming once.

### 2.6 ⚠️ `EvaluationRecord` "optional retrieval field" — it's an EXTENSION, not reuse
- **Evidence:** `Evaluator/runner.py:44` `@dataclass class EvaluationRecord`
  fields are all per-prompt-response: `case: PromptCase`, `response_text`,
  `validator`, `latency_s`, `raw_response`, `error`, `behavior`, `environment`,
  `judge`, `scoring`, `correctness`, `conversation_trace`. **No `retrieval`
  field.** The `status` property (lines 85–103) branches only on
  error/correctness/environment/judge — **no retrieval branch**.
- **Correction:** the doc (Section 6) calls this "Extend" — accurate — but the
  architect must define (a) the `retrieval` payload type and (b) how a retrieval
  score maps into the `status` pass/warn/fail ladder (recall@k/nDCG are
  continuous metrics, not boolean pass/fail; a threshold convention is needed).

### 2.7 ⚠️ experiment-tracking `primary_metric: ndcg@10` mapping — plausible, but format undefined
- **Evidence:** `shared/experiment_tracking/adapters.py:55–57` — the shared
  SFT/KTO lineage→RunRecord helper sets `run_type=f"cloud_{method}" if cloud
  else method`. There is a `cloud_` prefix convention (also `cloud_grpo`).
  `registry.py` supports eval→train run linkage (lines 125–133).
- **Correction:** mapping retrieval results → `RunRecord(primary_metric=ndcg@10)`
  is a **new adapter** that first needs a retrieval-eval lineage format defined
  (parallel to `evaluation_lineage.json`). The reuse claim holds at the
  *registry* level; the *adapter* is new work. Architect must define the
  retrieval lineage schema.

### 2.8 ✅ Recipe schema + method-local pin pattern — CONFIRMED
- **Evidence:** `Trainers/recipes/qwen35_2b_sft_smoke.yaml` shows the schema:
  top-level `name / target / method / provider`, `job.{image, pull_policy,
  transfer}`, **`setup.pip:[...]`** (overlay installs), `run.{method, trainer,
  ...}`, `model`, `dataset`, `training`, `lora`, `artifacts`. The doc's proposed
  `Trainers/recipes/embedding_bge_base_smoke.yaml` (`method: embedding`,
  `target: local|both`, `setup.pip` adding ST+faiss) **matches this schema**. ✅
- **Crucial corroboration for §3:** that recipe's own comment reads *"Qwen3.5
  needs transformers >=5.2.0. The current official Unsloth image still ships
  transformers 4.57.1, so pin a compatible stack here"* and overrides
  `transformers==5.5.0`, `trl==0.22.2`, unsloth/unsloth_zoo from git. **The
  repo's working pattern is: image ships old transformers → recipe `setup.pip`
  overrides.** This directly informs the §3 image-floor correction.

### 2.9 ✅ SFT trainer shape + legacy pins — CONFIRMED
- **Evidence:** `Trainers/sft/` = `train_sft.py` (52 KB), `requirements.txt`,
  `setup.sh`, `configs/`, `src/`. `requirements.txt` pins the **legacy** stack:
  `transformers==4.45.2`, `trl==0.11.4`, `peft==0.7.1`, `torch==2.4.1`,
  `huggingface-hub==0.25.0` ("DO NOT upgrade"), and "Do NOT install Unsloth from
  this file." This **confirms** the doc's "trainer-runtime pins are tightly
  version-locked" claim and the "method-local, don't touch legacy SFT pins"
  discipline. ✅ The new `Trainers/embedding/requirements.txt` must be its own
  pinned island.

---

## 3. Unsloth `FastSentenceTransformer` Path — CONFIRMED UPSTREAM, NOT RUNTIME-VERIFIED

### 3.1 What is confirmed (✅ upstream)
Via Unsloth's **official notebooks** (Context7 `/unslothai/notebooks`,
benchmark 86.6, the highest-coverage Unsloth source):

- `from unsloth import FastSentenceTransformer` is **real**.
  `FastSentenceTransformer.from_pretrained(model_name=..., max_seq_length=...,
  full_finetuning=False)` — signature confirmed. `full_finetuning=False` is a
  newer flag (the doc doesn't mention it; relevant to the `full` adapter mode).
- Official embedding notebooks exist for **EmbeddingGemma-300M**,
  **Qwen3-Embedding-0.6B** (a seed model), Qwen3-Embedding-4B, bge-m3,
  gte-modernbert-base, all-MiniLM-L6-v2, all-mpnet-base-v2.
- Pairs with stock `from sentence_transformers import SentenceTransformerTrainer,
  SentenceTransformerTrainingArguments, losses` — **confirms the doc's
  "same training code, two loaders" / ST-API claim.**
- `losses.MultipleNegativesRankingLoss(model)` used in examples (doc cites the
  `Cached` variant — both are real ST losses; cached = memory-efficient).
- LoRA config in examples uses **decoder target modules**
  `["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]` +
  **`task_type="FEATURE_EXTRACTION"`** + `use_gradient_checkpointing="unsloth"`.
- `BatchSamplers.NO_DUPLICATES` is used (matters for in-batch-negative losses —
  a detail the doc's loss section should inherit).
- Reranker/classifier support: the doc's claim is consistent with Unsloth's
  notebook collection description ("embeddings ... and other specialized tasks"),
  but I found **no explicit reranker fine-tuning notebook** in the Context7
  sample — so "reranker supported" is **upstream-claimed, not notebook-verified**
  (v2 concern anyway).

### 3.2 What is NOT verified / contradicted (⚠️/❌ — the de-risk priority)

**❌ The image-floor claim is contradicted by repo evidence.**
- Doc (lines 98–101): "Version floor: `transformers==5.1.0`, `trl==0.27.1`,
  PyTorch 2.9+. The repo's modern Unsloth image
  (`unsloth/unsloth:2026.1.2-pt2.9.0-cu12.8`) satisfies this."
- Repo reality:
  - `.skills/fine-tuning/configs/qwen3_4b_a100.yaml:12` pins
    `unsloth/unsloth:2026.1.2-pt2.9.0-cu12.8-update@sha256:5266c...` — but runs
    `method: sft` and does **not** assert a transformers version.
  - The SFT smoke recipe + `.skills/fine-tuning/SKILL.md:125` (dated 2026-04-22)
    state the official Unsloth image **ships `transformers 4.57.1`** and that
    Qwen3.5 runs need **config-level overrides** (`transformers==5.5.0`,
    `trl==0.22.2`, git unsloth/unsloth_zoo).
  - `cloud-training.md:208–214` documents that `unsloth/unsloth:latest` and
    `2026.2.1-...` tags **failed before trainer import** with numpy/scipy
    mismatches in Phase 1 HF Jobs smokes.
- **Conclusion:** there is **no repo evidence** that the cited image ships
  `transformers>=5.1.0` out of the box; the only hard datapoint says the image
  ships 4.57.1. The embedding recipe will therefore (correctly, per Section 8)
  need a `setup.pip` overlay pinning `transformers>=5.1.0`/`trl>=0.27.1` — and
  **even then**, the repo's history of image/numpy mid-session mismatches
  (`cloud-training.md`) means the exact working overlay is **unproven** and must
  be smoke-tested in the code phase. The Section-2 "satisfies this" sentence
  should be downgraded to "requires a pip overlay, version TBD by smoke test."

**⚠️ `FastSentenceTransformer` is not runtime-verified against any image the repo can reach.**
Existence is confirmed in Unsloth's notebooks; importability from the repo's
pinned/overlaid image is **not** statically verifiable and was (lead-confirmed)
out of PREPARE scope. **Architect must treat the dual-loader capability probe as
a real unknown.**

**⚠️ Model-id namespace question (fast path).** Unsloth's 4-bit notebook list
loads **`unsloth/`-prefixed** repos (`unsloth/Qwen3-Embedding-0.6B`,
`unsloth/bge-m3`). The doc's registry uses canonical `BAAI/`, `intfloat/`,
`thenlper/`, `Qwen/` ids. Whether `FastSentenceTransformer.from_pretrained`
accepts arbitrary canonical HF ids, or prefers/optimizes `unsloth/`-prefixed
mirrors (esp. for 4-bit/QLoRA), is **unverified** — architect should add a
`fast_path_hf_id` override to the registry or confirm canonical ids load.

### 3.3 sentence-transformers / faiss presence — ❌ ABSENT
Grep across the worktree (`.py`, `.txt`, `.yaml`, `.toml`):
- `FastSentenceTransformer` → **only in the design doc**, zero in code.
- `sentence-transformers` / `sentence_transformers` → **zero** anywhere.
- `faiss` → **zero** anywhere.
These are **all-new method-local dependencies** (confirms the doc's Section-8
"new deps" framing). Nothing to reuse; nothing to conflict with.

### 3.4 Contingency (lead's priority de-risk)
**If `FastSentenceTransformer` proves unreachable** in the repo's image (4-bit
incompatibility, numpy/scipy mismatch as seen in `cloud-training.md`, or import
failure), the dual-loader **fast path collapses** and the architect should
design **plain `SentenceTransformer` as the primary loader**, with Unsloth as an
opportunistic accelerator behind a capability probe rather than a load-bearing
assumption. This changes the image/dependency story in Section 8 (a plain
`sentence-transformers` + torch stack does not need the Unsloth image at all for
the fallback path). **Recommend the architect design the fallback as the
correctness baseline and the fast path as an optimization, not the reverse.**

---

## 4. Seed Models — CONFIRMED on HF (with registry corrections)

All four exist. Attributes vs the doc's `model_registry.yaml`:

| Model | Family | Pooling | Prompts | Dim | Max seq | trust_remote_code | Notes / corrections vs doc |
|---|---|---|---|---|---|---|---|
| `BAAI/bge-base-en-v1.5` | BERT | **CLS** ✅ | query prompt **OPTIONAL** (doc implies required) | 768 ✅ | 512 ✅ | No ✅ | "no instruction → only slight degradation." Registry should mark prompt optional. |
| `intfloat/e5-base-v2` | BERT | **mean** ✅ | **`query:`/`passage:` REQUIRED** ✅ | 768 ✅ | 512 ✅ | No | Matches doc exactly. ✅ |
| `thenlper/gte-base` | BERT | **mean** ✅ | **none** ✅ | 768 ✅ | 512 ✅ | No | Matches doc (doc omits dim/prompts; both default-fine). ✅ |
| `Qwen/Qwen3-Embedding-0.6B` | decoder/Qwen3 | **last_token** ✅ | `Instruct: {task}\nQuery:{q}` ✅ | **up to 1024, user-definable 32–1024 (MRL)** | **32768** (doc says 8192 ❌) | No explicit requirement (doc says `true` — verify) | Doc `max_seq_length: 8192` is **low**; native is 32k. MRL/Matryoshka is native → the `matryoshka_dims` registry field is genuinely useful here. Doc's `trust_remote_code: true` is **unconfirmed** — HF card shows standard usage; architect should verify, not assume. |

**LoRA target-module correction (decoder models):** Unsloth's official
Qwen3-Embedding/EmbeddingGemma notebooks use the **full** decoder target set
`[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]` plus
`task_type="FEATURE_EXTRACTION"`. The doc's registry lists only
`[q_proj, k_proj, v_proj, o_proj]` for the decoder entry and **omits
`task_type`**. Architect should reconcile (the MLP-proj modules
`gate/up/down_proj` materially affect adapter capacity).

**BERT LoRA targets:** doc uses `[query, key, value, dense]`. This is the
standard HF-BERT attention+output naming and is plausible, but I could **not
statically confirm** these exact module names resolve for bge/e5/gte under ST's
Transformer wrapper — **architect/code phase should verify module names via a
`named_modules()` dump** before trusting the registry defaults.

---

## 5. What Reuses Cleanly vs What Is New (architect summary table)

| Surface | Doc claim | Verified verdict |
|---|---|---|
| `TRAINING_METHODS` extension point | reuse | ✅ true (watch derived legacy-dir map, §2.1) |
| "automatic" pickup across CLI/backends | reuse | ❌ ≥4 hardcoded method tuples to edit (§2.2) |
| Mac backend method add | reuse | ⚠️ MLX/SFT-only today; ST-fallback is new (§2.3) |
| LoRA adapter **upload** | reuse | ✅ model-agnostic copytree (§2.4) |
| **Merge** / merged-16bit upload | reuse | ❌ hardcoded `FastLanguageModel` (§2.4) |
| Pluggable verifier seam | reuse | ✅ clean import-side-effect register (§2.5) |
| `EvaluationRecord` retrieval field | extend | ⚠️ new field + status-mapping needed (§2.6) |
| experiment-tracking `primary_metric` | reuse/extend | ⚠️ registry reuses; adapter+lineage new (§2.7) |
| Recipe schema + `setup.pip` overlay | reuse | ✅ schema matches; overlay is the right pattern (§2.8) |
| Method-local pinned `requirements.txt` | reuse pattern | ✅ confirmed island discipline (§2.9) |
| `FastSentenceTransformer` fast path | reuse Unsloth stack | ⚠️ exists upstream, NOT runtime-verified (§3) |
| Image satisfies transformers≥5.1.0 | reuse image | ❌ image ships 4.57.1; needs overlay (§3.2) |
| sentence-transformers / faiss present | new dep | ✅ confirmed absent → all-new (§3.3) |

---

## 6. Open Risks / Unknowns for the Architect to Resolve

**R1 — Dual-loader capability probe (HIGH, lead-priority).** `FastSentenceTransformer`
is unverified in the repo's reachable runtime. Design the probe so the **plain
`SentenceTransformer` fallback is the correctness baseline** and Unsloth is an
optimization (§3.4). Define probe semantics: import success? CUDA available?
4-bit support? Decide what happens on Mac/MPS/CPU explicitly.

**R2 — Merge-loader split (HIGH).** `shared/model_loading/merge.py` hardcodes
`FastLanguageModel`. Decide: parametrize the merge loader by family, or give
embeddings a dedicated merge entry point. LoRA-only upload is unaffected.

**R3 — Method-string duplication (MEDIUM).** Adding `embedding` requires editing
the hardcoded tuples in `llamacpp_backend.py:131`, `mlc_backend.py:69`,
`unsloth_backend.py:63`, `base_cloud.py:113 validate_training_method`,
`rtx_backend.get_available_methods`, and `mac_backend.get_available_methods`.
Decide add-string vs refactor-all-to-import-`TRAINING_METHODS`. Note: the
**evaluation** backends are eval-time, not train-time — confirm whether
`embedding` even belongs in those eval-backend tuples (it may not).

**R4 — Retrieval score → pass/fail mapping (MEDIUM).** `EvaluationRecord.status`
has no continuous-metric branch. recall@k/MRR/nDCG are continuous; define
threshold conventions per scenario (e.g. `min_ndcg@10`) so retrieval results
produce a pass/warn/fail.

**R5 — Retrieval lineage + tracking adapter (MEDIUM).** Define the
retrieval-eval lineage format (parallel to `evaluation_lineage.json`) before the
`primary_metric: ndcg@10` adapter can be written.

**R6 — `frozen_head` nonstandard-pooling caveat (MEDIUM).** Unsloth warns custom
heads/nonstandard pooling need verification (doc lines 104–106). The
`frozen_head` mode (Dense projection over a frozen base) and any cross-encoder
reranker (v2) must be validated against a plain-`SentenceTransformer` baseline.
Architect should make "compare-to-ST-baseline" a required gate for frozen_head.

**R7 — Registry attribute corrections (LOW-MEDIUM).** Fix per §4:
Qwen3-Embedding `max_seq_length` (8192→up to 32768), confirm its
`trust_remote_code`, add decoder MLP-proj LoRA targets + `task_type=
FEATURE_EXTRACTION`, mark bge prompt optional, add a `fast_path_hf_id` /
`unsloth_id` override for the fast-path namespace question (§3.2). Verify BERT
LoRA module names (`query/key/value/dense`) via `named_modules()`.

**R8 — QLoRA deferral is correct (LOW).** Doc defers QLoRA (Unsloth 4-bit
embedding path WIP). Confirmed consistent with Unsloth's own "QLoRA WIP" note;
the 4-bit model list exists but the doc's deferral is the safe call. No action,
just don't let QLoRA creep into v1 scope.

**R9 — verifier type-string vs filename (LOW).** Reconcile `retrieval_metrics`
(registry key) vs `retrieval_verifier.py` (filename) naming once (§2.5).

---

## 7. Recommendation to the Architect

1. **Design the loader as fallback-primary, fast-path-optional** (R1/§3.4) —
   this is the single highest-leverage de-risk and the lead's stated priority.
2. **Split the merge/upload path** (R2): LoRA upload reuses as-is; merge needs a
   family-parametrized loader.
3. **Treat method-string addition as a multi-file edit, not a one-liner** (R3) —
   or refactor the duplicated tuples to import `TRAINING_METHODS`.
4. **Downgrade the image claim**: plan for a `setup.pip` overlay pinning
   `transformers>=5.1.0`/`trl>=0.27.1` and a **mandatory smoke test** of that
   overlay (the repo has a documented history of image/numpy mismatches).
5. **Correct the registry seed entries** per §4 before locking the
   `EmbeddingModelSpec` dataclass shape.
6. Keep `Trainers/embedding/requirements.txt` a pinned island; never touch the
   legacy SFT pins.

The doc is a good blueprint; with the §2 corrections and §6 risks resolved, the
architect has solid anchors.

---

## 8. Evidence Index (for traceability)

| Claim | File:line / source |
|---|---|
| TRAINING_METHODS tuple | `shared/utilities/paths.py:11` (+ derived dirs L13–17) |
| Hardcoded method tuples | `tuner/backends/evaluation/{llamacpp:131,mlc:69,unsloth:63}_backend.py`, `tuner/backends/training/cloud/base_cloud.py:113` |
| Mac backend SFT-only | `tuner/backends/training/mac_backend.py` (`get_available_methods → ['sft']`) |
| LoRA upload agnostic | `shared/upload/strategies/lora.py` (`LoRASaveStrategy`, copytree, `requires_gpu→False`) |
| Merge causal-LM-coupled | `shared/model_loading/merge.py:91,100,108` (`FastLanguageModel`) |
| Merged16Bit strategy | `shared/upload/strategies/merged_16bit.py` (`Merged16BitStrategy`) |
| Verifier registration | `shared/verifiers/builtins/__init__.py` (`@register` side-effect) |
| EvaluationRecord shape | `Evaluator/runner.py:44` (dataclass, no retrieval field; status L85–103) |
| tracking adapter run_type | `shared/experiment_tracking/adapters.py:55–57`; registry linkage `registry.py:125–133` |
| Recipe schema + overlay | `Trainers/recipes/qwen35_2b_sft_smoke.yaml` (setup.pip transformers==5.5.0) |
| Image ships transformers 4.57.1 | `.skills/fine-tuning/SKILL.md:125` (2026-04-22), smoke recipe comment |
| Image-profile import failures | `.skills/fine-tuning/reference/cloud-training.md:208–214` |
| Legacy SFT pins | `Trainers/sft/requirements.txt` (transformers==4.45.2, peft==0.7.1, "no Unsloth here") |
| FastSentenceTransformer API | Context7 `/unslothai/notebooks` — EmbeddingGemma/Qwen3-Embedding notebooks |
| ST trainer pairing | Context7 `/unslothai/notebooks` (SentenceTransformerTrainer + losses.MNRL) |
| bge-base-en-v1.5 attrs | https://huggingface.co/BAAI/bge-base-en-v1.5 |
| e5-base-v2 attrs | https://huggingface.co/intfloat/e5-base-v2 |
| gte-base attrs | https://huggingface.co/thenlper/gte-base |
| Qwen3-Embedding-0.6B attrs | https://huggingface.co/Qwen/Qwen3-Embedding-0.6B |
| sentence-transformers/faiss absent | grep over worktree (zero hits outside design doc) |
