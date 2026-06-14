# Embedding & Reranker Pipeline — Phase 1 Architecture

**Phase:** PACT Architect
**Author:** architect (team pact-3c41533e)
**Date:** 2026-06-14
**Designs:** Phase 1 (v1 core) of `docs/EMBEDDING_RERANKER_PIPELINE_PLAN.md` v1.1
**Consumes:** `docs/preparation/embedding-reranker-phase1.md` (PREPARE verification — wins on conflict with the blueprint)
**Worktree:** `.worktrees/embedding-reranker-pipeline-v1`

---

## Purpose

Lock the Phase 1 contracts so the CODE phase can fan out across multiple coders
with minimal cross-talk. Every risk R1–R9 from PREPARE has a design decision
here, and every CODE work-unit has a defined interface + file-ownership
boundary.

This is **design only** — no application code. YAML schema sketches and
interface signatures in these docs are the contracts; coders implement against
them.

## Document Set

| Doc | Contents |
|-----|----------|
| [00_INDEX.md](00_INDEX.md) | This file — overview, risk-resolution map, reading order |
| [01_CONTRACTS.md](01_CONTRACTS.md) | The locked contracts: registry schema + `EmbeddingModelSpec`, dual-loader, family-parametrized merge seam, retrieval metrics + verifier + Evaluator + tracking, trainer layout, method-wiring, skill/fixtures/CI |
| [02_WORK_UNITS.md](02_WORK_UNITS.md) | CODE work-units with file-ownership boundaries, recommended coder split, shared-file sequencing, and the HANDOFF checklist |

## Reading order for CODE

1. Read this index for the risk map and scope fences.
2. Read `01_CONTRACTS.md` in full — it is the binding interface spec.
3. Read `02_WORK_UNITS.md` for your assigned work-unit and its boundaries.

---

## Scope Fences (non-negotiable)

**In scope (v1, Phase 1):**
- Model registry + `EmbeddingModelSpec`.
- `Trainers/embedding/` trainer — dual loader (fallback-primary), adapter modes `full | lora | frozen_head`.
- Family-parametrized merge seam in `shared/model_loading/merge.py`.
- `shared/ml/retrieval_metrics.py` + `retrieval_verifier` + Evaluator wiring + experiment-tracking adapter.
- Method-wiring (surgical add-string for train-time surfaces).
- `.skills/embedding-training/` skill + checked-in fixtures + CI smoke test.

**Out of scope (deferred):**
- **QLoRA** (R8) — Unsloth 4-bit embedding path is WIP; do not let it creep into v1.
- **Reranker / `Trainers/reranker/`** — Phase 3.
- **SynthChat triplet generation + hard-negative mining** — Phase 2 (the trainer consumes JSONL; it does not generate it). Checked-in fixtures stand in for generated data in v1.
- **Adapter/reranker serving registry** — v2.
- **MTEB/BEIR benchmarking adapter** — Phase 4.

**Repo rules that bind this design:**
- **No backward-compat shims** — when parametrizing `merge.py`, alter callers directly; no re-exports/dual signatures.
- **Config-driven, no hardcoding** — every model-specific detail lives in `model_registry.yaml`.
- **`.skills/` is canonical** — edit there, then `python3 .skills/scripts/sync_skill_trees.py` to refresh `.agents/skills` and `.claude/skills` mirrors.
- **Method-local pins** — `Trainers/embedding/requirements.txt` is its own pinned island; never touch the legacy SFT `transformers 4.45.2` pins.

---

## Risk Resolution Map (R1–R9)

Each PREPARE risk and where it is resolved in `01_CONTRACTS.md`.

| Risk | Summary | Resolution | Section |
|------|---------|-----------|---------|
| **R1** | Dual-loader capability probe (FastSentenceTransformer runtime-unverified) | Fallback-primary: plain `SentenceTransformer` is the correctness baseline; Unsloth fast path selected by a capability probe (import + CUDA + opt-in). Uniform `LoadedEmbeddingModel` return regardless of path. | §2 |
| **R2** | Merge loader hardcodes `FastLanguageModel` | Parametrize `merge.py` by `family` IN PLACE (lead-confirmed). Causal-LM path stays behavior-identical; embedding family dispatches to an ST merge. No parallel module. | §3 |
| **R3** | Method-string duplicated in ≥4 hardcoded tuples | Surgical add-string for **5 train-time** surfaces (`TRAINING_METHODS`, `SUPPORTED_METHODS`, `rtx_backend.get_available_methods`, **plus 2 gaps PREPARE missed**: `cli/parser.py --method choices`, `hf_jobs_backend` method list). **Exclude** `embedding` from the 3 eval-backend discovery tuples (correctness call — embedding eval flows through `retrieval_verifier`, not serving/quant backends). A `tests/` registration-sweep pin requires same-commit needle updates + a sibling embedding registration test. Broader dedup flagged as future cleanup. | §5, §5.1.1 |
| **R4** | Retrieval continuous metric → pass/warn/fail | Threshold convention in the scenario (`min_*` thresholds + a `warn_margin`); `retrieval_verifier` maps continuous metrics to a `VerifierOutput.passed` + a new `EvaluationRecord.retrieval` payload and a status branch. | §4 |
| **R5** | Retrieval lineage format + tracking adapter | Define `retrieval_eval_lineage.json` shape; new `retrieval_eval_to_run_record` adapter mapping `primary_metric=ndcg@10`, `run_type="embedding"`. | §4.4 |
| **R6** | `frozen_head` nonstandard-pooling caveat | `frozen_head` mode requires a compare-to-plain-`SentenceTransformer`-baseline gate. **v1 disposition (lead-accepted A′):** cloud-smoke-deferred, NOT a v1 runtime assertion — the MODE is unit-tested; only the fidelity GATE rides the deferred cloud-smoke checklist as a named, blocking pre-production gate. Phase-2 trigger: becomes a runtime assertion if `frozen_head` graduates to a primary mode. | §2.4, §7.4 |
| **R7** | Registry attribute corrections | Seed entries corrected: Qwen3 `max_seq_length: 32768`, `default_dim: 1024` + `matryoshka_dims`; bge prompt marked optional; decoder LoRA targets add MLP-proj + `task_type: FEATURE_EXTRACTION`; `fast_path_hf_id` override field added; BERT module names flagged for a `named_modules()` CODE-time verification. | §1 |
| **R8** | QLoRA deferral | Out of v1. `adapter_mode` enum is `full | lora | frozen_head` only; no `qlora`. | Scope Fences |
| **R9** | Verifier type-string vs filename | Registry **type key** = `retrieval`; **filename** = `retrieval_verifier.py`. (Type key is the contract; filename follows the `*_verifier.py` sibling convention.) | §4.2 |

---

## Key Anchor Verifications (file:line, confirmed this phase)

| Anchor | Confirmed |
|--------|-----------|
| `shared/utilities/paths.py:11` | `TRAINING_METHODS = ("sft", "kto", "grpo", "dpo")`; lines 13–17 derive dir maps (adding `embedding` auto-adds a dead `rtx3090_embedding` legacy entry — harmless). |
| `shared/model_loading/merge.py:91/100/108` | `merge_lora_checkpoint()` hardcodes `FastLanguageModel`; called via `find_or_create_merged` → `resolve_model_path`. This is the seam to parametrize. |
| `shared/verifiers/builtins/__init__.py:11-18` | Registration is an import-list side-effect + `__all__`; adding `retrieval_verifier` = one import + one `__all__` entry. |
| `shared/verifiers/contract.py` | `VerifierInput` is **per-completion** (`completion_text`, scalar `score`/`passed`). Retrieval is **corpus-level** — §4.2 resolves the impedance mismatch. |
| `Evaluator/runner.py:44-103` | `EvaluationRecord` has no `retrieval` field; `status` ladder (L85-103) has no retrieval branch. §4.3 extends both. |
| `shared/experiment_tracking/adapters.py:24-68` | `_training_lineage_to_run_record` helper + `run_type` prefixing. §4.4 adds a parallel retrieval adapter. |
| `shared/experiment_tracking/schema.py:49,61-62` | `run_type` enum lacks `embedding`; `primary_metric`/`primary_metric_name` already exist (clean fit for ndcg@10). |
| `shared/ml/metrics.py` | Classification/regression-only (sklearn, `task_type`). Retrieval metrics are a different family → `shared/ml/retrieval_metrics.py` is a correct **sibling**, NOT an extension of `compute_metrics`. |
| `shared/ml/` + `Trainers/ml/` | A non-causal-LM trainer family (`ml`) already exists — precedent for `embedding` as a peer family. |
| `Trainers/recipes/qwen35_2b_sft_smoke.yaml` | Recipe schema confirmed: `name/target/method/provider`, `job.{image,pull_policy,transfer}`, `setup.pip:[...]`, `run.{method,trainer,...}`, `model`, `dataset`, `training`, `lora`, `artifacts`. The comment confirms image ships transformers 4.57.1 → overlay pattern. |
| `Trainers/sft/requirements.txt` | Legacy pinned island (`transformers==4.45.2`, "Do NOT install Unsloth from this file"). Embedding requirements is a separate island. |

---

## CLAUDE.md note (HANDOFF flag)

`CLAUDE.md` is gitignored in the worktree — this design did **not** edit or
create it. One root-`CLAUDE.md` note for the orchestrator (outside the
worktree): once `embedding` is added, the project-structure `Trainers/` list in
the root `CLAUDE.md` would benefit from an `embedding/` entry. That is an
orchestrator action, not a CODE-phase file edit.
