# CODE Work-Units & Coder Split — Embedding Phase 1

Defines work-units with **explicit file-ownership boundaries** so concurrent
coders don't collide, plus the recommended coder assignment and shared-file
sequencing. Interfaces are in `01_CONTRACTS.md`; this doc is about *who owns
which files* and *what order*.

---

## File-Ownership Map

Each file is owned by exactly one work-unit. No two work-units write the same
file concurrently. Shared/sequenced files are called out in §"Shared-File
Sequencing".

### WU-A — Registry + Trainer + Dual Loader + Merge Seam (backend-coder)

**Owns (new):**
- `Trainers/embedding/train_embedding.py`
- `Trainers/embedding/requirements.txt`
- `Trainers/embedding/configs/config.yaml`
- `Trainers/embedding/configs/model_registry.yaml`
- `Trainers/embedding/src/registry.py`
- `Trainers/embedding/src/model_loader.py`
- `Trainers/embedding/src/data_loader.py`
- `Trainers/embedding/src/losses.py`
- `Trainers/embedding/src/evaluation.py`
- `Trainers/embedding/src/callbacks.py`

**Owns (modified):**
- `shared/model_loading/merge.py` — the family-param seam (§3). **Behavior-
  preservation critical**: default `family="causal_lm"` path unchanged.

**Contracts:** `01_CONTRACTS.md` §1, §2, §3, §6.

### WU-B — Retrieval Metrics + Verifier + Evaluator + Tracking (backend-coder #2 or database-engineer)

**Owns (new):**
- `shared/ml/retrieval_metrics.py`
- `shared/verifiers/builtins/retrieval_verifier.py`
- `Evaluator/config/scenarios/embedding_retrieval_smoke.yaml`
- `Evaluator/recipes/embedding_retrieval_eval.yaml`

**Owns (modified):**
- `shared/verifiers/builtins/__init__.py` — add `retrieval_verifier` import + `__all__` entry.
- `Evaluator/runner.py` — `EvaluationRecord.retrieval` field + status branch (§4.3) **AND the corpus-level routing branch** (§4.3 amendment): when a scenario declares `retrieval_config`, resolve `RetrievalConfig` + call `evaluate_retrieval` ONCE as a SIBLING to the per-completion loop (not inside it). Without this the feature is inert in prod.
- `shared/experiment_tracking/adapters.py` — `retrieval_eval_to_run_record` (§4.4).
- `shared/experiment_tracking/schema.py` — `run_type` enum comment += `embedding`.
- `shared/experiment_tracking/experiment_spec.py` — add `"embedding"` to the `validate()` method-allowlist set at L211 (§5.1.2). **Functional gate** — without it, an embedding `ExperimentSpec` fails validation.
- `Evaluator/requirements.txt` — add `faiss-cpu`.

**Contracts:** `01_CONTRACTS.md` §4, §7.3.

### WU-C — Method-Wiring + Recipe + Image Overlay (devops-engineer)

**Owns (modified):**
- `shared/utilities/paths.py` — add `"embedding"` to `TRAINING_METHODS` (§5.1).
- `tuner/backends/training/cloud/base_cloud.py` — `SUPPORTED_METHODS += "embedding"` (§5.1).
- `tuner/backends/training/rtx_backend.py` — `get_available_methods()` += `"embedding"` (§5.1).
- `tuner/cli/parser.py` — `--method choices += "embedding"` (§5.1, **R3 gap**).
- `tuner/backends/training/cloud/hf_jobs_backend.py` — method list += `"embedding"` (§5.1, **R3 gap**).
- `tuner/backends/evaluation/{llamacpp,mlc,unsloth}_backend.py` — **exclusion comment only** (§5.3); do NOT add embedding to the tuples.
- `tests/trainers/dpo/test_dpo_method_registration.py` — update the `test_named_gate_sites_source_contains_dpo` needles to the new 5-element literals (§5.1.1). **Same commit** as the wiring edits.

**Owns (new):**
- `Trainers/recipes/embedding_bge_base_smoke.yaml` (§7.2).

**Contracts:** `01_CONTRACTS.md` §5, §5.1.1, §7.

> **Pin-update note:** WU-C edits a test file (`test_dpo_method_registration.py`).
> This is a deliberate same-commit pin update justified by §5.1.1 — the
> registration-sweep test's exact-literal needles change because embedding is
> appended to the named-gate lists. The new sibling test
> `tests/trainers/embedding/test_embedding_method_registration.py` is owned by
> WU-D (test convention), not WU-C.

### WU-D — Skill + Fixtures + CI Smoke (docs/devops, or fold into WU-B for fixtures)

**Owns (new):**
- `.skills/embedding-training/SKILL.md` + `reference/{embedding-training,retrieval-eval,triplet-data}.md`
- `.agents/skills/...` + `.claude/skills/...` — **generated** via `sync_skill_trees.py` (never hand-edited).
- `Datasets/embedding/examples/{triplets_smoke,corpus,queries,qrels}.jsonl`
- `tests/` embedding smoke test (§8.3).
- `tests/trainers/embedding/test_embedding_method_registration.py` — sibling of the dpo registration-sweep (§5.1.1): asserts embedding registered at the 5 train-time gates AND explicitly NOT required at the 3 eval-backend tuples (locks the R3 exclusion as intentional).

**Owns (modified):**
- `.skills/fine-tuning/SKILL.md` + `reference/dataset-formats.md` — add `embedding` method/recipe/schemas.
- `.skills/evaluation/` — document `retrieval` verifier + scenario schema.
- `.skills/synethetic-data-generation/` — (light) note triplet scenarios as Phase-2 forward ref.

**Contracts:** `01_CONTRACTS.md` §8.

---

## Recommended Coder Split

| Work-unit | Recommended agent | Why |
|-----------|-------------------|-----|
| WU-A | `pact-backend-coder` | Trainer + loader + merge-seam Python; the largest unit. |
| WU-B | `pact-backend-coder` (#2) or `pact-database-engineer` | Metrics/verifier/Evaluator/tracking — data-shape + scoring logic, separable from WU-A. |
| WU-C | `pact-devops-engineer` | Wiring + recipe + image overlay — infra/config surface. |
| WU-D | `pact-devops-engineer` or a docs-focused coder | Skill + fixtures + CI; can run after A/B land the interfaces, or in parallel for fixtures. |

A 2-coder minimum (WU-A on one, WU-B+C on another) works; a 3-coder split
(A / B / C+D) maximizes parallelism. WU-D's CI test depends on A+B interfaces
existing, so it integrates last (or stubs against the contracts).

---

## Shared-File Sequencing (collision avoidance)

Most files are single-owner. The genuine cross-cutting concerns:

1. **`shared/model_loading/merge.py` (WU-A only).** No other WU touches it.
   The risk is *behavior preservation*, not collision — the causal-LM default
   path must stay identical. Test-engineer verifies SFT/KTO merge post-change.

2. **`shared/verifiers/builtins/__init__.py` (WU-B only).** Single small edit
   (one import + one `__all__` entry). No collision.

3. **`shared/experiment_tracking/{adapters,schema}.py` (WU-B only).** Additive
   (new adapter fn + enum-comment). No other WU writes these.

4. **`Trainers/recipes/embedding_bge_base_smoke.yaml`** — listed under WU-C
   (canonical owner: recipe/image). WU-D references it as a fixture but does NOT
   create it. **Sequence:** WU-C creates the recipe; WU-D's CI consumes it.

5. **The `embedding` method string** appears in WU-A (registry/trainer paths),
   WU-B (tracking `run_type`), and WU-C (`TRAINING_METHODS`). These are
   *different files* — no write collision — but they must agree on the literal
   `"embedding"`. It is the single canonical method string; no variants.

**No file is co-owned.** If a coder finds they need to edit a file owned by
another WU, that is a contract gap — surface it (SendMessage to lead) rather than
silently editing across a boundary.

---

## Cross-Cutting Invariants (all coders honor)

- **`shared/` purity:** `shared/verifiers/` and `shared/ml/` MUST NOT import
  `Evaluator/` or `Trainers/`. The retrieval verifier stays `shared/`-pure;
  registry specs for eval-time embedding are passed in via `RetrievalConfig` by
  the Evaluator caller (§4.2).
- **No backward-compat shims** (repo rule): the merge-seam parametrization
  alters callers directly.
- **Config-driven:** no model-specific behavior in Python; it all lives in
  `model_registry.yaml`.
- **Method-local pins:** `Trainers/embedding/requirements.txt` never perturbs
  the legacy SFT `transformers 4.45.2` island.
- **Overlay pins are TBD-pending-smoke-test:** do not invent version numbers;
  the CODE smoke test produces them.
- **`.skills/` canonical:** edit `.skills/`, then `sync_skill_trees.py`; never
  hand-edit the `.agents/skills` / `.claude/skills` mirrors.

---

## HANDOFF Checklist (for the architect's own HANDOFF; mirrored for CODE)

- [ ] All R1–R9 have a design decision (see `00_INDEX.md` risk map).
- [ ] Every WU has a defined interface (`01_CONTRACTS.md`) + file boundary (this doc).
- [ ] No file is co-owned across WUs.
- [ ] QLoRA (R8) + reranker fenced out of v1.
- [ ] Merge-seam decision (parametrize-in-place) + behavior-preservation constraint documented.
- [ ] Method-wiring decision (surgical add; eval-backends excluded) documented with the exclusion-comment instruction.
- [ ] Overlay pins flagged TBD-pending-smoke-test.
- [ ] CLAUDE.md note flagged (root `Trainers/` list; orchestrator action; not edited in worktree).
- [ ] Future-cleanup flag: dedup the hardcoded method literals to import `TRAINING_METHODS` (separate task).
- [ ] CODE-time verification flag: BERT `lora_target_modules` names via `named_modules()` dump; Qwen3 `trust_remote_code` confirm.

### Test-pin triage (spec-completeness sweep — done this phase)

Result of grepping `tests/` for every surface this design modifies:

| Touched surface | Existing pin? | Pin type | Action |
|-----------------|---------------|----------|--------|
| `parser.py` / `hf_jobs_backend.py` / `rtx_backend.py` named-gate literals | YES — `test_dpo_method_registration.py::test_named_gate_sites_source_contains_dpo` | exact-substring needle | **same-commit needle update** (WU-C, §5.1.1) |
| 3 eval-backend tuples | YES — `...::test_lifecycle_iteration_sites_include_dpo` | presence-marker (`"dpo"`) | **no change** — embedding excluded by design; `"dpo"` stays present |
| 3-method tuple scan | YES — `...::test_no_three_method_tuple_left_unregistered` | grep for 3-tuples | **no change** — 4→5-tuple unaffected |
| `shared/model_loading/merge.py` | NO existing pin | — | test-engineer ADDS behavior-preservation coverage (WU-D notes #1) |
| `Evaluator/runner.py` `EvaluationRecord` | YES (5 evaluator tests) | exercise correctness/judge ladder | change is **additive** (optional field default `None` + guarded branch); existing cases untouched — test-engineer confirms |
| verifier registry/builtins | YES — `test_verifiers.py` | registered-type checks | **additive** new `retrieval` type; `register()` raises only on duplicate — no collision |
| `experiment_tracking` `run_type` / adapters | YES — `test_schema.py`, `test_adapters.py` | `run_type` used as test DATA, NOT pinned as an enum set (run_type is a free string, not validated) | **additive** — no pin to update |

Conclusion: exactly **one** existing pin needs a same-commit edit (the named-gate
needles); everything else is additive or an intentional exclusion. No silent
spec-vs-implementation drift on the touched surfaces.

---

## Test-Engineer Notes (for the TEST phase, task #6)

Highest-value targets, given the design:
1. **Merge-seam behavior preservation** — assert `family="causal_lm"` path is
   identical to pre-change (SFT/KTO merge still works). This is the load-bearing
   regression risk of the whole PR.
2. **Capability-probe totality** — probe never raises; degrades to fallback on
   every failure mode (no CUDA, no unsloth, import error).
3. **Retrieval-metric correctness** — recall@k/MRR/nDCG/MAP against hand-computed
   expected values on a tiny fixture (continuous-metric correctness, not just
   "finite").
4. **Status-ladder branch** — `EvaluationRecord.status` returns pass/warn/fail
   correctly across the threshold boundary (the R4 convention).
5. **`frozen_head` R6 gate** — the compare-to-ST-baseline assertion fires.
6. **Registry validation** — bad family/pooling/qlora-mode → `ValueError`.
