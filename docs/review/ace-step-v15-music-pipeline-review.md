# Peer Review: ACE-STEP v1.5 Music Fine-Tuning Pipeline

> Branch: `feat/ace-step-v15-music-pipeline` · Commits: `7c39b44` (registration SSOT dedup) + `3e06e1d` (pipeline)
> Base: `origin/main` @ `c1737bc` · Reviewed: 2026-06-24 · Status: **PR not yet created** (push sandbox-gated)
> Panel: architect, test-engineer, backend-coder (cross-review #28), backend-coder-2 (cross-review #27), security-engineer (fresh)

## Verdict

**Sound work, mergeable after one cheap Blocking fix.** All five reviewers converged: design is coherent, the contract held without bending, the SSOT dedup is clean and test-enforced, subprocess handling has no injection sink, and `.gitignore` + Dockerfile integrity are good. Local verification is green (2306 passed, 0 regressions vs base; ace_step suite 33 passed / 3 GPU-skipped). GPU `.pt`-shape smoke + image-build pin-capture are CI-deferred per user decision.

- **1 Blocking** (cheap, CPU-runnable): untested argv-translation contract.
- **~9 Minor**: docs/honesty drift, defensive hardening, security foot-gun mitigation.
- **~7 Future**: tracked follow-ups; the weights-wiring item is the most important.

## Cross-corroboration (high-confidence signals)

- **lokr/lora silent-no-op is REAL**: backend-coder byte-confirmed against `args.py` source that `--rank/--alpha` live in a separate arg-group from `--lokr-linear-dim/--lokr-linear-alpha`; test-engineer flagged zero coverage of the branch; security confirmed argv is list-form (no shell injection). Three independent angles → B1 is the right call.
- **Serving-exclusion ratified on both levels**: backend-coder-2 (impl: byte-identical exclusion tuple across all 3 serving backends, negative-test-locked) + architect (domain: diffusion DiT ≠ causal-LM checkpoint, correctly unservable via unsloth/mlc/llamacpp).

## Findings

| # | Finding | Severity | Reviewer |
|---|---------|----------|----------|
| B1 | `config_translation.build_preprocess_argv`/`build_fixed_argv` (pure CPU fns, the subprocess-boundary contract incl. the lokr/lora silent-no-op branch) have **zero tests**. A wrong-branch bug silently mistrains with no error. | **Blocking** | test-engineer |
| M-a | Weights `snapshot_download` is **unwired** → the pinned model revision (`19671f40`) is decorative; a real GPU run would pull upstream `main`. (Not a blocker for THIS PR — GPU path CI-deferred; dry-run needs no weights.) | Minor (borderline) | architect |
| M-b | `local_run_handler` bind-mount not root-jailed: `cache_dir` is RW-mounted + root-chowned → careless/crafted path = host write-as-root + recursive chown. Operator foot-gun (not remote escape; `data_dir` is `:ro`). | Minor (security) | security |
| M-c | `test_pipeline_gpu.py`: stale `_module_available` dead-code guard + skip-reason strings misstate WHY deferred (now #28 landed). Honest relabel: preprocess/train = GPU-execution-deferred; generate = entry-not-built. | Minor | test-engineer |
| M-d | Docstring drift: "train.py preprocess" survives in 3 module docstrings while code correctly emits "fixed --preprocess". | Minor | architect |
| M-e | `schema.py:49` `run_type` doc-comment stale (omits ace_step; non-gate — free-form str). | Minor | backend-coder-2 |
| M-f | `subprocess_runner` Popen has no `cwd=` (verified benign; recommend defensive `cwd=resolve_ace_step_home()`). | Minor | backend-coder |
| M-g | Unknown/None `adapter.type` silently falls back to lora flags (fails loud at argparse downstream; wrapper-side allowlist friendlier). | Minor | backend-coder |
| M-h | Audio verifier: no test at duration exactly == min/max_duration_s (strict-inequality boundary; harden against future `<`→`<=` flip). | Minor | test-engineer |
| M-i | `ACE_STEP_HOME` env seam selects which `train.py` executes (pinned in-container; operator-trust on host). Doc-only. | Minor | security |
| M-j | Cosmetic: config comment lists `--lr / --learning-rate`; code emits `--lr` (both exist). | Minor | backend-coder |
| F-1 | `base_models.py:149` + `doctor_handler.py:652` discovery/doctor hardcode `(sft,kto,grpo,dpo)` → trained ace_step (and embedding) invisible to discovery/doctor. **Pre-existing**, not a #27 regression. SSOT-derive fix. | Future | backend-coder-2 |
| F-2 | `Evaluator/vllm_setup.py:32` shadow `TRAINING_METHODS=('sft','kto')` — name collision with SSOT, different value. Rename. | Future | backend-coder-2 |
| F-3 | Dockerfile base image pinned by tag not `@sha256`; extra pip layer unpinned; cu128 `--extra-index-url` index-confusion surface. Bundles with devops deferred pin-capture. | Future | security |
| F-4 | Main-thread stderr-tee theoretical deadlock under stdout-flood + full-stderr-pipe (low risk; line-buffered drain). | Future | backend-coder |
| F-5 | XL-variant subdir/revision is best-effort (correctly flagged in code). | Future | architect |
| F-6 | Registration-test conceptual overlap between `test_ace_step_registration.py` and the #33 sweep (different files; not harmful dup). | Future | test-engineer |
| — | Serving-asymmetry domain-correctness: **RESOLVED** (architect ratified; no action). | Closed | architect |

## Disposition (orchestrator recommendation)

- **B1** → auto-remediate before merge (test-engineer writes `tests/ace_step/test_argv_contract.py`, CPU P0, imports the pure fns; no source touch).
- **Fix-now cluster** (cheap + substantive): M-b (security containment warn + doc), M-c (honest skip strings), M-d/M-e (docstring/doc drift), M-f (defensive cwd), M-h (boundary test), M-i (ACE_STEP_HOME doc).
- **Defer/skip**: M-g (covered by B1 unknown-type cell), M-j (cosmetic).
- **Track as issues**: F-1, F-2, F-3, F-4, F-5; **M-a weights-wiring** as the top-priority deferred item (gated on the CI-deferred GPU/image smoke). F-6 skip.
