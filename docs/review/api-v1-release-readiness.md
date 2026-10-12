# API v1 main-merge readiness

Status: working checklist, 2026-10-07. Audited baseline engine source: `80d9a063940bb4381788f900c5311c56bdc7fd66`. The operator authorized fixing confirmed blockers and merging once the final candidate passes review and CI. This note is not itself a green gate or authority for publishing a general release. Final candidate identities and terminal checks must be recorded in the merge PR.

The [product boundary and frozen historical matrix](../architecture/submodule-first-training-v1.md) remain authoritative for their named source. The current branch adds a provider-neutral reference composition. Its phase-exit selection covers seven operational families: training, runs, artifacts, evaluation, chat, data, and pipelines. This is not an exhaustive public contract inventory; ingestion has a separate public facade. These are distinct contracts and optional host compositions, not one universally deployed service. The reference host accepts host-owned stores, authority, provider family and optional backends; its example stores are in memory. The checked-in minimal Modal consumer retains a local attempt claim, but does not supply a packaged, durable public host factory or complete coordinator restart recovery for an arbitrary consuming repository.

## Evidence at this checkpoint

| Area | Evidence and limit |
|---|---|
| API contracts and local behavior | A seven-facade phase-exit selection passed six fake-provider tests at the candidate checkout under local Python 3.13.9 / pytest 9.0.3. This is narrow provider-free evidence, outside the supported CI Python 3.12 / pytest 8 environment. The earlier 789 contract tests passed at `af04218c`; that result belongs to the older source. |
| Current candidate checks | Baseline GitHub CI runs [37680245959](https://github.com/ProfSynapse/Synaptic-Tuner/actions/runs/37680245959) and [37680250554](https://github.com/ProfSynapse/Synaptic-Tuner/actions/runs/37680250554) target `80d9a063`, before the preparation fixes below. Full CI failed on lint and core tests; both full-CI inference shards and the separate evaluator-callers shard passed. The combined provider-free inference shard remains in progress at this checkpoint. These are not green final-candidate results, and Python 3.12 CI does not validate Python 3.10. Read-only local checks reported `CURRENT` for the 98-member training runtime lock, 119-member inference lock, 66-member offline worker closure and packaged worker; skill trees were synchronized. These checks do not replace installed-package or independent release review. |
| Tracked-diff audit | A path-level audit of 1,611 changed tracked paths found no binary additions; the two added JSONL files are test fixtures. It found no obvious private environment, key, model-weight or checkpoint paths. This was not a full file-content or Git-history secret scan, and untracked local files were outside its scope. |
| Modal native training | Earlier exact-source live evidence established observation and verified streaming of the five training artifacts for its reviewed implementation. The current Modal descriptor advertises `observe` and `artifact_streaming`; `logs`, `cancel`, `reconcile`, and `cost_quote` are false. These flags do not themselves authorize start or qualify a new source revision. See [native training qualification](modal-native-training-qualification.md). |
| Recent 9B technical smoke | Operator-local records for an earlier exact-source two-step smoke show five verified artifacts and three naturally completed generation responses. The responses copied material and had poor writing quality. Natural completion checks the generation boundary, not writing quality. These private records are not portable CI evidence. |
| Latest full training attempt | Operator-local evidence shows a later full attempt reached remote `TRAINER_EXECUTE START`. No terminal training result, artifact verification, evaluation, or quality conclusion is recorded here. Its result must be checked against that exact attempt before changing this row. |

The reference implementation and the live Modal slice support a narrow engine API v1 claim. This merge targets a source-tree engine consumed as a submodule. The package-data selection in [`pyproject.toml`](../../pyproject.toml) includes specific runtime assets; broader CLI schemas, scripts, templates and profile assets remain source-tree assets. A complete wheel-only CLI release is outside this claim. The branch does not establish every provider, method, backend, deployment form, or quality target described in older plans. The frozen Phase 2 gate remains a separate acceptance criterion; historical green tests and Modal proof do not silently close it.

## Main-merge gates

The preparation patch makes both CI workflows include
`feat/submodule-cloud-api-v1-windows` in their push filters. Two unused test
bindings were removed without changing assertions or the `importorskip` side
effect. Independent review found no gate or test weakening and corrected the
facade-inventory wording above. PyYAML parsing and `git diff --check` passed.
Ruff 0.16.10 passed over tracked Python files with configured exclusions. The
two affected test selections passed all 31 cases under isolated Linux CPython
3.11.14 / pytest 8.4.2 with Modal 1.5.4 mocks; neither the active launcher nor
the training runtime environment was modified. These local patch checks are
separate from the still-running baseline CI and must not be described as a
green final-commit release gate.

The baseline full CI completed with 12,488 passed, 328 skipped and 33 failures
in its core shard; both inference shards passed (350 and 390 tests). In addition
to the 32 failures described below, the exact completion-call census omitted
the post-training evaluator. Its caller uses `Evaluator.vllm_client.VLLMClient`
and belongs to the existing `OTHER_PROTOCOL_CALLERS` category. Adding that exact
entry preserves the census assertion and does not change the runtime protocol.
The census also excludes the explicitly gitignored `unsloth_compiled_cache`
directory, so generated local trainer files do not masquerade as production
call sites. All four contract tests passed locally; independent review confirmed
the protocol classification and preserved exact-source assertion.

The baseline provider-free core completed with 5,524 passed, 280 skipped and
32 failures. Thirty failures were SDK-dependent diagnostic tests running without
the intentionally excluded Modal SDK; two retained obsolete artifact limits.
The SDK-dependent cases now explicitly require the SDK, and a separate required
CI job admits Modal 1.5.4 / protobuf 6.33.6 before running the full mock-only
diagnostic file. Local isolated validation passed 620 cases with the SDK and
313 cases with 307 skips when SDK imports were blocked. The SDK-free import
boundary remains unchanged. Reader tests now accept the current per-file bound
and reject exactly one byte over it; all 12 passed on Linux. Independent review
found no lost coverage or weakened gate. Neither fix changes runtime behavior.

| Gate | Current disposition | Required review evidence |
|---|---|---|
| Exact candidate and scope | Open | Freeze the proposed HEAD; inventory the branch delta against main, including public API contracts, providers, packaging, locks, tests and documentation. Review the unusually large delta as a whole, including unrelated changes, before merging. |
| Full CI and package checks | Open | Run the complete applicable CI on the exact proposed HEAD. Record test, import/installed-wheel, lock-regenerator and skill-tree sync results; distinguish local tests from CI. Repeat changed-source checks after any fix. |
| Independent review | Open | Obtain independent correctness, security and release review of the proposed diff, with findings resolved or explicitly accepted at the intended narrow scope. |
| Runtime and inference pins | Open | Verify the packaged training and inference lock inventories, source hashes and dependent build artifacts against the exact proposed source. A `CURRENT` hash check establishes agreement with a policy-valid lock, not independent approval of non-hash pins or a live image. |
| Claim and evidence wording | Open | State which API families, providers and workflows are actually qualified. Keep operator-local smoke records separate from portable CI and do not promote a pending full attempt or a natural-stop smoke into a quality result. |

## Outside this merge claim

A general public packaged host factory, universal restart recovery, streamed metrics and logs, checkpoint resume, provider parity across Docker/HF Jobs/RunPod, GGUF output, and a writing-quality qualification remain separate work. The optional reference evaluation, data, chat and pipeline implementations require their own host-supplied backends and authority; existence of a facade or fake test is not live qualification. The current ingestion claim covers the reviewed Markdown/YAML source path, not arbitrary file uploads. No new paid run is required merely to prepare this main-merge review.
