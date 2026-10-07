# Engine-owned rich training contracts

This slice relocates the immutable rich compilation DTOs and deterministic
compiler entry point from the public API package to `tuner.training.contracts`.
It preserves canonical JSON normalization, exact runtime/resource validation,
accelerator and plan digest domains, deterministic workload compilation, and
the static resolver-method binding that prevents dynamic attribute lookup from
becoming part of request resolution authority.

The migrated production consumers are coordinator material, request
resolution, SFT compilation, the planning service, recipe contracts, runtime
dispatch, and Docker model/preparation code. The old public training module is
intentionally unchanged in this bounded slice; the lead-owned atomic cutover
must remove its duplicate rich definitions rather than add aliases or
re-exports. Public execution operations, preflight, submission, outcome, and
facade contracts remain outside this module.

The migrated internal test imports are:
`tests/training/test_training_service.py`,
`tests/training/test_sft_compilation.py`,
`tests/training/test_coordinator_material.py`,
`tests/trainers/sft/test_runtime_v1.py`,
`tests/runtime/test_dispatch.py`,
`tests/runtime/test_artifact_verification.py`,
`tests/execution/providers/test_modal_coordinator_bundle.py`,
`tests/execution/providers/docker_provider_v1/conftest.py`,
`test_model.py`, and `test_preparation.py`. Legacy Modal-operation tests and
public-contract tests are lead-owned removal/replacement work, not compatibility
targets for this internal module.

`tests/training/test_training_compiler.py` mechanically preserves the former
public contract file's rich DTO and compiler regression coverage. It omits only
the four facade-specific tests for accepted verbs, stale preflight, failed
preflight, and typed-stage delegation; the lead-owned public facade cutover
replaces those separately.

The lead-owned cutover must also update `tuner.training.__init__`; the resolution
module no longer acts as a compatibility re-export for the extracted types.

The provider-free tests validate canonicalization, complete plan and host
provenance digest sensitivity, strict field constraints, the complete compiler
pipeline, and static resolver binding. They do not activate a provider or
qualify cloud execution. Runtime tests currently encounter the separately
known packaged offline-worker closure-manifest drift; this slice does not alter
that lead-owned manifest or its hashes.
