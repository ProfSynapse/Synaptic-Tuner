# Supported surfaces

Context: read this before invoking ingestion or designing an integration. It
separates stable public contracts from repository-private implementation.

## Key idea

Use either the checked-in `tuner.py ingest` command for the proven local
Markdown recipe or `synaptic_tuner.api.v1.IngestionAPI` with a host-owned
`IngestionOperations` implementation. Everything under `tuner.ingestion` is an
implementation detail, not an integration surface.

## Public CLI

The executable local recipe is:

```text
python tuner.py ingest --config CONFIG --select ALIAS=PATH [--select ALIAS=PATH ...] --json
```

The config is a strict UTF-8 JSON file of at most 1 MiB. It is read without
following a link and is checked for identity changes during the read. `--json`
is required. Each alias must be unique and each path comes from explicit user
authority. V1 retains fixed admission budgets; v2 adds strict member and
aggregate admission budgets under finite safety ceilings and binds the
null-capable versioned Markdown parsing profile. Config semantics and host
paths stay separated.

The command emits one compact JSON object. Expected success has `success: true`
and `status: verified`. Bootstrap, validation, selection, parsing, execution,
and verification failures use closed codes and must not expose source content or
absolute paths.

## Public Python API

Import public types from `synaptic_tuner.api.v1`. The lifecycle is:

1. `admit(SourceAdmissionRequest)` creates an immutable snapshot reference.
2. `propose(StructureProposalRequest)` may return advisory proposals; a proposal
   is never configuration authority by itself.
3. `plan(IngestionRequest)` returns a bound preview and fingerprint.
4. `preflight(plan)` checks the exact plan and has bounded validity.
5. `start(plan, preflight)` accepts only a ready, unexpired, exactly bound pair.
6. `show(run)` obtains the current sanitized outcome.
7. `resume(outcome)` is eligible only for an interrupted failed outcome.
8. `reconcile(outcome)` is eligible only for `reconcile_required`.
9. `result(outcome)` is available only for terminal outcomes.
10. `verify(result)` is eligible only for a succeeded outcome.

The host owns authorization, storage, configuration, credentials, and operation
implementation. The public facade detaches inputs, checks mutation, rebuilds
results, and verifies binding. Do not bypass it by importing private operations.

## Unsupported shortcuts

- Do not instantiate `ProcessLocalIngestionOperationsV1` in user integrations;
  the CLI already composes the proven local implementation.
- Do not call bundle writer/verifier helpers directly to simulate a successful
  ingestion.
- Do not serialize exception text, source values, or absolute paths into a
  replacement lifecycle.
- Do not build a corpus-specific public verb or runtime parser.
