# Protocol: diagnose ingestion

Context: planning, preflight, execution, reconciliation, or verification did not
produce a verified result.

## Mission

Resolve the typed, sanitized failure at the correct boundary without leaking or
silently rewriting source material.

## Steps

1. Read `../references/lifecycle-and-diagnostics.md` and classify the public
   `status`, `error_code`, and `diagnostic_codes`; do not seek private exception
   text first.
2. For `invalid_input` or `bootstrap_failed`, correct only config, invocation,
   or project composition, then rerun both validators.
3. For `invalid_selection`, `authority_unavailable`, `source_unsafe`,
   `source_changed`, `limit_exceeded`, or `admission_ineligible`, follow the
   admission-code table in the lifecycle reference. Preserve the closed code;
   never replace it with host paths or exception prose.
4. For `unmatched_source` or `ambiguous_source`, correct declarative include,
   exclude, or binding patterns. Never add a parser-specific repair for one
   corpus.
5. For `parse_failed` or `metadata_invalid`, determine whether the declared
   structure is wrong or the source is malformed. Optional frontmatter permits
   absence, not malformed syntax. If source correction is appropriate, stop and
   obtain separate explicit authorization before editing the source; then create
   a fresh snapshot and plan.
6. For `source_changed`, discard the stale plan and readmit the authorized
   selection. Do not pretend the old immutable snapshot still binds current
   bytes.
7. For `effect_uncertain` with `reconcile_required`, use the public API's
   `reconcile` transition for the exact outcome. Do not republish, delete stage
   material, or construct a new bundle as a repair.
8. For `bundle_invalid`, preserve the evidence, reject the handoff, and create a
   new authorized run only after the underlying integrity problem is understood.
9. Record only aggregate counts, identifiers, digests, and typed codes in the
   checkpoint. Stop if resolution requires new authority.

## Guidelines

- Pattern: fix config for scope/meaning errors and source only when separately
  authorized for genuine source defects.
- Anti-pattern: weaken validation, print the hidden path, or patch a normalized
  bundle in place to make a failure disappear.

## Next

After an authorized correction, return to `configure-markdown-recipe.md` or
`execute-ingestion.md` as appropriate. If authority is withheld, this protocol
is terminal and the blocked checkpoint is the deliverable.
