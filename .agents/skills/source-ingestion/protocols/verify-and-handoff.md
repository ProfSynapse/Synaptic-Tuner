# Protocol: verify and hand off

Context: the public command or API reports a successful ingestion outcome.

## Mission

Confirm the immutable normalized bundle and hand its declared projections to a
later dataset workflow without creating training examples here.

## Steps

1. Read `../references/privacy-determinism-and-bundles.md` and
   `../references/dataset-handoff.md`.
2. Require explicit public verification. For API integrations call
   `IngestionAPI.result(outcome)` only for a terminal outcome, followed by
   `IngestionAPI.verify(result)` for a succeeded outcome. For the CLI require
   `success: true`, `status: verified`, and no diagnostic codes.
3. Confirm the private bundle directory contains exactly `manifest.json` and
   `items.jsonl`. Do not rewrite either member or implement an independent
   verifier.
4. Bind the checkpoint to the public `structure_set_digest`,
   `manifest_digest`, `plan_fingerprint`, `outcome_digest`, `bundle_ref`, and
   `bundle_digest`, plus aggregate source and document counts.
5. When repeatability is required, rerun from independently admitted identical
   bytes and confirm byte-identical bundle members and the same bundle digest.
   Expect run, snapshot, plan, and outcome identities to differ.
6. Copy `../templates/ingestion-checkpoint.md` and fill only aggregate evidence.
   Never include source text, metadata values, or absolute host paths.
7. Hand off the verified bundle reference, structure identity, named text
   projection, metadata declarations, and checkpoint. State explicitly that
   chunking, prompt/response construction, labels, sampling, and train/eval
   splits remain undecided dataset work.
8. Stop when the recipient can consume the verified normalized source contract
   without relying on the original filesystem layout.

## Guidelines

- Pattern: distinguish semantic bundle identity from operational run identity.
- Anti-pattern: call `items.jsonl` a training dataset or infer row semantics
  from its current shape.

## Next

Run `validate-and-sync.md` before delivering changes to this project-local skill
or its checkpoint guidance.
