# Dataset handoff

Context: read after bundle verification, before passing normalized source to
synthetic-data generation or dataset construction.

## Key idea

Ingestion produces normalized source records, not training examples. The handoff
names what is verified and preserves everything still undecided.

## Required handoff

Provide:

- the private `bundle_ref` and `bundle_digest`;
- the `structure_set_digest` and declared structure name/version;
- the named text projection, currently `text` over the `body` field;
- declared metadata fields, currently the logical `path` in the minimal template;
- source, processed, and document counts;
- the snapshot manifest, plan, and outcome digests/references needed for audit;
- explicit verification status and diagnostic codes;
- the aggregate checkpoint created from the shipped template.

The recipient may read the normalized items only inside the authorized private
boundary. It must not depend on the original absolute source root.

## Still undecided

The handoff does not decide:

- segmentation or chunking;
- prompt, completion, message, preference, or reward shape;
- system prompts or tool schemas;
- labels, judges, or quality thresholds;
- sampling balance, duplication policy, or train/eval split;
- tokenizer, context window, or target model;
- publication destination.

Those belong to an explicitly aligned dataset recipe. Preserve the normalized
bundle so different recipes can consume the same verified source identity.

## Refusal conditions

Do not hand off a bundle when verification is false, diagnostics are nonempty,
the exact two-member inventory is absent, counts contradict the public outcome,
or required provenance is missing. Do not repair the bundle in place.
