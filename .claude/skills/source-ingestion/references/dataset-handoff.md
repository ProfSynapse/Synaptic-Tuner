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
- each selected context/full-document and target/body text projection, with its
  exact structure ref (name, version, digest), projection name, and `field_ref`;
  mark either role undecided when no projection has been selected. The minimal
  template declares only `text` over `body` and selects no context projection;
- declared metadata fields, currently the logical `path` in the minimal template;
- source, processed, and document counts;
- the snapshot manifest, plan, and outcome digests/references needed for audit;
- explicit verification status and diagnostic codes;
- the aggregate checkpoint created from the shipped template.

The recipient may read the normalized items only inside the authorized private
boundary. It must not depend on the original absolute source root.

Do not describe a selected target as prose-only without checking its declared
field selector: a projection over `document_text` includes valid YAML
frontmatter when the source has it.

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
