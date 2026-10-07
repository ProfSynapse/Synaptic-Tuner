# Skill spec: fine-tuning full-context reference corrections

Status: APPROVED -- user approved the narrow support-policy implementation and local guidance update on 2026-10-02. No packaging.

## Purpose
Correct the existing fine-tuning references to describe the tested generic
full-context evaluation size policy and the explicitly approved retrospective
support-document policy. These are local reference corrections, not a new skill
or workflow redesign.

## Trigger
Preparing complete context-to-completion datasets or diagnosing same-job
evaluations containing complete context bundles.

### Should not trigger
Unrelated legacy tool-call formatting work; independent skill packaging or installations.

## Workflow
1. Replace the stale 128 KiB statement with the implemented finite bounds.
2. Explain serialized envelope overhead, fail-before-submit checks, and unchanged
   serving/context/output limits without adding corpus-specific recipes.
3. Link the engineering verification note; check the reference and sync mirrors.
4. Document the approved exact-ID support opt-in, truthful ancestry, strict
   default behavior, target/revision exclusions, and conditioned-writing label.

## Inputs
The approved runtime change, independent boundary review, and passing regression
tests, plus the existing canonical fine-tuning reference.

## Outputs
Updated `.skills/fine-tuning/reference/modal-jobs.md` and `dataset-formats.md`,
with matching copies in `.agents/skills` and `.claude/skills`.

## Good output
User's requested behavior: "give a context bundle, and it generates the prose";
guidance preserves complete documents and explains actual transport limits.

## Bad output
User's concern: "not over indexing on my use case and file/system"; no vault,
story, metadata-schema, or model-specific transport semantics are introduced.

## Constraints
No new recipes, no skill-router rewrite, no package, no public publishing.
Preserve pins, signatures, no-truncation behavior, and independent serving limits.

## Out of scope
Changing runtime behavior as part of the skill edit, selecting chapter data,
launching training, or revising any other skill section.

## Done
The references match tested code, the stale bound is gone, support opt-in limits
are explicit, and canonical/mirror checks pass without altering unrelated
pre-existing changes.

## Delivery
Project-local source only. The user's explicit no-packaging instruction overrides
the skill-crafter packaging default.

## Open decisions
None for this approved scope. Any broader restructuring requires new alignment.

## Change log
2026-10-02: The previously approved local-only scope includes retrospective-support guidance.

## Proposed amendment: measured response-retention correction

Status: PENDING EXPLICIT APPROVAL; the earlier approved scope is unchanged.

The completed one-epoch run verified all training artifacts, but one chapter
response was discarded by the recorder's 64 KiB text limit after passing the
larger HTTP transport boundary. The proposed documentation-only amendment is to
replace that stale retained-response figure with the tested runtime bound, link
the focused regression evidence, and explain that finite byte/aggregate limits
remain distinct from a null request-level token budget. It changes only the
existing Modal reference and its synced mirrors after the code fix is reviewed.
All original triggers, good/bad examples, genericity constraints, and delivery
rules above remain: no new recipe, no router redesign, no packaging, no provider
submission, no truncation, and no claim that the lost response was recovered.
Done means the reference matches tested code and mirror verification passes.
