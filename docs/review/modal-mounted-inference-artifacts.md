# Modal mounted inference artifact transport

Engineering slice, 2026-09-10. Local implementation and qualification in progress.
This is engine-only byte transport, not a serving grant, remote launch admission,
inference runtime lock, Sandbox deployment or live model qualification.

`ModalMountedInferenceArtifactReader` accepts an already-authenticated native
inventory and a borrowed exact mount-root descriptor. Its only callback surface
is `read_artifact`, usable by the existing private SFT materializer. It correlates
the canonical five artifacts to effect-scoped paths and provider-entry identities,
then enforces bounded inventories, regular single-link files, exact sizes and
hashes, retained descriptor identities and EOF directory correspondence. Source
files are never modified; no model weights pass through the operator machine.
The trusted caller must authenticate the physical Volume mapping first: supplied
IDs and deterministic entry hashes prove consistency, not account provenance.

The shared materializer explicitly closes source iterators on completion and
consumer failure. Regression tests retained an otherwise-unclosed generator:
both cleanup cases failed before the fix and passed after it. An active failure
or interrupt is preserved if iterator cleanup also fails. The caller's borrowed
root descriptor is never closed by either materialization or transport.

Sol agents own the new transport, its separate acceptance module and independent
review in nonoverlapping worktrees. The lead owns iterator lifecycle integration,
documentation and qualification. Review identified an unbounded directory census;
it was replaced with streaming enumeration that rejects the sixth member. It
also required independent effect/Volume snapshots and EOF ancestor identity
checks. No cloud calls, credentials, image changes, dependency upgrades, EHR
edits, pushes or merges are part of this slice.

## Qualification

Pending final frozen-source test, package and independent-review results. Existing
training locks remain separate from the still-unimplemented inference closure.
