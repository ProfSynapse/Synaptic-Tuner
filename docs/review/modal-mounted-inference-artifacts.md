# Modal mounted inference artifact transport

Engineering slice, 2026-09-10. Local implementation, independent review,
combined provider-free and immutable package qualification complete.
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

The new reader module has 29 cases, passing in 0.07 seconds in its author's
isolated worktree. Independent qualification of all inference tests plus that
reader module passed 136 cases in 0.36 seconds on the frozen lead source. The
lead's materializer/retrieval selection passed 69 cases in 0.21 seconds, including
six retained-iterator regressions for malformed chunks, destination write errors
and interrupts, each with successful or failing iterator cleanup. The initial
two malformed-chunk cases were measured failing before the production fix.

The frozen source passed the defined combined selection: **2,403 tests in
337.62 seconds**, on clean CPython 3.12.9 / pytest 8.4.2 without Modal or PyTorch.
That is the previous 2,368-case coordinator/Foundation/provider/training/runtime/
inference selection plus 29 mounted-reader and six iterator-cleanup cases, not
the whole repository suite. Embedded pytest used `python -c`, preserving the
existing multiprocessing fixture's importable child entry point. No production
or test source changed during this run.

Both existing training source locks remain CURRENT: 97 Modal pins and 66 offline
trainer members (679,487 payload bytes). The canonical fine-tuning guidance and
both mirrors pass the checked-in synchronization check. These training locks
remain separate from the still-unimplemented inference closure.

## Immutable package qualification

Exact source commit `0ebaf3a090fe65fe70ee879c938e093642639c25` was archived and
built offline without dependency downloads. The wheel is 2,046,082 bytes with
SHA-256
`9e480a17db1980f21dff1635f1c070c7dcdef8da1bd90ab7fb08fb71267f9aa8`.
Installed-package qualification from a neutral directory passed the checked-in
CI snippet's 37 engine/Evaluator imports and two resource checks. Constructor,
callback and public/private materializer signature binding also passed. Modal
and PyTorch were absent from the disposable package environment; the trainer
environment was untouched.

Independent audit verified all 718 unique ZIP/RECORD entries and all 710 packaged
Python files byte-for-byte against the source archive. The mounted reader hash is
`5816f0c5662de8edad752554ba149d9527ced40ebdceeccc993c136ab099bff3`;
the shared materializer hash is
`77164b6cd53917e2dba9b35bcc23e5140507f674fefd33bdaec388f6cee2116d`.
The source commit has exactly ten intended paths. No unexpected secret/private
artifact filenames were found; this is not a content-level secret scan. No
provider execution, CI dispatch, model loading or live-chat qualification is
implied.
