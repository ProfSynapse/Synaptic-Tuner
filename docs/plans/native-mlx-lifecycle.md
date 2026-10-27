# Native MLX public lifecycle work

The owner requested public progress, cancellation and recovery after successful
native training. Model-quality tuning is outside this work. These operations
need implementation and qualification: the current family submits synchronously
and explicitly refuses public logs, cancellation and reconciliation.

This branch is an implementation candidate. Its first transport component does
not enable public capabilities or qualify a new Mac operation.

## Preserve the current boundary

Use `ProviderFamilyV1`, the existing coordinator/Foundation, public `TrainingAPI`
and `RunsAPI`, and host-owned record/stream ports. Do not create an engine
database, parse console text into progress, move thread-affine SQLite stores to
a worker thread, or infer signal authority from a persisted PID. Legacy default
synchronous submission and canonical command/workload bytes remain unchanged.
Each later live attempt still requires exact reviewed source/runtime and fresh
bounded admission. Retain every uncertain claim and never resubmit to recover.

Progress, stop request, confirmed termination, artifact verification and recovery
are distinct observations. Recovery here reconciles an existing attempt after
caller loss or restart; it does not imply optimizer/checkpoint resume.

## Caller-pumped owned transport

`NativeProcessSession` starts the existing guardian and returns after sending its
finite protocol. `poll` collects the terminal result on the constructing thread;
there is no internal store-writing background thread. The host must pump it and
close the session explicitly (or use its context manager). The existing guardian
owns the actual worker group and enforces deadline/output/file bounds even while
the host is between polls. Protocol delivery has a separate five-second bound;
this is not a physical deadline for arbitrary host-port or OS process-creation
calls.

Only the current owner's anonymous pipe can request stop. Cross-thread stop
writes no store and is idempotent while active; polling/persistence stay on the
owning thread. A new session cannot adopt or signal an existing live attempt.
Stored claim/spawn/outcome records retain the existing invocation bindings.
Missing terminal acknowledgement remains uncertain, including after a failed
spawn-record write. Source drift before a claim prevents dispatch; an occupied
claim prevents every later start. No restart recovery is claimed by this first
component.

CPU fixtures exercise actual guardian subprocesses with a simple worker and a
real thread-affine SQLite store. They never import MLX, alter pytest's hard
resource limits, admit hardware or qualify public lifecycle support.

The startup writer uses `selectors.DefaultSelector`, including for descriptors
above 1,023. A real-guardian regression covers R1's high-descriptor failure; the
earlier consumed attempt is not replayed or repaired.

## Bounded authenticated framing candidate

`native_lifecycle_protocol.py` is a stdlib-only source component, currently
unconnected to guardians, workers or public APIs. `NativeLifecyclePolicyV1`
declares finite progress-record and journal-byte limits. Each canonical numeric
event has an exact invocation/policy binding, sequence, prior-frame digest and
domain-separated HMAC. Only fixed phases and named numeric metrics are accepted;
terminal events require the existing closed confirmed-termination outcome.
Before public terminal admission, also check output counts against the exact
invocation's limits rather than just the codec's global ceiling.

The writer accepts a host-owned fresh private descriptor, checks its retained
identity/mode/link count/size/offset across appends and fsyncs each complete frame.
It reserves a full record for termination. A partial write poisons that writer;
it cannot continue or reopen the same attempt. The reader ignores a bounded
uncommitted partial tail, authenticates every complete frame and refuses bytes
after a terminal frame. Supplying the retained prior snapshot rejects rollback
and replacement of committed prefixes. The host must retain that snapshot and
its custody through the existing store/stream ports. A verified prefix alone
does not establish freshness, completeness, public authority or artifact validity.

`NativeEvidenceAuthority.lifecycle_key` derives an invocation/policy-scoped key
only if the host explicitly allows the new HMAC purpose. The original key stays
host-only. The proposed anonymous-channel delivery to trusted supervision is
not implemented; this component performs no key delivery, source admission,
process execution or public capability promotion. Independently review the
framing/derivation before implementing that delivery. The ML child must receive
only a bounded progress channel, never the derived authentication capability.

## Remaining joins

1. **Versioned lifecycle selection.** Bind the opt-in policy and its finite
   limits in planning/material authority. Preserve absent-policy legacy bytes.
   Public capability flags stay off until independently authenticated exact
   lifecycle proof exists; an internal method or matching digest is insufficient.
2. **Structured progress.** Use trusted worker callbacks with fixed phase/numeric
   fields, per-run binding, monotonic sequences and finite record/aggregate
   bounds. Accept only complete records; preserve committed prefixes across
   partial tail writes. Publish authenticated pages through existing host streams
   and cursors. No prompt, generated text, paths, argv, environment or exception
   strings belong in diagnostic progress.
3. **Public asynchronous submission and cancel.** Return an authenticated known
   run after owned dispatch, before training finishes. Reads and cancellation
   resolve original submit/material bindings. A stop receipt records a request;
   only trusted terminal evidence establishes cancellation. Test completion
   races and do not require mutable source files to mint new stop authority.
4. **Authenticated restart reconciliation.** The current parent-only terminal
   write is insufficient when the owner dies. Review an independently retained
   guardian terminal publication, scoped to one exact invocation and using
   host-controlled authentication. Any derived signing capability must remain
   in trusted host supervision, never the ML child, argv, environment, logs or
   files. The original long-lived key must not be copied. Interrupted or absent
   evidence remains uncertain, never absence proof or permission to replay.
5. **Qualification.** Exercise normal completion, live progress before terminal
   publication, cancellation, caller loss, interrupted acknowledgement and cold
   reconciliation through the public facade. Start with CPU fixtures, then real
   Darwin process tests and separately reviewed bounded actual MLX attempts.
   Verify packages/current CI and independent source/evidence/claim review before
   promoting the corresponding capability. Full desktop distribution is excluded.

The pinned MLX-LM 0.32.2 SFT trainer already accepts `training_callback` with
`on_train_loss_report` and `on_val_loss_report`. The native GRPO loop owns its
numeric update list. Reuse those source-verified seams; no upstream trainer
replacement or model-specific launcher switch is needed for progress.
