# Native MLX public lifecycle work

The owner requested public progress, cancellation and recovery after successful
native training. Model-quality tuning is outside this work. These operations
need implementation and qualification: the current family submits synchronously
and explicitly refuses public logs, cancellation and reconciliation.

This branch is an implementation candidate. Its internal transport/journal join
does not enable public capabilities or qualify a new Mac operation.

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
The legacy session leaves missing terminal acknowledgement uncertain, including
after a failed spawn-record write. The explicit journal opt-in below can recover
an independently authenticated terminal. Source drift before a claim prevents
dispatch; an occupied claim prevents every later start.

CPU fixtures exercise actual guardian subprocesses with a simple worker and a
real thread-affine SQLite store. They never import MLX, alter pytest's hard
resource limits, admit hardware or qualify public lifecycle support.

The startup writer uses `selectors.DefaultSelector`, including for descriptors
above 1,023. A real-guardian regression covers R1's high-descriptor failure; the
earlier consumed attempt is not replayed or repaired.

## Bounded authenticated framing candidate

`native_lifecycle_protocol.py` is a stdlib-only source component joined to the
opt-in guardian/session, but not ML callbacks or public APIs. `NativeLifecyclePolicyV1`
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
host-only. R2 independently accepted framing/derivation before implementation of
the delivery candidate below. Only fixed public test keys have exercised that
candidate. Independently review delivery before using actual host authority.
The ML child receives only a progress writer, never the derived capability.

## Trusted guardian/session join candidate

`NativeLifecycleLaunchV1` binds the policy and exact sibling codec source hash in
the request already hashed by `NativeInvocation`. The session verifies this
selection and source before claiming. After the claim, it creates a private
exclusive journal and retains its device/inode binding through the record port.
Existing collisions and interrupted setup stay consumed and uncertain.

The session delivers the scoped key through an anonymous pipe to the trusted
guardian. Descriptor numbers are control arguments; key bytes never enter
arguments, environment, logs or files. A separate domain authenticates the full
launch specification. Before loading the codec or spawning a worker, the guardian
checks that tag, independently joins argv/environment/limits to the invocation
digest, rechecks the exact request bytes/selection and verifies codec source bytes.
It closes the key pipe before spawning. Only the new progress writer is passed
to the worker; journal, owner and key descriptors are excluded. The omitted-policy
path preserves the original protocol and supervisor call/receipt shapes.

The guardian validates worker-origin events as data, injects its own authenticated
scope/sequence and publishes terminal evidence only after owned group cleanup
confirms termination. Worker-origin terminal/free-text events are refused. Invalid
progress or interrupted publication may leave the attempt uncertain even when
cleanup ran; neither a closed failure exit nor an unsigned stdout receipt proves
termination. Complete valid terminal publication precedes stdout acknowledgement,
so loss of that acknowledgement does not erase the independently retained proof.

Host snapshots retain the private root and exact journal identity, use bounded
`pread` over an observed prefix while permitting append growth, and authenticate
all complete frames before retention. Frames enter the existing **effects stream**
with exact sequence/readback checks, on the store-owning thread. The observation
partition may be truncated and must not hold recovery authority. Each snapshot
refreshes newer retained frames from other readers before checking rollback.
An authenticated retained terminal needs no mutable journal/source file to
establish the same outcome. Reads check the exact invocation's output limits;
missing/partial/unconfirmed evidence stays uncertain, with no adoption or replay.

CPU fixtures cover progress before completion, actual owner-process death,
cancellation, stdout/record/stream acknowledgement loss, cold reads, descriptor
exclusion, custody changes, concurrent-reader prefix rollback and launch tampering.
No Darwin/MLX or public capability proof is claimed. The current production ML
child still needs its versioned request/progress-argument and SFT/GRPO callback
joins; do not use this opt-in with it yet. Actual host authority delivery still
requires independent source review and later exact capture/admission review.
Hosts must own session lifetime explicitly: returning from a CLI and immediately
exiting intentionally loses ownership. Forked hosts must not leak the owner pipe
to descendants; that host behavior is not qualified by these CPU fixtures.

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
4. **Public authenticated restart reconciliation.** The legacy parent-only
   terminal write is insufficient when the owner dies. Independently review the
   opt-in guardian journal above, then bind its exact authenticated outcome into
   public reconciliation. The derived signing capability remains in trusted host
   supervision, never the ML child, argv, environment, logs or files. The original
   long-lived key is not copied. Interrupted or absent evidence remains uncertain,
   never absence proof or permission to replay.
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
