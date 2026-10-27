# Native MLX public lifecycle work

The owner requested public progress, cancellation and recovery after successful
native training. Model-quality tuning is outside this work. These operations
needed implementation and qualification: the original family submitted
synchronously and refused public logs, cancellation and reconciliation. The
opt-in implementation below is now CPU-tested; actual Mac qualification remains.

This branch is an implementation candidate. No default capability promotion or
new Mac qualification follows from its source/CPU results.

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
No Darwin/MLX capability proof is claimed. R3 independently accepted the internal
guardian/journal source before the worker/public joins below. The complete new
join still requires independent source review and later exact capture/admission.
Hosts must own session lifetime explicitly: returning from a CLI and immediately
exiting intentionally loses ownership. Forked hosts must not leak the owner pipe
to descendants; that host behavior is not qualified by these CPU fixtures.

## Worker and public API candidate

The explicit child request `synaptic-native-child-request/v2` adds exactly one
`lifecycle` selection. It must be paired with the guardian-owned `--progress-fd`;
legacy v1 accepts neither. The child imports only admitted captured source, then
binds the compiled workload and rejects insufficient progress budgets before ML
imports. Its write-only pipe contains no key and uses finite nonblocking delivery.
`NativeWorkerProgress` enforces numeric fields, monotonic update bounds and record
limits. Pinned SFT callbacks report training/in-loop validation; GRPO reports from
its actual update loop. Separate reload/publication events follow final saved
adapter checks. Existing artifacts and absent-callback call shapes stay unchanged.

`NativePlanningV1(lifecycle=...)` commits the exact captured codec/policy into its
profile digest and uses version2 execution descriptors. It requires a separate
`NativeLifecycleGrantV1` and independent authentication port. Qualified grants
also require the ordinary hardware qualification and distinct lifecycle evidence;
restricted grants claim no successful evidence and bind the original restricted
selection, run, attempt and hardware. Matching hashes or constructing a grant
object are not permission. Unsupported/missing authority closes preflight and
capabilities; it never selects a synchronous fallback. Omitted-policy bytes and
default behavior remain unchanged.

The consumer owns `native_lifecycle_session` as a long-lived context and supplies
that exact session/authority/record port to `native_mlx_family`. Public start
returns a known run after owned dispatch. `runs.outcome` pumps observation and
`runs.logs` delivers authenticated bounded pages/cursors from verified frames.
Log timestamps are authenticated host-first-observed times, not GPU event times;
private numeric messages retain exact event digests and stable timestamps across
reads. The host keeps all API/store calls on their owning thread.

The host binds cancellation with `NativeCancellationMaterialV1`, referencing the
original authenticated submit and invocation. `catalog.resolve_submission` uses
a create-only index and rechecks that original material; no source file is needed
to mint an already-owned stop. Cancellation records request acknowledgement
separately from terminal confirmation. It uses only the live pipe, or recognizes
an already terminal target. A zero exit racing cancellation/owner loss requires
the complete bound publication before resolving success. Current cancellation
authorization after preflight expiry uses the existing host's explicit
`allow_current_cancel_authorization=True` policy; it does not renew start authority.

Cold reconciliation reuses the existing Foundation adapter and quiescence ports.
Only authenticated terminal evidence or a signed read-only stage acknowledgement
can close a lost dispatch. Proofs bind the exact Foundation record/request, epoch
and witness; absence/finality proofs always refuse. Orphaned submissions/cancels
are looked up, never re-executed. Missing acknowledgements/terminal data remain
uncertain. The existing coordinator may advance a recovered stage into the
distinct, separately admitted submit effect; a retained-read-only plan cannot
dispatch it. Base host grants remain responsible for every new effect.

Material-key policy must explicitly allow `native-cancellation-material/v1` in
addition to its original purpose. Reader-key policy must explicitly allow the
new capability, cancellation acknowledgement, stage acknowledgement, progress
metadata and quiescence purposes. No default host policy acquires these rights.
The existing restricted host CLI still provides the legacy synchronous workflow;
a reusable config-first lifecycle qualification mode must be completed before
actual Mac cycles. Do not use a throwaway launch script or fabricate admissions.

## Remaining qualification and integration

1. **Source/host workflow review.** Independently review the complete worker and
   public join, complete the reusable host qualification mode, and bind its final
   source/configuration/operations into fresh reviewed root admission.
2. **Actual progress delivery.** Qualify real SFT/GRPO callback delivery before
   terminal publication, matching update counts and declared finite budgets. CPU
   doubles do not establish actual MLX callback behavior or hardware support.
3. **Actual cancellation.** Demonstrate the public request and confirmed bounded
   process cleanup during actual MLX work, plus completion races. Keep request,
   confirmed cancellation, successful completion and artifact verification distinct.
4. **Actual cold recovery.** Demonstrate caller/acknowledgement loss and fresh
   host reconciliation through the public facade on Darwin, retaining all consumed
   claims and original authority after start expiry. No persisted-PID signalling,
   adoption or uncertain replay; no optimizer/checkpoint-resume claim.
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
