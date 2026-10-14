# Reference host storage

`SQLiteDurableStoreV1` is an optional implementation of API v1's record and
stream storage ports. The consuming host chooses the database path and owns
its lifetime. Importing the public API does not open a database or import
SQLite.

Use an absolute path beneath an existing directory owned by the current user
with mode `0700`. The database must be a regular, singly linked file with mode
`0600`; a new file is created with that mode. The adapter targets a trusted
local POSIX filesystem. Its path checks do not provide a filesystem sandbox.

Pass the **same store instance** to both ports to enable atomic run-state and
history updates in the reference Data, Evaluation, Pipelines and training
workflow stores:

```python
from pathlib import Path

from synaptic_tuner.api.v1.reference.data import build_data_operations
from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1

# The host supplies its clock, configured Data ports and private state directory.
with SQLiteDurableStoreV1(Path(private_state_directory) / "host.sqlite3") as store:
    data = build_data_operations(
        records=store, streams=store, clock=clock, ports=data_ports,
    )
    # Use data through DataAPI, or compose the other configured host families.
```

The same `records=store, streams=store` pair can be supplied to
`ReferenceHostPortsV1`, `compose_reference_evaluation` and
`compose_reference_pipelines`. Keep the connection on its creating thread.
Another host process opens its own instance at the same path; it must also
supply its own configured clients, clock and other composition inputs.
The default SQLite lock timeout is five seconds and can be configured up to
60 seconds. It bounds lock contention, not arbitrary filesystem latency.

## Modal composition

`compose_modal_packaged_reference_host` and its qualification variant accept
optional `records` and `streams` arguments. Supply both; passing neither keeps
the existing in-memory defaults. Passing the same SQLite instance selects the
atomic workflow updates described below. The caller owns opening, closing and
reopening that connection. Store methods are checked before resolving named
secrets or using provider catalogs.

Recomposition still requires the original admitted prepared material, provider
configuration, signing keys and retained Modal call/binding catalogs. These
inputs are explicit warm constructor arguments; a workflow database does not
replace them. The retained reader below supplies only its authenticated known-run read
projection, without reconstructing executable source authority. New standalone
Modal training runs select one SQLite instance for both ports under their existing private run directory.

Provider-free tests close and reopen both the generic workflow database and the
existing Modal catalog storage, then construct fresh effects, adapter and host
objects against the same mocked provider. They exercise retained public reads
without another training submission. They do not restore process-local source
payloads or stage effects, enable cancellation/reconciliation, or recreate the
sealed qualification object needed for restricted logs. Live known-run provider
recovery remains a release gate; arbitrary continuation is not a claim of this
composition.

## Standalone run retention

Before a new standalone Modal run calls `TrainingAPI.start`, it saves the
original host signing keys separately from the exact composition inputs in its
private run directory. `authority-keys.json` and `readback-inputs.json` are
create-only files with mode `0600`; the directory has mode `0700`.
`workflow.sqlite3` holds the generic workflow records and history. The existing
attempt journal continues to hold one-shot claims and Modal transport catalogs.
The runner closes both stores when it returns or fails.

The key file contains the generated host authority/reader keys and runtime
qualification signing key. It does not contain Hugging Face or Modal API
credentials. The input file retains the original request, canonical workload and
configuration bytes, deployment layout, scope, runtime identities and planning
records. It excludes SDK handles, input source leases and the sealed log proof.
These files are private implementation state, not an export format.

Retention must finish and read back successfully before training starts. A
write failure preserves partial files for diagnosis and prevents stage/submit;
provider build and CPU qualification may already have occurred. Existing files
are never overwritten to retry the write. The loader checks private file
custody, byte integrity and run/scope associations. It does not replace typed
runtime validation or grant new execution authority.

The existing `open_retained_modal_runs` reader and generic runs CLI
(`show`, `observe`, `verify`, `download`) now consume those original retained inputs through RunsAPI. Exact known calls
can be observed without current recipe/source replanning; terminal verified reads
avoid provider polling. They cannot restore keys that older runs never retained,
renew a submit claim, discover an unknown call, reconstruct an input lease or
continue interrupted training. Original catalogs, scope and signing keys remain
required. Default/unsupported capabilities remain refused; real provider lifecycle
qualification is separate. See the [current operation matrix](../plans/api-v1-support-matrix.md).

## Transactions and compatibility

The optional `commit_record_and_append` operation checks both the expected
record revision and the expected last history sequence within one SQLite
write transaction. A stale precondition changes neither. A committed
transaction persists both the new head and its event. A process interruption
before commit leaves neither write committed.

Only a shared object exposing that operation selects the atomic path.
Existing hosts with separate record and stream implementations retain their
existing behavior. A failed atomic operation never falls back to separate
writes. Evaluation progress can advance the record revision independently of
the history sequence; both preconditions retain their own values.

Schema initialization is transactional and versioned. Unknown database
schemas are refused; there is no automatic import or migration of unrelated
databases. Future schema changes require a reviewed migration. Back up the
database using SQLite-aware tooling or while all writers are closed; copying
one file from an active database is not an application recovery procedure.

## Recovery scope

Reopening storage preserves opaque records, history and consumed claims. It
does not reconstruct a worker process handle, finish an interrupted LLM
request, resume row generation or establish that a remote job stopped.
Cancellation requests and confirmed cancellation remain different states.

The Data plan index is written separately from genesis. Its deterministic run
identity must prevent a missing index from authorizing another execution.
Existing ambiguous records are retained rather than silently repaired.

The training workflow store writes its plan index before atomic genesis.
If the host stops between those writes, ordinary plan lookup rejects the
index without a head. Atomic head/history storage does not close that third
write boundary. The existing direct-store creation contract still permits
identical creators to converge on one authenticated genesis; this does not
authorize a coordinator to replay a submission.

Training tests close the SQLite connection and compose new host operations
over the reopened database. They verify retained plans, preparations,
authorization commitments, effect records and workflow history, including
reconciliation and repeated cancellation without another submit or cancel
effect. These tests retain the same scripted fake provider as the external
provider and the same configured test secret. They establish host-storage
behavior, not fresh remote-provider qualification.

This store also accepts the other public storage partitions, but the atomic
head/history integration is scoped to the four stores named above.
Supported known-run recovery and optional durable local publication are described
below and in the current operation matrix. Real provider lifecycle parity,
HF final publication and final supported-host migration/release acceptance remain
separate gates; stored state does not promise arbitrary continuation.
See the [stable release plan](../plans/stable-v1-release.md).

## Optional durable publication records

`reference.publication_store.DurablePublicationStoreV1(records, lock_root,
create=True)` supplies the existing publication-store port. The caller owns
the generic record store, including its connection, and an existing private
0700 local POSIX lock directory. Publication records use the existing canonical
codec/transition kernel and generic effects-partition CAS; no second database
or state machine is introduced. The adapter is thread/process affine.

Nonblocking OS locks retain each transfer's file descriptor until completion,
uncertainty relinquishment or owner close. Private root and lock-inode records
prevent another opener from silently selecting a different lock root or a
replacement lock file. Stable lock files are never deleted. Closing or process
death releases local ownership; a fresh reader can fence an orphan only for
existing lookup recovery, never another transfer. Active owners cannot be
fenced. `create=False` requires the original root binding; ownership operations
also require their original lock bindings and files. Missing setup is refused
instead of created during cold reads.

Real SQLite/process controls cover an active competing opener, producer
`SIGKILL`, the first fresh database opener, authenticated fixture receipt lookup,
terminal reads and no repeated transfer. This qualifies local record/ownership
plumbing, not a concrete destination, durable artifact spool, live publication,
network-filesystem locks or distributed ownership. Default reference artifact
composition still has no publication destination; those adapters remain separate.


## Optional local publication spool and destination

`reference.local_publication` supplies `LocalArtifactSpoolV1`,
`LocalPublicationDestinationV1` and `LocalDestinationRegistryV1` for the existing
publication ports. The caller owns separate existing 0700 local POSIX spool and
destination roots, configured role-to-relative-filename mappings, finite byte
limits, an absolute monotonic deadline and the existing publication evidence key.
No default destination or public root export is added.

The spool retains directory and leaf descriptors and writes exclusive 0600 files.
Its total-byte limit applies across its entire lifetime, including partial and
aborted sinks, with at most 100 files and 100 publication directories. Each chunk
is at most 1 MiB and is conservatively charged before the first write. An uncertain
partial write permanently aborts that sink without refunding its reservation.
Per-file limits remain independent. The destination's byte limits apply to each
publication's exact source inventory; each publication has a fresh directory.

The destination copies the already verified spool streams, verifies size/hash/EOF,
fsyncs the files, then writes an exclusive authenticated receipt marker binding the
original command and transfer ownership. Files, marker and directories are
rechecked before returning. Missing, incomplete or invalid markers remain
indeterminate. Lookup verifies the original authenticated marker and files without
writing, adopting, overwriting or retransferring anything. Partial state is retained.

Actual SQLite/public-operation tests kill a producer before and after that marker.
The first fresh database opener uses the existing lookup-only recovery permit:
without a marker it remains ambiguous; with the original valid marker it verifies
without destination writes or source reopening. These tests use signed test-owned
source bytes, not a composed verified RunsAPI source. The native RunsAPI join below has separate CPU evidence; an HF destination
and live publication qualification remain separate nodes.

Retained descriptors and named identity checks detect observed path replacement,
symlinks, special files, ownership/permission changes, hardlinks and mutation during
reads. Cold reads authenticate content rather than claiming a persisted inode
attestation; a same-byte replacement is independently hash-verified. These checks
are not an atomic filesystem transaction, security sandbox, network-filesystem
qualification or physical I/O termination. Deadlines are cooperative and never
reset; filesystem calls may themselves block. Local fsync is not remote durability
or a live publication qualification claim.


## Optional verified run source join

`reference.publication_sources.RunsArtifactSourceV1` describes a source using
only `RunsAPI.show`. The trusted RunsAPI must expose artifact inventories in
stored outcomes only after native verification, as the reference implementation
does. A successful state alone is insufficient: the manifest must be nonempty.
The adapter signs the existing verified-source descriptor with the caller's
original publication authority. Its deterministic verification digest commits
the stored public run/state/artifact projection; it is **not** a native receipt
digest or a new verification/assessment. No provider observation, training start,
reverification or source preparation happens during description.

The adapter opens each role through `RunsAPI.artifacts` and checks equality with
the current stored manifest. Existing public streams and publication materializing
operations enforce exact run/role/bounds, chunk order, sizes, hashes and EOF.
A changed native artifact is refused before any destination write.

`compose_publication_artifacts` joins the caller-owned store, spool, registry,
RunsAPI, publication authority and clock into the existing `ArtifactsAPI`:

```python
from synaptic_tuner.api.v1.reference.publication_sources import compose_publication_artifacts
from synaptic_tuner.api.v1.artifacts_facade import PublicationRequest

artifacts = compose_publication_artifacts(
    runs=runs, store=publication_store, spool=spool, destinations=registry,
    authority_ref="host-publication", authority=publication_authority, clock=clock,
)
result = artifacts.publish(PublicationRequest(verified_run, destination_ref))
```

The caller retains all object lifetimes, connections, original keys and explicit
local configuration. The constructor closes nothing and does not alter default
empty destinations or create a new database. Existing-only cold stores/local
adapters can verify/list or recover an original marker without reopening a spool
or native source streams; source description remains a stored show-only read.

CPU tests join actual public TrainingAPI/native verification of five SFT roles,
public artifact streams and real local publication. Provider CLI replies and ML
outputs are scripted fixtures. Marker-boundary tests use real `SIGKILL`, forked
processes with retained fixture host inputs, and the first newly opened SQLite
connection/host in another process. Postmarker lookup verifies without native
streams, provider calls, grant binding or destination writes; premarker remains
ambiguous. This does not reconstruct an independent remote provider, qualify
live training/publication or establish arbitrary interrupted-work continuation.
