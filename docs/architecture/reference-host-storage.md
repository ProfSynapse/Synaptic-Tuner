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
inputs are explicit constructor arguments; a workflow database does not replace
them. New standalone Modal training runs select one SQLite instance for both
ports under their existing private run directory.

Provider-free tests close and reopen both the generic workflow database and the
existing Modal catalog storage, then construct fresh effects, adapter and host
objects against the same mocked provider. They exercise retained public reads
without another training submission. They do not restore process-local source
payloads or stage effects, enable cancellation/reconciliation, or recreate the
sealed qualification object needed for restricted logs. Live provider restart
and safe continuation remain separate release gates.

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

This supplies the original inputs needed for a future standalone reader.
There is no standalone readback command yet, and it cannot recover signing keys
that older runs never retained. In-flight recovery, continuation and disabled
provider capabilities remain open release requirements.

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
Complete training recovery, provider call catalogs, durable artifact
publication and end-to-end host migrations remain separate release gates.
See the [stable release plan](../plans/stable-v1-release.md).
