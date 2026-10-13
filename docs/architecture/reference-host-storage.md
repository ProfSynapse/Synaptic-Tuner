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
history updates in the reference Data, Evaluation and Pipelines families:

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

This store also accepts the other public storage partitions, but the atomic
head/history integration is scoped to Data, Evaluation and Pipelines.
Complete training recovery, provider call catalogs, durable artifact
publication and end-to-end host migrations remain separate release gates.
See the [stable release plan](../plans/stable-v1-release.md).
