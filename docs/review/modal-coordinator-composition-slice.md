# Modal coordinator engine composition

The internal composition entrypoint wires only existing reviewed components and
consumer-owned stores and authorities. It performs no provider read, deployment,
registration, credential lookup, or authority creation. It rejects a preflight,
deployment, explicit client, quote, or preparation snapshot that does not belong
to the same retained coordinator configuration before constructing the broker.
The preparation, operational preflight, and public service share one exact clock
instance. The factory reparses the retained execution source and binds its source
fingerprint and deployment-member digest before constructing anything. It also
constructs the read transport and run reader itself from the same facade,
deployment, catalogs, authenticators, recipes, and consumer evidence authority;
a caller cannot substitute a separately wired reader.

The result contains the concrete training service, existing generic run
operations, durable coordinator, retained Foundation boundary, and a provider
registration whose six advertised capability flags are false. The caller may place
that registration in its own lazy registry; this function never mutates a global
registry and does not enable or qualify those advertised surfaces.

The run operations share the same planning and workflow stores, coordinator,
retained Foundation, authenticated reader, authorities, and clock. `APIHost`
can therefore list and show retained workflows. Live outcome, log, and artifact
reads remain deliberately denied while the descriptor advertises the
corresponding flags as false. These flags are not a master start switch; exact
preflight, grants, and Foundation lineage remain the execution authority.
