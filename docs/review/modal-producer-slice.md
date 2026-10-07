# Modal coordinator producer slice

This slice publishes the result of one already-admitted Foundation `SUBMIT`
effect. It accepts only a `ModalWorkerInvocation` reconstructed by the pure
worker validator and a closed `ModalProcessResult`; it grants no authority and
does not convert a Foundation command into the legacy Modal operation model.

The producer reuses the existing bounded artifact, terminal, completion and
log-metadata schemas, with a separate v2 log-chunk wire. Their identity is derived from the exact submit command,
verified deployment, retained volume identifiers, job reference, and log
generation. A successful process must declare exactly the five artifact roles.
The complete inventory shape and bounds are checked before the first artifact
copy. All control records are constructed and authenticated before the output
directory is exclusively claimed. Each exclusive copy is then verified with a
bounded-memory streaming hash, and the claimed directory is checked for the
exact five regular names and sizes before any control write. The completion
manifest's MAC is written last. A failed process publishes no artifacts and no
completion manifest. Only the closed remote return codes 0 and 120--125 are
accepted, and the admitted chunk and terminal byte limits are enforced before
the first artifact mutation.

The coordinator producer writes only the generic-entry
`synaptic.modal-log-chunk/v2` wire. It captures one UTC timestamp from its
injected clock after invocation and result validation and before mounted I/O,
then constructs one exact `RunLogEntry`: sequence zero, `INFO`/`completed` for
success or `ERROR`/`failed` for a closed failure, with `size_bytes` equal to the
message's UTF-8 size. The v2 codec validates this entry and commits it into the
new chain-node digest. There is no timestamp synthesis from an operator step
and no legacy v1 fallback.

After all evidence bytes and tags validate, the producer exclusively claims
the submit effect's log and evidence directories as well as the successful
artifact output directory. Before the final completion MAC publication it
checks the exact single chunk name and size and the known evidence members.
These are collision checks at publication time, not immutability claims;
readers still relist and revalidate every retained member.

The submit digest commits the complete Foundation predecessor and preparation,
and the retained Host proof maps that predecessor to the authenticated stage
claim. The reused v1 terminal and completion documents do not, by themselves,
carry the launch-claim, stage-claim, or bundle digests. Consequently, a
standalone v1 terminal proves the exact submit/result identity but is not an
independent serialization of the complete Host launch proof. This slice does
not add a weak sidecar or change persisted v1 meanings.

Production composition and registration remain intentionally absent. The
remote worker cutover must compose this producer only after launch admission,
source preparation, fixed process execution, and the retained Host proof are
all wired together.

Correction (2026-09-09): the final publication marker is the completion MAC,
not the manifest body. Timestamped log chunks use the new v2 schema; the
surrounding metadata, terminal and completion records retain their v1 schemas.
