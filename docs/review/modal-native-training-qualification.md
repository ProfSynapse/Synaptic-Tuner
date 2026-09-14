# Native Modal training qualification — 2026-09-14

Attempt `modal-chat-20260914-h` completed training and native artifact
verification through the minimal consuming project's checked-in launcher.
The owning process emitted `NATIVE_TRAINING_QUALIFIED` and exited zero.
Engine source was `b5794518f427521cdc4bfcf474fd0ad791b57445`; consumer source
was `e4f30d026390dd48973bffd5f580842f01faa151`. The exact call was
`fc-01M2GG4T32SXB228AFPVNEEXSZ`, in `synaptic-smoke-v1`.

The [bounded evidence summary](evidence/modal-training-qualified-20260914h.json)
records the provider identities, native paths, five artifact hashes/sizes,
qualification digest, and review limits. Full consumer-owned records remain in
its private `.synaptic/state/modal-chat/modal-chat-20260914-h/smoke.sqlite3`.
Model, tokenizer, lineage, metrics and workload artifacts remain in the exact
dedicated Modal Volume. Nothing was published to the Hub.

Independent read-only review checked all 41 catalog payload hashes and canonical
JSON, every content-addressed qualification reference, source/call consistency,
and the recorded transitions: queued revision 5, succeeded-unverified revision 6,
verified revision 7. The live process authenticated the evidence and streamed
the five artifact bodies through verification. The later audit cannot rederive
the ephemeral HMAC key; it verifies structure and hash linkage, not fresh MAC
authentication. Configuration binds two training steps; the SQLite records alone
do not independently establish the executed step count. The exact console-log
read returned zero with empty output and no credential-shaped strings.

This qualifies native training observation and artifact verification for the
reviewed implementation. The subsequent descriptor change enables only
`observe` and `artifact_streaming`; logs, cancel, reconcile and cost-quote remain
disabled. It does not qualify inference serving, chat, main, or a release.
Updated source locks and a freshly built, CPU-qualified inference image are
required before the separately bounded public train-and-chat smoke. Existing
image captures must not be reused for changed locked source.

Failed attempts F and G remain preserved. F reached training but its restricted
environment omitted PATH, preventing GCC from finding its linker. G supplied
system paths but omitted the image's pinned Python directory and failed before
training deployment. H's authenticated PATH is
`/opt/conda/bin:/usr/bin:/bin`; image/runtime pins and credential isolation were
not weakened. H also exercises the retained-assessment fix after a real wait,
without generating replacement read authority.
