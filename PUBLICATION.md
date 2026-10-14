# Caller-owned local artifact publication

This host command joins the existing public `RunsAPI` and `ArtifactsAPI` with
caller-owned SQLite, transfer locks, bounded disk spool and local destination.
It publishes the five original native artifacts of a stored, successfully
verified HF training run. It does not train, observe a pending run, verify a
training run or upload a model repository.

Install the exact reviewed engine wheel and its pinned HF host dependencies in
an isolated environment. The checked-in engine Gitlink supplies only the host
example modules; production imports resolve from the installed package. Copy
`publication.example.json` outside the checkout and replace all placeholders
with the exact original retained consumer commitment, run, private destination
root and caller-owned publication SecretRef. Keep the original secret value
outside tracked files and preserve its key reference for cold lookup.

First use the existing training consumer's public `verify` action. Its stored
`show` must expose the verified native five-role manifest. Publication commits
that trusted public stored projection, not a separately reconstructed native
receipt digest.

From a neutral directory, with the named key loaded through the existing dotenv
credential path, run the installed interpreter against the absolute host script:

```sh
/ABSOLUTE/ENV/bin/python -I /ABSOLUTE/HOST/publish_artifacts.py publish --config /ABSOLUTE/PRIVATE/publication.json
/ABSOLUTE/ENV/bin/python -I /ABSOLUTE/HOST/publish_artifacts.py lookup --config /ABSOLUTE/PRIVATE/publication.json
```

`publish` creates its root exclusively. It never reuses an existing or failed
root. `lookup` requires the exact original config binding, key, run, destination,
SQLite schema and lock-root identity. It creates no spool and refuses a missing
publication. Existing kernel recovery performs destination lookup only, without
native streams, provider reads, grants or another transfer. Ambiguous or tampered
state stays retained and returns the fixed failure diagnostic; it is not retry
authority. SQLite may persist normal lookup/verification bookkeeping, while the
destination and spool remain unchanged.

Use a private local POSIX filesystem supporting retained descriptors and flock.
Configured byte limits bound artifact reads and aggregate transfer; the deadline
is cooperative filesystem timing, not syscall termination or a storage quota.
Partial state is preserved. The command never deletes resources or adopts a
different publication. Model archives remain original native archive bytes.

The tests use actual public native verification and SQLite with explicit
scripted provider/ML observations. Their fresh cold composition proves lookup
does not reopen native streams or write destination files. Genuine completed
training and its local publication remain separate live qualification evidence.
