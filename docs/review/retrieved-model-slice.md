# Retrieved model slice

This internal, unregistered SFT-only adapter requires a successful Runs API
receipt replay, the exact five-role verified inventory, and complete artifact
streams matching every retained size and digest. It writes only into a fresh,
exclusive attempt beneath a configured private root opened component by
component without following links. Failure removes only entries whose retained
file and directory identities prove this attempt created them. A substituted
path, unknown child, or otherwise uncertain ownership leaves a closed orphan
for explicit recovery; no recovery command is implemented in this slice.
Cleanup is identity-checked best effort under the private-root assumption,
not an atomic unlink-by-inode guarantee against concurrent hostile same-user
mutation between filesystem calls.

Before the standard semantic stream validator sees an archive, a raw 512-byte
header pass rejects non-regular types, PAX/GNU extensions, sparse/link members,
prefix fields, noncanonical or duplicate flat names, excessive members, and
individual or expanded size violations. It also requires two zero end blocks
and a bounded all-zero trailer. Because the producer's default PAX writer can
emit an extension for numeric fields beyond the USTAR range, this first build
deliberately supports a narrower safe subset and fails closed on otherwise
producer-valid very large PAX archives. Extraction uses no `extractall`; each
already-admitted member is created exclusively and streamed. The issued
in-process result rechecks root and attempt identities, exact inventories, and
every source and extracted file digest before returning model paths.

The injected Runs operations implementation is the trusted boundary. With the
reviewed durable `TrainingOperationsV1` and reader composition, receipt replay
and the immutable verified terminal inventory establish the descriptors;
complete streams bind local bytes to those descriptors. An arbitrary
`RunsAPI` wrapper is not authentication. Its separate reverify, outcome, and
stream calls carry no shared inventory digest that this materializer itself
authenticates. This slice does not manufacture closure authority or repeat
the original remote semantic proof. It has no process-restart receipt adoption,
registry, cache, provider SDK, model loader, CLI, GPU, or public export.
Factory-only construction provides controlled result provenance and integrity;
it is not a new authentication claim or a sandbox against hostile Python
already executing inside the host process.

Secure directory-relative and no-follow primitives are mandatory; unsupported
platforms fail closed rather than using a weaker path-based fallback.
