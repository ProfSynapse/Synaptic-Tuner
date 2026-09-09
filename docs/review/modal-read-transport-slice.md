# Modal coordinator read transport slice

This disabled internal slice implements provider reads only after reconstructing
the exact authenticated submit binding and retained stage-to-submit launch
envelope.  Host Foundation, assessment, binding, stage, and launch authorities
remain on the host; the adapter introduces no database, grant, or recovery
authority.  Exact client scope, deployment, configured Volume names, and the
semantic eight-member stage bundle are revalidated before evidence I/O.

Terminal state is accepted only from the bounded signed terminal pair.  Its
absence yields no authenticated generic phase: a provider timeout cannot
distinguish `QUEUED` from `RUNNING`. Returned values, lookup errors, and absence
do not imply completion or finality. Partial or malformed terminal evidence
never falls back to a provider-polled phase. Authenticated live progress needs a
separate future worker evidence contract.

Logs use the Foundation-native v2 chunk codec whose signed records are exact
`RunLogEntry` values, including their real timestamps.  The transport verifies
metadata, exact chunk relisting, sizes, hashes, identity, the complete chain,
and the admitted log policy before selecting a contiguous query page.  Page
byte accounting uses each selected entry's exact UTF-8 `size_bytes`. A
terminal watermark is exposed only after separate terminal authentication.

Artifact inventory authenticates terminal, full logs, and completion metadata,
then relists the exact five operation-scoped outputs without reading their
bodies.  Artifact retrieval is a separate bounded stream: it reauthenticates
the inventory, requires exact membership, verifies byte count and SHA-256, and
rechecks the complete output listing after consumption.  It never imports the
legacy run reader or bulk-reads model artifacts.

Composition, registration, credentials, provider calls, and live-read
qualification remain out of scope.
