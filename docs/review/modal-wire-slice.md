# Modal remote launch wire admission slice

Status: provider-free pure admission implemented; unregistered and not a remote
transport, worker invocation, or cloud activation.

`coordinator_wire.py` admits the existing signed
`synaptic.modal-launch-claim/v1` together with its exact signed stage claim and
opaque bundle. It bounds and canonical-parses control records before signature
verification and bundle hashing, reconstructs exact stage and submit commands
and their complete retained configuration/deployment bindings, and compares
all available signed projections against explicit worker expectations.

The trusted host dispatch expectation (supplied independently of worker claim
bytes) includes the exact canonical submit command, deployment
evidence bytes, provider/profile/account/namespace scope, app/function target,
executor identity, Volume IDs and configured names, and verification key. This
also fixes the exact stage-claim digest and bundle digest/size. It prevents a
valid launch for another run, target, or same-length signed material from being
replayed merely because it uses the same deployment. Configured Volume names
are rederived from the retained preparation snapshot before comparison; worker
claim fields are never promoted into trusted expectations.

Remote admission recomputes the stage provider reference as
`modal-stage-claim:<sha256(stage-claim)>` and reconstructs the corresponding
stage-evidence binding digest committed by the host launch claim. The included
Foundation binding, outcome, record, assessment, and receipt digests remain
host-attested public lineage: this pure worker routine does not possess enough
Foundation state or authority to rederive or authenticate those host records.

The returned frozen value contains only canonical stage/submit command bytes,
the validated preparation snapshot and deployment bytes, opaque bundle bytes,
the launch-claim digest, effect IDs, Volume IDs, and a key reference. These are
the minimum bytes needed to reconstruct the stage binding and parse and
independently recompile the bundled resolved material in the remote runtime. No provider
SDK/client, filesystem, process, source clone, catalog, grant authority,
Foundation authenticator, signing key, or host service crosses this boundary.
The verifier key can authenticate these scoped wire claims only; it cannot mint
or authorize a Foundation effect.

Provider-free tests are not evidence that the opaque bundle is executable, the
deployment exists, Modal was contacted, a job ran, training succeeded, or any
cloud authority was exercised.
