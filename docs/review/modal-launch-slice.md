# Modal Foundation launch-envelope slice

Status: provider-free preparation and admission implemented; not a remote wire
protocol and not registered.

`coordinator_launch.py` prepares and admits a distinct signed
`synaptic.modal-launch-claim/v1` record for purpose `modal-launch-claim/v1`.
It does not reinterpret any legacy Modal record, mint a grant, spawn a function,
or invoke a worker. The existing Foundation submit grant remains the only
mutation authority.

Both preparation and admission reconstruct and externally authenticate the
exact immutable stage and submit `ModalCommandBinding` values. They require an
exact `StageCommandV2` and `SubmitCommandV2`, byte-identical retained
preparation snapshots and deployment bytes, and a valid existing
`modal-stage-claim/v2` signature. They then build the stage intent and invoke
the generic coordinator `_derive_foundation` reducer over the complete stage
record and authenticated assessment. No partial receipt reducer exists here.

The submit command's complete `StagePredecessorV2` must equal the reducer's
fresh authenticated FOUND stage result: provider/profile/account/namespace,
project/run, plan, preparation, workload, stage effect, exact establishing
receipt digest, and Foundation record digest all match. The launch claim binds:

The FOUND provider stage reference must also be
`modal-stage-claim:<sha256(stage-claim-bytes)>`. This is the canonical identity
of the exact authenticated material: the claim itself binds its command,
retained configuration, Volume IDs, key reference, bundle digest, and size.
Thus another correctly signed material for the same stage command cannot be
paired with the recorded stage result.

- the complete canonical submit binding and submit command;
- submit effect and invocation nonce;
- the complete stage predecessor;
- stage binding, record, assessment, reducer binding/outcome, and bound-stage
  reference digests;
- the existing stage claim and opaque bundle digests and bundle size;
- exact control/artifact Volume IDs, retained configured Volume references,
  and key reference; and
- the retained immutable preparation configuration and verified deployment
  through the embedded submit binding.

The reusable admission routine reruns every reconstruction, authentication,
generic Foundation reduction, predecessor comparison, stage signature check,
claim reconstruction, and launch signature check before returning exact
provider-independent values. Its returned command bytes are the exact
`SubmitCommandV2.canonical_bytes`.

## Remaining dependencies

`ModalLaunchEnvelope` is a local host-side validation object. It contains host
Foundation objects and is not a remote serialization or verification protocol.
The remote worker must never receive host grant-signing secrets or instantiate
host authority services. A later remote-admission slice must consume only the
minimal signed launch/stage bytes under already permitted verifier keys and
reconstruct the public lineage needed by the worker.

The staged bundle remains opaque in this slice. No result here proves that it
contains a valid source, workload, artifact contract, runtime closure, or
remote invocation. Foundation-native bundle semantics, mounted admission,
function spawn, worker execution, and registration remain separate blocked
slices. Provider-free success is not evidence of a Modal deployment or job.
