# Modal coordinator reader slice

Status: provider-free reader boundary implemented; production composition and
remote-wire transport intentionally disabled.

## Implemented boundary

`ModalCoordinatorRunReader` implements the four generic run-reader operations
without wrapping `ModalVerifiedRunsOperationsV1` or its repository/lifecycle.
Before any injected provider read it reconstructs the exact
`ProviderRunReadRequestV1`, authenticates the Foundation grant, every receipt,
every invalid-evidence envelope, and the complete assessment, and checks the
record, binding, outcome, FOUND receipt, submit command, preparation, workflow
run, provider scope, and authenticated Modal command binding as one lineage.
It invokes the coordinator's complete Foundation reducer over a fresh submit
intent and requires the reducer's exact binding, outcome, and bound run to equal
the request values. Consequently a self-consistent substituted outcome/bound
run cannot replace the job reference carried by the authenticated fresh FOUND
receipt.

The reader and effects adapter share the exact immutable `ModalCommandBinding`.
The catalog resolves it by command digest. Before asking the independent binding
authority to authenticate anything, the reader rejects structural substitutes
and reconstructs a fresh binding from its exact command, preparation-snapshot,
and verified-deployment bytes. Reconstruction reruns the real preparation
adapter and proves the command, executor, payload, deployment selection, and
retained configuration equivalent. The authority then authenticates that fresh
complete snapshot. This local equivalence proof is not provider authentication,
quote-freshness evidence, source availability, or permission to execute.

The injected read transport returns internal immutable snapshots rather than
new persisted or signed schemas:

- observations retain `CrossPlaneIdentityV1`, canonical evidence, explicit
  provider phase, and exact terminal evidence for every closed state;
- logs retain cross-plane identity, chain digest, generation, high watermark,
  and terminal generation/phase together with bounded provider-neutral entries;
- artifact inventory retains an authenticated completion manifest and
  cross-plane identity without reading artifact bodies;
- artifact streams are checked independently for exact inventory membership,
  caller and policy bounds, nonempty exact byte chunks, total size, and SHA-256,
  with truncation or mutation failing before successful iterator completion.

Each internal snapshot's canonical evidence is reconstructed deterministically
from its typed fields; caller-supplied `{}` or contradictory evidence is not
accepted. The reader also requires the verified deployment selection to equal
the injected client binding, binds every cross-plane identity to the deployment
attestation and submit invocation nonce, requires exact operation-scoped output
paths, and retains bounded in-memory log commitments to reject generation/high-
watermark regression and same-query equivocation. Log pages must be contiguous,
query-bound, within requested entry/byte limits, and carry a truncation flag
consistent with their authenticated high watermark.

The reader creates only existing provider-neutral authenticated observation and
log envelopes. It adds no runnable capability, registration, public export,
database, cache, fallback, or compatibility alias.

## Reused lower-level Modal semantics

The transport contract is shaped for the existing real Modal primitives:
`ExplicitModal154ReadFacade` provides explicit-client scope binding, bounded
volume listing, and bounded complete/streaming reads; `TerminalControlPlane`
authenticates terminal records; `LogControlPlane` authenticates metadata and
re-lists, hashes, parses, and chain-validates chunks; and the completion types
retain the five-member artifact inventory and cross-plane commitments.

`CompletionControlPlane.validate()` itself is not suitable for the inventory
operation because it calls `verify_artifacts()` and eagerly reads every artifact
body, including the model. This slice therefore requires a distinct
authenticated inventory transport operation and a later bounded stream. It
does not weaken `artifacts()` into a bulk download.

## Deliberate integration blocker

There is not yet a production Foundation-native Modal read transport. The older
`TerminalExpectationV1`, `LogExpectationV1`, `CompletionExpectationV1`, and
`CrossPlaneIdentityV1` encode legacy operation/namespace assumptions. They may
be reused only after composition proves exact equivalence to the authenticated
Foundation submit command, complete record/assessment, provider outcome, sealed
Modal configuration, and deployment evidence. In particular, copying profile,
resource, quote, source, runtime, or secret digests out of the submitted command
would not independently recompute them.

The shared remote-wire cutover must retain enough sealed operational
configuration and authenticated evidence to recompute those commitments, bind
the Foundation-derived job reference to all terminal/log/completion evidence,
and implement non-eager authenticated inventory plus identity-preserving log
pagination. Until that exists and is independently reviewed, registration and
runnable capabilities must remain off.

## Provider-free verification

The focused tests cover Foundation-authentication-before-I/O, authenticated
observation and log projection, inventory without body reads, exact verified
stream completion, and truncated-stream rejection. They use an injected fake
transport only; they do not prove a Modal client, deployment, cloud lookup,
credential, spend, training run, or artifact retrieval.

Tests are run from the clean engine checkout with the candidate inserted into
`sys.path` inside the Python process, an empty environment, plugin autoload
disabled, `MODAL_IS_REMOTE=1`, no `PYTHONPATH`, and no source writes. The current
regression interpreter is CPython 3.12.9 with pytest 9.0.2; this is regression
evidence, not a declaration that the release supports pytest 9.
