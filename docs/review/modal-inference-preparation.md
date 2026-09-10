# Modal chat preparation binding

Engineering slice, 2026-09-10. Implementation and provider-free qualification in
progress. This slice binds a separately selected inference configuration and
resource quote into the existing Foundation preparation and commands. It creates
no grant, store, provider object, executable registration or serving session.

The source and workload inputs must be immediate results of the existing source
and retained-workload admission boundaries. This factory checks their complete
correspondence and stability; it is not a replacement for current run
reverification. Configuration and quote authentication use the existing evidence
verifier, trusted identities and freshness checks. Parsing or holding a
factory-issued value alone does not prove freshness, runtime qualification or
permission to allocate a GPU.

One exact configuration body describes the selected Sandbox app, explicit client
scope, image and separate inference runtime/source commitments, resources,
storage identities, named-secret requirements and bounded session policy.
Training deployment selection requires Function-specific facts and is not used
to invent a Function for a Sandbox. Existing resource-quote bytes are reused,
but must match the recomputed inference resources and selected scope/profile.
The inference profile is not required to equal the training profile.

Chat uses a distinct session identity, executor and input commitment. Its stage
and submit commands reuse the generic Foundation machinery. Submit requires the
same chat preparation's stage predecessor; the existing Foundation remains
responsible for authenticating the actual receipt, grant and durable effect.
Training authority cannot be reused to authorize a new serving allocation.

Signed configuration is still a statement from a trusted consumer, not an image
inspection. A separate reviewed inference lock, deployment/quote producer,
exact command retention, chat executor/transport and remote admission remain
required before launch. Provider idle timeout is defense in depth, not a proof
of user inactivity or a wall-clock billing deadline. No pins or account objects
are guessed here, and no model weights are downloaded to the operator.

Sol agents own the preparation module, separate tests and independent review in
nonoverlapping worktrees. The lead integrates, documents and qualifies the slice.
EHR, cloud resources, credentials, pushes and merges are out of scope.

## Qualification

Pending frozen implementation, independent review and measured test/package
results. Existing training source locks must remain unchanged and CURRENT.

Review corrections: configuration authentication uses a distinct inference
purpose, while quotes reuse the existing resource-only purpose. The selected
chat profile is independent of training; the app name must be distinct under
this plan. Source and workload snapshots remain stable around clock and
authentication callbacks, and model metadata is rederived from the exact
retained workload bytes. The shared `ChatSessionPolicy` and named-secret profile
validators are reused. Timeout bounds are expressed in seconds, not byte limits.

The retained result has one immutable canonical snapshot. Construction and
access share the same preparation derivation, so inconsistent retained
preparation digests cannot produce a different command. Evidence nonce values
are authenticated identity metadata, not standalone replay protection. Actual
receipt/grant authentication, freshness at execution and durable one-shot
consumption remain outside this preparation factory.
