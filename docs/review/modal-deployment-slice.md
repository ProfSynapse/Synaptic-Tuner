# Candidate Modal coordinator deployment

This internal builder defines—but does not deploy or invoke—the Foundation-native
Modal worker. Its static expectation is built from canonical deployment-selection
bytes, never from expiring or self-referential deployment attestation bytes.
Before constructing any SDK object it reconstructs the complete selection from the
exact provider profile and client fields, and matches the environment, named
Volumes, named Secret and key tuple, runtime-requirements digests, packaged image
reference, and fixed coordinator executor. The MAC and model-token environment
symbols must be valid, distinct members of that Secret requirement.
The declared key set is exactly those two symbols and rejects Modal host-token
names, even when a substituted profile and selection agree with one another.

The deployed function accepts exactly one dispatch byte string. Inside the call,
it constructs the fixed HMAC verifier/signer, dual-clone source materializer,
model-preparing credential-stripping process runner, completion producer, and
mounted coordinator worker. The required named Secret supplies only the evidence
MAC key and pinned-model token; Modal host credentials are not attached.
This is a declaration/provisioning constraint, not a read of Secret contents:
Modal's lookup checks required keys only, so the consumer must provision the
dedicated Secret without additional credential values.

The model-cache preparation commit occurs before the child trainer. After worker
completion, the artifact Volume is committed before the control Volume so signed
success evidence cannot become durable before its artifacts. Retries remain zero.

Public exports, deployment, provider registration, and qualification of the six
advertised read, lifecycle, artifact-streaming, and cost-quote surfaces are absent.
The lead-owned runtime-lock maintenance must add the candidate wrapper and
its complete image-bootstrap import closure; the separately authenticated 66-file
trainer closure remains unchanged.
