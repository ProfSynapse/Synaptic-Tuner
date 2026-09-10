# Modal chat Foundation effects

Status: implementation and local qualification in progress, 2026-09-10.
Engine-only; no Modal SDK transport or live serving qualification.

## Boundary

`ModalChatEffectExecutor` supports retained chat STAGE/SUBMIT commands only.
`ModalChatReconciliationAdapter` provides lookup for those exact commands.
Their chat-specific descriptors and resolvers use the existing Foundation
minting helpers, broker and reconciliation service. The consumer supplies its
catalog, complete-content authority and transport. No training grant, executor
or provider transport is reused. `ModalEffectOutcome` is reused only as the
existing result value; it does not supply authentication or finality authority.

Admission reconstructs the retained chat binding, compares exact command,
preparation, payload, provider/account/environment scope and descriptor, and
guards caller and owned inputs across callbacks. Lookup derives its resolution
identity from the exact command and configured adapter. A direct call to this
private executor is not a grant check: consumers must compose it through the
existing Foundation broker. The broker owns one-shot admission, verifies actual
authenticated predecessor evidence and retains authenticated result receipts.

Transport is called at most once per admitted dispatch or lookup. Ordinary
transport exceptions cross to Foundation, which exposes closed ambiguity or
interruption diagnostics. An exception is never definite absence, and a lost
create response never authorizes a second create. A returned absence claim
still requires the consumer's existing Foundation finality verifier. Process
control exceptions are not converted into provider evidence.

The provider-free tests use an in-memory Foundation repository and synthetic
content/grant authorities. They prove adapter composition, not production
persistence, live account identity, remote launch authorization or a mounted
Volume's authenticity. CANCEL is still excluded until the owned remote target
and cleanup authority are implemented.

## Qualification

Pending final source freeze, independent adversarial tests, broker integration,
regression selection, installed-wheel import checks and unchanged training lock
verification. Do not interpret this note as completed qualification yet.

## Remaining work

Implement authenticated remote launch and exact Volume-to-mount admission,
reuse the shared materializer and pinned-base model preparer on the worker,
capture a separately reviewed inference runtime/image/bootstrap lock, and add
the explicit-client SDK transport. Durable owned leases, bounded readiness and
requests, exact-target cleanup and the existing ChatSession integration remain
required before live serving. No operator weight upload or local Docker
orchestration step is introduced by this adapter.
