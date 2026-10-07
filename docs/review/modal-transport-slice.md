# Modal Foundation host transport slice

This internal slice composes the authenticated Foundation coordinator path with
the pinned Modal 1.5.4 SDK boundary. It is not a public cutover and does not
grant authority. The effect adapter authenticates the complete retained command;
the transport reconstructs and authenticates it again before any SDK read.

`ModalFoundationHostTransport` accepts two consumer-retained fact sources. One
resolves exact rich stage material by command digest; the other resolves the
exact authenticated stage-to-submit launch envelope. They are not databases,
registries, signing services, or authorities. The transport also receives the
existing authentication ports and exact recipe registry needed by host submit
preparation.

For stage, the transport checks the retained material and exact client,
claim content, bounds, and authentication before any SDK read. It then checks
deployment and volume identities and delegates to
`ModalFoundationVolumeWriter`. That writer writes or verifies exactly the
operation-scoped bundle, stage claim, and claim tag. A successful observation
uses the canonical stage-claim content identity.

For submit, all Foundation lineage and bundle semantics pass through
`prepare_modal_submit_dispatch` before provider access. The exact selected
function receives one positional argument: the canonical worker dispatch bytes.
The transport calls `spawn` once and returns only its bounded call ID. It never
uses `remote` and never retries.

For cancel, the authenticated `CancelCommandV2` supplies the exact provider call
ID and reason digest. Modal 1.5.4's `FunctionCall.from_id(id, client=...)` binds
the explicit client. The returned handle ID must equal the target. The retained
deployment wrapper digest is checked against the packaged reviewed lock; that
wrapper configures `single_use_containers=True` and provider retries at zero.
Only then does `cancel(terminate_containers=True)` cancel that exact call while
asking Modal to terminate paid work. It is invoked once.

Preparation, authentication, retained-fact, scope, deployment, and volume
mismatches fail before the mutating SDK method. Once stage provider access,
`spawn`, or `cancel` is entered, an exception is ambiguous and maps to
`INDETERMINATE`; no retry is attempted. Reconciliation authenticates the exact
binding but remains `INDETERMINATE` without provider access. The current command
contains no authenticated way to rediscover a lost submit call ID, so this
slice neither claims absence nor introduces an unsupported recovery store or
finality-proof service.

The tests use provider-free explicit-client fakes. They count stage files,
spawn and cancel calls, exercise a real eight-member semantic stage bundle,
prove mismatches stop before SDK access, and preserve ambiguity after a
mutating call begins. They do not contact Modal, use credentials, submit paid
work, or prove live provider behavior.
