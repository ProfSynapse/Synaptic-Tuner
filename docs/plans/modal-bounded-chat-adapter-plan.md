# Bounded Modal chat adapter

Status: engineering plan, 2026-09-10. Engine-only; no provider access or live
qualification is implied. The local run-chat composition is not a cloud service.
Slice 1 is complete locally at `ffbb363`, with 2,297 provider-free tests and
immutable-wheel/import qualification recorded in `../review/run-chat-slice.md`.
Slices 2–4 remain unimplemented; no Sandbox has been created for this plan.

## Product boundary

A consumer chooses a runtime for one verified training run and explicitly sends
chat requests. Preparation runs on the selected execution machine. Modal weights
must not travel down to the operator merely to be uploaded again. Successfully
saved training artifacts remain in their existing authoritative storage; chat
does not imply Hub publication, a second registry, or another database.

The runtime-first `open_run_chat(runs, run, *, runtime)` contract passes the run
before materialization. `LocalVLLMRunChatRuntime` retains destination/preparer
policy and preserves the local chain. A remote adapter receives the same run
plus its constructor-injected provider-specific authenticated catalogs. The
generic helper validates the returned run/model/artifact projections; the trusted
adapter must perform current reverification and admission before serving effects.
Local `RetrievedSFTModel`/`ServingTarget` values bind paths and device/inode
identities and must never be presented as remote filesystem proof.

`RunsAPI` alone does not expose native placement: its artifact values are
role/hash/size. The Modal authenticated inventory retains the exact artifact
Volume, effect-scoped paths and provider-entry identities. A remote source must
correlate that inventory and full retained Foundation/launch evidence to the
same public run outcome before mounting or streaming anything.

## Proposed resource ownership

Use one dedicated GPU Sandbox per chat session, with a separate reviewed
inference image/source contract and an existing explicitly selected app. Do not
reuse the training app/function or treat a training submit grant as chat
authority. This is an engineering choice for an exclusive session resource,
not a claim that Function/Cls can never serve models.

Read-only SDK measurement: installed distribution metadata identifies Modal
1.5.4. Its public `Sandbox.create` (`modal/sandbox.py`, line 549 at that pin)
accepts explicit `app`, `client`, `image`, `gpu`, `volumes`, `timeout`,
`idle_timeout` and `readiness_probe`. The similarly named `_create` at line 759
is private, not the public signature to emulate. `environment_name` on public
create is deprecated and only warns; resolve the exact existing app with
`App.lookup(..., client=explicit, environment_name=exact, create_if_missing=False)`
and retain its verified identity instead of passing a deprecated environment
override. GPU requests use the SDK's V1 Sandbox path even if its newer CPU-only
backend configuration is enabled. No ambient client fallback is permitted.

Modal documents Sandbox lifetime and idle limits, but active commands and
connections affect what counts as idle. Therefore provider idle timeout is only
defense in depth, not a user-inactivity timer. The exact scheduling/start point
of the provider lifetime clock remains unverified at this pin; do not claim a
hard wall-clock billing deadline from it alone. Retain a host deadline before
creation, a remote admission/session deadline, explicit startup/request bounds,
and provider lifetime configuration. Client death must not be the only cleanup
trigger. See the [Sandbox lifecycle documentation](https://modal.com/docs/guide/sandboxes).

Use an authenticated connection token for the exact admitted service port;
do not expose a public tunnel or perpetual endpoint by default. The token is a
host-side access credential, not a model/GPU credential. It must never enter
argv, logs, documents, retained command bytes or the inference subprocess.
Disable environment proxy/auth inheritance and redirects and keep HTTP responses
bounded using the existing client policy. The endpoint must be validated against
the retained exact Sandbox binding before attaching its token. See the
[Sandbox networking documentation](https://modal.com/docs/guide/sandbox-networking).

`terminate(wait=True)` has no public timeout argument in SDK 1.5.4. It must not
be wired directly into the existing finite `ChatSession` close contract. Use
bounded authenticated RPC/completion observation and retain unresolved cleanup
against the exact Sandbox ID. Detaching is not termination. A lost creation
response is indeterminate: no automatic second create and no listing-based
guess that the first allocation is absent.

## Implementation slices

1. **Runtime-first consumer boundary.** Move local preparation into the local
   adapter, retain portable model/artifact metadata plus optional local-only
   saved-file access, and prove a fake remote runtime performs no operator-side
   artifact download or model preparation. No compatibility wrapper.
2. **Authenticated remote source and mutation binding.** Correlate public run
   reverification/outcome with retained Modal native inventory without downloading
   model archives. Bind exact source, image, app/environment/client, resource and
   lifetime policy, quote, input artifacts, model revisions and session nonce.
   Reuse unchanged Foundation `STAGE/SUBMIT/CANCEL`, grants, predecessor receipts,
   one-shot attempt consumption and consumer-owned persistence with a distinct
   chat executor and exact chat binding/catalog. Foundation preparations contain
   opaque workload digests; a new effect kind or authority framework is unnecessary.
   The existing training-specific Modal coordinator transports are not reusable
   as chat transports.
3. **Remote worker and provider boundary.** Separate locked inference source/image
   and SDK adapter. Reverify exact mounted artifact bytes using existing bounded
   archive/descriptor-relative primitives; obtain pinned base weights on Modal
   through existing SDK model preparation. Keep named preparation credentials out
   of inference. Respect persistent data ownership and exact Volume identity.
   Create once only after durable authority consumption, retain the returned ID,
   and bound readiness, access-token issuance, request and exact-target cleanup.
4. **Session adapter and qualification.** Compose the existing `ChatSession`
   controller with the authenticated remote client and durable owned lease.
   Test full/LoRA flow, client disappearance, post-create ambiguity, startup and
   request failure, mismatch-before-I/O, exact-target stop, unresolved cleanup,
   replay/duplicate allocation, secret-shape redaction and no local weight upload.
   Independently review source/locks and installed package, then qualify the
   exact pushed source and selected account resources live. Keep raw credentials
   and unredacted provider responses out of evidence.

The current training runtime lock must not be relabeled an inference lock.
Image/ML stack and inference bootstrap inventory need separate deliberate review;
no opportunistic package upgrades or guessed image digest. Actual cloud resource
identities, quote/limits and required mutations remain explicit before a live run.
