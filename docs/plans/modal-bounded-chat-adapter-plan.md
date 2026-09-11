# Bounded Modal chat adapter

Status: engineering plan, 2026-09-10. Engine-only; no provider access or live
qualification is implied. The local run-chat composition is not a cloud service.
Slice 1 is complete locally at `ffbb363`, with 2,297 provider-free tests and
immutable-wheel/import qualification recorded in `../review/run-chat-slice.md`.
Slice 2a (native artifact-source binding) is implemented locally, with its
qualification recorded in `../review/modal-inference-source-binding.md`.
The workload/model-correlation portion of slice 2b is now implemented locally;
its qualification is recorded in `../review/modal-inference-workload-binding.md`.
Separate chat resource/mutation binding and slices 3–4 remain unimplemented;
no Sandbox has been created for this plan.

Correction (2026-09-10): slice 3 now has a private shared materializer and a
mounted-artifact byte transport under local qualification. This is partial
implementation, not an authenticated remote worker or a deployable adapter.

Qualification update (2026-09-10): this transport and its shared iterator cleanup
passed 2,403 combined provider-free tests, independent review and immutable-wheel
qualification; see `../review/modal-mounted-inference-artifacts.md`. Remote
launch admission, chat resource authorization and the inference runtime lock
remain open.

Preparation update (2026-09-10): the chat-specific configuration/resource
preparation is implemented and locally qualified with 2,438 combined tests,
independent review and an exact-commit wheel. It authenticates configuration and
quote inputs, not a serving grant or an inspected inference runtime. Exact chat
command retention, executor/transport, inference lock/deployment evidence
production and remote launch admission remain open; see
`../review/modal-inference-preparation.md`.

Sequencing correction (2026-09-10): implement exact STAGE/SUBMIT chat command
binding and consumer-owned authenticated retention before freezing the separate
inference lock. The remote bootstrap is not yet complete and no inspected
inference image/dependency/Python pins exist; a lock populated now would omit
future source or invent runtime facts. This changes implementation order, not
the required live gates. Current work is recorded in
`../review/modal-inference-commands.md`.

Command-retention update (2026-09-10): exact STAGE/SUBMIT binding and authenticated
consumer-catalog retention are implemented. Independent source review and focused
tests pass, with 2,495 combined regression tests and a corrected exact-commit
wheel independently qualified. A dependency-only install exposed and resolved
an undeclared HTTP-client dependency; CI now performs the same clean check.
The new metadata regression passed separately after the combined run's collection.
See the review note for exact counts and source/package boundaries. This is not
a Foundation dispatch, serving grant or runtime-lock gate. The next implementation
work is the authenticated chat worker and executor/transport, followed by its
complete inference lock and bounded session adapter.

Effects update (2026-09-10): chat-specific Foundation executor, reconciliation
adapter and resolvers are implemented and independently reviewed. The broad
regression selection passed 2,503 tests; the final focused selection passed 59,
with overlap and collection boundaries recorded in
`../review/modal-inference-effects.md`. The exact-commit wheel passed clean
dependency-only imports and an independent archive/RECORD audit. Consumer-owned
Foundation remains the grant, predecessor, one-shot and receipt authority.
Next is authenticated remote worker admission and SDK transport; the separate
inference runtime lock and bounded owned session cleanup remain unfinished.

Launch-admission update (2026-09-10): one chat-specific host-signed claim and
pure worker admission are locally qualified. They reuse actual Foundation stage evidence and
the shared preparation snapshot; do not duplicate the model/source/native
inventory into another runtime truth. Independently supplied configuration and
command expectations anchor admission. This does not implement SDK dispatch,
physical Volume authentication, an executable worker, inference-lock inspection
or owned cleanup. Independent review, all 2,613 selected regression tests and
the exact-commit installed wheel pass; measured results are recorded in
`../review/modal-inference-launch.md`.

Mounted-preparation update (2026-09-10): fresh launch admission is composed with the
existing mounted artifact reader, shared private materializer, pinned-base
preparer and `ServingTarget`. This is internal worker preparation, not an
executable bootstrap, SDK dispatch, inference-lock check or bounded session.
Independent review, 36 final worker tests, 513 affected regression tests and the
exact-commit installed wheel pass. Collection/revision boundaries and measured
results are recorded in `../review/modal-inference-worker.md`.

Runtime-prerequisite slice (2026-09-10): code inspection found that the reusable
vLLM controller always selected the current interpreter and the HTTP client did
not enforce an admitted request-body bound. Explicit interpreter selection and
bounded whole-body serialization are now implemented in those existing
components, with no parallel server/controller. Independent source and wheel
audits, 330 selected regression tests, 77 overlapping caller tests and the
exact-commit installed-wheel checks pass. This does not project or authenticate the
remaining Modal serving settings. Configuration or the separately inspected
runtime lock must still bind generation/readiness/vLLM choices before bootstrap
execution. Qualification and environment limits are recorded in
`../review/inference-runtime-prerequisites.md`.

Serving-projection slice (2026-09-10): explicit generation, readiness and vLLM
choices are required in the authenticated configuration and projected into the
existing worker/runtime values. The prepared result retains original launch
admission and checks expiry after preparation without renewing its lifetime.
The result is non-authoritative data, not an executable service. The future
bootstrap must reverify/rederive it and clamp time before and after startup.
Independent source/package review, 621 selected distinct regression tests and
installed-wheel import/contract checks pass. Qualification limits are recorded in
`../review/modal-inference-serving-projection.md`. Inference image/source-lock
verification, executable bootstrap, SDK transport and owned Sandbox cleanup
remain unfinished.

Shared-deadline prerequisite (2026-09-11): the existing runtime, verified-chat
composition and session controller now accept the same optional process-local
monotonic deadline. Startup cannot grant a new conversation lifetime; shorter
local limits still win, and late responses cannot commit history. Source
`c2efd11` passed 387 primary and three separate signed-worker tests, independent
review, and exact-commit wheel/import checks; overlapping caller/subset lanes
and environment limits are recorded in `../review/inference-shared-deadline.md`.
This is reusable timing
code, not the executable Modal bootstrap or a substitute inference lock. The
training lock is SFT-specific, and no reviewed inference image/Python/dependency
pins exist yet. Complete the inactive bootstrap's source/control boundary before
capturing its distinct inventory; runtime verification must gate actual serving
and provider activation, with no caller-supplied bypass verifier.

Bootstrap implementation checkpoint (2026-09-11): the remote-local worker
composition and concrete packaged inference-runtime verifier at source `165b000`
passed 64 integrated
provider-free tests and independent source review, with 387 existing regressions
passing separately. Exact-source installed-wheel checks also passed; see
`../review/modal-inference-bootstrap.md` for qualification limits. This connects
launch admission, remote preparation,
the original deadline and the existing chat runtime. It does not add the SDK
transport or an authenticated remote request service. The measured inference
runtime/closure/dependency resources are still absent, so the installed worker
must deny before preparation or serving. Next capture and independently review
those actual runtime pins, then finish provider creation, authenticated access
and exact-instance cleanup. No current image identity or live serving proof is
implied by the new verifier's source/configuration consistency checks.

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

   Slice 2a now uses `ModalInferenceSourceBinder`: the same consumer's current
   public verification and retained workflow must match the complete native
   manifest, including its source identity, and remain stable during admission.
   It retains immutable metadata projections without streaming artifact bodies.
   This is not a serving grant or a claim about actual full/LoRA archive kind.
   Slice 2b now correlates the authenticated retained workload/model commitments
   through `bind_modal_inference_workload`, matching the exact workload-record
   hash and size without artifact-body reads. It validates equal immutable
   model/tokenizer revisions and preserves training configuration as metadata.
   Separate inference deployment/resource policy and chat-specific execution
   binding remain open. Do not reuse a training grant or add a new generic
   authority system to fill that gap.

   Current implementation slice (2026-09-10): bind one separately authenticated
   Sandbox inference configuration and the existing resource-only quote into a
   chat-specific Foundation preparation. Keep source/workload freshness in the
   existing caller admission chain and perform complete consistency checks here.
   Parsing configuration is not authentication; authentication is not a grant
   or proof that a supplied inference image/lock has been inspected. The selected
   inference profile and chat control/key configuration need not equal training
   values. Exact authenticated artifact placement remains tied to the source run.
   See `../review/modal-inference-preparation.md` for implementation status.
3. **Remote worker and provider boundary.** Separate locked inference source/image
   and SDK adapter. Reverify exact mounted artifact bytes using existing bounded
   archive/descriptor-relative primitives; obtain pinned base weights on Modal
   through existing SDK model preparation. Keep named preparation credentials out
   of inference. Respect persistent data ownership and exact Volume identity.
   Create once only after durable authority consumption, retain the returned ID,
   and bound readiness, access-token issuance, request and exact-target cleanup.

   Correction (2026-09-10, adapter sequencing): connect retained chat commands
   through chat-specific Foundation effect/reconciliation adapters before adding
   the SDK transport and worker. Reuse Foundation grant consumption, predecessor
   authentication, receipt admission and recovery ownership unchanged. This is
   internal composition, not an extra operator step or a new authority system.
   The provider-free boundary is tracked in
   `../review/modal-inference-effects.md`; transport, worker, runtime lock and
   bounded owned cleanup remain separate unfinished parts of this slice.

   Implementation clarification (2026-09-10): existing filesystem materialization,
   `ServingTarget`, pinned-base preparation and vLLM process ownership can run on
   Modal itself. Reuse them there; do not add a parallel launch controller merely
   because the machine is remote. The implemented extraction shares only private
   post-admission byte-stream materialization. Public local RunsAPI verification
   and remote authenticated mounted-source admission remain separate callers.
   An internal byte reader is transport, not authentication or serving authority;
   it cannot substitute arbitrary source paths for verified materialized files.
   The checked-in vLLM image tag is not an inference image/runtime lock. Capturing
   a separate reviewed lock remains required before a real deployment.

   `ModalMountedInferenceArtifactReader` supplies bounded, descriptor-relative
   streams from one retained mount to that same private materializer. It checks
   canonical five-member projections, exact byte hashes and sizes, no-link file
   identities and output-directory identity at EOF. It borrows the mount root
   descriptor and closes its own descriptors when iteration ends or is aborted.
   These are filesystem consistency checks: a future trusted deployment must
   bind the exact selected Volume objects to the configured mount roots before
   worker construction. A claimed Volume ID and a deterministic provider-entry
   digest are not authentication.

   Trust-boundary correction (2026-09-10): physical mount mapping is owned by
   the explicit-client deployment adapter, which resolves and checks exact
   Volume objects and supplies those same objects in the provider mount map.
   The worker trusts Modal to honor that map, then retains no-follow directory
   descriptors and verifies the authenticated artifact inventory and bytes.
   No one-use mount-anchor files are required: they would add cloud writes and
   cleanup races without independently proving the provider's mapping. Mounted
   preparation can be composed and tested before the runnable worker/bootstrap
   and separate inference lock exist; that does not qualify a live deployment.
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
