# Modal Training v1 Reference

Modal training is a submodule-first execution provider behind the public
`TrainingAPI`. It is not a second training CLI and it is not launched with
`modal run`. A consuming host supplies request configuration, durable storage,
authorization grants, credential resolution, and evidence authentication. The
engine supplies strict contracts, planning, verification, staging, the mutation
broker, and the fixed remote worker.

## Product flow

1. The host calls `TrainingAPI.load(canonical_json)` through its configured
   loader, retaining the exact request and project identity.
2. `TrainingAPI.resolve(request)` obtains the exact source, model, dataset,
   configuration, runtime and artifact-policy fingerprints from the host's
   resolver; rich compilation contracts belong to `tuner.training.contracts`.
3. `TrainingAPI.plan(resolved, provider)` retains a generic immutable plan and
   provider context. Modal-specific configuration stays in authenticated
   consumer-owned preparation, stage and launch catalogs, not a second plan API.
4. `TrainingAPI.preflight(plan)` authenticates current source, deployment and
   quote facts before explicit-client Volume/Secret readiness checks.
5. `TrainingAPI.start(plan, preflight)` uses the generic durable coordinator;
   its host authorization port supplies the exact effect grants. A restart
   must reuse retained authority and outcomes, never infer a new submission.
6. Post-start observation, reconciliation and verification belong to `RunsAPI`,
   not `TrainingAPI.outcome` or a Modal-specific lifecycle facade.

Correction (2026-09-09): the old `ModalTrainingOperations`, repository and
composition entrypoints have been removed in the local coordinator cutover.
The replacement uses existing generic Foundation/coordinator ports and
`CoordinatorTrainingService`. The Modal registration's six advertised read,
lifecycle, artifact-streaming, and cost-quote flags remain false pending consumer
binding and security/release qualification. They are not a master start switch:
exact authenticated preflight, consumer effect grants, and Foundation lineage are
required. Do not treat these local code changes or provider-free tests as a newly
qualified live path.

`synaptic_tuner.api.v1.modal.compose_modal_coordinator` takes explicit
`ModalCoordinatorStorePorts` and `ModalFoundationCompositionPorts`, plus the
existing read-evidence collaborators. Its result supplies `training` and `runs`
for `APIHost(composed.training, HostPorts(runs=composed.runs, clock=clock))`.
Use the same clock object throughout composition. The factory performs no
provider I/O; it does not choose storage, mint credentials, or activate the
registration. Provider-free list/show proof does not enable live outcome, log,
or artifact reads while the descriptor capabilities remain false.

## Chat after training (embedded consumer workflow)

Use the checked-in `tuner.inference.run_chat.open_run_chat` entrypoint with the
consuming project's authenticated `APIHost.runs`, retained run reference,
selected runtime adapter. The adapter owns current run reverification, artifact
admission and model preparation on its execution machine, then one bounded session;
it does not add a CLI, registry, database, authority loader or publication step.
For Linux/WSL local vLLM, use `Evaluator.local_run_chat.LocalVLLMRunChatRuntime`.
Pass the private artifact destination and optional pinned-base preparer to that
local adapter constructor, not to the generic `open_run_chat` call. Generic
dispatch checks result consistency but does not authenticate an arbitrary adapter.
The embedding example and ownership contract are in the repository-root
`docs/architecture/verified-run-chat.md`. Use this existing composition instead
of adding throwaway retrieval/start/chat scripts.

Successful materialization persists the consumer's model files after session
cleanup. The helper sends no hidden prompt and does not save conversations
automatically. Full models need no upstream preparer; LoRA uses the existing
`PinnedModelPreparer` seam for the exact authenticated base revision. Preparation
belongs on the selected execution machine, not a manual local weight-staging
step for cloud execution. Inference children remain credential-free and offline.
Generic results expose run, artifact and model metadata; `local_model` is an
optional local-only path capability and must be absent for remote results.

Correction (2026-09-10): this is locally tested adapter composition, not live
Modal inference qualification. The current Modal deployment is training-only.
A Modal chat adapter still requires exact-source deployment, remote model and
artifact preparation, authenticated access, provider-side lifetime/cost controls
and separate live verification. A local session timer is not a cloud billing
guarantee. Do not silently route to another runtime if a selected adapter fails.

For Modal-native preparation, correlate the current verified workflow's complete
artifact manifest with the authenticated native reader inventory; public
role/hash/size values alone do not prove provider placement or the same attempt.
The internal `ModalInferenceSourceBinder` is the reusable admission boundary for
that correlation, composed with the same consumer's runs, stores and reader.
It must not stream model archives to the operator. Its metadata result is not
serving permission or proof that future mounted bytes are unchanged. Separate
chat authority and remote byte verification remain required; see
`docs/review/modal-inference-source-binding.md` for qualification limits.
After fresh source admission, the internal `bind_modal_inference_workload`
function reauthenticates the retained launch and matches the exact workload
bytes to the native workload-record hash and size. It exposes intended pinned
model/tokenizer identity, not actual archive kind or model usability. Its
`load_in_4bit` field describes training configuration, not inference precision.
Remote archive verification/loading and separate chat authority are still
required; this adds no operator-side download or new manual workflow.

Implementation reuse boundary: filesystem-local means local to the selected
execution machine, including a Modal worker. Reuse the existing materialization,
`ServingTarget`, pinned-base preparer and vLLM process controller there instead
of creating parallel serving mechanics. The private byte-stream materialization
extraction is implemented and locally tested; its reader supplies transport,
not authentication. Broader qualification remains separate.
Local RunsAPI verification and remote authenticated mounted-artifact admission
must remain outside that shared core. Never transfer operator filesystem receipts
as proof of remote files or treat a training image lock as an inference lock.

The internal `ModalMountedInferenceArtifactReader.read_artifact` is the mounted
byte transport for that shared core, not a new operator command. Construct it
only after authenticating the remote launch, native inventory and exact physical
Volume mapping. It checks bounded exact inventory, no-link descriptor-relative
files, hashes, sizes and retained identities, but cannot authenticate a mount
from a supplied Volume ID. The caller retains ownership of its root descriptor;
stream consumers must close abandoned iterators (the shared materializer does).
Remote launch composition, separate inference locks and chat authority remain
unimplemented; this transport is not live Modal-chat qualification.

Chat preparation must use its own session identity, inference resources and
Foundation commands. Reuse the resource-quote and evidence-verifier contracts,
but authenticate the separate inference configuration and recompute its resource
commitment; never inherit a training quote or grant. Source/workload values must
come directly from the existing fresh admission chain. A configuration signature
does not replace inference image/lock review or authorize a serving allocation.
This is internal adapter composition, not another operator step; current local
implementation status is in `docs/review/modal-inference-preparation.md`.

Retain chat STAGE/SUBMIT content through the internal
`ModalInferenceCommandBinding` and `retain_modal_chat_command` helpers, using
the consumer's existing `resolve`/`publish_if_absent` catalog semantics and a
complete-content authentication authority. `load_modal_chat_command` performs
authenticated read-only recovery. These helpers do not dispatch commands, issue
grants or authenticate a stage receipt. A storage error must not trigger an
overwrite or provider submission; the unchanged Foundation remains responsible
for grants, predecessor evidence and one-shot execution. CANCEL binding is not
available until an owned Sandbox target can be authenticated.

The internal `ModalChatEffectExecutor` and `ModalChatReconciliationAdapter`
connect authenticated retained chat commands to the existing Foundation broker
and reconciliation service. Consumers supply their catalog, content authority
and transport; use the chat resolvers with Foundation, not a direct executor
call as an authorization boundary. Foundation alone consumes grants, verifies
the actual stage predecessor and retains authenticated receipts. A lost dispatch
response remains unresolved/orphaned without resubmission; failed lookup remains
interrupted. These adapters do not yet supply a Modal SDK transport, authenticated
remote worker, owned cleanup lease or live serving qualification. See
`docs/review/modal-inference-effects.md`.

Build chat launch evidence through `prepare_modal_chat_launch` and verify its
single bounded argument through `admit_modal_chat_launch`. The host builder
authenticates the actual Foundation STAGE receipt/record and complete SUBMIT
predecessor; the worker admission compares the signed content to independently
supplied exact configuration, command and mount-path expectations. Source/model
and native inventory projections remain in the shared preparation snapshot.
These pure helpers do not consume a SUBMIT grant: future SDK transport must use
them only inside the existing authorized Foundation dispatch. Admission expiry
is not a GPU billing deadline. Matching mount-path claims is not authentication
of physical Volume mounts, and matching runtime commitments is not inspection
of an inference image or lock. No executable worker or operator command is added.
See `docs/review/modal-inference-launch.md` for the qualification boundary.

The private `prepare_modal_chat_worker` composition freshly admits that launch
before filesystem access, verifies mounted artifacts through the shared byte
reader/materializer and prepares the existing `ServingTarget`. Full models do
not invoke base-model preparation; LoRA uses the existing pinned-model preparer
on the execution machine. The trusted deployment must bind exact selected
Volume objects to the admitted mount roots. The worker checks retained local
directory identities, not the provider's mount implementation. Its destination
must be worker-private and outside the artifact/control/cache mounts. No mount
anchor, operator-side weight upload or local Docker launcher is needed. This
composition is not yet a runnable bootstrap, inference-lock check, serving lease
or SDK adapter; see `docs/review/modal-inference-worker.md`.

Correction (2026-09-10, implementation order): finish the exact chat command and
remote worker boundaries before freezing the inference runtime lock. The current
training lock and vLLM image tag cannot supply missing inference image, Python,
dependency or bootstrap pins. Retained configuration/quote evidence is parsed
structurally during recovery, not reclassified as fresh source admission or
runtime qualification. See `docs/review/modal-inference-commands.md`.

Runtime prerequisite update (2026-09-10): the existing vLLM startup spec accepts
an explicit `python_executable`; the runtime uses that exact canonical absolute
POSIX path rather than substituting the current interpreter. Selection alone
does not authenticate the executable or inference image. The existing verified
chat/local adapter also accepts `max_request_bytes` (default 1 MiB), enforcing
the complete serialized HTTP JSON body before transport creation. A Modal
composition must explicitly project its admitted interpreter and request limit
into these fields; this update does not yet provide that composition, a runtime
lock check or a runnable Modal service. See
`docs/review/inference-runtime-prerequisites.md` for measured qualification.

Serving-projection update (2026-09-10): the unreleased internal inference
configuration requires a complete `serving` section; older bodies without it
are rejected rather than filled with local defaults. Fractional generation/GPU
values use integer thousandths and probe time uses milliseconds, preserving the
existing integer-only canonical evidence format. Tensor parallelism derives
from the exact configured accelerator count. The mounted worker returns its
target inside `ModalChatWorkerPreparation.startup`, explicit generation/body
bounds, the configured session policy and a retained copy of the original
admission. It checks admission expiry again after model preparation.
This is data projection, not a session or authority receipt. The future locked
bootstrap must freshly verify and rederive these projections, then clamp startup
and session durations to the original remaining deadline before and after
startup. Never restart the full configured lifetime after preparation. No
inference image/lock or Sandbox cleanup is qualified by this change; see
`docs/review/modal-inference-serving-projection.md`.

Shared-deadline update (2026-09-11): the existing `start_vllm_runtime`,
`verified_vllm_chat` and `ChatSession` accept an optional absolute `deadline`
in the controlling process's monotonic clock domain. It is never a serialized
UTC timestamp. Pass the same deadline across startup and session composition;
local startup/session/request/idle limits can shorten it, never extend it.
Expired startup is rejected before filesystem/port/process work, with repeated
checks after potentially slow preparation, port probing and readiness. Session
watchdog/request waits use remaining time; late validated responses cannot
commit history. This closes the shared-timer prerequisite, not the Modal
bootstrap: the latter still must freshly authenticate the original claim,
derive remaining time without renewal, verify the separate inference runtime
and own Sandbox cleanup. No verifier callback or guessed runtime lock was added.
See `docs/review/inference-shared-deadline.md`.

Bootstrap implementation update (2026-09-11): the internal
`open_modal_chat_worker` composes signed launch admission, concrete packaged
inference-runtime verification, the existing mounted-model preparer and
`verified_vllm_chat`. It derives one conservative monotonic deadline before
preparation and does not renew it after startup. Fresh admission and rederived
serving settings are required before startup and before yielding the session.
No caller-supplied runtime verifier, alternate lock path, manual weight staging
or new operator command is accepted. The distinct inference runtime manifest,
worker closure and dependency lock are intentionally not supplied until their
image/Python/dependency/source pins have been measured and independently reviewed;
the concrete verifier therefore denies the current unqualified installation.
Do not replace that denial with training pins or a bypass callback. This worker
context is not yet a Modal deployment, authenticated remote request service or
Sandbox cleanup adapter. Runtime-file checks assume the trusted immutable image
and packaged source tree; they do not establish the provider's image identity
or prevent arbitrary mutation of already imported Python code.

Correction (2026-09-11, private chat connection): the executable worker reads one
bounded startup frame from the authenticated SDK stdin of its newly created
Sandbox, then serves bounded canonical chat frames on stdout. Use
`inference_entrypoint.encode_modal_chat_start` for the startup frame; it carries
the exact signed launch, independently supplied static expectation and declared
credential **names**, never values. The image must pre-create
`/workspace/modal-chat/{model,base,scratch}`. The deployment adapter remains
responsible for the exact physical Volume mapping. Worker bootstrap still uses
the concrete packaged runtime verifier; absent inference locks still deny.
The executable accepts no command-line arguments and emits no raw exception
diagnostics. This private module is not an operator-facing replacement for the
provider-neutral run-chat workflow.

The bootstrap yields the session plus its verified portable model identity;
the ready frame carries that identity. Never infer full/LoRA kind from training
configuration on the host. The host reconstructs `PreparedModelIdentity` from
the bound ready frame and must compare its model/tokenizer refs and revisions
to the authenticated workload before handing off the session.

The remote channel uses the existing `ChatSession` watchdog and never derives a
second deadline. It bounds startup-input waiting, frame size and blocked channel
I/O, retains sequential request/session/launch bindings, and closes the session
on EOF, stop or failure. Keep every public/encrypted service-port list empty:
exposing the raw vLLM port would bypass the session's idle, turn and history
limits. This replaces the earlier proposed HTTP access-token/tunnel path, not the
existing ownership and authorization requirements. Provider-free channel tests
are not live Modal qualification, and cleanup initiation is not proof of a
provider billing deadline.

For inference-runtime diagnosis, use the checked-in
`scripts/capture_modal_inference_runtime.py` with explicit `--app`,
`--environment`, exact digest `--image` and full `--source-commit` selections.
This is a maintenance probe, not a training/chat submission API. Its CLI reads
the environment-provided `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` pair by default.
For an operator's existing CLI login, explicitly select `--modal-profile NAME`;
that path reads only the named SDK profile's pair, with environment overrides
disabled. Missing or blank credentials fail closed; neither path falls back to
the other or calls `Client.from_env()`. Both construct an explicit client.
Never put credential values in argv, reports or new files. It resolves only
the selected existing app (`create_if_missing=False`) and creates one CPU-only
Sandbox: 1 CPU, 2048 MiB, 300-second provider timeout/idle timeout, no GPU,
Volumes, Secrets or runtime network. Image preparation is remote; no local
Docker image pull or model download is required. Treat image preparation as a
cloud operation, not a guaranteed free local preflight.

The remote stdlib-only `scripts/inspect_modal_inference_runtime.py` captures
actual interpreter identity/hash and installed distribution metadata without
importing ML packages. Missing Modal/vLLM are explicit candidate facts, not
successful runtime admission. Reports bind the copied inspection script's hash
and record image/engine commit as operator selections only: the base probe has
not installed that engine commit and does not attest the final image. Do not
turn candidate metadata into production pins without independent review.

The capture launcher has a 600-second host wait plus an independent bounded
cleanup attempt, with no automatic create retry. Ambiguous creation is not
absence. A delayed returned handle is retained for exact-target cleanup while
the owning process lives, but its daemon operation is not durable after CLI
process death; an indeterminate report is not proof of shutdown. Known resources
are checked through the exact returned handle, never listing/adoption. Keep the
closed ownership/cleanup report and resolve ambiguity before another attempt.

For a failed capture that returned an exact Sandbox ID, diagnose that same
instance through the maintenance command's explicit `--read-sandbox ID` mode.
Keep the original image/source and app/environment selections as provenance;
the latter flags alone do not attest a recovered Sandbox's environment. This
mode requires a stopped exact instance, reads only bounded stdout/stderr, and
performs no create, listing, termination or retry. It exposes only validated
candidate metadata or the inspector's closed error codes, never arbitrary
provider text. Unrecognized output retains the known exit code with a fixed
unclassified marker, not a guessed cause. Its 30-second read deadline starts
after explicit client construction; it does not bound SDK authentication.
Do not infer failure cause from the capture process's generic
`capture_failed` result or start another allocation to obtain diagnostics.

An inspector metadata rejection is not permission to drop packages from the
runtime inventory, accept duplicate distributions, raise limits or guess pins.
Keep its validation intact and use closed, distinct diagnostics for enumeration
limits, invalid names/versions, duplicate normalized names and metadata-read
failures. A follow-up allocation still needs its own applicable authorization;
reuse the exact stopped instance for read-only diagnosis first.

Correction (2026-09-11, source freshness): bind the source once when opening a
session. Before requesting its SUBMIT grant, use the existing binder's
`assert_current(source)` to recheck current workflow/Foundation/native metadata
without another verification transition or model-body read. Calling `bind`
again invokes `RunsAPI.reverify`, advances the workflow revision and invalidates
the earlier preparation even when the model is unchanged. A read-only guard is
not serving permission; the consumer still supplies separate exact STAGE and
SUBMIT grants through Foundation. This is internal adapter work, not an extra
operator command.

The internal `ModalRunChatRuntime` now connects those existing binders, consumer
catalog/content authority, exact grant port, Foundation broker and ready-lease
handoff to `open_run_chat`. Construct one runtime per session attempt; a failed
open must not silently start another attempt. The normal context exit closes
the owned Sandbox. Retain `runtime.owned_lease` when cleanup remains unresolved,
and retain the transport's pending ownership if no ready lease was obtained.
The engine neither selects persistence nor mints the consumer's grants. This
composition is provider-free tested, not yet live-image or serving qualification.

Correction (2026-09-11, inference image identity): the signed configuration's
`image.registry_reference` and `image.image_digest` select the pinned **base**
image. Required `image.provider_image_id` identifies the separately built final
Modal Image. The embedded runtime manifest uses `base_registry_reference`, with
no legacy alias. An image cannot practically embed a manifest naming that same
image's final OCI digest; do not create that circular commitment. Build and
review the source/dependency/runtime layers separately, then bind the returned
provider Image ID in the authenticated configuration. SUBMIT must resolve that
exact existing Image with the explicit client and verify its hydrated identity;
it must not construct a new image or add operator source at submission time.
Runtime-file checks and provider object identity checks are complementary, not
substitutes. Neither a signed ID nor a base-image candidate report qualifies the
actual installed runtime by itself.

## Frozen live evidence

This evidence describes the pre-coordinator implementation. It is historical
evidence, not a live qualification of the replacement worker or public cutover.

Modal SFT is live-proven for run
`modal-sft-20260826T144636Z-7aec224e893d` and provider call
`fc-01M0Z8K9MCPN3P368V3CK94TV2`. The host ledger records one submit attempt,
provider success, verification invalidation, a read-only reopen, and final
verification. The invalid result was a verifier location error: the logical
`/workspace/run` root corresponded to Modal's resolved physical Volume mount.
The correction is evidence-bound relocation, not an unbounded path search.

The exact five verified artifact roles were published to a host-selected local
destination, and repeated publication converged to the durable receipt. The
portable sanitized fixture is
`tests/fixtures/training_product/modal_live_v1/evidence-index.json`. Its typed
completeness is authoritative: lifecycle and publication digests/projections
are captured, but authenticated provider terminal/completion bytes and artifact
payloads are not. Do not cite the fixture as a raw provider transcript.

The cross-provider contract and proof matrix live in
`docs/architecture/submodule-first-training-v1.md`.

## Fixed v1 topology

- Exact Modal SDK `1.5.4` and one explicit authenticated client; no ambient
  profile or `Client.from_env()` fallback in engine code.
- Fixed app `synaptic-training-v1`; the exact function name is derived by
  `modal_function_name(deployment_ref)` and includes the deployment identity.
- One A10 GPU, one canonical command argument, `retries=0`, and one detached
  `.spawn()` call behind the authenticated Foundation effect broker.
- One digest-pinned Unsloth registry image with its inherited entrypoint
  cleared.
- One existing Modal Volume v1 for control/log/evidence records and one
  distinct existing Modal Volume v1 for input/output artifacts.
- Each effect is isolated below `operations/{effect_id}/`; jobs never share
  global `input/`, `output/`, `logs/`, or `evidence/` paths.
- The remote job independently clones the exact pushed host project and exact
  engine commit, verifies the project gitlink, and invokes only
  `Trainers/sft/runtime_v1.py --canonical-workload-stdin` without a shell.
- Before invocation, the remote worker retains the verified full engine checkout
  separately and stages only the authenticated offline-worker manifest members
  at the execution engine root. A full checkout is not a prepared worker: extra
  files, including Git metadata, fail the exact-closure guard. This preparation
  is internal to the provider, not an additional operator command. Collisions
  or invalid members fail before trainer invocation; never relax the guard or
  retry a submitted job to recover from them.
- The verified runtime is CPython 3.11.14 at `/opt/conda/bin/python3`; its
  executable, image, complete hash-pinned launcher dependency closure,
  deployment wrapper, remote worker/producer/runtime modules, SFT entrypoint,
  and ML stack are checked in `modal-runtime-v1.lock.json`. The host enforces
  this packaged lock before preflight and the remote materializer enforces it
  again against the reconstructed exact checkout.

## Storage and evidence

Model preparation belongs on the execution machine, before the offline trainer.
The provider must automatically obtain the exact configured model revision using
the Hugging Face SDK and reuse verified cached weights. Do not require operators
to download weights locally and upload them to Modal. Keep model credentials in
the preparation wrapper, never in the trainer subprocess; preserve the runtime's
exact revision, link-free snapshot, and offline checks. Cache integration must
treat the persistent Volume as untrusted and must not let SDK filesystem writes
follow attacker-controlled cache paths. This is an internal preparation phase,
not a new submission command or a separate downloader service.

Modal Volume is the authoritative provider-native artifact store for a Modal
run. Hub publication is optional and separately authorized. The remote producer
emits exactly five artifacts: workload record, training lineage, training
metrics, final model, and tokenizer.

The control Volume contains operation-scoped, authenticated structured logs,
terminal evidence, and the completion manifest. The host database stores the
expected workflow/effect identities, one-shot authority consumption, provider
job reference and verification result using the existing generic stores. Exact
Modal configuration, signed stage material and launch envelopes are published
through consumer-owned catalogs before the unchanged Foundation dispatches.
Restart reuses the retained signed launch and assessment. The engine does not
select or ship a concrete database and must not create SQLite state.

Mounted Volume writes are committed explicitly after the producer finishes.
The artifact Volume is committed before the control Volume so an intentionally
visible completion record cannot precede its artifacts. Any uncertainty after
staging or `.spawn()` is reconciliation-only; it does not recreate submission
authority.

Treat both mounts as hostile shared storage. On the locked Linux runtime,
reads and writes traverse through retained directory descriptors and open leaves
relative to those descriptors, preventing an ancestor substitution between
validation and I/O. Reads are bounded and compare descriptor identity before
and after; writes use exclusive leaf creation. Named Modal Secrets are the only credential path;
secret-like environment keys or symbolic secret values are rejected before
image construction.

## Three proof levels

1. **Provider-free barrier** — schemas, canonical parsing, hostile binding
   tests, exact SDK surface construction, image inspection, network-disabled
   image runtime checks, compilation, and packaging. No credentials, provider
   calls, GPU, or spend.
2. **Authenticated live preflight** — explicit-client account/workspace/
   environment binding, existing Volume identities, deployed Function version,
   Modal image identity, secret names/required keys, and a current quote. This
   may read provider state but may not submit training.
3. **Paid smoke** — after the exact tree is independently accepted, committed,
   pushed, and granted, stage once and call `.spawn()` once. Observe and verify
   by the durable provider job ID and Volume evidence. Never retry an ambiguous
   submission.

## Failure diagnostics

Raw trainer stdout/stderr, tokens, provider responses, and exception text do not
cross the remote contract. Persist closed status codes and redacted structured
records. For a failed live smoke, collect the provider call status, Modal logs,
operation-scoped Volume inventory, authenticated terminal/log records, exact
source/deployment/runtime locks, and host lifecycle history before changing
trainer hyperparameters.

Provider/runtime failures should be fixed in the provider profile, runtime lock,
deployment wrapper, or reusable engine contract. Model, dataset, tool schema,
and training choices stay in host configuration; do not hardcode the current
smoke into runtime code.

## Runtime-lock maintenance

When a file already named by `modal-runtime-v1.lock.json` changes, verify the
lock from the repository root:

```bash
python3 scripts/regenerate_modal_runtime_lock.py
```

The default is read-only and exits nonzero when a declared source hash is
stale. `CURRENT` means only that the declared source hashes agree with
the current canonical, policy-valid lock; it is not independent approval of
the dependency, image, Python, SDK, or ML-stack pins. After reviewing the
source change, refresh only those SHA-256 values and then verify again:

```bash
python3 scripts/regenerate_modal_runtime_lock.py --write
python3 scripts/regenerate_modal_runtime_lock.py
```

This is an offline local maintenance command. It does not contact Modal, load
the provider SDK, resolve packages, inspect an image, or authenticate source or
quote evidence. It preserves the exact reviewed inventory and preserves all
non-hash fields supplied by the current policy-valid lock without approving
them. Inventory or pin changes require a separate deliberate lock/schema
review; never use this command to discover, add, remove, or redirect locked
members. Its pathname and identity rechecks are
best-effort protection for local maintenance races, not hostile-volume
retained-directory-descriptor or compare-and-swap safety.

Correction (2026-09-09): extraction of low-level worker ports and source-staging
helpers deliberately expanded the lock, schema, runtime policy and maintenance
inventory together from eight to ten files. The new members are `worker_ports.py`
and `worker_source.py`; runtime/image/dependency/Python pins are unchanged.
This is the currently composed worker's source lock, not qualification of the
still-disabled Foundation worker. Its eventual production cutover must review
and lock the new bootstrap path before enabling execution.

Correction (2026-09-09, coordinator cutover): the reviewed declaration now has
97 members. The bounded static import audit plus explicitly reviewed lazy
imports covers 91 Python files and one resource. Three additional public API
files (`context.py`, `execution.py`, `sources.py`) remain conservative integrity
pins, and the launcher dependency lock and SFT entrypoint are separate runtime
pins. The wrapper digest now identifies `coordinator_deployment.py`; removed
legacy remote/producer files are not part of the new declaration. This audit is
not arbitrary dynamic-import discovery or a live-runtime proof. The separate
trainer closure still has 66 members: its old public rich-training module has
been replaced by `tuner/training/contracts.py`, with reviewed importer hashes.
