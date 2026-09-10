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
