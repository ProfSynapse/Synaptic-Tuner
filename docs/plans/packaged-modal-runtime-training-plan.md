# Approved Plan: Provider-Neutral Packaged-Runtime Training

> Status: APPROVED — Joseph Rosenbaum, 2026-09-22
> First qualified provider: Modal
> Legacy path retained: `superproject + dual_clone` developer integration

## Objective

Build a provider-neutral packaged-runtime execution mode for Synaptic Tuner.
A caller supplies a local path or upload plus declarative training
configuration. The trusted host prepares a private, content-addressed dataset,
resolves an immutable runtime release, and submits a canonical workload through
the public `TrainingAPI`. The provider runs a prebuilt Synaptic Tuner runtime.
No customer, host, or engine Git repository is cloned per packaged-runtime job.

Modal is the first implementation and live qualification. Hugging Face Jobs,
RunPod, local Docker, and future providers reuse the shared release, workload,
input, authority, and artifact contracts while retaining their native execution
and storage mechanics.

## Architecture Decisions

### Provider-neutral runtime release

Add `PackagedTrainingRuntimeReleaseV1`, containing:

- stable release reference and schema;
- exact engine wheel/package digest and source provenance;
- fixed worker entrypoint and worker-closure digest;
- immutable OCI image reference and digest;
- Python implementation, version, executable path, and executable digest;
- complete installed-distribution inventory digest and count;
- platform, CUDA, and runtime facts;
- compatible methods, models, immutable model revisions, and dataset formats;
- workload, prepared-input, and artifact-contract schema identities;
- canonical release-manifest digest.

Provider object IDs do not enter this manifest. A separate
`ProviderRuntimeBindingV1` binds the release digest to provider-native facts
such as a Modal image/app/function, HF image/job configuration, or RunPod
template/image identity. This prevents circular release identity.

Add `PackagedExecutionBindingV1`, binding one run to its runtime release,
provider runtime binding, prepared input, canonical workload/configuration, and
artifact policy. Rich training contracts accept the closed union:

```text
ExecutionMaterialV1 = GitExecutionSourceV1 | PackagedExecutionBindingV1
```

The public flow remains:

```text
load -> resolve -> plan -> preflight -> start
```

For v1 compatibility, `ResolvedTrainingRequest.source_digest` contains the
selected execution material digest. A later API version may rename that opaque
field; this work does not require a breaking public schema change.

### Local path or upload to private prepared reference

Add a host-only preparation step before canonical planning:

```text
local path / one-use upload stream
  -> stable no-link bounded read
  -> declarative normalizer/verifier
  -> private content-addressed host store
  -> PreparedTrainingInputIdentity
```

The resulting authority is `prepared://sha256/<semantic-digest>` plus revision,
byte digest, size, and format. Paths, filenames, upload handles, and prose never
enter the canonical request, durable launch authority, or provider dispatch.
Expose this through `TrainingAPI.prepare(...)` or an adjacent public training
source contract; retain `load(...)` for already prepared canonical requests.

Reuse `VerifiedTrainingInputSource` and its one-use stream lease. Each provider
stages the same identity through native storage with exclusive creation,
bounded streaming, digest verification, and readback:

- Modal: operation-scoped Volume object;
- HF Jobs: exclusive Bucket slot;
- RunPod: operation-scoped Network Volume path;
- local Docker: private copied workspace.

### Lifecycle and authority

The consumer continues to own configuration, credentials, grants, durable
coordinator/Foundation state, reconciliation authority, and final outputs.
Submission consumes one effect grant and makes one provider mutation. An
ambiguous result is reconciliation-only and never grants automatic replay.
Provider credentials enter only through named provider secret mechanisms.
Model preparation may use a model token; the offline trainer child remains
credential-free and offline.

### Modal-first adapter

Release time, not job time:

1. Build a derived immutable OCI image from the reviewed runtime base.
2. Install the exact Synaptic Tuner wheel and hash-pinned Modal bootstrap.
3. Embed the runtime release manifest and worker closure.
4. Capture the final Python and complete distribution inventory.
5. Publish the OCI image.
6. Deploy a non-serialized function implemented by the installed package.
7. Retain authenticated evidence for the exact Modal image, app, function,
   environment, and deployment generation.

Predeploy once per consumer profile and release. Shared Volumes and Secrets are
deployment-bound; every run remains isolated below its effect ID. A new runtime
release gets a new deterministic deployment identity instead of mutating a
floating function.

Per run, the adapter verifies the existing qualified deployment, stages input,
retains STAGE/SUBMIT evidence, and invokes `.spawn()` exactly once with one
bounded canonical dispatch document. It must not use
`add_local_python_source`, serialize host functions, clone Git, or install
packages. The worker verifies its embedded release, workload, mounts, input,
and artifact policy before model preparation or training.

Structured progress, terminal evidence, and exact artifacts remain in
provider-native storage. Observation and reconciliation use the exact returned
call ID. Artifact retrieval is streamed, bounded, rehashed, and compared with
the durable completion inventory.

### Other providers

Shared contracts stop at release identity, prepared input, workload, authority,
and verified artifact inventory. Provider families retain native semantics:

- Modal: predeployed Function, `.spawn()`, Volumes, Modal Secret, call ID.
- HF Jobs: immutable image, `run_job`, HF Bucket, HF job ID.
- RunPod: immutable image/template, Network Volume, Pod/job identity and owned
  cleanup.
- Local Docker: immutable image and private host-owned copy/artifact roots.

Do not add a universal lowest-common-denominator job object. Continue using
`ProviderFamilyV1` planning, preparation, executor, reader, and reconciliation
ports with provider-specific preparation documents.

### Provider-portability gate

Modal is a qualification target, not a dependency of packaged training. The
shared packaged worker and trainer seam must import no provider SDK and accept
no provider-native object, identifier, storage path, or lifecycle document.
It consumes only authenticated runtime-release, execution-binding, workload,
prepared-input, process-local path, and artifact-policy material.

Every provider adapter must prove the same boundary:

- stage the same prepared-input identity using provider-native storage;
- translate provider-native launch state into the shared admitted execution;
- invoke the same installed-package worker without cloning or installing the
  engine at job time;
- retain provider-native observation, cancellation, and reconciliation rules;
- return the same bounded, exact artifact inventory and terminal evidence; and
- keep provider SDK types and credentials below the adapter boundary.

RunPod and HF Jobs extension tests must exercise this boundary directly. A
change to the shared trainer solely to accommodate Modal mechanics is a design
failure unless the capability is independently useful to the other providers
and represented in provider-neutral contracts.

### Coexistence

Configuration selects exactly one mode:

```text
execution.mode = packaged_runtime | developer_integration
```

`packaged_runtime` rejects Git authority and clone fields.
`developer_integration` preserves the current dual-clone finalizer, runtime
lock, deployment, worker, and historical evidence unchanged. There is no
automatic selection or fallback between modes.

## Implementation DAG

Current checkpoint (2026-09-22): A–F have provider-free implementations and
focused tests. The Modal release deployment and CPU self-check are being wired
into a real callable and independently reviewed. No packaged image has been
deployed and no GPU training has run through this path. G–I remain ahead.

```text
A Shared release contracts
 |- B Host input preparation/API
 |- C Packaged worker and release tooling
 `- D Execution-material/coordinator generalization
       `- E Modal packaged adapter
            `- F Modal provider-free qualification
                 `- G Modal CPU live qualification
                      `- H Paid Qwen smoke
                           `- I Full 220-row training and verification

After I -> J HF Jobs extension contracts/tests
After I -> K RunPod extension contracts/tests
```

J and K are deferred until the full training run and artifact verification are
complete. The shared contract is already checked for provider-specific imports;
there is no need to implement another adapter before the Modal run.

Later capacity work: the current prepared-input contract admits one file up to
64 MiB. Larger uploads will need a versioned chunk or manifest protocol. The
packaged trainer also requires Linux container facilities, which is compatible
with the planned providers but does not provide native Windows or macOS training.

## Scope Contracts and Ownership

### A — Shared release contracts

Owns runtime-release and provider-binding immutable types, canonical schemas,
digests, parsing, and hostile-input tests. Expected surface:
`tuner/runtime/releases.py`, public v1 release contracts, and schemas. It does
not implement any provider or mutate runtime images.

### B — Host input preparation and API

Owns path/upload source contracts, stable descriptor reads, content-addressed
private storage, prepared-reference injection, and `TrainingAPI` ergonomics.
Expected surface: `training_facade.py`, new training-source contracts, and
`tuner/training/input_preparation.py`. It never stages provider storage.

### C — Packaged worker and release tooling

Owns the fixed packaged worker entrypoint, worker closure, and one checked-in
release-management CLI with plan/build/capture/verify/promote phases. It does
not implement lifecycle authority or provider submission.

### D — Execution material and coordinator

Owns the closed Git-or-packaged execution union, deterministic recipe/workload
compilation, coordinator material, and provider-neutral conformance tests.
Expected surface: `tuner/training/contracts.py`, `coordinator_material.py`, and
recipe compilation. It does not contain provider IDs or storage concepts.

### E — Modal packaged adapter

Owns new `packaged_*` Modal modules for release binding, preflight, deployment,
dispatch, worker composition, reading, and reconciliation. It reuses reviewed
generic Foundation and safe mounted-input/artifact primitives. It must not
rewrite the legacy dual-clone modules.

### F — Modal provider-free qualification

Owns construction, schema, security, hostile-storage, no-Git, no-source-copy,
single-spawn, ambiguous-result, and artifact-policy tests. No credentials or
provider calls.

### G — Modal CPU live qualification

Owns the bounded CPU self-check of exact image, package, Python, inventory,
closure, deployment lookup, staged fixture, and shutdown/ownership evidence.
No GPU or model training.

### H — Paid Qwen smoke

Owns one explicitly authorized minimal GPU smoke exercising model preparation,
forward/backward training, LoRA save, terminal evidence, and artifact readback.
It may adjust provider resource configuration after measured OOM evidence but
must not change the reviewed model/runtime identity.

### I — Full private training

Owns the exact 220-row prepared-v2 run, monitoring, completion, five-artifact
verification, adapter semantic verification, and durable research record. Chat
and evaluation are later, separate processes.

### J — HF Jobs extension

Owns provider-free binding and conformance tests mapping the shared release and
prepared-input contracts to immutable HF image execution and Bucket storage.
It does not alter Modal or claim live qualification without a separate run.

### K — RunPod extension

Owns provider-free binding and conformance tests mapping the shared contracts
to immutable templates/images, Network Volumes, exact job identity, and owned
cleanup. It does not adapt the legacy shell/GraphQL lane.

## Qwen 3.5 4B Qualification Gates

1. Provider-free release/schema and hostile-input tests pass.
2. The existing Qwen base `sha256:1644...`, Python 3.12.3, and 327-package
   inventory are candidate inputs only.
3. A derived image containing the exact wheel and Modal bootstrap is built;
   its complete final inventory and Python executable are freshly captured.
4. The release admits only Qwen/Qwen3.5-4B at revision
   `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, SFT, and the reviewed dataset
   and 32K prompt/completion controls.
5. Modal CPU self-check passes without Git, source injection, GPU, or trainer
   credentials.
6. One paid minimal smoke performs real optimization and saves LoRA weights.
7. Exactly five artifacts verify; `final_model` is a valid adapter and fits the
   192 MiB member and 256 MiB aggregate policy.
8. The full private 220-row training completes.
9. Final artifacts are re-observed, streamed, rehashed, and verified before
   success is declared.

## Failure and Security Rules

- User-selected arbitrary images or commands are rejected; config resolves an
  allowlisted release.
- Local input rejects links, path escape, races, oversize payloads, and digest
  changes.
- Provider staging is exclusive and collision-failing.
- Release, provider binding, workload, input, and artifact policy must all bind
  the same plan and effect.
- Missing or drifting deployments fail before paid submission.
- Credential values never enter config, argv, durable records, or diagnostics.
- Raw trainer/provider exception text is not persisted; use bounded closed
  diagnostics.
- Unknown submission outcome becomes orphaned/reconciling, never a retry.
- Provider-native storage is authoritative until verified publication or
  retrieval completes.

## Journaling Checkpoints

Update the existing project journal after:

1. architecture acceptance;
2. shared contracts and hostile tests;
3. final runtime image capture/promotion;
4. Modal CPU qualification;
5. paid GPU smoke, including bounded failure lessons;
6. full training and verified LoRA;
7. later chat/evaluation completion.

Do not publish journal content automatically.

## Genuine External Unknowns

- Confirm with official documentation and CPU proof how Modal 1.5.4 deploys a
  non-serialized function implemented only by a prebuilt image package.
- Select the final registry destination and release-signing authority.
- The Modal bootstrap may change or conflict with the Qwen image's current
  327-package inventory; only derived-image capture establishes the final lock.
- A10 feasibility and cost for the real 32K workload require the paid smoke.
- HF Jobs and RunPod implementation must use freshly verified official APIs;
  only their provider-free extension contracts are approved here.

## Completion Criteria

The architecture is complete when a clean consumer can provide a private local
path/upload and declarative config, receive a content-addressed prepared ref,
plan and authorize a packaged runtime through `TrainingAPI`, run the full Qwen
dataset on Modal without any job-time Git clone or source injection, and
retrieve a semantically verified LoRA plus its exact evidence and lineage.
