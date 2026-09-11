# Modal inference bootstrap and runtime verifier

Status: source and tests locally qualified; final package checks pending,
2026-09-11. Independent source review passed.
Engine-only; no cloud, GPU, model download, credential access or EHR changes.

## Implemented boundary

The internal `open_modal_chat_worker` connects exact signed launch admission,
the concrete packaged inference-runtime verifier, existing mounted artifact/base
model preparation and `verified_vllm_chat`. It is a worker-local context yielding
the existing `ChatSession`, not a deployed Modal endpoint or consumer UI.
It adds no operator command, authority store, downloader or local Docker hop.

One process-local monotonic deadline is anchored before initial admission and
the trusted UTC clock read. The remaining signed claim window, capped by the
configured absolute lifetime, supplies its duration. Using the earlier monotonic
sample is conservative for callback latency. Preparation and startup consume
that same budget; startup cannot grant a new session lifetime. Invalid/regressing
clock samples and expiry deny before the next owned effect or before yielding.
This bounds acceptance/cleanup initiation, not arbitrary syscall duration or
provider billing. Provider-side termination is still a separate requirement.

Before startup and before yielding, fresh signed admission rederives all serving
settings and compares retained run/artifact/model metadata. The original target,
retrieved-model and base-snapshot object identities and cheap path/file-identity
metadata are retained. These checks do not rehash multi-GB weights: the existing
preparer verifies bytes, and the runtime independently validates the target
immediately before spawn. Inference keeps the existing explicit credential-free,
offline environment and owned process cleanup. Saved model files survive exit.

Ordinary setup failures use a closed diagnostic; control-flow exceptions survive.
Pre-runtime collaborator exceptions cannot supply a cleanup handle. Only exact
runtime leases from the existing runtime-entry boundary are retained, without
executing exception properties. Once acquired, the existing context owns cleanup
and can attach its exact unresolved lease. Consumer exceptions retain identity.

## Concrete runtime verification

`verify_modal_inference_runtime(configuration)` accepts no alternate lock path,
lock bytes or verification callback. It reads the fixed packaged
`inference-runtime.lock.json` with bounded no-follow file access. Its manifest
names the fixed neighboring `inference-dependencies.lock` and
`inference-worker-closure.json`; the latter's source members must equal the
runtime inventory excluding those two resource files.

The verifier compares authenticated configuration against the packaged image
selection, SDK and Python pins; checks declared source file sizes/hashes and
closure consistency; verifies physical CPython version/executable identity/hash;
and compares the complete installed distribution name/version map. Modal must
be present at the same version as the SDK pin. ML packages are not imported.

Digest definitions are deliberately distinct: `runtime_lock_digest` hashes the
entire canonical runtime manifest including its trailing newline;
`source_lock_digest` hashes the canonical source inventory including its newline;
`dependency_lock_digest` hashes the exact dependency file bytes;
`worker_closure_digest` hashes the canonical closure document excluding its own
digest field, including the canonical trailing newline. The runtime inventory
also pins the complete closure resource's file hash and size.

The runtime manifest and its dependency/closure resources are intentionally
absent: there are no measured, independently reviewed inference pins yet.
The current installation therefore denies before model preparation or serving.
This is implemented verification logic tested with synthetic fixtures, not a
positive qualification of an inference image or an always-successful test hook.
Package-data declarations reserve the exact three resource names for their
eventual reviewed addition; no synthetic production lock is shipped.

The verifier assumes a trusted immutable image/package tree. Image selection
equality does not attest which image Modal actually mounted. Distribution
metadata is not a hash check of every installed dependency file. The dependency
lock is byte-bound, not semantically proved to be a complete hash-valid installer
closure. Minimum required source members and manifest self-consistency do not
discover complete dynamic imports or prove there are no other files in a full
engine installation. Those properties require the separate image, dependency
and complete source-inventory review before deployment. Already imported Python
code and arbitrary later filesystem mutation are not authenticated by this check.

## Qualification

The final integrated selection passed 64 tests in 126.55 seconds: 23 runtime
verifier tests, 22 bootstrap acceptance tests, four full/LoRA success/backend
failure compositions through the real runtime/controller/client with external
effects replaced, and 15 additional mutation/clock/API-boundary tests.
The positive bootstrap cases record the runtime verifier call; the actual
verifier's positive tests use a synthetic packaged runtime and distribution map.
Together these prove behavior and ordering, not a working production runtime.
Independent focused review passed 27 tests in 19.01 seconds, a subset of the 64.
The earlier 47-case integrated and 23-case verifier-only selections are likewise
not additive totals. Tests use CPython 3.12.9 / pytest 8.4.2 with automatic plugins
and pytest caching disabled, without Modal or ML packages installed.

The existing runtime/HTTP/chat/materialization regression passed 387 tests in
1.81 seconds. The unchanged training source lock remains CURRENT at 97 members;
the separate offline SFT closure remains CURRENT at 66 members and 679,487
payload bytes. Neither training inventory is repurposed for inference.
The two selections total 451 distinct tests. Black's single-worker Python 3.12
check passed for all six new Python files; skill synchronization and whitespace
checks passed. The first read-only-sandbox formatting check stalled and was
stopped; the scoped local rerun passed. No training/server process was involved.
Final wheel qualification remains pending.

## Next user-visible milestone

Capture and independently review the actual inference image, Python, dependency
and complete bootstrap inventory; package those resources. Complete explicit
Modal creation, authenticated request access, durable exact-instance ownership
and bounded termination/recovery, then demonstrate current-source training,
retained outputs, real model response and confirmed shutdown. No capability
flag, public provider activation, deployment, push, merge or release occurs in
this slice.
