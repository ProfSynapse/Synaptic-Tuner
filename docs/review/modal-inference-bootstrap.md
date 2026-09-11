# Modal inference bootstrap and runtime verifier

Status: locally qualified source `165b000ab8098e853fdb153c1b3d0e7821e1ddc1`,
2026-09-11. Independent source and archive-to-wheel audits, and isolated wheel
checks passed.
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

The final offline wheel from the exact source commit's completed archive is
2,085,701 bytes, SHA-256
`2d32e01d4b50b0b08ef28d0a7d91025fa6275e297366711c7881485225995e7f`.
An earlier extraction started before the archive command completed and reported
unexpected EOF. Its partial source/wheel were preserved separately as unqualified,
never installed or published, and excluded from qualification. The final build
used a fresh extraction after archive completion and successful archive listing.
Independent audit verified 727 unique ZIP entries and all 727 RECORD rows with
no missing/extra rows or hash/size mismatches. All 719 Python members matched
the completed archive byte for byte. Existing training runtime/closure resources
also matched; all three unreviewed inference resources were absent. Filename-only
inspection found no `tests/` tree or private artifact names, with only expected
secret-related source names and two pre-existing test-like trainer modules.
This is a filename check, not a credential/content scan.

The final wheel was installed offline without dependencies into the reused
isolated declared-dependencies-only wheel environment, not a freshly resolved
environment. From a neutral directory with isolated Python, the archive's exact
CI probe passed 46 engine imports and two existing training-lock resource checks.
Both new source hashes matched the installed modules:

| Source | SHA-256 |
| --- | --- |
| `inference_bootstrap.py` | `03231442306c0febc9c650626c7edf6387a69784efb8583f9b5f29a17c54d18f` |
| `inference_runtime.py` | `7c8045dbe38d1d638ab3cb4b8d1f0f7999193cf305b4c344970ca214023a9480` |

Installed signature binding accepted the two exact entrypoint argument sets and
rejected runtime-verifier, alternate-lock-path and alternate-clock keywords.
All three unreviewed inference resources were absent. The actual installed
verifier rejected a valid synthetic configuration with the closed missing-lock
diagnostic. The bootstrap was not invoked. Modal, Torch, NumPy, pandas and pytest
were absent in this wheel environment; these checks created no server or provider
session.

## Next user-visible milestone

Capture and independently review the actual inference image, Python, dependency
and complete bootstrap inventory; package those resources. Complete explicit
Modal creation, authenticated request access, durable exact-instance ownership
and bounded termination/recovery, then demonstrate current-source training,
retained outputs, real model response and confirmed shutdown. No capability
flag, public provider activation, deployment, push, merge or release occurs in
this slice.
