# Modal mounted chat preparation

Status: locally qualified, 2026-09-10.
Engine-only; local qualification performs no provider access, model downloads
or process startup. No live qualification is claimed.

## Boundary

`prepare_modal_chat_worker` freshly verifies the bounded signed launch before
filesystem access. It reconstructs the run, five artifacts, native members and
workload from the single admitted preparation snapshot. It owns retained
no-follow directory descriptors, rejects lexical overlap and equal physical
root identities, then composes the existing mounted artifact reader and private
shared SFT materializer. The artifact reader and materializer borrow their root
descriptors; the worker attempts to close every descriptor it owns on all exits.

Materialized run/artifact/model identities must match immutable authenticated
projections before pinned-base preparation. Full archives ignore the optional
preparer; LoRA uses the existing `PinnedModelPreparer` contract and serving-target
validation. The LoRA preparer may acquire the pinned base on the execution
machine. Archive semantics determine full versus LoRA; the training
`load_in_4bit` setting remains authenticated workload metadata but is not
projected into `RetrievedSFTModel`/`ServingTarget` or used by this worker.
The exact returned `ServingTarget` must retain the checked retrieved object.
Input/model/root checks run again after preparation. Ordinary failures use a
closed error; process-control exceptions and active failures survive cleanup.

The trusted deployment adapter owns exact Volume-object-to-mount mapping. This
worker verifies local descriptor and artifact integrity, not Modal's mapping
implementation. No mount anchors or separate evidence/authority framework are
added. The destination is worker-private and outside the artifact/control/cache
mounts. Existing materializer cleanup covers its failed attempts; successfully
materialized files remain under that private worker lifetime, including when a
later preparation step fails. Shared artifact/cache data is never torn down by
this composition.

This helper does not inspect an inference runtime lock, create a Sandbox, start
vLLM, expose an endpoint, enforce a GPU lease or authenticate provider cleanup.
It must not be wired into a runnable deployment before those boundaries exist.

## Qualification

Source commit: `4066e604522a95c274098177fdd40d56432b3ac2`. Independent source
review passed the final worker SHA-256
`c22ac53d70cc5b08d362a79bb0c042bb5fe4c08ff68454e2d31fd6f95d47d0a2`.
Review corrections added exact result types, guards before and after pinned-base
preparation, exact retrieved-object retention, immutable source/model comparison,
fresh root identity checks and normalization even when a collaborator raises the
worker's own exception class with arbitrary text.

The final integrated selection passed **36 tests in 115.73 seconds**: 28
independently authored cases and eight integration/cleanup cases. It covers
full/LoRA behavior, denial before filesystem access, byte corruption and links,
lexical overlap and duplicate physical root identities, root replacement before
and after preparation, a genuine second materialization substituted into a valid
target, callback mutation, duck results, process-control exceptions after opening
roots, closed diagnostics and cleanup failures without losing the primary error.

The fixture replaces producer-helper bytes before native evidence is created,
runs the actual source/workload/preparation and Foundation STAGE/signing chain,
and mounts bytes matching every signed hash and size. It does not rewrite an
authenticated snapshot or mock worker admission. This is producer-derived fixture
evidence, not a live producer filesystem-write proof. Both model-kind integration
cases use the actual `ModalPinnedModelPreparer`; only external snapshot acquisition
is replaced. Actual root checks, snapshot capture and serving-target validation run.

The affected regression selection passed **513 tests in 680.70 seconds** under
isolated CPython 3.12.9 / pytest 8.4.2 with automatic plug-ins disabled. It covers
inference, selected local chat/Evaluator callers, model snapshots, mounted I/O
and Modal inference tests. That run collected before the independent worker
matrix and six extra cleanup cases were integrated; it uses
the prior worker hash `f461ceca7df0346d4130e09a08876ee48050f6e4230ba4a9c9d6eb5f219b2127`.
The only final source delta removes the two-line same-class error passthrough;
the complete final 36-case run covers that revision, including its regression.
These selections overlap; neither is the entire repository suite. The preceding
launch slice's 2,613-test run is a broader pre-worker baseline, not a final-worker
whole-suite claim.

The exact Git archive produced `synaptic_tuner-1.1.0-py3-none-any.whl`,
**2,073,396 bytes**, SHA-256
`d3fb4d1cc7aa4eb7615c8d666ab08a56cd06092dfa4f0b60d18a090d51b85b20`.
Independent audit verified 725 unique ZIP/RECORD entries, every recorded hash
and size, and all 717 Python members byte-for-byte against the source archive.
The packaged worker matches the independently reviewed final hash.

A fresh dependency-only install passed the checked-in CI probe's **44
engine/Evaluator imports**, both packaged resources and a credential-free
`inspect.Signature.bind` check for the worker entry. The worker module came from
the installed environment under isolated Python in a neutral directory; Modal,
Torch, pytest, NumPy and pandas were absent. No entry point was invoked by the
signature probe, which also rejects hidden `**kwargs` parameters.

The unchanged 97-file training runtime lock and separate 66-member offline
worker closure (679,487 payload bytes) report `CURRENT`; both are schema-valid
in the wheel. Canonical skill mirrors match. A filename-only package scan found
no test paths or private artifact filenames; this is not a content-level
credential audit. CI's timeout is now 25 minutes because the preceding local
2,613-test regression took 15m38s, exceeding its former 15-minute limit. This is
headroom for verification, not a claim that CI ran. No provider calls, GPU work,
model downloads, push, merge or EHR changes were part of this slice.

## Remaining work

Complete the executable worker/bootstrap and separate inference runtime lock,
then explicit-client SDK transport and bounded owned session/cleanup integration.
Runtime/image/dependency pins must be measured rather than inferred from training.
