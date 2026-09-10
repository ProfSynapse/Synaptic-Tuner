# Modal mounted chat preparation

Status: implementation and local qualification in progress, 2026-09-10.
Engine-only; no provider access, downloads, process startup or live qualification.

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
validation. Archive semantics determine full versus LoRA; the training
`load_in_4bit` setting is not transferred into inference configuration.
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

Pending final independent tests/review, affected regression and exact-commit
installed-wheel qualification. The test fixture replaces producer bytes before
native evidence is created, runs the actual source/workload/preparation and
Foundation STAGE/signing chain, and mounts bytes matching every signed hash and
size. It does not rewrite an authenticated snapshot or mock worker admission.

## Remaining work

Complete the executable worker/bootstrap and separate inference runtime lock,
then explicit-client SDK transport and bounded owned session/cleanup integration.
Runtime/image/dependency pins must be measured rather than inferred from training.
