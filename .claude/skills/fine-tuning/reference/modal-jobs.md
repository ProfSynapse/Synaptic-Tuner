# Modal Training v1 Reference

Modal training is a submodule-first execution provider behind the public
`TrainingAPI`. It is not a second training CLI and it is not launched with
`modal run`. A consuming host supplies request configuration, durable storage,
authorization grants, credential resolution, and evidence authentication. The
engine supplies strict contracts, planning, verification, staging, the mutation
broker, and the fixed remote worker.

## Product flow

1. The host loads a `synaptic-training-request/v1` document.
2. `TrainingAPI.resolve` locks the host project, engine submodule, model,
   dataset, provider profile, runtime, and artifact policy.
3. `TrainingAPI.plan` produces one immutable plan whose canonical, non-secret
   `synaptic-modal-plan-context/v1` binds the verified deployment, explicit
   client scope, exact Volume IDs, quote/expiry, cost ceiling, and unique
   operation identity.
4. `TrainingAPI.preflight` performs read-only source, deployment, Volume,
   capability, quote, and authorization checks.
5. The host obtains an opaque grant and calls `TrainingAPI.start` once.
6. `TrainingAPI.outcome` reconciles the provider call and verifies the exact
   terminal records and artifacts.

The engine now provides `ModalTrainingOperations` and
`compose_modal_training_operations`. A consuming main project must still
provide a conforming durable repository and authenticated host ports before a
live provider proof; do not use a manual launch as a substitute.

## Frozen live evidence

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
- Fixed deployed app/function `synaptic-training-v1/run_sft_v1`.
- One A10 GPU, one canonical command argument, `retries=0`, and one detached
  `.spawn()` call behind `MutationBroker`.
- One digest-pinned Unsloth registry image with its inherited entrypoint
  cleared.
- One existing Modal Volume v1 for control/log/evidence records and one
  distinct existing Modal Volume v1 for input/output artifacts.
- Each effect is isolated below `operations/{effect_id}/`; jobs never share
  global `input/`, `output/`, `logs/`, or `evidence/` paths.
- The remote job independently clones the exact pushed host project and exact
  engine commit, verifies the project gitlink, and invokes only
  `Trainers/sft/runtime_v1.py --canonical-workload-stdin` without a shell.
- The verified runtime is CPython 3.11.14 at `/opt/conda/bin/python3`; its
  executable, image, complete hash-pinned launcher dependency closure,
  deployment wrapper, remote worker/producer/runtime modules, SFT entrypoint,
  and ML stack are checked in `modal-runtime-v1.lock.json`. The host enforces
  this packaged lock before preflight and the remote materializer enforces it
  again against the reconstructed exact checkout.

## Storage and evidence

Modal Volume is the authoritative provider-native artifact store for a Modal
run. Hub publication is optional and separately authorized. The remote producer
emits exactly five artifacts: workload record, training lineage, training
metrics, final model, and tokenizer.

The control Volume contains operation-scoped, authenticated structured logs,
terminal evidence, and the completion manifest. The host database stores the
expected operation identity, lifecycle, one-shot authority consumption,
provider job reference, and verification result. It implements
`ModalTrainingRepository`, including an atomic preparation commit. The engine
derives result expectations from the durable preparation plus canonical
attempt record; it does not select or ship a concrete database and must not
create SQLite state.

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
