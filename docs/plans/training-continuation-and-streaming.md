# Training continuation and durable streaming — scoped design

Status: provider-neutral intent parsing, recipe propagation, structural checks,
and read-only `RunsAPI` source-stream verification are implemented. Packaged
execution remains deliberately rejected; no current run is resumable by these
contracts. No existing runtime lock has been refreshed for this future lane.

## Existing boundaries

`TrainingAPI` owns canonical load, resolve, plan, preflight and one-shot start.
`RunsAPI` already has bounded paged logs, verification and role-keyed artifact
streams. The current packaged Modal SFT contract publishes exactly five terminal
roles: workload record, training lineage, training metrics, final model and
tokenizer. The current Modal reader explicitly reports logs unsupported. The
generic `ArtifactPolicy.retain_checkpoints` flag and trainer `save_steps` alone
do not publish a resumable checkpoint; the current five-role verifier and
download helper reject other roles. The first completed run cannot be promoted
to a full-state resume without a recorded optimizer, scheduler, RNG and data
cursor. Its verified final adapter could be an input to a separately authorized
warm-start only after the future lineage binding, transfer and runtime loading
path is implemented.

## Two disjoint continuation intents

- **Full-state resume:** source is a verified, committed `trainer_state`
  checkpoint containing model/adapter weights, optimizer, scheduler, RNG and
  exact data cursor, each covered by an authenticated manifest. Keep the same
  immutable prepared dataset, base model revision, adapter layout, runtime and
  training invariants (batching, seed, optimizer, loss/template controls and
  data order). A preserved scheduler keeps the original total horizon. A longer
  horizon is allowed only with a separately declared, hash-bound scheduler
  transition and tested migration semantics; never silently reinterpret an
  existing scheduler state. The new run has a new one-shot authority and records
  the parent run, checkpoint digest and resumed step.
- **Adapter warm-start:** source is a verified final adapter (or a separately
  defined adapter snapshot), with the same pinned base revision and compatible
  adapter layout. New prepared data and other training controls may be selected.
  Initialize from weights, reset optimizer/scheduler and start a new step/data
  cursor and run lineage. Do not label it a full-state resume or combine its
  metrics with the parent's optimizer trajectory.

`TrainingInputV1.continuation` and recipe `continuation` accept the same
versioned nested intent; absence adds no key to legacy canonical JSON or
packaged config. `verify_continuation_source` now checks one `RunsAPI.verify`
result and the complete public artifact stream against the requested role,
size and digest. This read-only receipt is not a durable transfer, provenance
manifest, parent training-identity proof, or submission authority. In
particular, `compile_packaged_sft_workload` rejects any continuation block
with a closed explicit unsupported-execution error before a plan or provider
effect. Future integration must authenticate parent lineage and bind the
source to the exact child attempt, then stage and verify the artifact again.

## Proposed minimal integration slices

1. **Provider-neutral config and identity:** add an optional continuation block
   to a versioned training recipe/compiled workload. Bind mode, parent run,
   artifact role/digest, immutable source identities, child identity, schedule
   policy and parent-child lineage into the plan fingerprint. Keep existing
   recipes byte-identical when absent. Reject unknown modes and incompatible
   identity before preflight. No new provider-specific public verb or engine DB.
2. **Checkpoint publication:** version the generic artifact contract rather
   than mutating the existing five-role schema. Write a checkpoint into a
   private run-scoped, exclusive staging location; include complete state,
   trainer/library version, step/cursor, and per-member digests. Commit artifact
   bytes before control evidence. Expose a checkpoint only after immutable
   inventory and end-to-end verification succeed. Define an explicit checkpoint
   size/count/retention budget from measurement; do not reuse the current
   192-MiB-per-artifact/256-MiB-total limits for unmeasured optimizer state.
   Never overwrite a checkpoint name or infer that a failed/partial checkpoint
   is resumable. Keep the terminal five roles intact for legacy runs.
3. **Durable progress:** emit finite structured step/epoch/loss records into an
   append-safe run-scoped log with monotonic sequence/cursor. Publish bounded
   committed segments independently of terminal success; redact prompts,
   tokens, secrets, exception text and raw provider payloads. Implement the
   provider reader behind the existing `RunsAPI.logs` pagination contract;
   preserve its 200-entry/256-KiB page bounds. Live stdout alone is not durable
   recovery evidence. Unknown/gapped or uncommitted tails stay inconclusive.
4. **Consumer download:** reuse `RunsAPI.artifacts` and the existing verified
   stream pattern (bounded chunks, exact digest/size, private exclusive local
   file). Add roles only under the versioned policy and measured bounds; no
   arbitrary Volume-path reads or unverified raw bucket/SDK downloads.
5. **Execution:** in a future reviewed runtime source change, load a verified
   checkpoint before optimizer construction/restore for full-state mode, or
   initialize adapter weights before a fresh optimizer for warm-start. Verify
   state and cursor against signed workload at both host and child boundaries.
   Fail closed on an absent/mismatched member. Never replay an uncertain prior
   or current submission. Refresh affected runtime/trainer locks only after
   independent review; this design lane performs no lock maintenance.

## Provider-free acceptance tests before a live qualification

Test each mode's config and lineage round-trip; same-data/same-config resume;
declared horizon extension and scheduler transition; changed-data rejection for
resume and acceptance only for warm-start; missing/tampered optimizer, RNG or
cursor; wrong source role/revision; parent/child run separation; checkpoint
partial/duplicate/stale publication; readback digest and size mismatch;
paginated logs across gaps, restarts, terminal failure and concurrent commits;
legacy five-role plans unchanged. Then test actual trainer resume against an
interrupted tiny run, comparing uninterrupted and resumed state/step order under
the pinned trainer. Provider-free passing does not authorize a paid run.
