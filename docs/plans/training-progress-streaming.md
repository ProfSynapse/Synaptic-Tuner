# Training progress chart feed

Status: numeric contract, bounded log projection and optional local callback capture implemented; no live Modal chart feed is qualified. The existing training attempt must remain untouched. Modal advertises observe and artifact streaming, but not logs. Pending status and a trainer-start phase do not establish optimizer progress.

This updates the plan retained at `fd2a4b14` against merged main `73a3ba2c`. The historical local-state callback no longer exists. Do not restore its assumptions or create a second telemetry schema.

## Existing boundaries

`synaptic_tuner/api/v1/observations.py` already defines a versioned, typed, append-only observation record with stream identity, sequence, timestamp, family and exact payload. Observations are non-authoritative: they cannot authorize execution, change lifecycle state or establish artifact verification. Reuse this contract for chart measurements. Its existing training-phase payload remains compatible; its arbitrary phase text is not a reviewed safe-text export boundary.

Current SFT metrics originate in `BaseMetricsCallback.on_log` (`Trainers/shared/callbacks/base.py`), with private JSONL output. The current samples-per-second calculation is estimated, timestamps are local, and current and peak GPU memory are distinct measurements. The packaged worker retains child stdout/stderr privately and publishes final `training_metrics.json` only after training. Neither private stdout nor final metrics constitutes a live public stream.

`RunsAPI.logs` already offers bounded pages and authenticated run-bound cursors. The legacy Modal coordinator has a private terminal log chain; the packaged Modal worker has no live log transport and advertises `logs=False`. Typed observation requests/pages do not yet have a public reader facade. Qualify a projection of the canonical observation record onto the packaged read path before promising live delivery; do not invent a competing training-event vocabulary inside message text.

## Dependency-ordered slices

1. **Numeric contract and local capture (implemented, not live).** `TRAINING_METRIC_OBSERVED` reuses observation identity, sequence and timestamp; the typed payload has exact numeric fields and null unknowns. Loss is objective-specific and may be negative. Time scopes, current versus peak GPU bytes, measured token throughput and estimated sample throughput stay distinct. No private text or paths belong in it. The optional callback captures independently of checkpoint retention, but has no packaged sink or provider capability.
2. **Packaged capture and durable transport (proposed).** Bind an explicit fixed capture configuration into the packaged workload only when requested; absent configuration must preserve current canonical bytes. Feed the existing typed callback sink through the authenticated packaged child invocation, carry bounded numeric frames over a dedicated child-to-parent pipe rather than stdout, and attach retained project/run identity, sequence and UTC receipt time only in the trusted parent. Store bounded committed progress separately from the five-artifact completion, with explicit finite capture-budget exhaustion. Review affected locked inventories before edits.
3. **Authenticated readback qualification.** Test the actual provider chain, run/cursor isolation, byte/page bounds, partial committed reads, reconnect ordering and privacy. Carry numeric payloads through the real authenticated read path. Keep Modal logs disabled until provider-free checks, independent review and a separately authorized live transport qualification pass. A contract test or successful training run alone is not transport qualification.
4. **Host chart integration.** Consume accepted observations with run-bound cursors; persist chart points and cursor together. Show stale data as stale, not as zero. Surface gaps and invalid cursors; do not fabricate missing optimizer steps. The consuming host owns SSE/WebSocket/browser state. Browser disconnects never replay training.

## Numeric contract acceptance

The numeric contract was checkpointed at `9b235855`. The internal
`tuner/execution/observation_logs.py` projection now encodes the full canonical
metric observation into the existing log message, with a 4,096-byte UTF-8 bound.
It requires the expected training run's exact project/run identity and UTC Z
timestamp, preserves the original sequence, and verifies outer timestamp,
sequence, event, level and byte count on readback. Duplicate keys, nonfinite
numbers, alternate JSON encodings, wrong identities and malformed inputs fail
with a fixed diagnostic. No arbitrary phase text is admitted. Matching a run
name is not source authentication; this codec must be composed behind the
authenticated producer/read boundary.

Tests carry fractional metrics through the real Modal log-chunk parser and
the coordinator's authenticated public paging/cursor fixture. These are
provider-free format/read-path checks, not a live append qualification.

Test exact required field sets, null unknowns, finite numeric values, rejection of booleans and overflow, bounded integer steps and memory bytes, step/total consistency, utilization range, schema parity, public exports, and observation family/page binding. Existing observation kinds and wire shapes must remain unchanged. Loss may be negative for objectives that allow it; memory and rates cannot be negative. Memory occupancy must never be presented as measured GPU utilization.

Runtime validation is authoritative for exact Python integer types, finite-number conversion and cross-field step/total comparisons that standard JSON Schema cannot fully express. The no-text property is local to the new payload: existing stream identifiers and phase text still require source binding and privacy review before public transport. Existing observation pages have a count bound, not a serialized-byte bound; the later projection must apply its explicit byte budget.

Adding a kind to the closed v1 vocabulary is additive for current code, but an older consumer can reject a newly emitted kind. No emitter is enabled by the contract slice. Coordinate consumer support before the later emission rollout rather than assuming older clients ignore unknown kinds.

The later transport acceptance must additionally cover runs without checkpoint retention, configured sampling cadence, interrupted-stream recovery, complete UTF-8 bounded projection, authenticated multi-page reads, explicit capture exhaustion and unchanged terminal artifact verification. Completion and writing quality remain separate from progress observations.

## Live bridge constraints found in source review

The local `MetricsTableCallback(..., metric_sink=callable)` seam now forwards
typed numeric snapshots at its configured training-log interval, independently
of checkpoint saves. It is disabled by default and is not injected by the CLI
or packaged child. The sink receives a payload, not a run identity or a generic
log dictionary; a future bound adapter must provide sequence, UTC time and
retained execution identity before using the projection.

Capture is primary-rank-only and requires optimizer loss or learning-rate
evidence, so evaluation-only logs are not charted as training steps. Training
elapsed time is monotonic; the single-process sample-rate estimate subtracts
the resumed initial step. Unavailable or invalid measurements stay null;
whole-job time, token throughput and GPU utilization are not inferred. CUDA
byte counters come directly from the active device. Optional capture/setup or
sink failure disables only that sink with one fixed diagnostic, not an exception
containing private values. Existing table/JSONL behavior remains unchanged.

The user explicitly approved admitting the existing observation and timestamp
contract modules to the offline trainer inventory, increasing it from 66 to
68 members. The independent enumeration and hashes were updated; an isolated
import test excludes site packages and the source checkout to verify this
dependency closure. No package or image pin, trainer entrypoint, child transport
contract or provider capability changed. This admission enables the local
optional import; it does not qualify a remote metric feed.

The *legacy coordinator* finalizes one terminal log record at sequence zero,
binds an immutable terminal chain digest, requires a contiguous zero-based
chain and caps that whole chain at 256 records. Those rules belong to its old
reader; they are not the packaged worker's transport contract. The packaged
worker currently publishes signed completion and five artifacts, but no log
stream. Its host reader rejects log requests before reading completion. Do not
flip that capability or repurpose the immutable terminal chain as a live feed.
Existing observations permit sequence gaps; transport must preserve them rather
than renumbering. Bounded pages/chunks are necessary, but 256 points is not a
whole-run budget for a potentially long training job.

The Foundation canonical evidence limit is 16 KiB, as exercised by the worker
reproducer; the public query's 262,144-byte message allowance is not an
evidence allowance. Bound each canonical marker/evidence object including
serialization overhead, each chunk and page, and the run's configurable finite
capture budget independently. On budget exhaustion, publish an authenticated
closed exhaustion status; do not silently truncate a chart into apparently
complete data. Capture remains best-effort and must never stop training.

The isolated child has a bound run reference, but no full project identity or
authenticated whole-job start timestamp. Bind identity from retained execution
context in the trusted parent; leave whole-job elapsed time null until a bound
origin exists. Callback capture sends numeric fields only, never generic
trainer JSONL. The approved observation/timestamp offline closure admission
above enables the typed payload locally; it does not create a remote stream.

## Proposed minimal packaged bridge

Add an optional exact capture object to `tuner/training/modal_recipe.py`,
`tuner/training/packaged_compilation.py` and the packaged workload schema. Its
cadence and finite run budget are bound into the compiled workload; omission
leaves invocation/workload bytes unchanged. In `Trainers/sft/train_sft.py`,
enable the existing `MetricsTableCallback(metric_sink=...)` only for that
configuration. Extend `tuner/runtime/packaged_sft_execution.py` and
`packaged_sft_child.py` with one verified, dedicated inherited progress pipe;
keep the sealed dataset FD and private stdout/stderr contracts intact. The
child writes only bounded canonical numeric frames. The parent drains them
during execution, validates payloads, attaches identity from the retained
dispatch/binding, assigns sequence and UTC receipt time, and applies the
existing `tuner/execution/observation_logs.py` projection. No arbitrary module
path, environment plugin, transcript text or child-supplied project identity.

In `tuner/execution/providers/modal/packaged_worker.py`, publish create-only
bounded chunks and an authenticated marker *after* each complete batch under
the existing private control Volume, separate from signed final completion.
Commit the Volume writer so SDK readers can observe the batch. These SDK reads
are not mounted-reader cache reads and do not require `.reload`. A marker
binds run identity, previous marker digest, committed chunk digests and high
sequence; an exhaustion marker makes the finite capture limit visible. In
`packaged_reader.py` and `tuner/training/modal_host_reader.py`, verify the
largest contiguous authenticated committed prefix while the call is pending,
then page it through existing run-bound cursors. Host-retained cursor/high-water
state can reject rollback of a prefix it has already seen. An unseen tail may
be absent, a published prefix may lag, and even a completed run has **no
telemetry completeness or freshness guarantee**. Progress never changes
lifecycle authority, signed completion, or the five-artifact inventory; a new
completion schema is unnecessary for this best-effort contract.

Test exact absent-config byte compatibility, cadence and capture exhaustion,
numeric-only child frames, FD/identity isolation, partial writes and committed
visibility, marker tampering/rollback, interrupted and multi-page reads,
canonical evidence/chunk/page limits, privacy, and unchanged completion and
artifacts. Relevant suites are packaged compilation, SFT callback, packaged
child/execution, Modal packaged worker/reader and host reader/public paging.
The coordinated child transport, signed progress and readback change awaits
scoped approval. Only after provider-free tests, independent review and
separately authorized live qualification may `modal_host_composition.py`
advertise logs; until then this protocol and capability remain **proposed**.

## Deliberately separate work

Checkpoint resume/continuation, evaluation-only recovery, tagged desktop releases, MLX, GGUF, richer GPU power sampling and cost charts are not prerequisites for the numeric chart contract. Do not relaunch or modify the active training attempt to add observability. No source-only contract change qualifies a model, hardware profile or live provider feature.
