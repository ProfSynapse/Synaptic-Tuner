# Training progress chart feed

Status: staged implementation; no live Modal chart feed is qualified. The existing training attempt must remain untouched. Modal advertises observe and artifact streaming, but not logs. Pending status and a trainer-start phase do not establish optimizer progress.

This updates the plan retained at `fd2a4b14` against merged main `73a3ba2c`. The historical local-state callback no longer exists. Do not restore its assumptions or create a second telemetry schema.

## Existing boundaries

`synaptic_tuner/api/v1/observations.py` already defines a versioned, typed, append-only observation record with stream identity, sequence, timestamp, family and exact payload. Observations are non-authoritative: they cannot authorize execution, change lifecycle state or establish artifact verification. Reuse this contract for chart measurements. Its existing training-phase payload remains compatible; its arbitrary phase text is not a reviewed safe-text export boundary.

Current SFT metrics originate in `BaseMetricsCallback.on_log` (`Trainers/shared/callbacks/base.py`), with private JSONL output. The current samples-per-second calculation is estimated, timestamps are local, and current and peak GPU memory are distinct measurements. The packaged worker retains child stdout/stderr privately and publishes final `training_metrics.json` only after training. Neither private stdout nor final metrics constitutes a live public stream.

`RunsAPI.logs` already offers bounded pages and authenticated run-bound cursors. Modal has private chained log transport, but its logs capability remains disabled. Typed observation requests/pages do not yet have a public reader facade. Choose and qualify a projection of the canonical observation record onto the existing log transport before promising live delivery; do not invent a competing training-event vocabulary inside message text.

## Dependency-ordered slices

1. **Numeric contract (current slice).** Add `TRAINING_METRIC_OBSERVED` and one exact numeric payload to the existing observation family, JSON schema and public exports. Reuse record identity, sequence and timestamp. Measurements name their units where applicable and use null when unknown. Step fields are counts; epoch and loss are dimensionless, and loss is objective-specific rather than comparable across arbitrary training methods. Separate training time from whole-job time, current memory from cumulative peaks, measured token throughput from estimated sample throughput. No free text, paths, credentials or manuscript data belongs in this payload. The codec validates shape, not whether a producer actually measured a claimed value. Preserve existing record bytes and phase semantics. This slice does not wire callbacks or enable provider capabilities.
2. **Capture and durable transport.** Adapt the existing callback independently of checkpoint retention. Emit canonical observations at configured trainer log intervals. Normalize emitter timestamps to UTC without changing legacy timestamp admission. Define durable ordered private storage, finite byte limits and the observation-to-log projection. Keep the five-artifact completion inventory unchanged. Review the locked runtime inventory before editing its members.
3. **Authenticated readback qualification.** Test the actual provider chain, run/cursor isolation, byte/page bounds, partial committed reads, reconnect ordering and privacy. Carry numeric payloads through the real authenticated read path. Keep Modal logs disabled until provider-free checks, independent review and a separately authorized live transport qualification pass. A contract test or successful training run alone is not transport qualification.
4. **Host chart integration.** Consume accepted observations with run-bound cursors; persist chart points and cursor together. Show stale data as stale, not as zero. Surface gaps and invalid cursors; do not fabricate missing optimizer steps. The consuming host owns SSE/WebSocket/browser state. Browser disconnects never replay training.

## Numeric contract acceptance

Test exact required field sets, null unknowns, finite numeric values, rejection of booleans and overflow, bounded integer steps and memory bytes, step/total consistency, utilization range, schema parity, public exports, and observation family/page binding. Existing observation kinds and wire shapes must remain unchanged. Loss may be negative for objectives that allow it; memory and rates cannot be negative. Memory occupancy must never be presented as measured GPU utilization.

Runtime validation is authoritative for exact Python integer types, finite-number conversion and cross-field step/total comparisons that standard JSON Schema cannot fully express. The no-text property is local to the new payload: existing stream identifiers and phase text still require source binding and privacy review before public transport. Existing observation pages have a count bound, not a serialized-byte bound; the later projection must apply its explicit byte budget.

Adding a kind to the closed v1 vocabulary is additive for current code, but an older consumer can reject a newly emitted kind. No emitter is enabled by the contract slice. Coordinate consumer support before the later emission rollout rather than assuming older clients ignore unknown kinds.

The later capture/transport acceptance must additionally cover runs without checkpoint retention, configured sampling cadence, interrupted-stream recovery, complete UTF-8 bounded projection, authenticated multi-page reads and unchanged terminal artifact verification. Completion and writing quality remain separate from progress observations.

## Deliberately separate work

Checkpoint resume/continuation, evaluation-only recovery, tagged desktop releases, MLX, GGUF, richer GPU power sampling and cost charts are not prerequisites for the numeric chart contract. Do not relaunch or modify the active training attempt to add observability. No source-only contract change qualifies a model, hardware profile or live provider feature.
