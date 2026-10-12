# Training progress chart feed

Status: approved numeric child capture, signed remote progress storage and
authenticated host readback are implemented. Provider-free verification does
not qualify live delivery. Modal's logs capability remains disabled. The active
training/evaluation attempt is unchanged; no paid qualification was launched.

## Contract and scope

Reuse ObservationRecordV1 and its typed TRAINING_METRIC_OBSERVED payload,
not a second telemetry vocabulary or private stdout export. Observations cannot
authorize execution, change lifecycle state or establish artifact verification.
The existing signed completion and five-artifact inventory remain unchanged.

Loss is objective-specific and may be negative. Current and peak GPU memory,
training and whole-job elapsed time, measured token throughput and estimated
sample throughput stay distinct. Unavailable values remain null. Memory
occupancy is not GPU utilization. The parent supplies authenticated project/run
identity, sequence and UTC receipt time; whole-job elapsed time remains null
without a bound origin. Callback timestamps are not receipt timestamps.

Earlier local checkpoints: numeric contract `9b235855`, canonical log projection
`2a8aff56`, optional callback `d9d63feb`, legacy paging fix `3f0498f4`.
The approved offline inventory admission added the existing observations and
timestamp modules (66 to 68 members). This bridge refreshes affected source
hashes without changing package/image pins or adding offline inventory members.
The provider module is covered by the exact-source packaged wheel, not the
isolated trainer inventory or legacy bootstrap inventory.

## Opt-in configuration

The recipe and compiled workload accept this exact optional object:

```yaml
observation_capture:
  schema_version: synaptic-training-observation-capture/v1
  every_n_steps: 5
  max_records: 10000
```

Cadence is an integer from 1 to 1,000,000 optimizer steps; the capture budget is
an integer from 1 to 10,000 metric records, plus one closed exhaustion record.
These example values are not enabled by default. Omission preserves the
previous canonical workload/invocation shape; explicit null is rejected.
Capture is independent of checkpoint retention and primary-rank-only.

## Implemented path

1. MetricsTableCallback emits the typed numeric payload through an optional
   packaged sink. Evaluation-only callback logs are not training measurements.
2. A dedicated nonblocking inherited pipe carries canonical frames of at most
   4,096 bytes. The sealed dataset FD and private stdout/stderr remain separate.
   Parent validation checks exact fields, finite numbers, cadence and monotonic
   optimizer steps. Pipe, callback and queue failures disable telemetry without
   failing training; they do not imply clean budget exhaustion.
3. The parent signs each bounded envelope with purpose
   `modal-packaged-progress/v1`, binding the retained run, command, execution,
   provider call and budget. Files live under `progress-v1-<effect digest>`,
   outside the finalizer's create-only `operations/<effect>` root. Shards have
   at most 256 metric/status leaves; each envelope is at most 8,192 bytes.
   A bounded asynchronous queue avoids making training wait on remote I/O.
4. Ordinary Modal Volume background commits publish these files. There is no
   new per-metric commit RPC. SDK readers see the committed server view; they
   are not mounted readers requiring `.reload`. See
   [Modal background commits](https://modal.com/docs/guide/volumes#background-commits).
5. The packaged reader authenticates bounded pages, then the host reader uses
   existing reader authority and RunsAPI.logs projection. It does not poll
   completion to obtain metrics. Foundation evidence contains entry digests,
   not a duplicate unbounded copy of the log page.

At exactly the configured record budget, the parent may append an authenticated
TRAINING_PHASE_OBSERVED record with the sole admitted payload
`{"phase":"metric_capture_exhausted"}`. This is an ordinary public log entry,
not hidden verification metadata. The progress codec allows that exact status;
the original metric-only codec stays strict. Arbitrary phase text, paths,
prompts, exceptions and trainer JSONL are not exportable through this lane.

## Readback and consumer limitations

Progress is best-effort: no final completeness or freshness guarantee, even
after training completes. A missing tail is not zero loss, zero utilization,
training completion or authorization to replay training. Queue overflow leaves
an open prefix, not a false exhausted status. The page watermark names its
last returned record or cursor anchor, not the global provider head.

Each resumed numeric cursor authenticates its anchor and the chain within the
returned page. It does **not** prove the chain back to genesis or provide
durable cross-request rollback/fork detection. Consumers may retain canonical
record digests and flag a changed sequence, but that is not a stronger server
guarantee. Do not claim arbitrary reconnects prove a complete history.

The existing public API returns next_cursor only when a page is truncated.
For live tail polling, retain the most recent non-null cursor, repeat from it
when the current tail ends, and deduplicate by run and sequence while checking
canonical bytes for disagreement. Do not replace that retained cursor with
null on an empty/end page. Before the first full page, repeat from null. Host
chart points and retained cursors belong to consumer persistence, not a new
engine database. SSE/WebSocket/browser behavior is also host-owned.

Modal documents periodic background commits but not atomic visibility of an
in-progress final-path write. A short/torn committed envelope currently fails
closed and can temporarily reject a page containing an otherwise good prefix.
No unauthenticated point escapes. Before enabling logs, qualify overlapping
background commit/write behavior or review an atomic final-leaf publication
mechanism. Do not promise consistently readable prefixes before that evidence.

## Verification and remaining gates

Provider-free tests cover absent-config compatibility, numeric-only frames,
cadence, exhaustion, FD/callback failures, queue overflow, create-only namespace
separation, delayed/partial committed visibility, torn-tail rejection, signed
status at the exact budget, cursor 10,000, identity/tamper rejection, bounded
host readback, empty-to-growing pages and public exhaustion visibility. The
public fixture also covers retaining a tail cursor and deduplicating replay.
Independent review found no remaining blocker within this disabled scope.

Live qualification remains separate and requires authorization. Only after
provider-free checks, independent review and successful live qualification may
modal_host_composition.py advertise logs. A successful training run, local
callback test or CPU package proof alone is not live transport qualification.

Checkpoint continuation, evaluation-only recovery, releases, MLX, GGUF, richer
GPU sampling and cost charts are separate work. Never replay or alter an active
attempt to add observability. The old coordinator's immutable terminal chain
and 256-record cap are not this packaged progress protocol.
