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

## Standalone consumer retention

With capture configured and the selected public descriptor advertising logs,
the standalone wait loop reads one page per wait tick (at most 100 entries and
64 KiB), including the completion tick. It performs no final drain. Transient
read failures retry on the next tick with the same cursor and retained prefix;
invalid/conflicting records, storage failures or exhausted local bounds disable
polling. Training continues independently of telemetry.

Canonical typed records are written create-only at mode 0600 under the private
owner-only state root, using `progress-<sha256>.jsonl`, where the digest binds the
canonical `TrainingRunRef` dictionary. Replayed run/sequence pairs must have
identical canonical bytes. The optional result `progress` reports only closed
status, `records_path` and `retained_records`; it emits no arbitrary exception
text or extra stdout stream. Omitted capture omits this result field. With
capture requested and Modal logs disabled, `status` is
`progress_capability_disabled` and `records_path` is null; no public log call or
retained progress file is created. Process-local cursor/dedup state is not
restored from JSONL and does not recover coordinator/Foundation state.

## Bounded live qualification proposal (2026-10-12)

This is a reviewable proposal, not an executable approval or an already
implemented qualification mode. The public Modal registration still advertises
`logs: false`. Ordinary consumers must honor that descriptor. Turning it on to
make a smoke run work would advertise an unqualified capability.

### Implementation and admission gate

Extend the existing checked-in qualification workflow with a separately reviewed,
restricted metrics qualification operation. Reuse its exact-source build, CPU
receipt, retained deployment/call binding, host-owned authority and one-shot
attempt journal. Do not introduce an alternate launcher, direct Volume probing,
provider-specific public logs verb, or runtime secret channel. Existing
`examples/modal_chat/launch.py --mode qualify-training` qualifies native training
and artifacts; it does not supply this metrics operation. Standalone
`train --qualify` is a CPU self-check and cannot qualify GPU metric delivery.

The proposed operation must admit only an exact reviewed source/configuration,
synthetic fixture digest, selected provider/environment and bounded attempt.
Its isolated qualification host may compose the existing public `RunsAPI.logs`
reader under explicit qualification authority, while ordinary host descriptors
remain disabled. Review this composition and its separation from production
capability advertisement before implementation. All reads must retain the
submitted assessment and current Foundation revalidation. No disabled-capability
fallback to a private reader is permitted in normal training.

The provider-free barrier and independent review must first cover this admission,
absence of production activation, signed readback, delayed visibility, torn tails,
canonical conflicts, telemetry failure isolation, bounded polling and consumed
claims. If the operation cannot preserve those boundaries, stop at this gate.
A later successful live receipt and review authorize a separate capability change;
a smoke submission alone cannot turn on logs.

### Fixture and bounded configuration

Use newly authored harmless synthetic messages (for example short greetings and
arithmetic explanations), with explicit disjoint train/validation rows, declared
lineage and an immutable prepared-data digest. Use the existing
`python tuner.py prepare-dataset --config <reviewed-config.json> --json` surface
and review representative complete rows plus split counts. Public sanitized
fixtures under `tests/fixtures/training_product/modal_live_v1/substitutes` are
artifact placeholders, not training data or authenticated live evidence. Existing
Qwen smoke recipes reference private prepared data and must not be launched as-is.

Select an already qualified exact model/runtime profile and retain its model and
tokenizer revisions, recipe/render choices, image inventory and source locks.
A proposed dedicated recipe uses `observation_capture.every_n_steps: 1`,
`observation_capture.max_records: 32`, `training.max_steps: 32`, one accelerator,
and `job.timeout_seconds: 900`, with post-training evaluation omitted. These are
experiment limits for review, not evidence that 32 steps last long enough for a
background commit. If training finishes before any public metric is observed,
record an inconclusive transport result; never extend or replay that attempt.

Proposal: one integrated attempt with one CPU qualification submission and
one GPU training submission (the standalone run already performs its CPU check;
do not add a separate `train --qualify` submission),
zero automatic retries, and an operator-approved all-in budget ceiling of USD 5.
The existing recipe `maximum_cost_minor_units` and quoted GPU timeout estimate
are admission checks, not an all-in provider billing cap; builds, CPU, memory and
storage may be excluded. Obtain current scoped rates and review those exclusions
before choosing the exact minor-unit threshold. Do not quote historical prices
or imply the ceiling is technically enforced. A cancel is a separately approved
provider action; timeout or a failed watcher alone grants no cancel/replay authority.

Existing local planning and read-only rate surfaces, after a concrete reviewed
recipe exists, are:

```bash
python tuner.py train --job-config <reviewed-recipe.yaml> --plan --json
python tuner.py train --job-config <reviewed-recipe.yaml> --quote --modal-profile <name> --modal-environment <name> --json
```

Neither command proves live transport. The future qualification invocation must
be documented from its implemented parser/help before paid approval is requested.
For the standalone packaged route, preserve the exact clean engine commit and
reviewed source-wheel build; `prepare_current_source_wheel` explicitly builds
without a customer repository. The clean pushed host superproject/exact engine
gitlink requirement belongs to the older minimal consumer route and must not be
imported into standalone admission. CPython 3.11.14 POSIX, hash-pinned launcher
dependencies, the integrated current CPU qualification and exact effect authority
remain prerequisites. Preserve fresh attempt resources and any failures.

### Acceptance evidence

Retain a bounded chronological ledger of public page requests, receipt times,
cursors, canonical record digests and the exact provider call's pending/terminal
observations, with source/configuration/profile and fixture digests. Acceptance
requires at least two distinct authenticated numeric observations returned by
`RunsAPI.logs` while that same call is independently observed pending. A final
training_metrics artifact or points first read after completion does not pass.
Repeat the retained cursor across empty/end pages and show growing pages produce
one locally retained record per run/sequence, with canonical-byte equality on
replay. Compare the retained file and public projected records by digest.

Capture the actual background-commit latency and whether overlapping writes
cause temporary page rejection. The current protocol fails closed on a committed
short/torn envelope, potentially hiding a valid prefix. Demonstrate delayed
empty-to-growing readback and at least one overlapping publication window; report
what was actually observed. If no partial write is encountered, live partial-write
behavior remains unqualified: provider-free torn-tail tests are supporting evidence,
not a live experiment. Do not inject corruption into another attempt. A separate
bounded synthetic transport fixture or reviewed atomic-leaf publication is needed
if natural overlap cannot establish the required behavior.

Finally authenticate the unchanged outcome and five-artifact inventory, retain
closed failures and consumed claims, and record resource/time/cost observations.
No failed/inconclusive watcher, missing metric tail or download failure authorizes
training replay. General freshness, complete history, restart recovery, browser
charts and SSE/WebSocket behavior are outside this qualification.
