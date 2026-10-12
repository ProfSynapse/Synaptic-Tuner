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
restored from JSONL and does not recover coordinator/Foundation state. Failed
local writes/fsync can preserve the accepted prefix alongside a partial trailing
line; there is no automatic repair/reopen. `retained_records` counts the accepted
in-memory prefix and can differ from crash-durable content after fsync failure.
Once created, `records_path` remains reported even if no full record was retained.

## Restricted live qualification workflow (2026-10-12)

The checked-in restricted operation below is provider-free implemented and
reviewable. Its fixture and limits remain candidates for separate paid approval;
no live metrics qualification has been performed. The ordinary Modal registration advertises
`logs: false`. Ordinary consumers must honor that descriptor. Turning it on to
make a smoke run work would advertise an unqualified capability.

### Implementation and admission gate

`train --qualify-metrics` extends the existing standalone workflow with an exact
canonical admission and reviewed SHA-256. It reuses the exact-source build,
integrated CPU receipt, retained deployment/call binding, host-owned authority
and one-shot attempt journal. The separate qualification composition verifies
its admission and durable claim before exposing the existing public
`RunsAPI.logs` within that attempt. Ordinary composition continues to reject
qualification authority and advertises logs disabled. There is no new engine
logs verb, direct Volume probe or private reader fallback in normal training.

The candidate binds the final clean engine commit, profile and build-intent
hashes, compiled workload/configuration and prepared-data/content digests,
expected account/workspace, selected profile/environment, exact one-shot attempt,
GPU count and finite step/time/cost limits. The reviewed bytes are checked before
credentials or build/CPU/GPU effects; expected live provider scope is checked
before runtime preparation. A durable global claim consumes the attempt even
when a later outcome is uncertain. `--fresh-attempt` is rejected for this mode.
Do not change the attempt reference to reinterpret a failed submission as retry.

Candidate generation is part of the provider-free `train --plan` surface. It
writes create-only canonical bytes to an existing owner-private directory outside
the engine checkout, reports their SHA-256 and `execution_authorized: false`, and
makes no cloud calls. Generate after integrating and reviewing all source edits,
not from a dirty worktree or a circular checked-in source-SHA manifest. There is
no checked-in paid admission for the synthetic fixture.

The provider-free barrier and independent review cover admission/scope binding,
ordinary capability isolation, public authenticated readback, bounded persistence,
delayed/torn-tail rejection, pending observations bracketing reads, finite ledger
and consumed claims. Existing `train --qualify` remains CPU-only, and minimal
`examples/modal_chat/launch.py --mode qualify-training` remains native
training/artifact qualification; neither substitutes for this operation. All
reads retain the submitted assessment and current Foundation revalidation.

### Fixture and bounded configuration

The concrete [synthetic fixture](../../examples/training_metrics_fixture/README.md)
contains eight harmless question/answer pairs with declared context/target lineage,
complete-document order and an explicit six-train/two-validation sequence-tail
split. Its checked-in ingestion/preparation configs use the existing CLI; all
eight complete prepared rows and the CPU-only pinned-tokenizer mask check were
reviewed. Local evidence and exact bundle/dataset/content digests live beside the
fixture. Rebuild using its README; generated publications remain under `.tracking/`. Public sanitized
fixtures under `tests/fixtures/training_product/modal_live_v1/substitutes` are
artifact placeholders, not training data or authenticated live evidence. Existing
Qwen smoke recipes reference private prepared data and must not be launched as-is.

Select an already qualified exact model/runtime profile and retain its model and
tokenizer revisions, recipe/render choices, image inventory and source locks.
The dedicated `Trainers/recipes/qwen35_4b_modal_metrics_qualification.yaml` uses `observation_capture.every_n_steps: 1`,
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

From the final clean integrated engine checkout, rebuild the fixture, review the
plan and generate one admission outside the checkout. The selected names below
were confirmed by read-only discovery for this machine: profile/account/workspace
`synaptic-labs`, existing environment `synaptic-smoke-v1`. Revalidate actual scope
before execution. Substitute an unused private output path and the reviewed
attempt reference; placeholder values are not a runnable approval.

```bash
python tuner.py train --job-config Trainers/recipes/qwen35_4b_modal_metrics_qualification.yaml --plan --json
python tuner.py train --job-config Trainers/recipes/qwen35_4b_modal_metrics_qualification.yaml --plan --metrics-qualification-attempt <reviewed-attempt> --qualification-account synaptic-labs --qualification-workspace synaptic-labs --qualification-output <private-directory>/admission.json --modal-profile synaptic-labs --modal-environment synaptic-smoke-v1 --json
```

Review the complete canonical admission and returned digest. A later read-only
rate observation uses the existing surface, separately from candidate generation:

```bash
python tuner.py train --job-config Trainers/recipes/qwen35_4b_modal_metrics_qualification.yaml --quote --modal-profile synaptic-labs --modal-environment synaptic-smoke-v1 --json
```

A read-only rate observation on 2026-10-12 at 10:39:10 local used the
`synaptic-labs` profile in `synaptic-smoke-v1`, with isolated CPython 3.11.14
and the hash-locked 37-package launcher closure including Modal SDK 1.5.4.
It returned USD 1.95000/hour for one L40S; this gives a calculated GPU-only
estimate of USD 0.4875 for the 900-second timeout. This excludes build, CPU, memory, storage and usage beyond timeout;
it is neither an all-in billing cap nor execution approval. No job was submitted.
Refresh the scoped quote if required when reviewing the final admission. After
provider-free checks and independent review, request paid approval for the exact
admission digest, source, selected scope, one integrated CPU/GPU attempt and budget.
Only that later approval authorizes:

```bash
python tuner.py train --job-config Trainers/recipes/qwen35_4b_modal_metrics_qualification.yaml --qualify-metrics <private-directory>/admission.json --qualification-sha256 <reviewed-sha256> --modal-profile synaptic-labs --modal-environment synaptic-smoke-v1 --json
```

The effectful launcher remains CPython 3.11.14 with Modal SDK 1.5.4 and its
hash-pinned closure; a globally available newer discovery CLI is not a substitute.
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
observations, with source/configuration/profile and fixture digests. The delivery verdict
requires at least two distinct newly persisted authenticated numeric observations
returned by `RunsAPI.logs`, bracketed by pending observations of that exact call
before and after the read, and successful terminal native artifact verification. A final
training_metrics artifact or points first read after completion does not pass.
The finite private ledger binds the admission, run, exact provider call, receipt
time, cursors, closed read status, new record digests and pending/terminal states.
The JSON result reports `delivered_precompletion`, `inconclusive` or
`evidence_invalid`; CLI success requires delivered plus acknowledged durable
report publication. The ledger compares bounded retained JSONL bytes against its
accepted canonical records before a delivery verdict. Saved report bytes are
provisional (`durability_acknowledged: false`); only the returned digest-bound
result acknowledges successful report write/fsync with
`durability_acknowledged: true` and `report_sha256`. A saved report alone cannot
establish that acknowledgment after interruption or fsync failure.
`transport_qualified` and
`partial_write_behavior_qualified` always remain false. A delayed first read,
insufficient points or invalid local evidence cannot promote ordinary logs.
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
