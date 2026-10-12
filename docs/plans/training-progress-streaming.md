# Training progress chart feed

Status (2026-10-12): numeric child capture, signed remote progress storage and
authenticated host readback are implemented. Restricted paid attempts 03 and 04
completed independently accepted pre-completion delivery and native artifact
verification on exact source `01383fa81c8ed3847903095456e9b80350cd28a6`;
see [accepted evidence](#accepted-restricted-delivery-evidence-2026-10-12).
Modal's ordinary logs capability remains disabled. General transport and live
partial-write behavior remain unqualified. The earlier proposal and failed
attempt checkpoints below retain their historical scope.

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
reviewable. One separately approved attempt failed at local build-input admission
after provisioning; it never reached image build, CPU qualification or training.
The checkpoint below records the consumed claims and next preparation work.
The ordinary Modal registration advertises
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

The normal standalone execution automatically prepares the pinned bootstrap
wheel cache and exact current source wheel before claims or Modal effects.
`--plan` only reports local cache readiness; it does not download inputs.
For an optional provider-free preparation review, use the exact locked launcher:

```bash
python tuner.py train --job-config Trainers/recipes/qwen35_4b_modal_metrics_qualification.yaml --prepare-local --json
```

This may acquire public exact-pinned wheels and validate/build local inputs.
It is not a claim or execution grant, and temporary source-wheel staging is
removed on exit. Empty caches are handled automatically on the normal path;
corrupt/insecure inputs reject. See the canonical
[local preparation guidance](../../.skills/fine-tuning/reference/modal-jobs.md#standalone-local-preparation-and-cold-cache).

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
it is neither an all-in billing cap nor execution approval. The rate query itself submitted no job.
Refresh the scoped quote if required when reviewing each exact admission. After
provider-free checks and independent review, retain the admission digest, source,
selected scope, one integrated CPU/GPU attempt and budget. The user's subsequent
explicit bounded iteration authorization covers reviewed distinct attempts within
that scope; do not ask again for the same authority. Execution uses:

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

## Approved attempt checkpoint (2026-10-12)

Source `9c7392f2780077bab967b55adbec08c9eb6b107d`, attempt
`metrics-20261012-01`, used reviewed admission SHA-256
`a81af6d58b6c10cc37d9bd6689d6b036ed16b15dd604cf4782ca1c476f2730d0`. Initial local state-root mode 0775 was rejected before the
metrics claim; correcting it to 0700 did not consume or replay a cloud attempt.
The next invocation consumed the fixed metrics claim and build claim, provisioned
three Volumes, two Secrets and a build App, then stopped at the closed
`BUILD_INPUTS / INVALID` diagnostic at `runtime_build.prepare_build_inputs`.
No remote image build, CPU qualification or GPU training ran. No public metric
was delivered and neither transport qualification nor ordinary logs promotion
occurred. No delivery report acknowledgment was produced. A later read-only
lookup of the exact retained build-App name matched one App: stopped, zero
tasks, with a stop timestamp. No cleanup mutation was performed; this observation
is not deletion, reuse or adoption authority for its Volumes/Secrets. Preserve
the exact attempt's retained evidence and resources.

The failure cause was independently reproduced locally: the profile artifact
directory contained `profile.yaml` but none of its nine pinned bootstrap wheels.
The first required `grpclib` wheel read raised `FileNotFoundError` in the exact
build-input preparation path. The launcher environment's installed packages are
not those wheel files. This is a local dependency-preparation failure, not an
established image, trainer, GPU-capacity or metric-transport failure.

The retained build claim prevents repeating this submission. No cloud retry was
performed. The repair implements the reusable local workflow: prepare only the declared
bootstrap wheel inventory through hash-pinned acquisition, verify exact file
bytes and private-state-root modes provider-free before consuming claims or
provisioning, and retain source/runtime pins. The earlier derived-image helpers
validated/copied already-present artifacts; the repair now acquires the declared
exact inventory on the normal path. Review the integrated preparation surface
and provider-free barrier before a new clean source/admission. The later bounded iteration authorization below supersedes
the former requirement to request another approval for each reviewed fresh attempt. Do not
remove claims, invent another state root or rotate attempt identifiers to replay
this approved attempt, and do not delete or adopt its resources without authority.

## Preparation and second-attempt evidence (2026-10-12)

On source `840949aff3717859e84b3d63b0ad192e1430bd9f`, the optional local
preparation completed from an absent bootstrap cache and again from the warm
cache. Both returned `local_inputs_ready` with nine exact verified bootstrap
wheels, no provider effects and no execution authority. The retained journal
comparison showed unchanged attempt rows (two) and catalog rows (zero).
Private evidence is retained under `local-preparation-proof-20261012` in the
host training-state directory; these local checks do not prove live delivery.

The distinct `metrics-20261012-02` attempt used that source and admission
SHA-256 `654c4ec93995fb05d0691147770801f65b3f9a8beab6b620ff9bb6d0ad3d79c3`.
It passed image build and integrated CPU qualification, and captured 32 numeric
records plus one exhaustion record across nine growing pages in 69 polls.
Every growing page was bracketed by pending observations of the exact provider
call; all 33 record digests matched the retained progress JSONL. The run then
failed at `RUN_WORKER_SFT_ARTIFACT`. Terminal native artifact verification did
not pass. Its retained result is `inconclusive`, with
`terminal_artifacts_verified: false` and `durability_acknowledged: false`.
Visible numeric records alone do not make this a successful qualification run
or a `delivered_precompletion` verdict. Preserve its consumed claims, evidence
and resources; do not replay it. Ordinary logs and partial-write/transport
qualification remain disabled or unqualified. This attempt counts toward neither
of the two completed same-code runs required below.

A targeted provider-free reproduction through the real captured execution and
native verifier reproduced artifact rejection: the optional `metric_fd` was
included in the committed child transport and execution digest, but omitted when
the final verifier reconstructed that transport. The resulting digest mismatch
is consistent with attempt 02's closed artifact-stage failure; it does not
establish an introspected remote exception beyond that retained stage code.
The repair reads the bounded canonical saved child document, validates the
optional strict integer descriptor only for configured capture, reconstructs
and compares the complete canonical child bytes, then retains full execution
evidence equality. It preserves the fixed packaged-worker inventory and runtime
pins while refreshing the affected source hashes. Tests and independent review
passed before admission 03 was generated from the clean integrated source below;
that source subsequently completed both bounded live proofs.

## Accepted restricted delivery evidence (2026-10-12)

The repaired integrated source is
`01383fa81c8ed3847903095456e9b80350cd28a6`.
`metrics-20261012-03` used admission SHA-256
`24a1de6af9127a70e2f052badafd8226365766e34c48eca2ce443c00d211d8e0`.
Independent review accepted its first completed restricted proof: 71 polls,
seven growing pages and 32 numeric records plus one exhaustion record. Every
growing page was bracketed by pending observations of the exact provider call.
The five native artifacts and their exact hashes were independently verified.
The returned result acknowledged durable report publication and reported
`delivered_precompletion`; retained progress SHA-256 is
`7bd4be7ac9b2b4d340130568d3b8144377cbe3837684887e68ce4b43692093a3`
and report SHA-256 is
`8afb797725b4241610fa6c74c3820e5b7d133cd418d2c5750b240ff498f517a3`.
The saved report's provisional `durability_acknowledged: false` is expected;
the digest-bound returned acknowledgment establishes durable publication.

The fresh `metrics-20261012-04` admission SHA-256 is
`808ffa9f369567212c8caaa4417c220690b71f3cb16fb6f7f5e0cd3f0831542d`.
It binds the same unchanged source; only the reviewed attempt reference differs.
Attempt 04 completed 77 polls and nine growing pages with 32 numeric records
plus one exhaustion record, all growing reads bracketed by exact-call pending
observations. Its run is `modal-3234bf404adf0e4d95dee9f5` and exact provider call
is `fc-01M4RFNT6Q9P5STJ2ETR50KZ7N`. It returned acknowledged durable
`delivered_precompletion` with five verified native artifacts. Retained progress
SHA-256 is
`e34d9b75e7f28e75784c3a67c255edf5a801d872fff8d59ec4afea97902bf231`
and report SHA-256 is
`33fb155934ad6fddd35101b2336f1db87b7cc2ab7995a4e02d7636cd46a7a684`.
Independent audit accepted both completed attempts on the unchanged source:
the bounded two-run acceptance condition is met. This documentation update is
a later docs-only descendant; the live evidence binds the exact `01383fa8`
source above, not a retroactive qualification of another commit.
Ordinary logs remain disabled; transport and live partial-write qualification
remain false. These bounded results do not prove complete/fresh history, restart
recovery, cancellation, general production readiness or success on every call.

## Authorized reliability cycle (2026-10-12)

After the failed approved attempt, the user explicitly authorized repeated bounded
fix/test/live cycles until the supported path is reliable. That session authority
supersedes the earlier requirement to ask again for every fresh attempt. Preserve
the exact failed admission and claims; each new attempt must have its own reviewed
admission and finite source/input/provider/step/time/cost binding. No uncertain
submission is replayed. Ordinary logs remain disabled and earlier failed resources
are not deleted or adopted by this authorization.

The implemented repair automatically prepares hash-pinned bootstrap inputs on
the normal path, with private-root and actual local-input/source validation before
claims/provisioning. The optional `--prepare-local` surface uses the same service. A fresh clone must not depend on undocumented wheel files or
operator package installs. An empty cache normally acquires exact pinned artifacts; missing means inputs
still unresolved after preparation, such as an unavailable offline cache.
Verification must reject unresolved/corrupt inputs, unsafe
private-root modes and dirty/mismatched source before provider effects while
preserving exact launcher/runtime pins. Provider-free tests and the optional local
preparation report establish local behavior; the accepted live pair above
provides the bounded same-code delivery and artifact evidence.

Reliability acceptance requires at least two completed bounded live qualification
runs on the same final code, each with a distinct retained admission/claim and
reviewed fresh-attempt semantics, acknowledged durable pre-completion metric
delivery and native artifact verification. A subsequent code change requires a
new same-code pair. This demonstrates repeatability for the supported fixture;
it is not a guarantee against provider outages, external state changes, partial
write behavior, incomplete tails or every future configuration. Keep external
provider failures distinct from reproduced local defects, and avoid unrelated
feature work during the cycle.

## CPU transport qualification implementation

The next transport qualification uses the existing `train --qualify`/normal
captured-runtime self-check for a recipe with `observation_capture`, on the
supported bound-v2 runtime. No separate CLI or GPU training attempt is required.
The mandatory original CPU proof publishes before an optional synthetic probe
in the same fixed 1-CPU/512-MiB/120-second call. Primary marker/root or CPU
admission failure rejects. Finite optional probe failures preserve valid core
proof but leave logs disabled. Default absent-probe canonical bytes are retained.

The signed v2 dispatch binds a fresh claimed marker, exact release/deployment
and the explicit `modal_sdk_create_only/v1` production strategy. Complete
canonical bytes use the existing authenticated v1 Volume/client with
`batch_upload(force=False)` and mode 0600. There are no held stages, manifests,
setup commits or rename fallback. One intentional SDK collision must preserve
the original bytes. Accepted complete record pages require pending observations
of the same exact call and terminal-receipt absence before and after the page.
The signed terminal receipt is then written and committed on the control Volume;
after function completion the host verifies the exact nine-file inventory and
all hashes again. This measures pending delivery and post-commit preservation,
not rename atomicity, periodic commits, freshness or complete production history.
The CPU publishes only; host reads verify the preserved original and final files.
No redundant remote HTTP readback, network-policy relaxation or legacy SDK bypass
is part of this workflow.

Only sealed process-local reader proof matching the core receipt, release,
deployment, call, probe and durably retained capture can pass the composition
gate. JSON summaries/readback cannot rehydrate proof or restart authority.
The probe's eight synthetic numeric fixtures and one exhaustion fixture are
transport evidence, not a manufactured training outcome. Live CPU evidence and
independent review remain pending; the historical GPU pair above retains its
exact tested source and narrower qualification. See the canonical
[CPU workflow](../../.skills/fine-tuning/reference/modal-jobs.md#cpu-progress-transport-qualification)
for the actual command, finite bounds and limitations.

### CPU candidate attempts 01/02 (historical, not qualification)

Both used source `2f31010005d10da5934a1ac2ac1a34d20cd0d9d9`.
Attempt 01 did not select the authorized dotenv input and retained no provider
claim or journal rows; the normal global `--env-file` selection corrected that
operator boundary. Attempt 02 passed the mandatory CPU proof but its optional
rename candidate returned signed `PUBLISH_RENAME_INVALID`. Its durably retained
capture had 13 empty polls and no delivery; logs stayed disabled. EINVAL does
not establish a broader backend cause. Preserve both attempts and consumed state.
The new SDK strategy is an explicitly reviewed selection, not a failure fallback.

Attempt 01 stdout SHA-256:
`75e664c661c74faba04e3aa87cc66aca4ec60f1e5c5aa87616e7a5b39fd219be`.
Attempt 02 stdout SHA-256:
`7b2e9e768edd23761ee8e350b020266a24d77dcb67d4efc9749bf0dbc2427a03`;
capture SHA-256:
`c754fcd0550466f63cce6c6901a82faad66976519cd1510ba759b32ad278ca11`.
These CPU attempts do not alter the earlier GPU/native-artifact pair's exact
`01383fa81c8ed3847903095456e9b80350cd28a6` source qualification.

### CPU SDK candidate attempt 03 (historical, not qualification)

Attempt 03 used source `807071c985430613faebdf190b3b471dc658a74b`.
Its mandatory CPU proof passed, but the optional SDK probe returned signed
`SDK_PUBLISH_IDENTITY`. The retained capture contains eight polls, seven accepted
empty pages and zero records; logs remained disabled. Preserve its consumed
claims and resources without replay.

The closed live code locates an identity-check boundary; it does not identify
which field failed. A separate no-network reproduction against pinned Modal
1.5.4 established a representation defect: the public synchronized Volume has
no `_metadata` attribute, while `_get_metadata().version` supplies the valid
version. Hydration, object ID and exact client identity matched in that
reproduction. The reviewed fix corrects metadata access while retaining those
identity checks; it is not a relaxed admission or fallback strategy. This failed
attempt is not delivery or final-inventory qualification.

Attempt 03 stdout SHA-256:
`52f702a957af502f20ede4dac92df8b5210e6ff76e26d0b710081bba7d0d133e`;
capture SHA-256:
`cddbb067b32219077f5765d587a73dfe9bf442b02eab783bfdad287c82d76990`.

### CPU SDK candidate attempt 04 (historical, not qualification)

Attempt 04 used source `bfad73c99a72c1fe473078cc961b4a57a129f737`.
The core CPU proof passed; the optional probe returned signed
`SDK_PUBLISH_UNAVAILABLE`. Its retained capture contains eleven polls and one
record: the first complete publication was delivered, but this is not a complete
transport qualification. Logs stayed disabled; claims and resources remain intact.

A separate controlled-RPC reproduction against real Modal 1.5.4 verified normal
upload and `FileExistsError` collision mapping. Its `read_file` always uses
`VolumeGetFile2` plus HTTP download URLs, including for tiny files. Remote HTTP
readback is therefore a plausible attempt-04 boundary, not a cause established
by the closed live code. The reviewed simplification removes the redundant
CPU read after collision; host pending reads and post-completion exact inventory
and hashes retain preservation verification. Workload network restrictions and
permissions remain unchanged, without a legacy-RPC bypass.

Attempt 04 stdout SHA-256:
`0ece563047db637d25798a6e195dd35c6be2bc7330aabf5822507ef5fcb088cd`;
capture SHA-256:
`80883b010dc64d2e6fe917b3b8ddca6514c906af5f79521b6ec1c50e4ee4237c`.
