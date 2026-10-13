---
status: proposed
scope: OSS API v1 stable workflow release
baseline: 321a085f
updated: 2026-10-13
---

# Stable v1 release gate

This is the canonical preparation plan for the first broad stable OSS release.
The selected scope is public Python API v1 scenarios, environment creation and
execution, SynthChat generation/judging/improvement/export, dataset validation,
SFT, GRPO, evaluation/reports/comparison and artifacts, with durable host
composition, restart recovery, cancellation and reconciliation. SFT parity across
local Docker, Modal and Hugging Face Jobs is mandatory. GRPO includes one qualified provider first; Modal is the selected initial
candidate because its GRPO stack exists, not because that route is qualified.
Docker/HF GRPO parity is deferred. The baseline is main
`321a085f`; selecting this scope does not establish that its gates have passed.
No version bump, tag, package publication or provider execution is authorized by
this document. Live qualifications require reviewed finite scope and authority.

RunPod, DPO/KTO and other additional training methods, Syntunia integration,
browser transport,
managed multi-tenant serving and training-quality claims are deferred. Evaluation, scenarios, environments and SynthChat are included release families,
not optional scope merely because they have CLI entrypoints. Ingestion, chat and
pipeline compositions retain truthful backend status; no facade or CLI alone
establishes a stable installed operational contract.
The Python API is a library contract, not a deployed HTTP/MCP service.

## Current evidence and gaps

Historical claims remain bound to their tested sources. The
[October 7 readiness checkpoint](../review/api-v1-release-readiness.md) and
[frozen architecture matrix](../architecture/submodule-first-training-v1.md)
are not current release verdicts. The
[integration handoff](api-v1-integration-handoff.md) and
[streaming evidence](training-progress-streaming.md) describe later additions.

| Area | Evidence at this baseline | Stable gate still required |
| --- | --- | --- |
| Public contracts | Import-light typed API v1 and provider-neutral reference composition; provider-free tests. | Exact release contract inventory, compatibility/error policy and supported compositions. |
| Docker | Adapter/conformance implementation and provider-free tests are present. | Real SFT lifecycle, cancellation and restart/reconciliation evidence through the public API; fake tests do not satisfy this. |
| Modal | Historical GPU pair on `01383fa8` verifies restricted numeric delivery and native artifacts. CPU SDK pair on `9252e924` verifies bounded complete-byte transport and collision preservation. | Final-source stable lifecycle/recovery/cancellation closure. Logs remain conditional on exact matching per-call CPU proof, not globally enabled by historical evidence. |
| HF Jobs | Existing legacy CLI/protected operational lanes. | Public API v1 adapter and same stable lifecycle semantics; legacy job completion does not establish parity. |
| Host durability | Generic stores/authority ports and reference compositions; minimal consumer retains attempt claims. | Packaged usable durable host composition, migrations and complete restart recovery. In-memory reference stores and a local attempt journal do not provide this. |
| Packaging/platforms | Distribution version `1.1.0`, Python metadata `>=3.10`, selected wheel assets and Python 3.12 CI. The initial baseline omitted public schemas and workflow assets; later installed preparation evidence is recorded below, while the full script/asset inventory remains open. | Complete supported distribution inventory, clean installed consumer, demonstrated Python/platform matrix and release artifacts. Metadata is not qualification. |
| Optional families | Public facades exist; some operations require host-supplied backends. | Per-operation backend status, filtering and explicit supported/experimental/deferred boundaries. |

### Included workflow families: baseline versus release acceptance

| Family | Current evidence category | Required stable acceptance |
| --- | --- | --- |
| Scenarios | Checked-in CLI/config/schema workflows; not proof of a complete public authoring API. | Safe author/validate/version workflow with immutable scenario/rubric refs, installed templates/schemas and explicit public contracts. |
| Environments | Reviewed configured local DataAPI generation/improvement, immutable inputs, fixed worker, installed workflow and stalled-child cancellation. | Complete durable restart/reconciliation and supported backend qualification beyond the local fixture workflow. No security-sandbox claim. |
| SynthChat | Existing generate/judge/improve/export CLI and public data facade/backend composition. | Real installed public backend supports every advertised operation, usage, progress, cancellation and recovery; facade existence is insufficient. |
| Datasets | Existing validators and prepared training identities. | Bounded validation, provenance/splits/format contracts and immutable validated exports consumed unchanged by training. |
| SFT | Modal exact-source evidence; other provider/live parity gates remain open. | Docker/Modal/HF public API parity and qualified recipes, full lifecycle and native artifacts. |
| GRPO | Existing trainer/environment-backed CLI paths; no universal public/provider qualification claimed. | Explicit public backend/recipe/reward/environment contracts and bounded end-to-end proof; Modal selected first; live/public-backend qualification remains open. |
| Evaluation | Public facade and existing evaluator CLI/backends. | Installed execution, status, cancellation, metrics/judge reports and comparisons using immutable model/scenario/environment refs. |
| Artifacts | Public streaming/publication contracts with incomplete durable composition. | Durable verified host-local and HF model-repository publication, restart and ambiguous-outcome reconciliation. |

### End-to-end workflow acceptance

- [ ] Inventory actual exports and backends for scenario author/validate/version
  → environment config/create/validate/execution → SynthChat generate/judge/
  improve/export → dataset validation → SFT or GRPO → evaluation/reports/
  comparison → artifacts. Mark missing public operations explicitly; an existing
  legacy CLI cannot silently substitute for the promised API v1 contract.
- [ ] Keep authoring config-first: scenario, rubric, schema, tool shapes, reward
  and environment declarations are bounded reviewed data, not hardcoded dataset
  assumptions or arbitrary executable authority. Version references immutably.
- [ ] Inventory and install every advertised family's schemas, templates,
  configs, runtime assets and dependencies. Run its real backend from a clean
  installed consumer; no developer-checkout lookup or optional-facade shortcut.
- [ ] Verify environment tool allowlists, side-effect scope, deadlines, resource
  isolation and failure feedback into judge/improver inputs. Environment errors
  must not masquerade as model quality or be repaired by ad hoc runtime changes.
- [ ] Exercise SynthChat generation, judge and improvement, bounded export,
  actual usage accounting, progress/cancel and restart. Validate complete rows,
  lineage and intended source coverage before approving training inputs.
- [ ] Qualify SFT and GRPO separately; bind model/tokenizer/reward/environment
  revisions, prepared inputs, recipe/profile and finite hardware/time/cost.
  Preserve trainer-native artifacts and exact evaluation lineage.
- [ ] Execute evaluation and compare reports through the declared public backend;
  verify score/judge provenance, bounded response retention and explicit natural
  completion versus quality conclusions. Training completion is not evaluation.
- [ ] Demonstrate complete bounded end-to-end workflows on the exact final source
  and installed assets, including failure/cancellation/restart boundaries for
  every advertised family. Live-provider evidence requires separately reviewed
  scope; fake-provider tests do not establish real backend acceptance.

Exit: each included family has a truthful installed public contract, operational
backend and bounded evidence; both SFT and the selected GRPO route connect the
workflow without undocumented manual repair. Qualify the selected Modal GRPO route before claiming stable acceptance;
additional GRPO providers remain deferred.

All milestones below are **open**. Record terminal evidence against the exact
candidate source and artifact digest; no historical result automatically closes
a new gate. Retain failed attempts, consumed claims and uncertain submissions.

## Current preparation progress (2026-10-13)

PR #178 merged as `79f45ae2` with all required checks passing, packaging the
51 public schemas and defining these release gates. The next integrated preparation source, `268fc534`, adds
the reviewed evaluator/SynthChat/GRPO configuration inventory and two canonical
tool catalogs, explicit installed catalog resolution, and lightweight evaluator
startup. Missing or corrupt selected catalogs reject instead of becoming empty
defaults. Optional UI imports no longer install dependencies; disabling the live
dashboard preserves the final formatted summary.

Both checkout and source-archive-built wheels passed exact inventory/byte checks
(nine asset checks each). A fresh base installation in a neutral directory passed
real configuration/catalog loading, host manifest/capabilities and evaluator help
with process/network access prohibited inside help. The integrated focused suite
passed 76 tests with one Rich-dependent skip; a separate Rich-enabled run passed
five checks with only the optional ML identity check skipped. Both Modal source
inventories remain current. This is installed foundation evidence, not complete
workflow execution, provider parity or stable release acceptance.

The accepted fixed local worker at `27dde56e` reuses the existing environment
runtime interface. Its provider-free tests demonstrate real child fixture I/O,
cancellation/deadline interruption, confirmed termination and identity-safe
cleanup; the existing neutral installed-consumer CI step now exercises the fixed
`-I` child, read/write, default snapshot, exit and empty scratch. This qualifies
the primitive and its packaging; composed public evidence follows below.

Environment admission and immutable scenario/rubric/configuration/dataset
consumption are already reviewed. Optional host session-factory integration
connects those inputs to configured execution and per-candidate feedback. Runtime
`8de9e430` passed 916 combined tests with one intentional skip. Its fresh
sdist-built wheel passed nine asset checks and 51 session/public workflow tests
with the same intentional non-worker SIGSTOP skip, from a neutral directory under
isolated Python. Actual stalled-worker cancellation confirms exit and cleanup;
cold-host reads/cancellation and repeated starts preserve the existing claim.
Durable worker restoration, resumable rows and multi-host transition recovery
remain open. The selected direction keeps the
normal DataAPI lifecycle and host record/stream stores. The proposed additional
per-effect grant/episode DTO and parallel versioned lifecycle/result/list family
were cancelled after user review; they are not stable-release prerequisites.
Unknown active attempts must remain truthful and cannot authorize automatic
replay. This simplification does not waive cancellation or restart acceptance.

The worker has finite IPC and owned-child termination waits. Its `Popen`
construction and parent filesystem setup/cleanup are synchronous OS calls, so
these checks are not a universal wall-clock guarantee. Copy/move require temporary
copy capacity and are nontransactional. Process descriptors and uncertain
cleanup remain host-private, process-local custody; they establish neither a
security sandbox nor durable ownership reconstruction after restart.

The [operation support matrix](api-v1-support-matrix.md) records actual backend
and lifecycle gaps. The [public environment design](public-environment-execution.md)
defines contract/admission → local adapter/feedback → installed lifecycle
acceptance. Contract work does not enable execution before adapter review. No
version, tag or provider resource was changed by this preparation.

### Durable retained run readback preparation

The generic SQLite host store now commits workflow heads and event history in
one transaction, including training transitions. Modal composition accepts the
same caller-owned record/stream ports. Newly launched standalone runs retain
original host signing material, typed composition inputs and workflow state
privately before training starts.

The readback layer exposes `synaptic runs show|observe|verify|download` for an
exact retained run. It reconstructs the existing public read ports and authenticated
workflow; it does not prepare, stage or submit another job. Provider-free CLI
integration removes the original local recipe/configuration/dataset, reopens
stores and performs actual public verification/download against the same fake
provider. These initial CLI checks use fresh objects and reopened stores within
a process; the separate process-loss test below extends that evidence without
qualifying a live provider.
A fresh source-archive-built wheel passed the base neutral consumer, nine asset
checks and 81 installed retention/readback/storage/CLI cases. All nine changed
production modules matched the tested wheel byte-for-byte; the pinned launcher
imports and runtime inventory checks passed. The local proof uses copied package
files, matching CI's pip installation, because shared hard links are rejected by
the existing runtime source checks.

The subsequent known-call reader wheel passed 87 installed cases, the neutral
base consumer and nine asset checks. All five changed production modules matched
the wheel bytes. Seven pinned Modal SDK tests include actual synchronized call
result decoding: an exact builtin polling timeout preserves pending state, while
remote execution timeout and a malformed successful `None` result are refused.
These checks use controlled RPC responses and perform no provider actions.

The reader checks original plan/material/resource/cost bindings and opens only
existing databases. Known calls in `QUEUED`, `RUNNING` or `CANCEL_REQUESTED` can
be observed once through the original call binding. A pending observation
preserves the stored phase and revision; it does not claim current GPU activity.
Unknown or failed provider reads leave that checkpoint unchanged. Signed
completion advances through the existing workflow reducer. Terminal
`SUCCEEDED_UNVERIFIED`, `VERIFIED` and `VERIFICATION_FAILED` reads remain
supported. Already verified runs support show/download;
re-verification, logs, cancellation, reconciliation and in-flight continuation
remain unavailable. Historical runs without retained signing keys cannot be
recovered. See the [operational commands](../../.skills/fine-tuning/reference/modal-jobs.md#read-a-retained-run).

### Known-call process-loss qualification

A POSIX provider-free test kills the submitting host with `SIGKILL` after its
exact call has been retained, while the original connections and runner stack
are still alive. A fresh isolated process makes the first database open after
the kill and observes pending without changing the stored records or history.
After the scripted provider completes, another fresh process authenticates the
original-key HMAC completion, verifies and downloads all five artifact hashes.
A final process reads the persisted verified outcome without polling or reading
provider artifacts. Original source/preparation files are removed, and database
recovery files remain intact throughout the test.

The installed retention suite passed all 88 cases. After integrating the current
main branch, the fresh source-archive-built wheel also passed the restart proof
and evaluator comparison regressions (38 cases). Production imports were checked
against the isolated installation; the subprocess proof uses that same wheel.

This establishes host reconstruction at the retained known-call boundary against
an explicit disk-backed SDK fixture. It does not establish live Modal behavior,
uncertain-submission recovery, cancellation, cross-machine transfer or recovery
at every possible crash boundary.

### Installed base compatibility evidence

The separate base-package matrix on PR #191 (`6f010cac`) passed on Linux with
Python 3.10, 3.11, 3.12, 3.13 and 3.14, and on Windows/macOS with Python 3.12.
Each job builds an sdist and then a wheel, installs only base dependencies, and
checks exports, resource bytes, real configuration/catalog loaders, help and the
console/project-manifest path from outside the checkout. This qualifies those
base-package paths; provider launchers, POSIX stores, environment workers and GPU
execution retain their separate platform and runtime requirements.

### Ordinary HF Jobs SFT preparation

The explicit HF host assembly now uses the same public training/run APIs and
caller-owned records, grants and signing keys. It binds a named runtime profile,
exact source/input bytes, declared resources, account/namespace, existing bucket
and fresh prefixes before one-shot staging and submission. Partial uploads and
unknown submission replies remain consumed. The protected HF smoke and legacy
CLI lanes are unchanged.

The candidate uses isolated hash-pinned Hub 1.33 host dependencies and the
unchanged Qwen runtime profile's Hub 1.32 model helper. Public-model preparation
is anonymous; the explicitly named job secret stays in the privileged parent
and the trainer receives its reset environment. Parent source is a separate
fixed 16-member inventory, maintained by `scripts/regenerate_hf_parent_closure.py`;
the offline trainer retains its existing 68-member closure.

A fresh source-archive-built wheel passed the neutral base consumer, 30 installed
public lifecycle/bootstrap/native verification/fault/cold-store cases with the
real Hub 1.33 SDK, and two controlled HTTP model-preparation cases with Hub 1.32.
These include copied-parent imports in a fresh isolated process, actual bootstrap
and worker filesystem operations with injected ML effects, multi-MiB artifact
streaming, and refusal to replay uncertain effects. Both installed dependency
lanes assert their exact SDK version before collecting tests.

Initial native acceptance is limited to prepared `syntunia-sft-row/v1` inputs up
to 2 MiB, exactly five final native artifact roles and no checkpoint retention.
Logs, cancellation and reconciliation are unavailable. Cold reads retain the
original admitted material and keys; they do not restore an uncertain remote
submission. Shared-bucket readback is not an atomic-ownership or remote-fsync
guarantee. CPU tests on Python 3.12.15 do not qualify the candidate image's locked
Python 3.12.3, GPU execution or live HF behavior. Those release gates remain open.

## 1. Freeze the public support contract

- [ ] Publish the provider/method/operation matrix for plan/preflight/start,
  observe, outcomes, logs, artifacts/verification, cancel and reconcile.
  Distinguish contract-only, implemented/provider-free, live-qualified and
  unavailable states. Unsupported operations must be filtered before execution.
- [ ] Define stable error categories, uncertain outcomes, identity/provenance,
  pagination, bounds, compatibility and deprecation policy. Provider parity means
  consistent public semantics, not identical hardware or provider internals.
- [ ] Separate base library Python/platform support from exact provider launcher,
  image and trainer pins. The measured Modal launcher is POSIX CPython 3.11.14
  with Modal 1.5.4; Python 3.12 contract CI does not prove Python 3.10 or arbitrary
  launcher compatibility. Do not relax image pins to accommodate the operator.
- [ ] Select and qualify explicit stable SFT and selected GRPO recipes,
  model/tokenizer revisions,
  runtime profiles, native artifact contracts and hardware/platform combinations
  for each provider. Model/hardware success is not universal recipe support.
- [ ] Rewrite README quickstart and install guidance to the supported release
  contract; label legacy workflows and unqualified examples explicitly.

Exit: reviewed support matrix and executable examples agree with actual exports,
capabilities, package metadata and each advertised provider.

## 2. Complete safe packaging and a clean consumer

- [ ] Inventory required schemas, profiles, locks, scripts, trainer assets and
  templates for every supported installed operation. Resolve source-tree-only
  dependencies rather than silently discovering a developer checkout.
- [ ] Verify wheel and source distribution in neutral working directories and a
  fresh consumer, with no checkout imports, ambient credentials or undeclared
  runtime downloads. Keep lightweight contract imports free of SDK/ML effects.
- [ ] Document exact install extras, bootstrap preparation, offline/runtime
  boundaries and host configuration. Secret material never enters packages,
  recipes, runtime image environment or the isolated trainer.

### Baseline blockers and preparation status

- The baseline wheel inventory omitted Evaluator's 16 configuration YAML files and one
  recipe, SynthChat's seven configuration YAML files, 19 rubrics and 11 scenarios,
  12 GRPO configuration YAML files, four flywheel YAML files and Tools JSON.
  Explicitly review required members rather than packaging all repository files.
- `EnvironmentValidator()` without installed configs silently obtains an empty
  tool schema and `default_action=simulate`. A successful simulated fallback is
  not faithful configured execution. Installed execution must require the
  admitted assets and fail closed when absent; preserve intentional simulation
  only as an explicit configured mode.
- Evaluation CLI help on a minimal install fails through eager retrieval →
  `shared.ml` imports of undeclared pandas. Resolve import boundaries and declared
  optional dependencies; do not prescribe an indiscriminate ML-stack installation
  to make a lightweight help/contract path work.

- The initial reference data composition supplied `environment_validator=None`.
  Reviewed admission, retained input snapshots and parser/catalog injection now
  supply the input seams. Optional `ReferenceDataPortsV1.environment_sessions`
  and the host-configured `CapturedEnvironmentSessionFactory` are the current
  integrated path, using the fixed worker through `runtime_factory(limits,
  boundary)`. The installed local workflow now demonstrates generation/improvement,
  failure feedback, cancellation and cleanup with deterministic injected clients.
  Paid model/backend acceptance and complete durable recovery remain open.
- The paid evaluation judge path rejects with `backend_unmetered`. Select and
  qualify the intended judge backend, usage/accounting contract and bounded
  evaluation benchmark before a stable judging claim. Do not bypass metering or
  treat a facade/mock response as paid backend qualification.

The reviewed package inventory, catalog resolution and lightweight help fixes
above address those specific preparation defects. Complete installed execution,
broader environment/backend qualification, complete public lifecycle recovery, paid judging and
the broader asset inventory remain open release gates.

Exit: clean installed consumer can perform provider-free planning and compose
all supported providers using only declared assets/dependencies.

### Bounded public dataset validation

The reference validator bounds each physical line at 4 MiB and each invocation
at 64 MiB and one million physical lines. The line limit includes terminator
bytes; invocation limits include rescanned cursor prefixes,
blank lines and lookahead. Exceeding a read/work bound refuses the page with
`manifest_invalid`; oversized or malformed cursors refuse with `cursor_invalid`.
Existing row-count pagination and LF/CRLF/bare-CR/UTF-8 replacement behavior are
preserved. These finite read limits do not qualify OS-call timing, immutable
exports, provenance, split correctness or a complete dataset-to-training handoff.

The focused, reference and public Data API suites passed 97 cases. A fresh
source-archive-built wheel passed the neutral base consumer and 145 installed
Data/environment/durable-store cases, with one intentional non-worker SIGSTOP
skip. Installed checks inspect the loaded package files and include the full
reference Data suite and byte-bound controls; no production source is copied
into their test directory.

## 3. Ship durable host composition and migrations

- [x] Provide an opt-in host-owned SQLite implementation of the existing
  record/stream ports, with transactional schema initialization and atomic
  Data/Evaluation/Pipeline/training-workflow head/history updates. Cold-open and killed-process
  regressions cover storage commit boundaries; see
  [reference host storage](../architecture/reference-host-storage.md).
- [x] Reopen SQLite and reconstruct a reference training host against the same
  scripted fake provider, preserving plans, grants, effects and authenticated
  history without duplicate submit/cancel effects. This verifies host storage;
  live provider restart and the separate plan-index crash boundary remain open.
- [x] Let the packaged Modal composer accept paired host-owned stores. Verify
  reopened workflow and call catalogs with fresh host objects against a mocked
  provider, retaining the original inputs and keys. Process-local source
  payloads and live restart/continuation remain open.
- [x] Persist new standalone Modal workflow records and original private signing
  inputs before training starts. Partial retention prevents stage/submit;
  typed reconstruction and known-call standalone readback are implemented. Live
  recovery and uncertain-submission reconciliation remain open.
- [x] Kill a standalone host after it retains an exact Modal call, then use fresh
  processes to observe pending, verify/download signed fixture artifacts and
  read the persisted verified outcome. The provider remains a scripted SDK
  fixture; live recovery and other interruption boundaries remain open.
- [ ] Compose the existing generic persistence/authority ports into a supported
  durable host boundary. The host owns credentials, configuration, grants,
  coordinator/Foundation records, prepared-input catalogs and product state.
  Do not introduce a provider-owned lifecycle database.
- [ ] Persist plans, grants, one-shot attempt claims, exact source/runtime/input
  identities, provider call bindings, outcomes and artifact verification state.
  Define transaction ownership, concurrency/fencing and schema versions.
- [ ] Add reviewed migrations, rollback/forward-compatibility behavior and
  interrupted-migration recovery. Document deployment and backup/restore limits.
- [ ] Provide a clean-consumer composition example that uses the durable path;
  in-memory examples remain explicitly instructional.

Exit: restart reconstructs authorized read/continuation state without minting a
new submission grant or renewing a deadline.

## 4. Prove crash, uncertainty, cancellation and reconciliation

Docker now has a concrete bounded CLI transport for present-image checks,
create/start and scoped lookup. It binds the explicit local daemon identity,
full existing labels, deterministic container name and immutable image content
ID. Entrypoint, environment values and mount mappings are supplied by the host;
create never pulls an image or adopts an existing container. Selected mount and
ancestor identities are rechecked immediately before create, but this is not
atomic custody of the daemon's path resolution.

Scripted process-reply tests exercise these ports and preserve uncertain
per-session attempts without replay. This transport does not yet supply durable
host composition, source sealing, artifact streaming, cancellation or container
lifetime supervision. No live daemon or SFT parity is qualified by these tests.

The source-archive-built wheel passed the neutral base consumer and 187 installed
transport/model cases. The installed transport matched the reviewed source bytes;
runtime inventories and pins remain unchanged.

The host-local artifact port now supplies the existing Docker reader with signed
inventories and bounded chunk/EOF streams for explicitly configured artifact
roles. It checks the exact terminal container and retained directory/file
identities, enforces profile byte limits and a cooperative deadline, and verifies
the final byte count and hash. Tests pass through the authenticated coordinator
reader, including multi-megabyte output, changed files and invalid EOF signatures.
This does not freeze the host filesystem or make daemon and file observations
atomic. Source sealing, durable host composition, cancellation and live SFT
qualification remain separate.

All 44 focused artifact cases passed from a fresh source-archive-built wheel,
including the authenticated reader chains. The neutral base consumer passed,
and the installed artifact and transport modules matched the reviewed source.

The Docker cancellation port now authenticates the exact container, labels and
image, makes one bounded stop request, and retains its signed receipt in the
caller's existing durable record store. An uncertain response or receipt write
keeps the consumed claim; reopening storage cannot replay the stop. Historical
receipts remain readable after the CLI deadline, which bounds provider commands
rather than host-owned synchronous signing/storage callbacks.

A receipt records the successful stop request and observed nonrunning state.
Natural completion can race the request, so the public reader continues to
report the observed success/failure outcome; it does not infer `cancelled`.
Scripted tests include actual SQLite reopen, concurrent requests, failed writes,
lost responses, changed provider identity and the existing executor/reader path.
Full host composition and live provider qualification remain open.

The combined artifact/cancellation suite passed all 114 cases from a fresh
source-archive-built wheel. The neutral base consumer and byte comparison of all
three Docker transport modules passed; runtime inventories and pins are unchanged.

Docker source staging now publishes the existing factory-built worker bundle,
locked source members and one retained prepared-input stream into fresh private
paths. It verifies input size/hash/EOF before publication, retains directory and
file descriptors, and rejects changed bytes, permissions or path identities at
submit and immediately before container creation. Source identity remains the
original execution-source fingerprint; the staged inventory has separate evidence.

The source mount is read-only to the container; prepared input in writable state
is checked separately. Partial publication and uncertain results remain consumed,
with no adoption or automatic retry. This primitive provides process-local
source custody. Model preparation and live Docker SFT qualification remain open;
the composed host evidence below extends the persistence coverage. Legacy
compositions without the explicit checks gain no source guarantee.

All 162 installed source/artifact/cancellation cases passed from a fresh
source-archive-built wheel, including the signed executor path and changes made
after image inspection. The neutral base consumer passed and five production
modules matched source bytes. Runtime inventories and pins are unchanged.

A separate worker cleanup regression now opens an independent directory cursor
for every bounded scan. A child killed during enumeration could leave the
inherited directory description at EOF, causing parent cleanup to miss owned
files. Deterministic controls failed against the previous implementation and
pass with independent cursors; timeout and uncertainty behavior are unchanged.
The related worker/runtime suite passed 74 cases. A fresh source-archive-built
wheel passed the neutral base consumer and 219 installed environment, Data,
storage and worker cases with one intentional non-worker SIGSTOP skip. Five
production files, including the unchanged Docker host files, matched that wheel
byte-for-byte. The triggering CI trace suppressed the underlying cleanup error;
the controlled reproduction establishes the defect rather than retrospectively
proving that trace's cause.

The explicit Docker host assembly composes those existing ports with the public
Training API, caller-owned durable stores and named original signing keys. Source
publication occurs inside the claimed stage; both submit checks remain active.
Its public plan commits the complete original material, including runtime and
artifact-policy fields that the Docker profile does not otherwise represent.
Staging, container mounts and artifact reads must select the same host paths.

Scripted Docker CLI tests use real SQLite close/reopen and the public lifecycle:
native verification and direct artifact downloads, cancellation after the start
grant expires, natural completion races and refusal to replay uncertain submits.
Verification requires successful exact-container exit evidence and the existing
native SFT workload, lineage, model and tokenizer checks. Matching file hashes
alone cannot produce a verified run.

A fresh source-archive-built wheel passed the neutral base consumer and all
50 installed host cases, including changed cold-host keys and container identity.
All four changed production modules matched the wheel bytes. The affected source
suites passed 180 cases, followed by six additional cold-host negative controls;
independent review also ran the original 44 host cases. Runtime inventories and
pins remain unchanged.

Cold callers must retain the original rich material, profile and named keys;
the assembly neither discovers these nor resumes an incomplete stage. It keeps
existing capability flags, with logs unavailable. These tests use synthetic
trainer outputs and scripted CLI replies. Image/model-cache provisioning, real
Docker/GPU execution, actual process-loss recovery and live cancellation remain
separate qualification gates.

The packaged Modal cancellation machinery now supports an explicit host opt-in
for a current cancellation grant after the original start grant expires. It
authenticates the recorded cancellation intent, original successful submit,
retained binding and exact call before consuming the existing attempt claim.
The original commitment and staging/submission expiry remain unchanged.

Scripted public-API tests cover repeated requests, lost replies, a late
acknowledgment after a bounded wait, reopened workflow/journal state and a race
with successful completion. Actual Modal 1.5.4 SDK controls verify hydration,
client/call identity and cancellation acknowledgment without network requests.
An acknowledgment produces `cancel_requested`, not proof of stopped compute;
uncertain results retain the consumed claim and never repeat the provider call.
Ordinary Modal cancel/reconcile capabilities remain disabled, including the
retained reader and CLI. Live terminal evidence and full restart composition
are separate acceptance gates.

The fresh source-archive-built wheel passed the neutral base consumer and all
102 installed retention/readback/cancellation tests. Its five changed production
modules matched the reviewed source bytes. Both actual pinned-SDK controls also
passed; the packaged runtime inventories and pins remain unchanged.

- [ ] Fault-inject before/after claim, provider submission, call-ID retention,
  terminal observation and artifact acknowledgment; kill/restart the host at
  each boundary and verify exact source/run ownership.
- [ ] A timeout or missing catalog row never implies no provider call. Preserve
  uncertain claims; reconcile exact provider identity without automatic replay,
  resource adoption or cleanup. Bound every provider operation and its attempts.
- [ ] Specify cancel requested/accepted/terminal states and races with completion,
  repeated requests and restart. A cancellation request is not proof of stopped
  compute or refundable cost. Verify owned-provider terminal state independently.
- [ ] Reconciliation handles missing, stale, conflicting and unavailable evidence
  with finite closed diagnostics; it cannot overwrite contradictory verified
  state or infer successful training from missing progress.

Exit: the same fault/restart/cancel suite passes against all three providers,
with independent review of submission and cleanup authority.

## 5. Make final artifact publication durable

The reference publication composition currently uses in-memory state and an
empty destination registry. It does not establish durable host-local or Hugging
Face model-repository publication. HF bucket progress is a different operation
and cannot close this gate.

- [ ] Support final verified native artifact streaming to host-local storage and
  HF model repositories through the public artifact/publication boundary.
  Persist exact source/run/artifact hashes, destination identity, authority and
  publication attempt before writing; do not infer authority from a path or run.
- [ ] Persist claimed, transferring, committed, verified and ambiguous outcomes
  with versioned transitions, concurrency fencing and bounded deadlines. A
  remote commit and its durable local receipt are distinct facts.
- [ ] Bound source streams, spool files, per-file and aggregate bytes, staging
  lifetime and readback. Verify exact destination inventory, sizes and hashes
  before reporting completion; partial spools do not become final artifacts.
- [ ] Use destination-appropriate no-replace/CAS publication. For HF model
  repositories, bind the expected parent revision and retain the exact returned
  immutable commit; verify readback at that pinned commit rather than mutable
  HEAD. Conflicting concurrent writers must reject rather than overwrite.
- [ ] Recover external success followed by local receipt loss through bounded
  exact-identity readback/reconciliation. Unknown completion remains ambiguous;
  restart cannot blindly repeat destination writes or create another commit.
- [ ] Test crashes during transfer, before/after destination commit and before
  durable receipt/verification, plus competing host processes and destination
  conflicts. Reconcile only the exact authorized publication, preserving failed
  attempts and inconsistent evidence without adoption or cleanup authority.

Exit: both host-local and HF model-repository publication survive restart and
concurrency without duplicate destination writes/commits, and expose verified
or explicitly ambiguous results through the supported durable host composition.

### Progress retention and reconnect are a separate gate

- [ ] Define signed/verified bounded progress, cursor and duplicate/conflict
  semantics, durable retention acknowledgment and restart reconstruction.
- [ ] Qualify local progress publication and HF progress storage through their
  real public readers, including delayed visibility, transient reads,
  malformed/partial content, truncation, conflicting duplicates and interrupted
  local fsync. This is not final model publication.
- [ ] Reuse the Modal SDK create-only progress strategy and exact per-call proof
  gate where applicable. Historical CPU fixtures do not qualify GPU metrics on
  a new source, nor prove freshness/completeness or atomic server transactions.
  Preserve the finite per-call read/page/ledger/deadline limits.
- [ ] Keep progress best-effort unless a stronger guarantee is separately proved;
  verified terminal outcomes/artifacts determine completion, never missing tails.

Exit: documented progress durability and reconnect behavior is demonstrated,
not inferred from process-local cursors or retained JSONL alone. Neither this
exit nor bucket progress substitutes for durable final artifact publication.

## 6. Close provider parity in order

1. **Docker conformance:** run the complete public SFT lifecycle and recovery
   ladder against a supported real Docker/GPU environment. This server has no
   usable local GPU; use an authorized qualified execution machine. Verify
   native artifacts, isolated credentials, filesystem custody and owned cleanup.
2. **Modal closure:** apply the same ladder to the final candidate, preserving
   exact launcher/runtime locks, per-call log proof and one-shot claims. Close
   cancellation/reconciliation and restart gaps without weakening admission.
3. **HF Jobs API v1 parity:** implement the provider-neutral adapter using the
   existing SDK/storage mechanisms, then run the same ladder. Preserve protected
   anchor/secret/source semantics; do not merely wrap legacy CLI stdout as proof.

Exit: one shared conformance matrix has exact provider-specific terminal evidence
for every advertised stable operation. Required live cost/time/hardware bounds
are reviewed before execution; failed runs remain evidence, not replay authority.

## 7. Audit included families and optional backend status

- [ ] Inventory every exported family and optional backend. Separate included
  stable workflow
  integration needs from experimental reference implementations and deferred
  features; absence of a backend yields explicit unavailability.
- [ ] Pipelines, evaluation and chat cannot infer authority from a completed
  training run. Data and ingestion retain their own input/custody contracts.
- [ ] Examples and capability negotiation omit unsupported cancel/recovery,
  inference, model conversion or publication operations rather than advertising
  them and failing later. DPO/KTO and additional methods plus RunPod stay outside
  this tag. GRPO is included on one qualified provider first (Modal selected
  candidate); Docker/HF GRPO breadth is deferred.

Exit: the public package and docs have no ambiguous universal-provider promise.

## 8. Review and publish the exact release candidate

- [ ] Freeze candidate source, version, dependency inventories and built artifact
  hashes. Run complete applicable CI, installed wheel/sdist tests, clean-consumer
  tests and supported OS/Python combinations; record skips and limits explicitly.
- [ ] Verify training/inference/bootstrap closure hashes and skill-tree sync.
  Hash agreement is not independent approval of runtime pins or live images.
- [ ] Audit release contents and repository history for secrets/private data;
  verify licenses and provenance. No model weights, credentials or private live
  evidence are silently included in distribution artifacts.
- [ ] Resolve independent correctness/security/release review and documentation
  findings. Current README, support matrix, examples and release notes must agree.
- [ ] Define reviewed version/tag/build/signing-or-attestation/publication steps,
  artifact custody and withdrawal procedure. Existing CI wheel checks are not
  package publication; current version `1.1.0` does not establish a stable release.
- [ ] Publish only after the authorized final release gate, with immutable source
  and artifact identities and honest provider/platform limitations.

Stable acceptance requires all selected gates above to be closed. If a provider
or lifecycle gate cannot be met, change the proposed release scope explicitly;
do not silently relabel a narrower preview as this broad stable release.

### Immutable text-conversation preparation

An explicitly registered normalizer now prepares ordered text conversations
through the existing training-input ports. Declarative field/role mapping, an
explicit assistant target and group-disjoint train/validation selection preserve
the selected prefix and target without invented ingestion lineage. The private
publication commits source, configuration, optional Data outcome and exact row
bytes; independent verification and retained source reads reject changed
identities. Historical provenance remains the preparer's committed assertion.

A fresh source-archive-built wheel passed the neutral base consumer and all 49
installed preparation/source tests. Three production modules and three packaged
schemas matched the reviewed source bytes; the permanent CI lane resolves schema
resources from that installation. Existing v1/v2 publisher behavior is preserved.

The new `synaptic-conversation-sft-row/v1` format now has CPU-tested SFT
compiler, loader, preprocessing and native verification support. Existing
packaged runtime release allowlists remain unchanged and reject this format;
an additive candidate selection and installed Data-to-training planning join
are recorded below. Candidate image capture, execution and provider
qualification remain separate follow-up gates.

### GRPO trainer core preparation

The legacy GRPO CLI and a new retained-input entry share configuration building,
trainer construction and training calls. The new entry consumes explicit input
bytes, reward documents and a local model, checks their declared commitments
before loading ML dependencies, and requires adapter and tokenizer saves.
Metrics report actual trainer observations; step counts do not prove a parameter
update or useful reward variation. Existing CLI behavior remains available.

A fresh source-archive-built wheel passed the base-only neutral consumer and
40 installed core/configuration tests. One additional retained-input test with
the real Dataset library initially hit the local harness's overly long Unix
socket temporary path, then passed with a short private scratch path and the
same wheel. All five changed production files matched reviewed source bytes.
The installed CI lane uses Dataset 5.1.0 and pytest 8; trainer/model effects remain
explicit CPU substitutes. This is not validation of the GPU image's ML stack.

The public GRPO compiler, separate offline inventory, provider dispatch, native
verification and live candidate qualification remain open. This entry alone
does not authenticate a public plan, verify model member custody or advertise
GRPO through the public API.

### Conversation SFT compilation and execution preparation

The SFT path preserves the selected ordered prefix and assistant target, uses
preassigned group splits, and masks all prefix tokens. It rejects truncation,
missing target tokens and mismatched row formats. Compiled chat-template options
now pass the same bounded validation at the offline worker boundary. Existing
legacy and v1/v2 behavior is retained; release format allowlists are unchanged.

A fresh source-archive-built wheel passed the neutral base-only consumer and
47 installed tests, including six real Dataset cases and the real isolated
decoder/worker fixture. The latter uses a copied Python interpreter so resolving
its executable retains the test installation's dependencies. All ten changed
production files and five lock/closure documents matched the wheel bytes.
Runtime lock refreshes change source commitments only: inventories, image/SDK/
Python pins and release admissions remain unchanged. This is CPU execution and
packaging evidence, not a provider run or GPU training qualification.

### Public retained evaluation comparison

The public evaluation facade compares two existing result requests through an
optional read-only reference callback. It verifies retained report identity,
size and digest before parsing; reopened SQLite records and host artifacts work
without resolving a backend or rereading deleted scenario configuration.
Reports capture the prompt, validator, scoring and generation controls actually
consumed. The selected model or backend may vary explicitly; missing controls,
external judge/environment inputs, partial results and ambiguous identities do
not silently become comparable. Error and inconclusive cases remain distinct
from model pass/fail changes. Configuration provenance does not attest a remote
model or service implementation.

A fresh source-archive-built wheel passed the neutral base-only consumer and
209 installed tests. All six changed production modules matched reviewed source
bytes. Tests include cold retained-report reads, public callback isolation,
bounded strict payload validation and existing Markdown/console rendering.
The permanent CI lane copies only tests and installed non-Python assets into a
neutral directory. Comparison limits reports to 64 MiB, comparison output to 4 MiB and
10,000 cases, retaining full counts when displayed IDs are capped. These limits
do not themselves bound upstream HTTP acquisition. The local reference backend
now has the separate transport bounds below; in-flight recovery and live
backend/judge qualification remain open.

### Local evaluation HTTP bounds

The local HTTP reference backend now defaults to finite 1 MiB request and
decompressed response body limits. Hosts can configure exact positive integer
limits through 64 MiB; invalid values reject before client creation or secret
resolution. The existing transport rejects oversized encoded requests before
HTTP and closes streamed responses and sessions at the response boundary.
An omitted token ceiling remains independent of these byte limits. Legacy
client-factory calls keep their previous constructor arguments.

A fresh source-archive-built wheel passed the base-only consumer and all 226
installed evaluation/comparison/transport cases. The 17 new cases include exact
UTF-8 byte boundaries, split codepoints, oversize rejection, resource closure
and complete text in verified public report readback. Both changed production
files matched the wheel. These limits do not provide an aggregate wall-clock
deadline, cancellation or recovery; HTTP timeouts bound inactivity. No network
provider or model-quality qualification was performed.

### Installed Data-to-conversation-SFT planning handoff

Two installed joined cases now exercise actual public Data generation, judging,
improvement and bounded validation, real configured filesystem assertions,
conversation normalization and immutable preparation, then public SFT resolve
and plan. Both explicit Data outcome binding and optional generic provenance
are covered. A generated outcome cannot replace the selected improved outcome;
a historical v2-only release cannot admit the new conversation format. Prepared
bytes, identity and plan remain unchanged after the source datasets, metadata,
scenario and configuration are deleted, without another generation call.

The separate unqualified conversation runtime profile and compatibility assets
match the checkout-built wheel, source-archive-built wheel and installed files.
The host-owned build profile binds the previously accepted Stage2 wheel and is
excluded from both wheels to avoid a circular hash. A fresh base-only consumer
also passed. Synthetic provider/release and Git observations are explicit CPU
planning fixtures; they are not source admission, image capture or deployment
evidence. Existing profiles remain unchanged. Candidate image qualification
and training remain open, and the ordinary Modal recipe CLI retains its
existing v2 path. This is a Python API planning handoff.
