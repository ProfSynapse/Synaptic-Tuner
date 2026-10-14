---
status: release-preparation
scope: OSS API v1 stable workflow release
baseline: 321a085f
updated: 2026-10-14
---

# Stable v1 release gate

This is the canonical preparation plan for the first broad stable OSS release.
The selected scope is public Python API v1 scenarios, environment creation and
execution, SynthChat generation/judging/improvement/export, dataset validation,
SFT, GRPO, evaluation/reports/comparison and artifacts, with durable host
composition, restart recovery, cancellation and reconciliation. SFT parity across
local Docker, Modal and Hugging Face Jobs is mandatory. GRPO includes one qualified provider first; Modal is the selected initial
candidate because its GRPO stack exists, not because that route is qualified.
Docker/HF GRPO parity is deferred. `321a085f` is the historical planning
baseline, not current main or the accepted preparation stack. Selecting this
scope does not establish that its gates have passed.
No version bump, tag, package publication or provider execution is authorized by
this document. Live qualifications require reviewed finite scope and authority.

RunPod, DPO/KTO and other additional training methods, Syntunia integration,
browser transport,
managed multi-tenant serving and training-quality claims are deferred. Evaluation, scenarios, environments and SynthChat are included release families,
not optional scope merely because they have CLI entrypoints. Ingestion, chat and
pipeline compositions retain truthful backend status; no facade or CLI alone
establishes a stable installed operational contract.
The Python API is a library contract, not a deployed HTTP/MCP service.

## Current release status — 2026-10-14

Version **1.1.0** is the intended stable package version. Its classifier records
that intended channel, not completed acceptance. The first stable GitHub release
will bind exact source, wheel, sdist, checksums and qualification scope after the
gates below pass; no PyPI publication is promised. The dated evidence at this
checkpoint remains scoped, with final live provider/lifecycle gates open. Main is `1603b60b` through merged [#249](https://github.com/ProfSynapse/Synaptic-Tuner/pull/249).
HF publication [#238](https://github.com/ProfSynapse/Synaptic-Tuner/pull/238),
launcher closure [#239](https://github.com/ProfSynapse/Synaptic-Tuner/pull/239),
cold HF cancellation [#240](https://github.com/ProfSynapse/Synaptic-Tuner/pull/240),
native GRPO publication [#241](https://github.com/ProfSynapse/Synaptic-Tuner/pull/241),
HF material preparation [#242](https://github.com/ProfSynapse/Synaptic-Tuner/pull/242),
Modal capture diagnostics #243, configured Modal lifecycle #244 and finite
CPU-parent diagnostics #245, orphaned-cancel observation #246, same-model consumer
#247, its token-default repair #248 and source-root fixture correction #249 each
passed all 18 checks on their accepted heads; each merged tree matches its reviewed head. No stable version,
tag or package publication has been selected. The dated log below retains exact
source/wheel hashes, failures and evidence limits; older unavailable/baseline
summaries are historical rather than current backend verdicts. Counts below are
overlapping installed lane totals, not a cumulative unique-test count.

The reviewed HF batch staging/local-read diagnostics are installed accepted in
[#250](https://github.com/ProfSynapse/Synaptic-Tuner/pull/250) at `5248e576`, open
against main. The portable CPU sealed-input probe fix is also installed accepted
in [#251](https://github.com/ProfSynapse/Synaptic-Tuner/pull/251) at
`58225583be620625febba9c7e175af696bf71a5d`, stacked on #250. Both remain
separate from merged main.
Exact-head CI and merge remain pending at this checkpoint. Independent source review, narrow installed
readback evidence and local CPU experiments do not substitute for live training
qualification. The merged lifecycle implementations do not establish earlier
attempt causes or physical provider quiescence.

| Supported preparation/composition | Current accepted evidence | Remaining release work |
| --- | --- | --- |
| Durable host records/history | Caller-owned SQLite implements existing record/stream ports and atomic Data/Evaluation/Pipeline/training-workflow updates; retained plans, claims and original keys support bounded cold reads. | Final supported-host/recovery qualification; a database does not restore credentials, source leases or active processes. |
| Data/environment/evaluation consumer | **356 installed controls** exercise public generate/improve/validate, actual local fixture workers, immutable exports, reports/comparison and SQLite readback. Bounded live Luna attempt D on checkout consumer `7b8639ea` and engine wheel `d9e14291` completed two one-row Data operations, two one-case evaluations and four measured `FOUND` spend effects. Improvement validated an already-passing row without changing its JSONL bytes; comparison remained incompatible. | Representative scenario/source coverage, advertised cancellation behavior and broader backend/account qualification. The bounded synthetic smoke does not establish quality ranking, immutable model identity or invoice enforcement. |
| Ordinary SFT consumers | Shared checkout consumer with installed production passed **Docker 61 / HF 80** controls, including public start, known reads, native five-role verification/download and uncertain-submit non-replay. Modal also has the normal retained-run/public lifecycle and historical exact-source live evidence. | Final-source **Docker/Modal/HF** live SFT parity, real cancellation/reconciliation and known-run recovery with original inputs/keys. Historical Modal success does not qualify the final stack. |
| Modal GRPO | Public compiler/preparation, fixed isolated child, native verification, normal runner and retained known-run reads are implemented. The actual captured SmolLM candidate passed **256 installed controls** and retains the observed **396-distribution base**. | Qualify the actual augmented image/CPU closure, pinned model/tokenizer and one bounded **Modal GPU GRPO** run with real update/grouped reward evidence; constant/unchanged observations remain inconclusive. Other GRPO providers remain deferred. |
| Artifact publication | Caller-owned durable store/spool/create-only destination passed **377 installed controls + one existing skip**; native RunsAPI join passed **73 installed Docker/publication controls** with five-role streams and cold marker lookup. Optional HF SFT publication passed **115 installed controls**, including owned snapshot lookup after process death. The native GRPO renderer/publication join passed **145 combined installed HF controls**, **256 GRPO controls** and **39 source-closure controls** (overlapping lane totals) on wheel source `371d4b05`. | Live repository/account/destination qualification remains open; expected-parent submission and verified owned immutable snapshot do not establish RPC causality or an introducing commit after a lost reply. The GRPO join uses actual native verification over synthetic CPU model files; genuine GPU training and live publication qualification remain open. |
| Environment owner death | Guardian-backed fixed worker passed **388 installed controls + one existing skip**, including real owner SIGKILL/stopped-worker termination and retained-identity cleanup after acknowledged custody. | Keep pre-handoff owner loss, guardian/machine loss and unconfirmed cleanup explicit. This is local ownership cleanup, not resumable generation or a security sandbox. |
| Packaging/platforms | Fresh sdist-built wheels, neutral consumers, installed origins and declared assets are tested. The corrected offline source archive passed **39 source controls**, including a real pinned build, and **163 installed Modal controls**. Host examples/build declarations remain checkout-owned. The complete launcher passed **11 source controls / 163 installed Modal controls**, with 40 hashed dependencies plus engine and a passing dependency check. Fresh attempt C on exact `439608b6` passed actual four-row preparation, normal `--plan` and `--prepare-local`, producing its real source wheel. | Freeze advertised inventory/platform/install guidance and release artifacts; metadata `>=3.10` is not qualification. C authorized no provider effects and does not qualify a later final source, augmented image or model. |

**Recovery means retained known-run/status/history/artifact access and truthful
uncertainty without replay.** Implemented cold readers reuse original stored
bindings and keys; they do not renew submission authority, discover/adopt an
unknown job, replay a lost request, resume arbitrary generated rows or continue
training from an unadmitted checkpoint. Cancellation requests, SDK acknowledgment,
observed terminal status and confirmed owned-process termination are distinct.
The real provider cancellation/recovery/parity gates remain required; this scope
clarification does not waive them.

The archive correction is merged and actual normal local preparation succeeded on
`439608b6`; failed attempts A/B remain retained. The HF publication implementation
is installed accepted. Configured source-free known-job HF cancellation and
reconciliation passed **133 installed HF controls** and **163 paired installed
Modal controls** on fresh-wheel source `5c8542f2`; these are overlapping lane
totals. Initial descriptor opt-in and current permission are required, defaults
remain off, and lost acknowledgments never authorize replay. The native GRPO publication join is installed accepted with synthetic CPU model
files, without fabricating a TrainingAPI success. These implementations are now
merged; final supported-composition review remains open. Live qualification remains SFT on all
three providers, GRPO on one and the admitted publication destination. The bounded
paid nontraining smoke below is complete; representative coverage and its other
advertised behavior remain qualification work. Then complete the existing final contract/platform/
distribution review and choose a stable release.
The fresh CPU qualification attempt D on exact `4b77bd1f` stopped with
`CAPTURE_OUTPUT / INSPECTOR_REJECTED` and no qualification receipt. Its consumed
claim and local records remain retained. That code identifies the failed
inspection boundary, not its underlying cause or confirmed cleanup; no GPU
training or model preparation was performed. Its retained evidence cannot locate
the underlying failure more precisely.
A separate isolated CPU experiment reproduced a build defect: the captured base
already contains Hyperframe 6.1.0, and pip skips its same-version local wheel
under the former install flags, leaving the required wheel provenance absent.
Adding `--force-reinstall` produced the exact local-wheel URI/hash and nine
byte-identical installed members. This supports the installer correction; it
does not retrospectively establish attempt D's cause. The correction and bounded
inspector diagnostics are merged in #243: **630 installed controls** passed on
wheel `9bf5bdd67076d49e76405a6c24dd527dc6882db62b804704bae687c1a1c5b3dd`,
with all 18 checks green on accepted source `4c15db98`.

Fresh CPU qualification E used that exact source and failed once at
`CALL_INSTALLED_CHILD / UNAVAILABLE`. The exact returned failure frame establishes
that image capture/build, deployment and parent-runtime inspection completed,
then an unclassified installed-child qualification operation failed. No CPU
qualification receipt, native child code or specific failing operation was
retained; the underlying cause remains unknown. Preserve its consumed claims and
known call. There was no model download or GPU training. More precise finite-stage
diagnostics are merged in #245; a separately bound fresh attempt
is still necessary. This does not retrospectively establish E's cause.

The merged #244 Modal lifecycle implementation joins recipe opt-in and current
caller permission to the existing public cancellation/reconciliation operations. Acknowledgment is
distinct from an independent provider terminal observation. If an acknowledged
cancel loses its Foundation receipt before persistence, merged #246 retains
the consumed effect and permits known-run reads through the original authenticated
`FOUND` submit. Pending preserves `CANCEL_INTENT_RECORDED`; independent terminal
observations preserve the unadvanced cancel intent without inventing a receipt,
acknowledgment or bound cancellation. Native success can verify and stream the
five artifacts. Cancel/reconcile for that orphaned intent remain refused even
after terminal observation; physical quiescence is not established. Do not claim
arbitrary crash recovery or live cancellation qualification from the local tests.

Merged #246, accepted source `9ea8a004`, passed **178 installed coordinator/Docker controls**
and **426 installed native custody controls**, with one portable SFT control
deselected in the real-seal Python 3.14 lane. Source, wheel and installed production
bytes matched on wheel
`2b390c59073600d9396fd395ea734906bcfa3e26a8ef2a6dffdb6011b8a72a7a`.
Merged #247's source `c52be233` passed **345 installed evaluation/nontraining
controls** in 57.00 seconds on wheel
`d9e14291368baaa2c2011ed41c0b544b98b4ac2e52409eddb9624e783756b0d9`,
with source/wheel/installed production equality. After nine additional optional
consumer effort controls, merged #247 at `cf049501` passed **354 installed
controls** in 57.04 seconds using the same engine wheel. The changed consumer is
checkout-owned; the installed engine bytes are unchanged. Its optional
`provider.thinking_effort` forwards to the existing Data client constructor only
when selected; omission preserves defaults. An external evaluation judge uses
its own configuration, where null effort/temperature omit those parameters
rather than requesting explicit effort `none`. Its retained public evaluation
plans bind configuration-ordered run refs, including two runs of the same model;
legacy distinct-model attempts remain readable. The model alias is not an
immutable checkpoint, and external judging remains unverified/incomparable.
These are overlapping provider-free installed totals; CI, merging and live
qualification remain separate.

Bounded paid nontraining attempt D completed on `7b8639ea` with unchanged installed
engine wheel `d9e14291` after **356 installed controls** passed in 59.88 seconds.
The single reviewed execution exited zero with empty stderr. Credential-free
readback verified two Data operations with one row each, two evaluations with one
case each and no request errors, and four measured `FOUND` spend effects totaling
1,422 input and 382 output tokens. Generated and improved JSONL share SHA-256
`713ac3623203fc890a2eb980c7ed9308103eb7b4dbb5444b7939d33dad7177a1`: improvement
validated an already-passing row, not a textual revision. All model roles selected
`gpt-6-luna`, a mutable alias rather than an immutable checkpoint. Comparison
remained `comparable=false` with `comparison_basis_unverified` for the external
judge, not a model-quality ranking. The shared 16-call/300-second limits were
enforced; no observed live call count is exposed. Claim-to-last-closure timestamps
span approximately 21.5 seconds, not a billing or universal wall-clock bound.
Evidence is retained in `nontraining-qualification-20261014/attempt-d/readback.json`
and `root-execution-review.json`. Attempt C remains failed and conservatively
indeterminate after the pre-inner-dispatch token guard refusal; it was never
replayed or rewritten. The separate host-only default repair follows the existing
SynthChat client convention and does not change engine code. The reviewed #247/#248 heads passed exact full CI and merged. Remaining stable
qualification gates stay open.

### Current HF, Modal CPU and Docker prerequisites

HF SFT attempt A remains **stage-only indeterminate**, with no authenticated
submit or known provider job. The original workflow is
`stage_reconcile_required`; its consumed stage effect remains indeterminate and
must not be replayed. A bounded read-only batch experiment retrieved all 87
source files (982,148 bytes) in a three-second UTC measurement window and matched
the source-owned bytes. That result does not establish the original stage failure
cause or authorize adoption/re-upload. The locally integrated HF fix passed a
narrow installed legacy `show`: original 12 records and four streams remained
logically unchanged, the result retained reconciliation-required/no-provider-
evidence/no-replay status, and no HF credential/client, source preparation or
grant was opened. The fresh copied installation passed **171 controls** in 257.66 seconds on
wheel `72203ac13235d685b53de98853067be7f76cf80efedc7545336503201a1fe390`.
The initial default uv hardlink installation produced 23 asset-guard failures;
that evidence remains retained. Installing with `--link-mode copy` corrected
physical custody without changing a guard. Exact-head CI/merge and another
admitted training attempt remain pending. Evidence is retained under
`hf-sft-final-bound-plan-20261014/`, including the stage summary, batch-read
experiment and installed legacy-show record.

Modal GRPO CPU F on merged source `cf049501` ran the normal reviewed command once
and failed `CALL_INPUT_FD / UNAVAILABLE` at
`packaged_training_worker.input_fd`. Its build, deployment and qualification
claims and exact call remain retained; no signed CPU receipt was returned. This
boundary precedes `os.memfd_create`/`MFD_ALLOW_SEALING`, not the later seal step;
it cannot identify the remote exception or retrospectively settle E. The exact
copied local CPython 3.11.14 lacks those Python entrypoints. The existing training
helper succeeded locally through its reviewed libc fallback, checked real kernel
seals/hash/readback and refused a write with `EPERM`. The separate probe repair
reuses that authenticated helper and guarded core seal constants, without a new
ABI, fake Python constants, inventory/pin change or guard waiver. Source controls
passed **239 with one existing optional-Torch skip** on real portable 3.11; peer
controls passed **228 with one skip** (overlapping totals). The combined fresh wheel passed **237 installed controls**, with one optional-
Torch skip and two checkout-only controls deselected (covered by source tests).
All 1,050 source-owned members matched source, wheel and installation and had
link count one; 2,772 RECORD hashes were verified. All seven maintenance checks
are current.
A new admitted live CPU attempt remains pending. F retains no local image/
Sandbox identity or affirmative cleanup record; no GPU/model weights or replay
occurred. Evidence is `grpo-normal-cpu-qualification-20261014-f/outcome.json` and
its local existing-helper experiment.

The authorized Windows desktop is now reachable and has a ready WSL Docker
daemon through an explicit Unix socket; its default Windows `npipe` context is
unsuitable for the selected WSL/POSIX consumer. No global context change is
required. The idle RTX 3090 was observed; this is hardware readiness, not SFT
qualification. The physical Docker VHDX drive had 170 GiB free. A proposed increment of at most
100 GiB with at least 60 GiB reserved is planning, not a quota. Root approved
only the exact public Linux/amd64 `1644…` base-image pull, not GPU execution.
The single pull has started with a 900-second bound; no completed result is
recorded at this checkpoint.
The core preparation ports exist, but a reusable checkout-consumer preparation
CLI still needs its reviewed design/implementation before a GPU plan is ready.
No Docker training, model download or final-source lifecycle parity is established
by these observations or by a base pull alone.

No new grant, lifecycle DTO, store or scenario CRUD framework is required merely
to close this plan. Supported host composition and installed/source prerequisites
are detailed in the [operation matrix](api-v1-support-matrix.md) and
[host storage guide](../architecture/reference-host-storage.md).

## Historical planning evidence and gaps

These earlier planning summaries preserve their original scope. Current support
is the dated summary above and the updated operation matrix; release acceptance
criteria below are unchanged.

Historical claims remain bound to their tested sources. The
[October 7 readiness checkpoint](../review/api-v1-release-readiness.md) and
[frozen architecture matrix](../architecture/submodule-first-training-v1.md)
are not current release verdicts. The
[integration handoff](api-v1-integration-handoff.md) and
[streaming evidence](training-progress-streaming.md) describe later additions.

| Area | Earlier planning evidence | Gate recorded at that checkpoint |
| --- | --- | --- |
| Public contracts | Import-light typed API v1 and provider-neutral reference composition; provider-free tests. | Exact release contract inventory, compatibility/error policy and supported compositions. |
| Docker | Adapter/conformance implementation and provider-free tests are present. | Real SFT lifecycle, cancellation and restart/reconciliation evidence through the public API; fake tests do not satisfy this. |
| Modal | Historical GPU pair on `01383fa8` verifies restricted numeric delivery and native artifacts. CPU SDK pair on `9252e924` verifies bounded complete-byte transport and collision preservation. | Final-source stable lifecycle/recovery/cancellation closure. Logs remain conditional on exact matching per-call CPU proof, not globally enabled by historical evidence. |
| HF Jobs | Existing legacy CLI/protected operational lanes. | Public API v1 adapter and same stable lifecycle semantics; legacy job completion does not establish parity. |
| Host durability | Generic stores/authority ports and reference compositions; minimal consumer retains attempt claims. | Packaged usable durable host composition, migrations and complete restart recovery. In-memory reference stores and a local attempt journal do not provide this. |
| Packaging/platforms | Distribution version `1.1.0`, Python metadata `>=3.10`, selected wheel assets and Python 3.12 CI. The initial baseline omitted public schemas and workflow assets; later installed preparation evidence is recorded below, while the full script/asset inventory remains open. | Complete supported distribution inventory, clean installed consumer, demonstrated Python/platform matrix and release artifacts. Metadata is not qualification. |
| Optional families | Public facades exist; some operations require host-supplied backends. | Per-operation backend status, filtering and explicit supported/experimental/deferred boundaries. |

### Earlier workflow-family snapshot versus release acceptance

| Family | Earlier evidence category | Required stable acceptance |
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

The release-exit criteria below remain authoritative. Implementation evidence
closes only its stated slice, not the broad release milestone. Record terminal
evidence against the exact candidate source and artifact digest; no historical
result automatically closes a new gate. Retain failed attempts, consumed claims and uncertain submissions.

## Preparation checkpoint — 2026-10-13 (historical)

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
- [x] Define stable error categories, uncertain outcomes, identity/provenance,
  pagination, bounds, compatibility and deprecation in the reviewed
  [API v1 compatibility policy](../architecture/api-v1-compatibility.md).
  Provider parity means consistent public semantics, not identical hardware or
  provider internals. This policy does not close operational/live qualification,
  package metadata or the remaining contract inventory gates.
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

### Historical baseline blockers and preparation follow-ups

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
- At the initial checkpoint the paid evaluation judge path rejected with
  `backend_unmetered`; the later accepted metered judge closes that implementation
  gap, while live qualification remains open. Select and
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
3. **HF Jobs API v1 parity:** qualify the existing installed provider-neutral
   adapter and host consumer with the same ladder. Preserve protected
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

The additive public compiler and preparation evidence are recorded below.
The additive public GRPO compiler and isolated packaged candidate now have
separate implementation nodes. The candidate preserves exact compiled core
bytes, retained dataset/model custody and native five-role verification, with
fixed GRPO offline131/outer13 inventories and an admitted child-only deadline.
Focused CPU controls cover the actual owned child path using explicit ML/image
substitutes, and separately the actual selected import guard and complete wheel
Python checks. They do not qualify the GPU stack, parameter updates, reward
dispersion, provider execution or public cancellation. Provider routing and
live candidate qualification remain open; no current SFT release is relabeled.
See the canonical GRPO reference for maintenance and deadline scope.

The integrated isolated candidate passed a fresh sdist-built wheel's base-only
consumer and all 341 installed conversation/SFT/GRPO controls (10.44 seconds).
The wheel SHA-256 is
`75f8ca2af1849b7c911dd2ec9493b6eba9146206c11b072babd14d3d331dace1`.
All 24 changed packaged source/schema/manifest assets matched reviewed bytes;
five source-only manifest-maintenance controls also passed. Existing lock
refreshes retain their inventories and runtime pins. The installed trace fixture
selects engine-owned module namespaces independently of manifest membership;
neighboring third-party packages are excluded without hiding undeclared owned
imports. Actual child/model/image observations remain explicit CPU substitutes.

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
and training remain open. The later normal Modal host integration described
below also admits the conversation format; this handoff itself covers Python
API planning.

### Installed public GRPO compilation and preparation

The existing training input now accepts explicit GRPO hyperparameters and
retained declarative reward documents. Its separately registered recipe compiles
through the existing generic registry; default SFT registration and canonical
documents remain unchanged. Immutable prompt preparation preserves original
UTF-8 JSONL and reward columns without inventing SFT targets or splits. Dataset
identity binds bytes, format and prompt column, so legitimate model, reward or
learning-rate changes can reuse verified data. Each preparation request checks
actual retained-core admission before returning its verified input lease.

Source-archive-built wheel
`1233f4739d40893cd51717eb7c75293049ea0477a4fb22899be27ee72d320af8`
passed a fresh base-only neutral consumer and 233 installed tests. Six production
modules, three schemas and four refreshed source inventories matched the
reviewed checkout and installation byte for byte. Another 253 source and closure
checks passed. Existing 98/119/68-member inventories and runtime pins remain
unchanged; only reviewed source hashes and derived closure values were refreshed.

This is CPU preparation and pure developer compilation. It does not enable
packaged GRPO execution, qualify a model/runtime image, or establish reward
dispersion and parameter updates. Separate isolated execution, native artifact
verification, ordinary host integration and bounded live qualification remain
release requirements. See [the declaration contract](public-grpo-compilation.md)
and the canonical fine-tuning skill's GRPO reference for the exact scope.

### Shared packaged training input custody and child supervision

The packaged SFT controller now reuses an internal retained-input helper while
keeping its original compiler, trainer invocation and SFT evidence parser.
Cleanup failures reject execution. The shared raw child runner also supports an
explicit absolute deadline with an owned process group and a finite five-second
reap-confirmation wait; unconfirmed cleanup reports failure. Existing SFT calls
retain their prior externally bounded behavior. This helper alone does not
enable GRPO, provider cancellation or an end-to-end workflow timeout.

Fresh source-archive-built wheel
`3cd95da74a29fee96e6d0dfde98b74243b037ec240ad3aecd7269be78c679a8c`
passed the base-only neutral consumer and 298 installed conversation, SFT
execution, custody and process-control cases. Both changed modules and the
refreshed five-member manifest matched source and installation exactly. The
existing five-member inventory and runtime pins are unchanged.

The broader local Python 3.12 run passed 637 cases and skipped one; twelve
existing Linux sealing checks could not run because that portable interpreter
lacks the required `fcntl` constants. The affected child/worker modules passed
196 cases with one skip under an isolated system Python 3.14 environment that
provides them. No runtime fallback or substituted constants were introduced.
GitHub's supported Python 3.12 CI remains required before merge.

### Normal Modal conversation preparation

The existing Modal SFT recipe and host now select either the historical v2 or
conversation format explicitly, pass that selection through named runtime/build
compatibility checks, and retain the original immutable dataset bytes. One
private-copy path selects the existing format verifier; it also supports GRPO
prompt custody without enabling GRPO planning or execution. Historical profiles
remain unchanged and reject the conversation format.

Source-archive-built wheel
`671022228f663a4b2dce24903e41018ffc7377149f66a57d392c924af6cebae1`
passed a fresh base-only consumer and 113 installed host tests. The joined
conversation case reaches the actual normal runner and public preparation,
resolve and start with scripted provider replies. Existing v2 CLI, retained
readback, process restart and cancellation controls remain covered. Four changed
production modules and two installed candidate assets match the reviewed source
byte for byte. Existing runtime inventories and pins are unchanged.

The candidate remains unqualified. Git/source and provider observations in these
CPU tests are explicit fixtures, not live source admission, image capture or
training evidence. That build planner originally derived its source root from its
module location and required Git HEAD there. The later explicit source handoff
below removes that checkout-location assumption for installed callers. The host-owned build profile stays
outside the wheel. GRPO execution and live model qualification remain separate.

### Explicit source selection for installed Modal hosts

The existing `SYNAPTIC_ENGINE_ROOT` setting and `ProjectContext.engine_root` now
carry the selected engine build source through planning, local preparation,
revalidation and capture. Preparation retains the selected directory identity,
clean commit and build intent. Installed policy assets remain installed; both
the selected source and the built wheel must contain the expected inspector
before provider claims. Existing checkout defaults and restricted metrics source
rules remain unchanged. A local clean commit does not replace pushed-source or
host Gitlink authority.

Fresh source-archive-built wheel
`5b4df495f19282745674bfe253dbb3899eaef95ef19befe7e4264406a8dcf6fa`
passed a base-only neutral consumer and 122 installed host tests. Seven changed
production modules matched the source and installation byte for byte. The full
267-case legacy runner suite also passed, and existing inventories remain current
without hash or membership changes. The permanent installed lane uses actual
test-owned Git repositories and checks missing/dirty/replaced sources and changed
wheel inspectors, including the final pre-claim source recheck.

Wheel building and provider replies are scripted in those controls. They prove
source propagation and local consistency, not an actual provider image build,
pushed-source admission or model/hardware qualification. Installed training still
requires a separately admitted engine source checkout; the wheel is not a Git
source repository. GRPO execution and provider parity qualification remain open.

### Fresh-process Modal cancellation controls

Two installed CPU cases now restore the full host in separate interpreters using
its original retained keys, workflow, catalogs and attempt journal, after the
original submit authorization has expired. Cancellation acknowledgment stays
`cancel_requested`; a lost reply stays `reconcile_required`. A second restored
host preserves the same consumed claim and issues no new cancel, stage or submit.
The original submit commitment is unchanged, and training input access is denied.

These test-only capability settings and scripted SDK replies exercise existing
machinery. Production capability flags remain disabled, and acknowledgment does
not establish physical termination. The two cases passed against the accepted
PR212 installed wheel; this change adds tests and permanent installed CI coverage
without changing production code.

### GRPO closure regression correction

The broad core suite caught a reverse import from the shared SFT manifest parser
into GRPO. GRPO now supplies its private fixed identity and inventory to the
shared parser; SFT keeps its original fixed defaults and rejects overrides.
Bootstrap and inference static closure audits pass without widening inventories.
The schema test resolves resources from the actual installed production package,
including editable installations, and the package-data test includes all four
reviewed manifests.

All 59 targeted controls and 341 installed SFT/GRPO controls pass after correction.
The fresh sdist-built wheel is
`5d7a451b1d92d406f022aa278decc74342edf1610147d195c704e35467531703`;
its base-only consumer passes and both changed production files are byte-exact.
Independent review confirms only required source hashes, sizes and aggregate
digests changed in the fixed inventories; all six checks are current.

### Packaged GRPO material admission

The existing host material boundary now selects the exact GRPO core and packaged
wrapper from the original resolved configuration and retained runtime context.
It rechecks the public declaration, prepared input, artifact policy and execution
binding. Retained readback uses that same context; SFT compilation and canonical
bytes remain unchanged. This does not add an ordinary GRPO recipe or dispatch.

A fresh sdist-built wheel passed the base-only consumer and 247 installed public
GRPO/declaration/preparation/material controls (1.65 seconds). Its SHA-256 is
`bb4f08da70da6ee449d23d3c14995da2b5b6aae588b71ff93ce76818a02357d1`.
Both changed production files matched reviewed bytes; all six source lock checks
remain current without additional refresh. Explicit test-package imports keep
the combined installed test collection independent of pytest's import order.
Independent source and CI review passed; provider/model qualification is open.

### Fixed GRPO worker routing and native readback

The existing Modal packaged dispatch and executor now select the exact GRPO
compiler and isolated runtime. GRPO native verification must finish before
signed completion is published. The existing reader authenticates completion
and streams all five native roles; GRPO evaluation and SFT-only execution hooks
are rejected explicitly. SFT callback behavior and capability flags are retained.

The fresh sdist-built wheel
`5c94c413aa4c8c5f6cf070486eed5e12834b94b29f76e9565deb421ad12534b7`
passed the base-only consumer and 355 installed SFT/GRPO tests (13.96 seconds);
all three changed production modules matched reviewed source bytes. The 239
focused source cases and independent source/CI review passed. All six lock checks
remain current without inventory or hash changes. The joined CPU test uses the
actual isolated owned child, native verifier, HMAC completion and reader streams
with explicit model/image/provider substitutes. Fresh reader construction and
wrong-signature rejection are covered; full public-host SQLite restart, ordinary
recipe launch and live GRPO qualification remain separate integration gates.

The diagnostic readers now recognize the same nine fixed GRPO failure stages as
the worker issuer. Unknown and private values still reject; no additional
diagnostics are exposed. The correction passed 896 source controls with the
pinned Modal 1.5.4 SDK and 249 installed-reader controls (11.48 seconds). The
fresh sdist-built wheel is
`fc09461f14a4639aefcf26b0cc65c3227d1afb017734d2d3ab220aac2cdac35c`;
its base-only consumer passes and the changed reader is byte-exact. All seven
source lock checks remain current without refresh.

### Strict GRPO recipe planning

The normal Modal planning surface now accepts a strict GRPO recipe and an exact
GRPO-compatible named runtime profile. It retains and validates prompt bytes,
resolves the public GRPO declaration, and binds final material only to genuinely
prepared local inputs and the original runtime intent. The host-owned planning
store survives close/reopen after source and recipe removal. Ordinary execution,
restricted metrics qualification and quoting remain closed pending the normal
runner integration and live qualification. No candidate profile is promoted.

Fresh sdist-built wheel
`c4f4bc5123a6d93fb13f1f095f64189ec6785a9ac9c5a927127a915eecef8064`
passed the base-only consumer and 163 installed host, recipe and profile tests
(133.02 seconds). All six changed production modules match reviewed source bytes;
all six lock checks remain current without refresh. Snapshot callbacks receive
an independently parsed view of the same retained bytes before final identity
checks; SFT, conversation and GRPO callback controls pass together. Independent
source and integrated CI review passed. These CPU controls establish planning
and custody behavior, not cold full-host recovery or provider qualification.

### Method-aware packaged CPU qualification

The existing CPU self-check derives SFT or GRPO from the validated release
contracts, admits the corresponding fixed closure, and compiles the selected
trainer source from the pinned wheel. Parent and child use the same method;
legacy SFT remains supported. Unknown or mixed contracts reject before spawning.
The CPU protocol and signed receipt do not change.

Fresh sdist-built wheel
`42893dda1a7305d68115393685bec7afa2b1c6aca866f4278ea184c58c5448a2`
passed the base-only consumer. All 140 focused source controls passed. Locally,
393 installed controls passed on Python 3.14, whose real sealing constants allow
the isolated CPU protocol; the remaining portable SFT control correctly rejected
3.14 and passed separately against the same wheel on supported Python 3.12.
Permanent CI runs all 394 together on Python 3.12 without exclusions. All three
production modules matched source bytes. All six lock checks are current;
independent review confirms fixed inventories and non-hash pins are unchanged.

The tests exercise the actual isolated protocol, sealed input, root custody,
import guard and source compilation with explicit package/image/device-namespace
observations. They do not prove container isolation, live image compatibility or
training. Ordinary GRPO runner integration and cold full-host recovery remain
separate gates.

### Bounded GRPO parameter and reward evidence

An optional closed `limits.qualification_evidence` policy measures the complete
trainable population at the actual training callbacks and the values returned by
the configured reward function. Bounded original-dtype CPU chunks, finite
counts and the original child deadline constrain measurement. Signed zeros are
normalized only in detached CPU copies. Native verification checks the same
policy-bound evidence in lineage and metrics without adding artifact roles.

Actual parameter change and usable grouped reward variation are separate
requirements. Missing or constant observations, unchanged parameters and batch
reward scaling remain explicitly inconclusive. These are trusted locked-child
measurements; historical before-tensors are not retained. The optional policy
preserves existing canonical bytes when absent and does not promote a runtime
profile or claim GPU, model-quality or provider qualification.

Independent source, CI/skill and fixed-inventory review passed. The root source
suite passed 69 controls; the fresh sdist-built wheel
`3b7ae76f7120ab0524249107afee3575466d95b6bee0b374c3e4b6983ea691b2`
passed the base-only consumer and 69 installed evidence controls using exact
CPU Torch 2.9.0 (21.26 seconds). The custody suite passed 393 cases on local
Python 3.14; its sole portable SFT version rejection passed on Python 3.12
(0.84 seconds). This is split local evidence, not one 394-case run; CI retains
the complete suite on Python 3.12. All eleven changed production/resource files
are byte-exact in the wheel and installed package. Only reviewed source hashes
changed in the fixed 131/13-member GRPO inventories; all seven checks are current.

The same installed wheel also passed all 163 recipe, retention and profile
controls (132.31 seconds) and 247 public GRPO/material controls (1.70 seconds).

### Ordinary Modal GRPO runner

The normal host runner now accepts a GRPO recipe with an explicitly compatible
host-owned named profile. It retains the same prepared local inputs through
actual CPU qualification, final material binding, public planning/start and
cleanup. CLI execution and CPU qualification report distinct GRPO result tags;
SFT defaults remain unchanged. Quote and restricted metrics qualification remain
unavailable for GRPO. No candidate profile is bundled or promoted.

The joined CPU controls exercise the real isolated CPU qualification protocol,
signed receipt, public lifecycle, owned GRPO child, native verification, signed
completion and five public artifact downloads. Explicit image/provider/ML
substitutes keep this provider-free. A bad qualification signature prevents
training; a lost submission reply retains the consumed uncertain claim. These
controls do not establish cold-host recovery or live GPU qualification.

Independent source and CI/skill review passed. Root source tests passed 32
joined/recipe controls (16.23 seconds). Fresh sdist-built wheel
`40e1aa15b91d863c4fd7965424fef8ca7f6362262b24768e2ea137b561f8f964`
passed the base-only consumer and 402 installed custody/runner cases on local
Python 3.14 (31.27 seconds); the sole portable SFT version rejection passed on
Python 3.12 (0.99 seconds). This is split local evidence; CI retains all 403
on Python 3.12. All three changed production modules match reviewed bytes.
All seven source lock checks remain current without refresh.

The same wheel passed the complete existing installed recipe/retention/profile
lane: 163 controls in 132.37 seconds, including the updated recipe gates.

### GRPO recovery across host process loss

A CPU test now kills the actual normal GRPO producer after its durable known-call
checkpoint. It removes original source/configuration/preparation inputs, then
opens the original SQLite state and authority in separate processes. A pending
read preserves the original census; an authentically signed completion for the
wrong execution is rejected. The real signed native output then advances the run
and all five artifacts download with verified hashes. A final process reads the
persisted verified outcome and listing without provider polls. No submission is
replayed and no new qualification or authority is minted during readback.

This is test-only host reconstruction using disk-backed SDK replies and explicit
CPU image/ML fixtures. It does not qualify live Modal recovery, cancellation,
process-local descriptors, logs or GPU execution. Independent source/CI review
passed, including the original SFT process regression. The unchanged accepted
runner wheel `40e1aa15b91d863c4fd7965424fef8ca7f6362262b24768e2ea137b561f8f964`
passed installed GRPO recovery (14.76 seconds) and installed SFT recovery
(10.40 seconds) using each case's existing fixture lane. CI keeps SFT in the
retention lane and adds GRPO to the custody lane (404 total), avoiding duplicate
collection without its recipe fixtures. No production or lock bytes changed.

### Known-run observations after uncertain cancellation

Public outcome reads now accept `cancel_reconcile_required` for an originally
authenticated known run. Pending reads preserve that state; actual terminal
success, failure or cancellation updates run truth without manufacturing a
cancellation acknowledgment. The original cancellation intent, uncertain
Foundation history and absent bound cancellation remain unchanged. Both Modal
readers share this eligibility; provider capabilities remain disabled as before.

Independent source, CI and fixed-inventory review passed. Root validation passed
31 source controls, including real SQLite reopen and a killed producer followed
by fresh pending/native-completion readers and five verified downloads. Fresh
sdist-built wheel
`b85b9dd512f1a467e0a192e9b1cb4b6af2f8a25285d06be45665f1357376ce7c`
passed the base consumer, 170 installed generic lifecycle tests (31.70 seconds),
404 installed custody cases on Python 3.14 (64.38 seconds), the sole existing
portable SFT case on Python 3.12 (0.77 seconds), and all 163 installed Modal
retention controls (136.25 seconds). The portable case rejects Python 3.14 as
required; its first failure log is retained. CI keeps all 405 custody cases on
Python 3.12. Four production modules and three lock resources are byte-exact
in the installed wheel. All seven maintenance checks are current; runtime
inventories and non-hash pins are unchanged. Explicit SDK/image/ML fixtures
mean this is host conformance, not live provider cancellation qualification.

### HF Jobs cancellation and retained-receipt reconciliation machinery

The ordinary HF public host now supports the existing cancellation command and
reconciliation ports internally. The original authenticated submission, exact
known job, current Foundation record, original retained assessment and request
material must agree before provider access. A durable receipt after the pinned
SDK acknowledges the one-shot request is the sole source of cancellation
`FOUND`. Lost, late or unstored acknowledgments retain consumed uncertainty;
terminal provider status cannot manufacture acknowledgment or authorize replay.
Known stage/submit/cancel receipts may be looked up; unknown submissions are
never discovered or adopted. A completion race preserves natural success and
the real native verifier plus all five public artifact streams.

Default cancel/reconcile capability flags remain disabled. Current cancellation
authorization is an explicit exact-boolean host option; existing start expiry
is preserved. These SQLite reopening controls retain the original preparation
object: source-free cold reconstruction and live qualification remain separate.

Independent source and CI/skill review passed. Root passed all 52 affected
source controls with Hub 1.33.0 (56.42 seconds). Fresh sdist-built wheel
`966568f4128eef2347deb97e74e1fd1d438118ff5b0159191a4ecac487a9cb38`
passed the base consumer and the complete installed pinned-SDK HF lane: 52
controls in 57.21 seconds. All three changed production modules match reviewed
bytes in the wheel and installed package. All seven maintenance checks remain
current without inventory or pin changes. No live provider effects occurred.

### Metered Data, environment and evaluation joins

Four installed CPU controls now join the public Data generate/improve/validate
paths to an actual fixed local worker, controlled repair feedback and measured
scripted LLM calls. Every generator, judge and improver call contributes to one
spend record per run. Published artifacts retain their bytes and hashes through
SQLite reopening after input/config removal. Missing usage keeps the affected
run failed and its spend indeterminate. Separate controls compare two retained
evaluation reports after reopening SQLite without opening a backend or grant;
comparable scores do not erase accounting uncertainty.

Independent source/CI review passed; root passed all four source tests in
5.11 seconds. The unchanged accepted HF wheel
`966568f4128eef2347deb97e74e1fd1d438118ff5b0159191a4ecac487a9cb38`
passed the complete installed evaluation/report lane: 230 controls in
5.75 seconds, including these four joins and explicit production-origin checks.
No production or lock bytes changed. Skill mirrors are synchronized; the GRPO
quick reference now distinguishes normal runner support from unproven live
image/model qualification. These are scripted accounting and retained-read
controls, not live account/model quality, lost-request recovery or restoration
of an active worker. Paid evaluation judging remains a separate metering gap.

### Paid evaluation judge accounting

The reference evaluation composition now accepts a named paid judge through
the existing metered client and actual JudgeValidator. Paid response and judge
calls share one spend record only when provider, account, namespace and currency
agree; an unpaid response backend contributes no tokens to the paid judge's
spend. Model authorization covers both paid model identities. Missing usage,
malformed provider completions and uncertain client construction remain
indeterminate rather than becoming zero-cost success. Legacy unpaid judging
is unchanged. External judge comparisons retain unverified scorer provenance.

Independent source and CI/skill review passed. Root passed 89 focused source
controls in 2.75 seconds. Fresh sdist-built wheel
`4921b43bf5acd41c0a972d5308a13f0d2801cea8874abacd747faf6a145ec6b1`
passed the base consumer and all 266 installed evaluation/report controls in
5.71 seconds. Both changed production modules match reviewed bytes in the
wheel and installed package. All seven maintenance checks remain current
without inventory or pin changes; skill mirrors match. No live provider
effects occurred. This closes the paid judge accounting gap above; live
qualification still needs bounded HTTP response/deadline handling and review
of retries after uncertain requests. Authorization caps are not invoice proof.

### HF known-job recovery without original source or preparation

Ordinary HF host composition now retains bounded original read material in its
existing record store. A fresh read-only composition authenticates the original
plan, descriptor, workload, submit command, retained assessment and current
Foundation record before reading the exact known job. It reconstructs the
original submitted request and uses the existing native verifier and artifact
streams. Missing or conflicting records, altered request fields and unknown
submissions reject without adoption or replay. Cold cancellation, reconciliation,
logs and reverify remain unavailable even for an originally enabled descriptor.

Independent source and CI/skill review passed. Root passed all 69 source controls
in 75.52 seconds. The process test kills the real public producer, removes its
source/input/model-cache/preparation directories and opens original SQLite in
fresh readers before parent recovery access. It checks pending and mismatched
provider observations, verifies five native artifacts, and reads persisted
verified results in another process without provider polls. Provider, image and
trainer boundaries remain explicit fixtures; bootstrap/source-worker custody and
native verification are real. This is known-job host recovery, not live HF
qualification or active training restoration.

Fresh sdist-built wheel
`2759af578fde6b1e3f6a5e0c7f4bd90c2ec99c7b80801f6cfa4eaa5242bfc206`
passed the base consumer and all 69 installed HF controls with pinned Hub 1.33.0
in 84.89 seconds. All four changed production modules match reviewed bytes in
the wheel and installed package. Seven maintenance checks remain current without
inventory or pin changes; skill mirrors match. No live provider effects occurred.

### Stop subsequent paid calls after unknown usage

Paid Data and evaluation clients now refuse subsequent requests once their
shared run accumulator becomes indeterminate. Exceptions, malformed responses,
missing usage and failing usage accessors retain uncertainty; a refused retry
does not dispatch or add another actual-call count. Paid response and judge
wrappers share this state. The option defaults off for legacy/local clients,
and known measured retries remain available. This prevents later wrapper
dispatches; it does not terminate an in-flight request, control provider-internal
retries or authorize replay after restart.

Independent review found and corrected the usage-accessor exception edge case.
The final source passed 123 author regressions and 18 independent focused
controls, including the actual improvement retry loop, a swallowed JudgeService
failure, sibling wrappers and local compatibility. Fresh sdist-built wheel
`14dc85d4d2a1b90b00c52e25005096c48c145518faef3c770f42cf22095e2ad1`
passed the base consumer and all 284 installed evaluation/report controls in
5.92 seconds. All three changed production modules match reviewed source in
the wheel and installed package. Seven maintenance checks remain current and
skill mirrors match. No live provider effects occurred. HTTP response bounds,
whole-operation deadlines and live account/model qualification remain separate.

### Training-base inventory capture preparation

The existing capture and inspector scripts now offer an explicit training-base
mode with an immutable Unsloth image, exact interpreter and Modal builder pin
`2025.06`. The pinned SDK's registry commands must contain only the selected
`FROM`; modes that install Python, packages or an engine wheel reject before
provider access. The inspector observes full bounded distribution metadata,
Python executable and sysconfig facts, and Torch's actual CUDA build value.
It distinguishes the registry base from the derived script-inspection image.
This produces candidate facts only; no runtime profile or qualification is
inferred, and no real image capture has been performed for the GRPO candidate.

Independent source and CI/guide review passed. Root ran 174 capture/inspection
controls with actual pinned Modal 1.5.4 in 1.26 seconds, with no skips. The
existing dedicated SDK CI lane now includes the new 33 controls. Both scripts
match their reviewed Git blobs; all seven maintenance checks remain current
without inventory or pin changes and skill mirrors match. These are checkout
maintenance tools: the current wheel and sdist omit both scripts. This evidence
does not claim installed-script packaging, live provider effects or GPU/model
compatibility. Genuine base capture is still required before authoring the
named GRPO candidate profile and its bounded live update plan.

### Bounded paid completion HTTP

Paid OpenRouter and OpenAI Responses clients now support explicit request and
response byte limits and a monotonic caller deadline. The opt-in path dispatches
one POST with redirects and transport retries disabled, accepts only identity
response encoding, and decodes bounded UTF-8 JSON with nonfinite numbers rejected.
The worker owns transport resources and JSON decoding; the caller stops waiting
at its deadline. A late request can still finish, so a timeout retains unknown
usage and the existing paid-call latch refuses subsequent requests. This is not
physical request termination, an invoice cap, or a bound on owner-side request
serialization allocation/time. Omitted controls preserve the legacy path.

Independent source, CI and hash-refresh reviews passed. The 36 new controls use
an actual loopback HTTP server and cover redirects, UTF-8 byte boundaries, slow
responses, nonfinite exponent overflow, late-worker dispatch admission, immutable
request bytes and paid response/judge forwarding. Fresh sdist-built wheel
`30d5f1b1b92f6826c2b624ca88daa32d0e8437595adfd93f689ae813d7fad863`
passed the base consumer, 320 installed evaluation/report controls in 21.18
seconds, and 247 GRPO input/material controls in 1.68 seconds. The packaged worker
lane passed 404 controls under CPython 3.14 with real Linux sealing in 68.39
seconds; its separate portable SFT case passed under CPython 3.12 in 0.72 seconds.
The four changed production modules and two refreshed GRPO manifests match the
reviewed source, wheel and installation exactly. The fixed 131/13 inventories
and non-hash policy are unchanged, all seven maintenance checks are current,
and skill mirrors match. No live model or training-provider effects occurred.
Live account/model qualification and stable release acceptance remain open.

### Public nontraining consumer

The checkout-owned `examples/nontraining` consumer composes the existing public
Data and Evaluation APIs, host SQLite store, fixed filesystem worker, named
credentials and metered provider clients. Planning performs no model calls or
durable writes. Execution creates private attempt state, generates and validates
one row, improves the unchanged generated artifact, validates the improved row,
and runs two bounded evaluations. Existing attempt state refuses replay.

Readback opens existing storage without model clients or credential resolution.
Retained public Data plans bind each output path to its original run and artifact
hash/size. Both changed bytes and swapped generated/improved files reject.
Evaluation comparisons follow the two distinct configured model references,
independent of list order. Unknown upstream usage stops downstream execution;
the shared call budget and explicit output-token ceiling include judge calls.
External judge comparison remains unverified, and no active-request recovery,
security sandbox, invoice cap or model-quality claim is made.

Independent review resolved output ownership, comparison ordering and checkout
import-shadowing issues before acceptance. Seventeen focused controls passed.
The unchanged HTTP wheel
`30d5f1b1b92f6826c2b624ca88daa32d0e8437595adfd93f689ae813d7fad863`
passed all 337 installed evaluation/consumer controls in 39.58 seconds from a
neutral directory under isolated Python. Only the example and tests were copied;
engine production modules came from the installed wheel. The example is a
checkout consumer, not a newly packaged engine entrypoint. No runtime source,
dependency, asset or lock inventory changed. Live account/model qualification
and the overall stable release gate remain open.

### Durable publication records and local transfer ownership

An optional publication-store adapter now persists the existing publication
records and transitions through the generic host store. Retained POSIX file
locks establish local transfer ownership. Original directory and lock-file
identities prevent another opener from replacing an active owner's lock.
A fresh process may recover an orphan for lookup only; it cannot repeat the
transfer. Default artifact composition and destination support are unchanged.

Root review corrected a wrong-thread close that could release the owner's
descriptors. The regression uses an actual competing process to confirm that
ownership remains active. All 129 source publication controls passed in 1.71
seconds, including process death and first-reader recovery with a signed
destination fixture. Fresh sdist-built wheel
`11e5c7c5f5a1ff3806870eed928117490ee8214a12adb261f7183e80cccc9d78`
passed the base consumer and 348 installed lifecycle/publication controls in
55.83 seconds, with one existing non-worker cancellation skip. The new module
is byte-exact in reviewed source, wheel and installation. An older deep-JSON
test was corrected to accept both supported closed rejection outcomes observed
on Python 3.12 and 3.14; runtime behavior did not change. All seven maintenance
checks remain current and skill mirrors match. Concrete destinations, a durable
spool, verified-run composition and live publication remain separate gates.


### Installed public SFT consumer acceptance — 2026-10-14

The checkout-owned shared launcher and Docker/HF adapters exercise existing
TrainingAPI/RunsAPI with original named keys and caller-owned SQLite. The root
integration of reviewed source `4869cb7e`/`db21073b` and their CI changes passed
the exact installed Docker lifecycle lane (**61 tests**) and pinned Hub 1.33.0
lane (**80 tests**) under CPython 3.12.15. Both ran from neutral directories with
explicit checkout example/test copies and installed production-origin assertions.
The unchanged engine wheel is SHA256
`11e5c7c5f5a1ff3806870eed928117490ee8214a12adb261f7183e80cccc9d78`;
these example, test, workflow and guidance additions change no production engine
module or runtime lock. The root initially selected Python 3.14 for the Docker
lane; its native-artifact fixture correctly refused the existing portable SFT
runtime requirement (`>=3.11,<3.14`). That failed run remains retained; acceptance
uses the supported interpreter without changing the requirement.

The tests exercise public planning/preflight/start, create-only attempts, known
reads, native five-role verification/download and uncertain-submit non-replay.
Docker cancellation preserves acknowledgment versus observed completion and
unpromoted capability refusals. Docker cold reads still require original image
qualification assets and mount roots; HF cold reads use the retained reader and
remain read-only. These CPU controls do not establish live SFT provider parity,
GPU training, provider billing, physical cancellation or model quality.


### Captured GRPO candidate installed acceptance — 2026-10-14

The additive `smollm2-135m-grpo-candidate-v1` profile retains the actual
396-distribution CPU base inventory (SHA256
`46f0feddd1ba6d5e155dfb227d92679f9afd65634b98d80260cf0517b4b102cd`).
Its source-owned build profile starts from a real previously accepted wheel;
the existing current-source build path binds the actual replacement wheel.
The four-message example is prepared by the existing service with real byte
and semantic commitments. It selects three steps, four generations, L40S,
300-second child and 900-second provider bounds with finite G4 measurements.

Independent review accepted source `fb01d666` and CI/guidance `8e2d27ea`.
A fresh sdist-derived wheel, SHA256 `14b699380b762fc27d42c83f827f9a844cf94feac075608d91573d45aebd2026`, passed
base resource checks and **256 installed GRPO tests** under CPython 3.14.4,
including all nine candidate controls without skips. Runtime YAML/inventory
are byte-exact in source, wheel and installation; the host build profile is
excluded from the wheel. Existing runtime code, old profiles and locked
member inventories/pins are unchanged.

These are candidate configuration and CPU control-flow results. The test
Git source and replacement-wheel fixtures do not establish live source/build
authority. The captured base is not the augmented training image; actual
image build/CPU qualification, model preparation and GPU GRPO remain open.
Constant rewards or unchanged parameters remain inconclusive rather than a
qualification pass. No provider effect was performed for this change.


### Local publication installed acceptance — 2026-10-14

The optional local spool and create-only destination reuse the existing public
publication operations, authority and durable store. They retain private file
descriptors, verify streamed bytes and persist an authenticated original receipt
marker. Fresh-process recovery after a killed producer performs lookup only;
missing/incomplete markers stay ambiguous. No default destination was enabled.

Root review accepted `d0ae02d0` with correction `a381fc52` and CI/docs
`03e4597e`. Review corrections preserve immutable leaf permissions/identity
after writes and conservatively reserve chunks before possibly partial writes.
Final focused controls passed **29 tests**. Fresh sdist-derived wheel SHA256
`3664d77d001e11d842ccb59e1f360b123c7e5f6633416e7b375b584389f8b483` passed base resources and the exact installed
lifecycle/publication suite: **377 passed, one existing non-worker cancellation
skip**, under CPython 3.14.4. The new module matches source, wheel and install.

This qualifies the local adapter and lookup-recovery behavior with fixture-signed
sources. Joining actual native-verified RunsAPI artifacts and implementing an HF
model destination remain subsequent nodes; no remote publication was performed.


### Fixed worker owner-death installed acceptance — 2026-10-14

The existing fixed environment worker now uses a small guardian that owns the
exact child and retained scratch descriptors. After acknowledged custody, owner
death closes a dedicated liveness pipe; bounded termination/wait precedes
identity-preserving cleanup. A stopped worker cannot stop its guardian.

Root review corrected partial pipe allocation, startup backpressure/cancellation
and pending or partially consumed custody frames during teardown. **36 focused
controls passed**, including actual owner SIGKILL and stopped-worker cleanup.
Fresh sdist-derived wheel SHA256
`8196f5cc04175ca0a7fa07b417b74c8ecc9b68a665851f6650874a5391119a7c` passed base resources and the exact installed
lifecycle/publication lane: **388 passed, one existing non-worker cancellation
skip**, on CPython 3.14.4. Both existing worker modules match source, wheel and
installation; source `361a4c4c` includes accepted author fixes through `ffa8f5f2`
and the test-only EOF outcome correction `6cd77b50`.

Pre-handoff owner death, guardian death and machine loss remain outside this
cleanup guarantee. Unconfirmed worker exit preserves scratch and uncertainty;
cleanup retry requires confirmed exit and relinquished guardian custody. This
is local process supervision, not a security sandbox or durable continuation.
No generation row or uncertain paid request is replayed, and process cleanup
alone does not establish a terminal DataAPI outcome.


### Native run publication join installed acceptance — 2026-10-14

The optional source adapter now joins the trusted stored verified RunsAPI
manifest and public artifact streams to caller-owned publication ports and the
existing ArtifactsAPI. Its deterministic signed source identity commits the
public verified projection, not an invented native receipt digest. Default empty
destinations and caller object lifetimes are unchanged.

Independent review accepted `dd58dc4f`; root reviewed CI/docs `f8ac22ed`.
Fresh sdist-derived wheel SHA256
`d8f21284319d424a0fd9323e089d056437319392a45c520080f66a490ace2b61` passed base resources and **73 installed
Docker/publication tests** under CPython 3.12.15. The source adapter is identical
in source, wheel and installation. The controls connect public native five-role
verification to local publication, then test terminal reads and marker-boundary
recovery without provider calls, native source reopening or a second transfer.

The process controls use real SIGKILL and a new first SQLite opener, retaining
scripted fixture host inputs through fork. They do not establish independent
remote-provider reconstruction or live training/publication. HF repository
publication remains a separate destination and qualification gate.

### Real offline source-wheel build acceptance — 2026-10-14

Normal local GRPO preparation on `9285cd1d` failed before provider effects:
the retained build reported `OFFLINE_WHEEL_FAILED`, and exact subprocess
diagnostics identified the missing `_build_support` packaging helper. The fixed
source archive includes that helper, `MANIFEST.in`, both declared tool catalogs
and the public schemas, alongside the existing package trees. It retains the
clean source/commit checks and exact offline builder pins.

Root integration `e3b92ef2` passed **39 source controls** in 16.67 seconds,
including an actual CPython 3.11.14 offline build from a genuine Git snapshot.
The pre-fix allowlist reproduced the closed build failure; emitted schemas,
catalogs and candidate resources now match their source bytes. Fresh sdist-built
wheel SHA256 `d464d3ec1fd15037fb11482d4f6e0f7fb8c60e317d03a9a18a6e8a67c6e93320`
passed neutral base-resource checks and **163 installed Modal controls** under
CPython 3.12.15 in 130.95 seconds. All seven runtime/source lock checks remain
current without pin or hash changes. The same real build control is included in
the existing pinned Modal diagnostic CI job, triggered by its packaging inputs.

This closes the demonstrated source-archive omission. The original failed
attempt remains retained; a new normal preparation binds the final source.
No image, provider resource, model download or GPU qualification was performed
by these controls.

### Native GRPO publication installed acceptance — 2026-10-14

Wheel source `371d4b05` passed the base consumer, 256 installed GRPO controls
(2.05 s), 39 source-closure controls (2.04 s), and the corrected combined HF
publication suite: 145 passed (250.09 s). These totals overlap. All seven source
closure checks were CURRENT. The original combined run retained 142 passes and
three test portability failures: an import census counted third-party installed
files, and a golden fixture embedded the author's interpreter path. Corrected
test-only fixtures passed against the same production wheel; this was not a
production transport defect. The retained corrected log is
`grpo-model-publication-installed-20261014/installed-hf-grpo-publication-corrected.log`.

The real native verifier, original five roles, trusted source port, admitted
renderer and public ArtifactsAPI consume explicitly synthetic CPU model files.
Warm and uncertain-reply/cold immutable-snapshot controls do not fabricate a
genuine TrainingAPI success or qualify GPU training, live uploads, model loading
or model quality. The reviewed inference inventory adds exactly five pure parser
members (119 to 124); GRPO 131/outer 13 refresh only source hashes. Runtime,
dependency and image pins remain unchanged. Final-source live gates remain open.
