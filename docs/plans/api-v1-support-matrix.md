---
status: scoped-engine-release-candidate
baseline: e2e8cd4f
updated: 2026-10-28
---

# API v1 support and qualification matrix

## Current engine release candidate — 2026-10-28

`1.1.0rc1` is the first scoped OSS engine release candidate. Its integration
baseline is merged `e2e8cd4f631d6b015ea323fe038944f3d438b64e` (PR #279),
whose tree equals independently accepted `9524bae2`; all 18 checks passed on
that reviewed head. The release changes documentation, package version metadata
and the corresponding source-closure hashes, preserving execution logic and
runtime dependency pins. See the [release notes](../releases/1.1.0rc1.md) and
[developer guide](../development.md) for installation and supported boundaries.

The selected engine/Mac integration is complete: SFT and GRPO, live numeric
progress, confirmed cancellation, abrupt caller-loss/cold reconciliation,
saved-adapter verification and five-role artifact delivery passed on the exact
M4 fixture below. Final independent review accepted the evidence/package/claims;
post-expiry original-authority reads passed without retraining or renewal.
Hosts still authenticate qualification for their exact selection. This preview
does not close broader stable/provider gates or ship a desktop app.

## Integrated engine evidence — 2026-10-26

The current owner-selected work is the open-source engine API and reliable Mac
training integration. Full desktop installation/signing, bundled-runtime UX,
managed-download UI and local inference conversion are outside this increment.
Candidate `06e0fcf24ca152621044736f65f87ce6e176d0f4` integrates the release/GRPO,
Windows SFT and native MLX branches with main's host-trust corrections. Fixed
closure inventories and non-hash runtime pins are preserved. Independent source
review accepted it for bounded qualification, and a neutral base-only installed
consumer passed. That increment did not publish a package or deploy Syntunia;
the release candidate above is the subsequent publication scope.

Native public composition is available explicitly through
`synaptic_tuner.api.v1.native_mlx`, using the existing `TrainingAPI`/`RunsAPI`
and host-owned authority/storage ports. The exact M4/Qwen3-0.6B/CPython3.11.14/
MLX-LM0.32.2 fixture passed fresh raw and assistant-only runs on this integrated
source, each with three actual updates, finite changed saved adapters, separate
saved-adapter reload validation, public verification and five-role delivery.
Real Mac process and numerical controls passed. See the
[exact source-scoped report](../review/native-mlx-sft-qualification.md) and
[host integration contract](../architecture/native-mlx-host.md).

The rows below reconcile later GRPO/Windows/native evidence without relabeling
older execution sources as this candidate. A host must authenticate qualification
for its exact selection; no default native admission is shipped. Public native
logs/cancel/reconcile remain unavailable for legacy selections. The separately
authenticated versioned lifecycle has bounded M4 evidence below. Ordinary
qualification stays host-authenticated.
Broader stable-v1/provider lifecycle acceptance remains separate from the
completed bounded native lifecycle fixture.
HF Jobs qualification remains owner-paused. Existing Windows/Modal runs are not
retrained to repair documentation, packaging or local readback.

2026-10-27 increment: two practical conversation SFT datasets each completed
50 updates and saved-adapter reload evaluation on all 20 held-out rows at exact
`d57dc55e`; their preparation example retains the source-specific evidence.
Native GRPO at exact `ad17275a` completed one separately admitted three-update
M4 mechanics fixture, with varied rewards, saved parameter changes, reload
consistency and five verified/delivered roles. See the
[bounded GRPO report](../review/native-mlx-grpo-qualification.md). These exact
attempts do not relabel earlier evidence or qualify training quality/general
native support.

2026-10-27 lifecycle increment: execution `e29c7da8` passed two normal
three-update completions, confirmed public cancellation and abrupt caller-loss
with cold reconciliation for each of SFT and GRPO. All 20 delivered role hashes,
saved parameter changes and final-adapter reload checks passed; real Darwin
186 controls and five real MLX numerical cases passed. See the
[exact evidence and limits](../review/native-mlx-lifecycle-qualification.md).
Only the explicit policy/session/independently authenticated lifecycle selection
can expose these operations. Final independent review, all 18 checks and
post-expiry readback passed before merge. No default authority or broad
stable-v1 claim follows.

## Historical integration baseline — 2026-10-20

The consumer target is **Modal SFT and GRPO through public API v1**, with
retained known-run reads and verified artifacts. HF Jobs qualification is
**paused by the owner**; it is not a prerequisite for starting Syntunia's
integration. The owner explicitly keeps Modal GRPO on the release and Syntunia
readiness path. Local desktop training is also required on NVIDIA and Apple
silicon: retain Docker/CUDA for NVIDIA and qualify a native macOS GPU path
separately. Neither is deferred. Modal execution itself has no Docker-host prerequisite. This is an integration baseline, not a published stable
release or a claim that every lifecycle operation is qualified.

Merged source `eb8a722e8cb640558c0783bb281c29d1ec17a7b2` (PR #258) has the same
tree as tested candidate `864a9bccbc8bd866c56c021c131f12959f8a8649`. Its two
separately admitted Modal SFT runs completed, each with five verified native
artifacts and two numeric records delivered before completion. The second also
passed fresh-process public observation and retained readback. Evidence binds
that exact source/fixture; it does not qualify arbitrary recipes or models.
The accepted distribution was built from `af1cb989`; the later test-only delta
was checked as excluded from wheel, sdist and runtime inventories.

The [release plan](stable-v1-release.md)
retains exact runs, source provenance and failure evidence. The Docker root
preparation correction in PR #259 is separate from that live evidence. Current
source changes must be reviewed against the relevant package/runtime boundary;
no documentation update relabels earlier runs.

Use the [durable Modal consumer walkthrough](../../examples/training_qualification/modal-consumer.md)
for the existing executable path and its Python host boundary. The engine is a
Python library: Syntunia still owns HTTP/authentication, credential selection,
storage, admission and product state. No HTTP service or multi-tenant host is
provided by this baseline.

Status vocabulary:

- **Contract-only:** typed surface or planned semantic operation without a
  usable accepted backend/composition for this claim.
- **Implemented/provider-free:** code and local/mock behavior exist; installed
  acceptance is stated separately for each composition, and live provider
  qualification remains a separate requirement.
- **Installed accepted, bounded CPU:** a neutral consumer exercises installed
  production modules with declared fixtures/local processes. It does not
  qualify a live model, GPU, provider account or every Python/platform.
- **Live-qualified, bounded:** evidence establishes the stated operation only on
  its exact source/configuration. It does not qualify current main automatically.
- **Unavailable/disabled:** absent implementation, missing required host port or
  capability refusal. Legacy CLI behavior is identified separately.

The [API v1 compatibility policy](../architecture/api-v1-compatibility.md) defines
public compatibility; this matrix records operational support and qualification.

Package version `1.1.0` and its intended stable classifier do not certify every
composition. This dated matrix records scoped evidence; [GitHub release records](https://github.com/ProfSynapse/Synaptic-Tuner/releases)
bind published source/artifacts and final accepted operations. The distribution
route is reviewed GitHub wheel/sdist assets, not an unverified PyPI install.

## Included workflow operations

Installed counts below retain their original source-scoped evidence, detailed in
the release plan. They are not new tests of every current composition. Pending
coverage means qualification/review work, not an instruction to reimplement the
existing APIs. Nontraining smoke results remain bounded synthetic evidence.

| Family / operation | Current accepted support | Backend evidence and stable gap |
| --- | --- | --- |
| Scenarios: author/edit/version | Host-owned config-first YAML and immutable captured refs; no scenario CRUD facade | Existing declarative files/schema/loaders feed DataAPI admission. Document and qualify the supported author/validate/version workflow and actual source coverage; do not invent a separate authoring authority merely to replace host file ownership. |
| Scenarios: validate/select | Installed loaders, selected dependency capture and bounded validation | Actual scenario/rubric/tool declarations are consumed from retained snapshots. Structural validation does not establish semantic coverage or permission for environment effects. Representative full scenarios remain a release qualification gate. |
| Environments: config/create/validate | Reviewed admission, retained inputs and installed local host session composition | Optional `ReferenceDataPortsV1.environment_sessions` connects a host-configured factory to existing runtime methods. Public planning/preflight remain effect-free; selected execution requires that factory. No new environment facade or grant framework is required. |
| Environments: execute/tools/feedback | Installed public local worker/session execution and acknowledged-custody owner-death cleanup | Configured tools, fresh fixtures and corrective feedback use normal DataAPI. Real stalled-child/cancel/owner-SIGKILL controls pass; guardian/machine loss and pre-handoff death preserve their limits. No arbitrary row continuation, security sandbox or universal wall-clock claim. Live nontraining/backend qualification remains open. |
| SynthChat: plan/preflight/generate | Installed public DataAPI with configured reference ports | The candidate 356-control consumer lane joins actual fixed workers and measured scripted LLM calls. Requests and selected inputs are committed before stages; missing clients/factories or changed inputs reject. Bounded live Luna smoke D generated one row; representative scenario/source and broader model/account acceptance remain open. |
| SynthChat: improve/judge | Installed improve plus configured judge behavior | Actual generated bytes are improved and validated, with structured environment feedback and per-run export binding. Judge is workflow behavior rather than a separate DataAPI mode. Live D validated an already-passing row with unchanged JSONL bytes, so that smoke does not demonstrate a textual correction. Representative scenario/source coverage remains open. |
| SynthChat: export/datasets/validation | Implemented/provider-free result/dataset operations and validators | Bind immutable export, format, source coverage and split lineage; no manual relabeling. Installed source assets and exact downstream training identities must be checked. |
| SynthChat: usage/progress/cancel | Installed accounting, stored observations and cooperative cancellation | Paid unknown usage latches refusal across sibling run clients; measured known retries/local legacy callers retain their behavior. Cancellation prevents later work and requests owned worker termination; it does not stop an in-flight LLM/provider-internal retry. Cold reads preserve original claims/spend without replay or resumable rows. |
| Text-conversation training preparation | Installed normalizer, independent verifier and immutable source extension | Declarative mapping/target/group splits preserve ordered text messages without invented ingestion ancestry. All 49 installed cases pass. SFT compilation/loading/masking/native verification also pass 47 installed CPU cases. Two installed joined cases now carry real configured Data generation/improvement and environment assertions through immutable preparation and public resolve/plan after raw input deletion. A separate unqualified candidate declares the new format; historical releases still reject it. The normal Modal recipe/copy path now selects it explicitly, covered by 113 installed host cases including scripted public start. Installed hosts now use explicit engine source selection, covered by 122 installed host cases with real local Git and scripted builds. A separately admitted source checkout, image capture and live training qualification remain required. |
| Dataset validation | Implemented/provider-free `DataAPI.validate` and legacy validators | Real configured catalog/default assets must load in installed consumers; missing defaults cannot silently become empty schemas. Validation does not attest semantic provenance by itself. |
| Evaluation: plan/preflight/start/status/results | Implemented/provider-free optional `EvaluationAPI` reference composition | Requires explicit backend registry, scenario resolver and host stores; grants/spend apply to paid participants. Local HTTP body limits default to 1 MiB each, host-configurable through 64 MiB; exact UTF-8 transport and retained report checks pass installed. Metered/unmetered paid adapters exist; neither a registry entry nor CPU transport tests qualify a live backend. |
| Evaluation: judge/usage | Installed named metered paid judge and same-scope response/judge accounting | Local response plus paid judge counts only attributable judge usage; paid response/judge share a meter only for the same account/currency. Missing/wrong-shape/raised usage remains indeterminate and subsequent paid calls stop. Optional finite HTTP transport is implemented; D records four measured paid spend effects under its exact configured scope; broader backend/account qualification and invoice enforcement are not established. |
| Evaluation: reports/comparison | Installed public retained-report comparison and SQLite readback | Signed run/role bindings and consumed input/scoring/generation controls determine comparability. Missing/external judge/environment provenance remains unverified; comparison does not settle unknown spend. The 356 installed lane includes actual public comparison/renderer/consumer joins. Live D compared two same-alias runs but retained comparison_basis_unverified/comparable=false for its external judge. This is configuration provenance and mechanical score comparison, not remote model or quality attestation. |
| Evaluation: progress/cancel/restart | Stored lifecycle/progress/cancel state and cold result/report reads | Retained reports read/compare without reopening backends. Interrupted active work remains truthful and non-replayable; no mid-case continuation or forced in-flight HTTP termination is claimed. Advertised real backend cancellation/recovery still needs qualification. |
| Artifacts: run inventory/verified download | Installed public stream/download plus bounded current Modal live proof | The two `864a9bcc` SFT runs verified five native files each, with independent local hash checks. Original role/hash/size plus chunk/EOF bounds remain enforced. This does not qualify other providers or arbitrary artifact sets. |
| Artifacts: destinations/publish/verify | Installed durable local composition plus scoped live local publication; default registry remains empty | Live local publication on host `5473f600` with engine `1da0a808` verified five native SFT files and cold lookup without changes; this remains bound to that earlier source.  Caller supplies original publication authority and existing stores/spool/registry. Local exclusive files and authenticated marker are implemented; source join uses trusted RunsAPI stored verified manifests and actual public streams. Optional HF SFT model-repository publication passed 115 installed controls with parent CAS and verified native/rendered snapshots; live destination/account qualification remains open. The native GRPO renderer join passed 145 combined installed HF controls, alongside 256 GRPO and 39 source-closure controls (overlapping totals) on `371d4b05`. It uses actual native verification over synthetic CPU files, not a fabricated TrainingAPI success or live/GPU/model-quality proof. |
| Artifacts: restart/ambiguous publication | Installed local actual-process marker lookup recovery | Real killed-publisher/first-new-SQLite-opener controls verify original markers without source streams or new writes; incomplete markers stay ambiguous, never absent/retry authority. HF installed process-loss lookup also verifies original signed ownership and exact native/rendered immutable snapshots without another commit. It does not establish an introducing commit or RPC causality after an unknown reply; bucket progress is not final model publication. |
| Host record/stream storage | Optional caller-owned SQLite plus atomic heads/history and retained catalogs | Data/Evaluation/Pipeline/training stores preserve exact claims/plans/history; local publication adds retained OS ownership. [Host storage](../architecture/reference-host-storage.md) states cold prerequisites. Storage does not recreate credentials, active worker/source lease or remote continuation authority. |

## Training provider and lifecycle matrix

| Route / operation | Current support | Evidence and release requirement |
| --- | --- | --- |
| SFT generic plan/preflight/start | Implemented/provider-free public `TrainingAPI` and SFT compiler | Compile exact recipe/input/runtime identities, then host-owned grants and coordinator effects. Not every named recipe/model/hardware combination is qualified. |
| SFT Docker start/observe/artifacts | Live-qualified, bounded original Windows/WSL2 NVIDIA run | Three-update RTX 3090 run `docker-sft-venv-6874-r50-20261025` executed source `6874e9d3935065d3df650be953de141eb581ad0d`. Reviewed reader `b7d5e840fca8a94c13292dec2e40c6b1ca30315c` verified and delivered all five original artifacts (23,714,452 bytes), retaining earlier rejected receipts. Integration preserves the declared-venv launch and raw optional-control verifier fixes. This is not execution of the integrated source, repeated reliability, general Windows Python support or cancellation qualification. |
| SFT Docker logs/cancel/reconcile | Profile-selected cancellation and retained known-receipt reconciliation; logs unavailable | Exact-container stop is one-shot; acknowledgment and natural terminal truth remain distinct. Default/unpromoted capability refusals remain enforced. The older same-process composition still rejects cancellation. Final-source live cancellation/recovery parity remains open. |
| SFT Modal start/observe/verified artifacts | Live-qualified, bounded on `864a9bcc` | Runs `modal-4bcace8af9a4e37d00805ffd` and `modal-1e3c4eaf107a1411b87f6176` each completed and verified five native artifacts. The second passed fresh-process observe/show. Exact launcher/runtime/profile, input and source pins remain required. This is SFT fixture qualification, not arbitrary model/hardware or quality acceptance. |
| SFT Modal logs | Conditional implementation; restricted precompletion delivery proved | Both current SFT runs delivered two numeric records before completion. General transport and partial-write qualification remain false. Default logs stay disabled; only the existing exact reader-issued transport proof can enable its capability. A UI must tolerate missing progress and use verified terminal outcomes for completion. |
| SFT Modal cancel/reconcile/cost quote | Merged public configured lifecycle; default flags remain off | Initial recipe opt-in binds the descriptor before planning; current caller permission and the original authenticated submit/assessment/call target are required. Cancel acknowledgment and provider-input terminal observation are separate. Merged #246 permits known-run observation after an orphaned cancel receipt, preserves its unadvanced intent/history, and refuses cancel/reconcile replay. Neither observation nor acknowledgment proves physical quiescence. Quote/log/capability promotion and real final-source lifecycle parity remain separately gated. |
| SFT HF Jobs public API v1 | Implemented/installed; owner-paused qualification | Job `6ad7403a5275124786e158ba` ended ERROR with closed DISPATCH/23 (nonzero trainer subprocess). Cause remains unknown; retained bucket contains input only. No replay or cancellation occurred. The unused cancellation lane is not replacement-job authority. No new HF work is planned. HF model-repository publication is a separate operation and is not implicitly waived. |
| GRPO Modal public training route | Live-qualified, bounded on `ba228d6c` | Run `modal-10201ea947d40c7bffa0c92c`, call `fc-01M5JJG31MGCHJGRNW37BZSWSZ`, completed three updates, changed the aggregate 120-tensor parameter-set digest, and produced grouped rewards 0/1 with variance 0.25 over three nonconstant callbacks. Public verification and five delivered artifacts passed. This does not prove every tensor changed, writing quality, arbitrary recipes or execution of the later integrated closure. Other GRPO backends remain deferred. |
| Apple-silicon local training | Live-qualified, bounded native API v1 SFT fixture | Exact integrated `06e0fcf2` raw r69 and assistant-only r70 passed on M4/24GiB/macOS27.0.1 with the pinned private runtime/model and unchanged prepared publications. Each completed three updates and five verified/delivered MLX roles; independent saved-adapter changes and final reload losses passed. The public optional module and host-owned synchronous consumer are implemented. Real Mac51 process/host controls and three realMLX loss/autograd cases passed. Ordinary qualification remains independently authenticated. The explicit lifecycle selection has the separately scoped real-M4 evidence below; legacy lifecycle flags stay off. Checkpoint resume and broad model/OS support are unavailable. |
| Native MLX GRPO | One bounded M4 mechanics fixture at `ad17275a` | Fresh signed attempt completed three actual updates, three varied reward groups/48 calls and 102 generated tokens. All 40,960 saved parameters changed; final saved-adapter reload replay objective is finite. Public verification and five-role download/hash checks pass. See [exact report](../review/native-mlx-grpo-qualification.md). This historical run alone qualifies neither repeatability nor lifecycle. The later lifecycle fixture below adds repeated completion and scoped interruption evidence; ordinary authority and quality remain separate. |
| Native MLX public lifecycle | Bounded real M4 SFT/GRPO fixture at `e29c7da8`; independent authority required | Two completions per method, live numeric logs, confirmed public cancellation and abrupt caller-loss/cold reconciliation; one process claim per attempt. Four complete runs passed saved-adapter reload and five-role verification/delivery. Darwin CPU controls separately cover lost acknowledgements. See [exact scope](../review/native-mlx-lifecycle-qualification.md). Legacy synchronous flags stay off; no adoption, automatic replay, checkpoint resume or arbitrary model/OS claim. Final evidence review/current CI remain integration gates. |
| Training restart/recovery, all providers | Exact known-run cold reads; bounded Modal/Mac evidence | Fresh-process observation/show passed for the scoped Modal SFT evidence and Mac consumer. Both integrated Mac r69/r70 passed fresh-process observation, reverification and five-role redelivery after signed start expiry with original keys/catalogs and identical manifests. Use `runs.reverify` on already verified artifacts. Native active caller-loss/public cancellation now has the separately scoped lifecycle proof above; other providers and broad stable lifecycle acceptance remain open. No unknown-job adoption, automatic resubmission, arbitrary checkpoint continuation or restoration of a lost source lease is promised. HF qualification is paused. |

## Earlier exact live evidence

The current two-run Modal pair is recorded above; these older entries retain
their original scope and are not current-source capability grants.


| Evidence | Exact source and scope | What it does not establish |
| --- | --- | --- |
| Modal native training attempt H | Engine `b5794518f427521cdc4bfcf474fd0ad791b57445`; public observation and five verified artifacts. [Evidence](../review/modal-native-training-qualification.md) | New-source qualification, cancellation, recovery or inference/chat. |
| Restricted GPU metrics attempts 03/04 | `01383fa81c8ed3847903095456e9b80350cd28a6`; each 32 numeric records + fixture exhaustion, acknowledged report and five native artifacts. [Evidence](training-progress-streaming.md#accepted-restricted-delivery-evidence-2026-10-12) | General logs promotion, partial-write/freshness/completeness or current-source GPU qualification. |
| CPU SDK transport attempts 05/06 | `9252e924985b7bf41430e57eb49429f4501b59bf`; nine complete fixtures, pending exact-call/terminal-absent observations, collision preservation and post-completion final inventory, durable acknowledgment. [Evidence](training-progress-streaming.md#accepted-cpu-sdk-transport-evidence-2026-10-12) | GPU training, rename/background atomicity, general transport freshness/completeness, cancel or recovery. |

Failed GPU and CPU attempts retain their claims/resources and diagnostic limits
in the streaming evidence. No failure or historical success grants cleanup,
adoption, automatic replay or new provider authority.

## Source versus installed consumer

The base distribution metadata declares Python `>=3.10`; CI's Python 3.12 and
provider-free imports do not qualify every advertised Python/platform. Effectful
Modal launching requires exact POSIX CPython 3.11.14 / Modal 1.5.4 and its pinned
closure. Do not equate these with a universal library install requirement.

Declared public schemas, catalogs, selected runtime profiles and separate fixed
SFT/GRPO/HF source/dependency inventories are packaged and exercised through fresh
sdist-built wheels with installed-origin/byte checks. Base installs include
`jsonschema`; SDK/ML stacks remain explicit optional/pinned environments, not
implicit imports or runtime upgrades. CPU evidence uses supported 3.12 consumers
and selected real-seal/worker controls on 3.14; the portable SFT fixture deliberately
rejects 3.14. Neither that split evidence nor metadata proves all interpreters.

The `synaptic` console entrypoint is packaged. Host consumers under
`examples/nontraining` and `examples/training_qualification`, maintenance capture
scripts and source-owned image-build YAML are checkout inputs, not installed
engine Python. CI copies those declared consumers/tests/data while importing
production modules only from the wheel. Selected runtime YAML/inventory and
trainer schemas/manifests stay installed; packaging a model profile does not
qualify its image or GPU.

An installed Modal host can explicitly select an accepted clean engine source
through `SYNAPTIC_ENGINE_ROOT` / `ProjectContext.engine_root`. It builds that exact
source while retaining installed policy/inspector agreement, original source
intent and existing pushed/gitlink requirements where applicable. A wheel alone
is not a Git build checkout; cold known-run reads do not replan current recipes
or rebuild an image. The omitted source-build helper/schema/tool assets are now
included in the reviewed archive. An actual pinned offline build passes within
39 source controls, with 163 installed Modal controls on a fresh sdist-built wheel.
The archive correction is merged. Actual fresh local attempts C (`439608b6`) and
D (`4b77bd1f`) passed normal planning and source-wheel preparation without
provider effects; this is not augmented image or GPU qualification. Exact
launcher and provider image/child pins stay separate
from the library's broad Python metadata. Linux/POSIX descriptor/lock/guardian
operations and no local GPU are explicit boundaries, not Windows/macOS runtime
qualification. Final supported platform/asset/release artifacts remain a gate.

HTTP/SSE/browser transport, auth service, credentials, durable grants/stores and
product state belong to a consuming host; API v1 exports do not supply a
universally deployed service. Publication/recovery/concurrency need their own
host and provider evidence. Optional ingestion/chat/pipeline backends remain
explicitly host supplied and outside any automatic universal stability claim.

## Reviewed implementation references

- [Public exports](../../synaptic_tuner/api/v1/__init__.py) and [capability contract](../../synaptic_tuner/api/v1/providers.py).
- [Reference composition](../../synaptic_tuner/api/v1/reference/composition.py), [provider family](../../synaptic_tuner/api/v1/reference/provider_family.py), [run operations](../../synaptic_tuner/api/v1/reference/runs.py).
- [Docker composition](../../tuner/execution/providers/docker_provider_v1/composition.py) and [Modal preparation descriptor](../../tuner/execution/providers/modal/coordinator_adapter.py).
- [Conditional Modal host composition](../../tuner/training/modal_host_composition.py), [CPU qualification](../../tuner/training/modal_host_qualification.py).
- [Data modes/operations](../../synaptic_tuner/api/v1/data_facade.py), [reference data ports/execution](../../synaptic_tuner/api/v1/reference/data.py).
- [Reference evaluation backends/cancellation](../../synaptic_tuner/api/v1/reference/evaluation.py), [reporting](../../Evaluator/reporting.py), [comparison CLI](../../Evaluator/compare.py).
- [Reference artifact defaults](../../synaptic_tuner/api/v1/reference/artifacts.py), [public publication](../../synaptic_tuner/api/v1/artifacts_facade.py), [SFT method compiler](../../tuner/training/methods/__init__.py).

## Environment lifecycle scope

The selected implementation preserves normal DataAPI operations and existing host
record/stream stores. The proposed additional grant/episode contract framework and
parallel v2 outcome/result/list family were cancelled; they are not acceptance
requirements. Host-supplied policy and a session factory do not automatically
qualify restart, cancellation or safe continuation. Unknown active attempts retain
their uncertainty and cannot be replayed or falsely terminalized. The fixed worker guardian owns acknowledged local custody and cleans up only
after confirmed child exit. Pre-handoff owner loss, guardian/machine loss and
unconfirmed cleanup stay outside that guarantee; no durable controller or row
continuation is reconstructed from a PID.

### Historical local preparation checkpoint — 2026-10-14

The launcher closure at that checkpoint contained 40 exact hashed dependencies; the engine wheel
makes 41 installed distributions and `pip check` passes. Eleven source dependency
controls and 163 installed Modal controls are accepted. Actual private attempt C
on exact `439608b6` used genuine host Git/engine gitlink and the existing four-row
preparation service, then passed normal `--plan` and `--prepare-local` with a real
source wheel and nine bootstrap wheels. It resolved the former archive/dependency
setup defects without model downloads or provider effects. It does not qualify
the augmented image, GPU/model route or a later final source.

Attempt D repeated local preparation on exact `4b77bd1f`, then ran the bounded
normal CPU qualification in the selected Modal scope. It stopped with
`CAPTURE_OUTPUT / INSPECTOR_REJECTED` and no qualification receipt. The consumed
claim remains retained. The diagnostic identifies a boundary, not the underlying
cause or confirmed cleanup. No GPU training or model preparation was performed;
the augmented image remains unqualified.
