---
status: release-preparation
baseline: df166a75
updated: 2026-10-13
---

# API v1 support and qualification matrix

This matrix inventories the selected stable-v1 scope at `df166a75`; it is not a
release certification. Subsequent work must update individual rows only after
implementation and exact-source acceptance. Environment rows include later
admission/snapshot, fixed-worker and installed public local workflow evidence
through runtime `8de9e430`. Complete durable recovery remains open. The [stable release plan](stable-v1-release.md)
is the acceptance authority. SFT requires Docker, Modal and HF Jobs parity;
GRPO selects Modal as the first candidate, still unqualified. DPO/KTO, RunPod,
Syntunia and broader GRPO provider parity are deferred.

Status vocabulary:

- **Contract-only:** typed surface or planned semantic operation without a
  usable accepted backend/composition for this claim.
- **Implemented/provider-free:** code and local/mock behavior exist; real backend,
  installed consumer and live lifecycle acceptance remain separate requirements.
- **Live-qualified, bounded:** evidence establishes the stated operation only on
  its exact source/configuration. It does not qualify current main automatically.
- **Unavailable/disabled:** absent implementation, missing required host port or
  capability refusal. Legacy CLI behavior is identified separately.

## Included workflow operations

| Family / operation | Status at this baseline | Backend evidence and stable gap |
| --- | --- | --- |
| Scenarios: author/edit/version | Existing config-first YAML/CLI authoring; complete public authoring operation unavailable | Scenario/rubric schemas and loaders exist. `DataAPI` consumes scenario refs; it is not a scenario CRUD/version authority. Define immutable public authoring/validation references and installed resources. |
| Scenarios: validate/select | Implemented/provider-free loaders and validation | Config/scenario validation does not establish semantic input coverage or permission for environment effects. Qualify actual installed assets and representative full scenarios. |
| Environments: config/create/validate | Reviewed admission, retained inputs and installed local host session composition | Optional `ReferenceDataPortsV1.environment_sessions` connects a host-configured factory to existing runtime methods. Public planning/preflight remain effect-free; selected execution requires that factory. No new environment facade or grant framework is required. |
| Environments: execute/tools/feedback | Installed public local generation/improvement and bounded worker cancellation verified | Runtime `8de9e430` exercises configured tools, fresh fixtures and candidate feedback through DataAPI. Actual stalled-worker cancellation verifies exit/cleanup; uncertainty stays nonterminal and repeated starts do not replay. Complete durable restart/reconciliation remains open. Simulation is explicit; no sandbox or universal wall-clock claim. |
| SynthChat: plan/preflight/generate | Implemented/provider-free public `DataAPI` + optional reference backend | Host supplies exact `ReferenceDataPortsV1` and LLM clients. Installed local environment workflows use deterministic injected clients; actual model/backend qualification remains open. |
| SynthChat: improve/judge | Improve mode implemented/provider-free; judge is configured workflow behavior, not a standalone `DataAPI` mode | `DataMode` exposes generate/improve. Judge/rubric client, usage and input lineage need bounded backend qualification; CLI availability is not a public judging-service guarantee. |
| SynthChat: export/datasets/validation | Implemented/provider-free result/dataset operations and validators | Bind immutable export, format, source coverage and split lineage; no manual relabeling. Installed source assets and exact downstream training identities must be checked. |
| SynthChat: usage/progress/cancel | Implemented/provider-free reference accounting/observations and cancellation state | Paid clients require authenticated spend scope; unmetered paid backends refuse. Cancellation state is not proof of interrupted in-flight work. Restart and real usage settlement remain open. |
| Text-conversation training preparation | Installed normalizer, independent verifier and immutable source extension | Declarative mapping/target/group splits preserve ordered text messages without invented ingestion ancestry. All 49 installed cases pass. SFT compilation/loading/masking/native verification also pass 47 installed CPU cases. Existing runtime release allowlists still reject the new format; additive candidate admission, the complete Data-to-training join and live qualification remain open. |
| Dataset validation | Implemented/provider-free `DataAPI.validate` and legacy validators | Real configured catalog/default assets must load in installed consumers; missing defaults cannot silently become empty schemas. Validation does not attest semantic provenance by itself. |
| Evaluation: plan/preflight/start/status/results | Implemented/provider-free optional `EvaluationAPI` reference composition | Requires explicit backend registry, scenario resolver, grants/spend and host stores. Local HTTP body limits default to 1 MiB each, host-configurable through 64 MiB; exact UTF-8 transport and retained report checks pass installed. Metered/unmetered paid adapters exist; neither a registry entry nor CPU transport tests qualify a live backend. |
| Evaluation: judge/usage | Configured backend support exists; selected stable paid judge unqualified | Unmetered paid execution refuses `backend_unmetered` before client opening. Select and qualify intended judge/backend accounting and bounded benchmark; do not bypass refusal. |
| Evaluation: reports/comparison | Installed public comparison of verified retained reports | Existing result requests select exact retained reports verified against stored run bindings; comparison reads survive SQLite reopening without backend execution or scenario rereads. Consumed prompt/scoring/generation controls determine comparability; missing or external judge/environment controls remain unverified. All 209 installed CPU cases pass, including existing renderers. This is configuration provenance, not remote model attestation or quality qualification. |
| Evaluation: progress/cancel/restart | Observations and cancel-request transitions implemented/provider-free | A cancel request is not terminal cancellation. Durable restart, in-flight interruption/reconciliation and backend conformance remain open. |
| Artifacts: run inventory/verified download | Implemented/provider-free; bounded historical Modal live proof | Public `RunsAPI` artifact streaming preserves per-file/aggregate bounds and end-to-end hashes. Historical five-file native proof is scoped below; new-source installed acceptance remains required. |
| Artifacts: destinations/publish/verify | Implemented/provider-free contracts; default reference publication unavailable operationally | Default composition uses empty destination registry, unavailable verified source, in-memory spool/publication store. Host-local and HF model-repository durable transfer/commit/verification must be supplied and qualified. |
| Artifacts: restart/ambiguous publication | Contract/mechanism pieces, complete supported durable host unavailable | Persist claim/transfer/commit/verified/ambiguous states, expected HF parent/pinned commit and exact reconciliation. HF bucket progress is not model publication; receipt loss cannot authorize duplicate writes/commits. |
| Host record/stream storage | Optional SQLite implementation; atomic Data/Evaluation/Pipeline/training-workflow head/history updates | [Host storage](../architecture/reference-host-storage.md) documents explicit composition, schema initialization, process-crash boundaries and reopened fake-provider host tests. Reopened records do not restore worker ownership or complete provider/publication recovery. |

## Training provider and lifecycle matrix

| Route / operation | Current support | Evidence and release requirement |
| --- | --- | --- |
| SFT generic plan/preflight/start | Implemented/provider-free public `TrainingAPI` and SFT compiler | Compile exact recipe/input/runtime identities, then host-owned grants and coordinator effects. Not every named recipe/model/hardware combination is qualified. |
| SFT Docker start/observe/artifacts | Implemented/provider-free explicit durable host composition | Scripted CLI tests exercise exact source staging, native verification and reopened SQLite reads/downloads with original material and keys. Real public SFT/GPU conformance remains open; use an authorized machine with usable GPU, not this server's failed GPU. |
| SFT Docker logs/cancel/reconcile | Explicit durable host has opt-in bounded cancellation; ordinary logs/reconcile unavailable | Exact-container stop requests retain one-shot claims and authenticated receipts. Stop acknowledgment does not prove causal cancellation; natural completion retains its actual result. The older same-process composition still rejects cancellation. Actual process-loss and live lifecycle parity remain open. |
| SFT Modal start/observe/verified artifacts | Implemented plus bounded exact-source live qualification | Historical native and restricted GPU pair verify reviewed runs; they do not establish final stable candidate or every recipe. Exact runtime/launcher/profile and source pins remain required. |
| SFT Modal logs | Conditional implemented gate plus bounded CPU transport qualification | Generic preparation descriptor has logs false. Supported host composition enables logs only with exact matching reader-issued CPU probe proof and durable capture. Historical success/JSON cannot mint new authority. |
| SFT Modal cancel/reconcile/cost quote | Public contract and internal mechanisms; ordinary operational descriptor disabled | Preparation descriptor advertises observe/artifact streaming, with cancel/reconcile/cost quote false. Existing quote/example lanes are not universal public capability. Stable cancellation/reconciliation and restart closure remain open. |
| SFT HF Jobs public API v1 | Implemented/provider-free explicit ordinary SFT composition | Public start/observe/native verification/download uses existing host stores and authority, pinned SDKs and the declared runtime profile. Initial native acceptance requires prepared v1 inputs up to 2 MiB and exactly five final roles without checkpoint retention. Controlled SDK/bootstrap/SQLite tests do not qualify live HF, GPU, cancellation, reconciliation or full restart recovery. Legacy/protected lanes remain separate. |
| GRPO Modal public training route | Selected candidate; public compiler/backend and live acceptance unavailable for this claim | GRPO trainers and environment-backed CLI/runtime assets exist, but `tuner.training.methods` exports the SFT compiler only. Bind explicit GRPO workload/reward/environment contract and qualify actual provider route. Candidate stack presence is not admission or success. |
| Training restart/recovery, all providers | Durable host stores and bounded retained reads implemented; complete recovery unavailable | Modal known-call tests include an actual killed host and fresh-process public readback; Docker/HF tests reopen SQLite with the same scripted provider and original material/keys. These establish their stated boundaries, not every crash point or live remote recovery. Preserve consumed uncertain submissions without replay or renewed authority. |

## Exact historical live evidence

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

Public schemas and workflow asset/catalog packaging are release preparation
work. PR #178 merged the public schema inventory; preparation source `268fc534`
passed checkout and sdist-built wheel asset checks plus a fresh base consumer in
a neutral working directory. Catalog resolution and evaluator startup are tested
without optional ML dependencies. Snapshot workflow source `f7d2df31` passed 793
combined checks, including retained-input generation and improvement after source
deletion. Fixed-worker source `27dde56e` adds accepted process/fixture primitives
and the permanent neutral installed smoke. Public local runtime `8de9e430` passed
916 combined tests with one intentional skip; its sdist-built wheel passed 51
session/public cases with the non-worker SIGSTOP variant skipped. Permanent CI
runs those tests with installed production modules and no checkout imports.
Durable worker restoration and resumable continuation remain unqualified.
Selected ML/GPU operations still require their real
dependencies. Source checkout success alone is not installed workflow acceptance.

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
their uncertainty and cannot be replayed or falsely terminalized. Fixed-worker
process descriptors are local custody, not a restored durable controller.
