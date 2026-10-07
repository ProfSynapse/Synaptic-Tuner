# PREPARE: engine reference implementations of `TrainingAPI`, `RunsAPI`, `ArtifactsAPI`

Status: preparation evidence only. No design decisions are made here; every
claim carries a `file:line` reference against the worktree
`feat/modal-coordinator-adapter`. Ruling basis:
[`api-facade-decisions-20260917.md`](api-facade-decisions-20260917.md) line 16,
which states the engine ships reference in-process implementations over
host-supplied ports so hosts no longer hand-write the operations. Headline
correction to that premise, established below: the engine already contains a
complete in-process implementation of all three facades' operations. What is
missing is export, a provider-neutral composition entry point, and default port
implementations.

---

## 1. Current contract surface

### 1.1 `TrainingAPI` — `synaptic_tuner/api/v1/training_facade.py`

| Element | Location | Notes |
|---|---|---|
| `TrainingRequest` | `training_facade.py:15-24` | `request_id`, `project_ref`, `canonical_json`; all `required_text` |
| `AuthorizationRequirement` | `training_facade.py:27-71` | `operation`, `paid_effect`, `maximum_cost_minor_units`, `currency`; amount and currency are supplied together, currency is an uppercase 3-letter code |
| `TrainingPreflight` | `training_facade.py:85-184` | schema `synaptic-training-preflight/v1`; `binds(plan)` at `:124`, `is_expired(now)` at `:129`; a not-ready preflight requires a diagnostic code (`:115`) |
| `TrainingStart` | `training_facade.py:187-196` | `run: TrainingRunRef`, `accepted: bool` |
| `TrainingOperations` Protocol | `training_facade.py:199-204` | `load`, `resolve`, `plan`, `preflight`, `start`; `Clock` Protocol at `:207` |
| `TrainingAPI` | `training_facade.py:211-250` | `load`/`resolve`/`plan` delegate unchecked; `preflight` and `start` re-check exact type, plan binding and expiry |

There is **no error-code enum** for training; failures are `TypeError` /
`ValueError` raised in the facade (`training_facade.py:229-249`). Planning types
live in `planning.py`: `ResolvedTrainingRequest` (`:12-66`, five SHA-256 digest
fields), `TrainingPlanBasisV1` (`:69-143`, `basis_digest` at `:141`),
`ProviderPlanContextV1` (`:146-190`), `ProviderPlanRef` (`:193-208`),
`TrainingPlan` (`:211-249`, schema `synaptic-training-plan/v2`,
`plan_fingerprint` at `:247`).

### 1.2 `RunsAPI` — `synaptic_tuner/api/v1/runs_facade.py`

| Element | Location | Notes |
|---|---|---|
| `RunOutcome` | `runs_facade.py:90-133` | schema `synaptic-run-outcome/v1` |
| `RunListRequest` | `runs_facade.py:136-145` | `limit` 1..100; ASCII cursor ≤256 bytes (`:77-87`) |
| `RunPage` | `runs_facade.py:148-170` | `next_cursor`/`truncated` must agree |
| `RunLogLevel` | `runs_facade.py:173-179` | trace, debug, info, warning, error, critical |
| `RunLogEntry` | `runs_facade.py:182-234` | message ≤4096 UTF-8 bytes; `size_bytes` must equal the measured length |
| `RunLogsRequest` | `runs_facade.py:237-250` | `limit` 1..200, `maximum_bytes` 4096..262144 |
| `RunLogPage` | `runs_facade.py:253-281` | sequences strictly increasing; `total_bytes` must equal the sum |
| `RunVerification` | `runs_facade.py:284-294` | RFC3339 `checked_at` |
| `RunArtifactRequest` / `RunArtifactStream` | `runs_facade.py:297-318` | stream exposes `run`, `artifact`, `maximum_bytes`, `iter_bytes()` |
| `RunOperationCode` | `runs_facade.py:321-333` | 13 closed codes: `run_missing`, `cursor_invalid`, `capability_unavailable`, `read_ineligible`, `provider_read_invalid`, `log_bounds_invalid`, `cancel_ineligible`, `artifacts_unverified`, `artifact_role_missing`, `artifact_limit_exceeded`, `artifact_content_invalid`, `state_conflict`, `integrity_error` |
| `RunOperationError` | `runs_facade.py:337-342` | `ValueError` carrying an exact `RunOperationCode` |
| `RunsOperations` Protocol | `runs_facade.py:345-354` | `list`, `show`, `outcome`, `logs`, `cancel`, `reconcile`, `verify`, `reverify`, `artifacts` |
| `RunsAPI` | `runs_facade.py:357-546` | rebuilds every request, passes a detached copy, and re-validates the original and the presented copy after the callback returns **or raises** (`_call` at `:402-418`) |

### 1.3 `ArtifactsAPI` — `synaptic_tuner/api/v1/artifacts_facade.py`

| Element | Location | Notes |
|---|---|---|
| `PublicationState` | `artifacts_facade.py:13-19` | claimed, transferring, committed, verified, ambiguous, failed_before_effect |
| `ArtifactDestination` / `DestinationPage` | `artifacts_facade.py:49-73` | destination refs unique and ascending |
| `PublicationRef` / `PublicationRequest` | `artifacts_facade.py:76-95` | |
| `PublicationResult` | `artifacts_facade.py:98-176` | schema `synaptic-publication-result/v1` |
| `PublicationPage` / `PublicationVerification` | `artifacts_facade.py:179-204` | |
| `ArtifactsOperations` Protocol | `artifacts_facade.py:207-211` | `destinations`, `publications`, `publish`, `verify` |
| `ArtifactsAPI` | `artifacts_facade.py:214-318` | same detach/rebuild/mutation-detection discipline as `RunsAPI` |

The publication error vocabulary is **not** in this module: `PublicationCodeV1`
and `PublicationErrorV1` are at `coordinator_v1/publication.py:45` and `:58`,
re-exported through `synaptic_tuner/api/v1/publication.py:27-28`.

### 1.4 Shared result types and composition root

- `TrainingRunState` (8 values), `TrainingRunRef`, `VerifiedArtifact`:
  `results.py:10-60`. `ProviderCapabilities`, six closed booleans `observe`,
  `logs`, `cancel`, `reconcile`, `artifact_streaming`, `cost_quote`:
  `providers.py:34-73`.
- `APIHost` and `HostPorts`: `host.py:53-71`. `HostPorts` carries **only**
  `runs: RunsOperations` and `clock: Clock`; `APIHost.__init__` takes
  `TrainingOperations` separately. There is no `ArtifactsAPI` slot.
- Canonical-JSON primitives in `_contract.py`: `required_text` (`:14`),
  `canonical_integer` (`:26`), `exact_integer` (`:37`), `digest_text` (`:45`),
  `exact_fields` (`:52`), `canonical_bytes` (`:73`, `sort_keys`, `(",",":")`,
  `ensure_ascii=False`, `allow_nan=False`), `contract_digest` (`:88`,
  domain-prefixed SHA-256). RFC3339: `_timestamps.py:15`. Lazy export table:
  `__init__.py:15-166`.

### 1.5 Host ports — `synaptic_tuner/host/v1/ports.py`

Eleven Protocols, no implementations (`__init__.py:1`): `Clock` (`:14`),
`AuthorizationGrant` (`:18`), `GrantAuthority` (`:23`), `EvidenceStore` (`:27`),
`SourceReader` (`:33`), `OpaqueProviderPreparation` (`:37`, `runtime_checkable`),
`PreparationRepository` (`:54`), `ProviderSession` (`:60`),
`ProviderSessionFactory` (`:65`), `RunOutcomeRepository` (`:71`),
`TrainingPlanRepository` (`:77`).

**Finding.** No engine module imports `synaptic_tuner.host.v1`. These ports are
declared but unconsumed; the live composition uses the internal
`tuner/execution/coordinator_v1/ports.py` set instead (§2.2).

## 2. How the reference host composes the engine today

`examples/host-project/` contains **no Python composition** — it is
`synaptic.yaml`, `experiments/*.yaml`, `providers/modal-a10-v1.yaml` and
`plugins/example.py`. The hand-written host composition the ruling refers to is
`examples/modal_chat/` (≈6,089 lines across 21 modules).

### 2.1 Composition root

`examples/modal_chat/host.py:159-342` (`compose_modal_chat_host`) builds every
port, then calls `compose_modal_coordinator`
(`tuner/execution/providers/modal/coordinator_composition.py:74-264`) and
finally `APIHost(composed.training, HostPorts(composed.runs, clock))` at
`host.py:329`.

Engine modules the host imports directly (`host.py:8-39`):
`synaptic_tuner.api.v1.host`; `coordinator_v1.cursors.HMACCursorAuthorityV1`,
`coordinator_v1.foundation.FoundationRecordAssessmentAuthorityV1` and the four
`coordinator_v1.stores` in-memory stores; `foundation_v2` `GrantAuthorityV2`,
`InMemoryEffectRepositoryV2`, `ReceiptAuthorityV2`,
`InvalidEvidenceAuthorityV2`, `ProviderReaderFactoryRequestV1`; and five
`providers.modal.coordinator_*` modules.

### 2.2 Training path: load → resolve → plan → preflight → start

The host **does not bypass the facade**. `examples/modal_chat/consumer.py:83`
(`submit_training_once`) takes a real `TrainingAPI` and calls
`load`/`resolve`/`plan`/`preflight`/`start` at `consumer.py:117-127`, claiming a
durable attempt at `consumer.py:106-115` *before* provider planning.

| Step | Facade | Engine implementation |
|---|---|---|
| `load` | `training_facade.py:218` | `CoordinatorTrainingService.load` `tuner/training/coordinator_service.py:57` → host `ModalChatTrainingRequests` (`examples/modal_chat/requests.py:26`) |
| `resolve` | `training_facade.py:221` | `coordinator_service.py:65` → same host object as resolver (`host.py:311-312`) |
| `plan` | `training_facade.py:224` | `coordinator_service.py:87`; calls `PlanningPortV1.describe`/`.context` on `ModalOperationalPreflightAdapter`, then `put_context_if_absent`/`put_plan_if_absent` on the host planning store |
| `preflight` | `training_facade.py:227` | `coordinator_service.py:116` → `ModalOperationalPreflightAdapter.preflight`; issues `AuthorizationRequirement("training.start", True, …)` at `tuner/execution/providers/modal/coordinator_preflight.py:228` |
| `start` | `training_facade.py:239` | `coordinator_service.py:123` → `TrainingCoordinatorV1.start` (`tuner/execution/coordinator_v1/coordinator.py`), returning `TrainingStart(workflow.run, True)` |

`CoordinatorTrainingService` (`coordinator_service.py:34`) is therefore the
engine's existing `TrainingOperations` implementation, deliberately not exported
(`__all__: list[str] = []` at `:139`).

### 2.3 Runs path

`TrainingOperationsV1` (`tuner/execution/coordinator_v1/operations.py:205`) is
**misleadingly named**: its nine public methods are exactly the `RunsOperations`
verbs — `list` `:359`, `show` `:383`, `outcome` `:386`, `logs` `:409`,
`cancel` `:483`, `reconcile` `:506`, `verify` `:526`, `reverify` `:556`,
`artifacts` `:590`. It is constructed at
`coordinator_composition.py:254-260` and handed to `HostPorts` as `runs`.
`operations.py:1` states it is "deliberately not package-exported".

Capability gating inside it: `observe` `:391`, `logs` `:415`, `cancel` and
`reconcile` `:486-487`, `reconcile` again `:509`, `artifact_streaming` `:529`,
`:559`, `:595`; a missing capability raises
`RunOperationCode.CAPABILITY_UNAVAILABLE` (`operations.py:302-305`). Host call
sites: `launch.py:663` (`runs.outcome`), `:669` (`runs.verify`), `:466` and
`chat.py:254` (passing `host.api.runs` into the chat composition). Engine-internal
consumers: `tuner/inference/retrieved_model.py:824-842` and
`providers/modal/inference_binding.py:519-534`.

### 2.4 Artifacts / publication path

`PublicationOperationsV1` (`coordinator_v1/publication.py:2128`) is a complete
`ArtifactsOperations` implementation — `destinations` `:2454`, `publications`
`:2474`, `publish` `:2493`, `verify` `:2591`. Unlike the other two it **is
already publicly re-exported**, through `synaptic_tuner/api/v1/publication.py:10-45`,
with its ports (`DestinationPublicationPortV1` `:812`,
`ArtifactDestinationRegistryPortV1` `:823`, `PublicationStorePortV1` `:1581`,
`ArtifactSpoolPortV1`, `SpoolSinkPortV1`, `VerifiedArtifactSourcePortV1`,
`EvidenceAuthorityPortV1`) and `StrongInMemoryPublicationStoreV1` (`:1945`).

**Finding.** No `DestinationPublicationPortV1` implementation exists anywhere on
this branch: a repo-wide search returns only the port definition, the public
re-export and two test files. There is no local-filesystem and no Hugging Face
destination adapter, and the host does not compose `ArtifactsAPI` at all.

### 2.5 Generic vs host-specific in the hand-written host

GENERIC — no host-specific input beyond a key or a clock, so the engine could own
them: `UTCClock` (`authority.py:39`); `PlanningStore` (`:182-233`, the only
`PlanningStorePortV1` implementation outside tests); the publish-if-absent maps
`CanonicalCatalog` (`:235`) and `_ExactCatalog` (`host.py:63`);
`HMACAuthenticator` (`:54`), `ReaderEvidenceAuthority` (`:92`),
`ObservationAuthenticator` (`:166`), `LogAuthenticator` (`:174`);
`FoundationAuthenticator` (`:307`), which only wires three engine authorities;
`RetainedBindingAuthority` (`:276`), `RetainedInputs` (`:292`); and the denying
stubs `UnavailableRecoveryVerifier` (`:415`) and
`UnavailableQuiescenceEvidence` (`:423`).

HOST-SPECIFIC — the four categories the ruling reserves for hosts:
`BoundedAuthorization` (`authority.py:321`) and `_AuthorizationSlot`
(`host.py:94`) encode one approved cost and currency (`host.py:112-116`) and
refuse reconciliation grants outright (`host.py:140-141`);
`ModalChatTrainingRequests` (`requests.py:26`) is loader, resolver and
run-identity over host config; `ModalChatRichTrainingResolver`
(`resolution.py:116`) reads the host source tree; `ModalChatArtifactVerifier`
(`artifacts.py:27`) binds a host key and a provider reader; `ModalChatStorage`
(`storage.py:221`), `SQLitePublishIfAbsentCatalog` (`:121`) and
`SQLiteOneShotAttemptStore` (`:193`) are SQLite-backed (`:276`);
`ModalChatEvidenceReplay` (`replay.py:14`) is durable replay admission.

### 2.6 Where host state lives

The host writes only through `ModalChatStorage` (`storage.py:221`), a private
SQLite database opened at `:276` with `isolation_level=None` and a private-path
check at `:74`. The coordinator and Foundation stores are **all in-memory**
(`host.py:266-277` and `:202`), so nothing restores them after a process exit.
`AGENTS.md:111` says so explicitly: the attempt journal "is not durable recovery
of all coordinator/Foundation state".

## 3. Capability matrix versus the code on this branch

The frozen matrix in
[`submodule-first-training-v1.md`](../architecture/submodule-first-training-v1.md)
lines 58-70 is pinned to engine commit `31d2683…`. Against the current branch:

| Matrix row | Matrix status | What the branch actually contains |
|---|---|---|
| Operational `RunsAPI` cancel/reconcile in the reference host | `CONTRACT_ONLY` | The **implementation exists and is exercised**: `operations.py:483` and `:506` run against the fake provider ladder in `tests/execution/test_fake_provider_v1_conformance.py:576`, `:599`, `:894`, `:912`. Two things block it in the reference host: the Modal descriptor declares `cancel=False, reconcile=False` (`coordinator_adapter.py:50`), and `_AuthorizationSlot.issue_reconciliation_grant` raises unconditionally (`host.py:140-141`) |
| Local final publication | `LIVE_PROVEN` | Not reproducible here. `PublicationOperationsV1` exists but no local `DestinationPublicationPortV1` adapter does (§2.4) |
| Hugging Face final publication | `IMPLEMENTED_FAKE_TESTED` | Same: no HF destination adapter on this branch |
| Local Docker through the public training API | `NOT_IMPLEMENTED` | Confirmed. `compose_docker_same_process_coordinator_v1` (`tuner/execution/providers/docker_provider_v1/composition.py:366-428`) returns `_DockerSameProcessRuntimeV1`, a coordinator wrapper. It never builds `CoordinatorTrainingService` or `TrainingOperationsV1`, so no `TrainingAPI`/`RunsAPI` reaches Docker |
| HF Jobs / RunPod through the public training API | `NOT_IMPLEMENTED` | Confirmed; no provider modules |
| KTO/DPO/embedding/GRPO through the public training API | `NOT_IMPLEMENTED` | Confirmed |
| Evaluation and pipelines through public APIs | `NOT_IMPLEMENTED` | Confirmed; `EvaluationAPI` and `PipelinesAPI` do not exist |

Two further drifts between the documents and the code:

1. The checkpoint note at `submodule-first-training-v1.md:85-86` says the Modal
   descriptor's "six advertised … flags remain false". The code declares
   `ProviderCapabilities(True, False, False, False, True, False)` at
   `coordinator_adapter.py:50`: `observe` and `artifact_streaming` are true.
2. The roadmap at
   `submodule-first-training-product-roadmap-plan.md:116-117` assigns
   `outcome`, `verify` and `reverify` to `TrainingAPI`. The code places all
   three on `RunsAPI` (`runs_facade.py:348`, `:352`, `:353`), and a contract
   test pins exactly that split (`tests/contract/test_public_runs_api_v1.py:49`).

### 3.1 What a reference implementation would have to write versus compose

**Composable from existing code:** the three operations implementations
(`coordinator_service.py:34`, `operations.py:205`, `publication.py:2128`), the
four in-memory coordinator stores, `InMemoryEffectRepositoryV2`,
`HMACCursorAuthorityV1`, `FoundationRecordAssessmentAuthorityV1`,
`GrantAuthorityV2`, `ReceiptAuthorityV2`, `InvalidEvidenceAuthorityV2`,
`StrongInMemoryPublicationStoreV1`, and the fake provider family.

**Must be newly written:** a provider-neutral composition function, since the
only one is Modal-specific and names `ModalPreparationAdapter`,
`ModalOperationalPreflightAdapter`, `ExplicitModal154ReadFacade` and
`VerifiedModalDeploymentIdentityV1` as exact types
(`coordinator_composition.py:96-111`); a `PlanningStorePortV1` implementation;
default authenticator and catalog objects equivalent to the GENERIC list in
§2.5; at least one `DestinationPublicationPortV1` adapter if `ArtifactsAPI` is
to be demonstrable; and an `ArtifactsAPI` slot if it is to hang off `APIHost`.

## 4. Persistence

### 4.1 What `synaptic_tuner/api/v1/persistence.py` gives

It is a pure re-export surface (`persistence.py:1-62`) over
`tuner/execution/contracts.py`, `tuner/execution/operation.py`,
`tuner/execution/evidence.py` and `tuner/execution/lifecycle.py`. It exposes the lifecycle record model (`LifecycleRecord`, `LifecycleEvent`,
`LifecyclePhase`, `EventCode`, `MessageCode`), the effect model (`EffectRecord`,
`EffectIdentity`, `EffectKind`, `EffectState`, `EffectDisposition`,
`EffectObservation`, `GrantBinding`, `ExecutionScope`), `AttemptAdmission`,
`VerificationStatus`, `LifecycleRepository`, `EvidenceReplayRepository`,
`OperationBindingV1`, `apply_lifecycle_event`, and six error types.
`LifecycleRepository` (`tuner/execution/contracts.py:776-808`) is a six-method
Protocol: `create`, `load`, `append`, `compare_and_consume_attempt`,
`record_attempt_outcome`, `list_runs`.

**Critical finding.** This model is **not** used by the coordinator path. A
search for imports of `tuner.execution.contracts` inside
`tuner/execution/coordinator_v1/` and `tuner/execution/foundation_v2/` returns
nothing; the only consumers are the legacy `tuner/execution/service.py:9` and
`tuner/execution/broker.py:113`, plus test fakes
(`tests/execution/fakes.py:9`). The live v1 lifecycle uses `WorkflowRecordV1`
(`coordinator_v1/model.py`) behind `WorkflowStorePortV1`
(`coordinator_v1/ports.py:71-89`) and `EffectRecordV2`
(`foundation_v2/repository.py`). A reference implementation built on the
coordinator therefore inherits **none** of the public `persistence` types.

### 4.2 Repository ports a reference implementation must satisfy

From `tuner/execution/coordinator_v1/ports.py`: `PlanningStorePortV1` (`:64`,
4 methods), `WorkflowStorePortV1` (`:71`, 6 methods including
`compare_and_swap`), `PreparationStorePortV1` (`:97`),
`ExecutionGrantStorePortV1` (`:102`), `ReconciliationGrantStorePortV1` (`:112`),
`CursorAuthorityPortV1` (`:92`). Publication adds `PublicationStorePortV1`
(`publication.py:1581`).

Existing in-memory implementations: `InMemoryWorkflowStoreV1` (`stores.py:228`),
`InMemoryPreparationStoreV1` (`:565`), `InMemoryExecutionGrantStoreV1` (`:644`),
`InMemoryReconciliationGrantStoreV1` (`:850`),
`StrongInMemoryPublicationStoreV1` (`publication.py:1945`). The gap is
`PlanningStorePortV1`. The engine imports **no `sqlite3` at all** across
`tuner/`, `synaptic_tuner/` and `shared/`, so a file-backed reference repository
would be the first persistent storage the engine ships — which touches
`AGENTS.md:134` ("the consuming host owns … coordinator/Foundation persistence")
and `AGENTS.md:133` ("never … an engine-owned database").

### 4.3 The `<project>/.synaptic` carve-out

`ProjectContext.host` (`tuner/project/context.py:75-89`) fixes
`mutable = project_root / ".synaptic"` with five roots: `artifacts`, `state`,
`tracking`, `cache`, `tmp`. `_validate_context_roots` (`manifest.py:240-268`)
rejects any writable root inside the engine checkout or not strictly below
`.synaptic`. Contract tests assert that discovery and manifest loading **do not
create** `.synaptic` (`test_embedded_host.py:38`, `:57`, `:87`) and that
engine-managed writes reject `engine://`, `project://` and `config://` with
`PROJECT_WRITE_DENIED` (`test_engine_read_only.py:33-47`).

## 5. Testing conventions to follow

**Facade contract tests** live in `tests/contract/`: `test_public_training_api_v1.py`,
`test_public_runs_api_v1.py`, `test_public_publication_v1.py`, plus
`test_provider_neutral_foundation_v1.py` for the planning/preflight chain. Each
defines a local `class Operations:` inside the test body and asserts the object
it receives is **not** the object passed in
(`test_public_runs_api_v1.py:78`, `:82`, `:87`). Representative signatures:
`test_runs_api_has_only_the_accepted_verbs()` (`test_public_runs_api_v1.py:49`);
`test_runs_facade_rejects_presented_input_mutation_on_return_and_raise(verb, raises)`
(`:141`); `test_unstartable_preflight_never_reaches_operations(mode)`
(`test_public_training_api_v1.py:109`, parametrized
`["wrong_plan", "expired", "not_ready"]`);
`test_artifacts_api_rejects_callback_destination_drift()`
(`test_public_publication_v1.py:326`). Invariants: exact verb sets; result-to-request
identity binding; input immutability across both the return and the raise path;
`type(x) is T` rather than `isinstance`; rejection of `MappingProxyType`, `dict`
subclasses and `str` subclasses as field names, asserting no user callback ran
(`test_public_runs_api_v1.py:236`, `:247`).

**Example tests** — 22 files in `tests/examples/` over `examples/modal_chat`.
Facade-level: `test_modal_chat_consumer.py:91` (claim precedes provider
planning), `:125` (a not-ready preflight never reaches paid start);
`test_modal_chat_host.py:114`; `test_modal_chat_public_reverification.py:37`,
which builds a real `TrainingOperationsV1` at `:56` and `RunsAPI` at `:76`.
Where only the facade contract matters, `RunsAPI(object())` is passed
(`test_modal_chat_consumer.py:193`).

**Fakes** — `tuner/execution/fake_provider_v1.py` is a full provider family:
`FakeEffectExecutorV1` (`:228`), `FakeReconciliationAdapterV1` (`:258`),
`FakeProviderEvidenceAuthorityV1` (`:291`), `FakeProviderRunReaderV1` (`:345`),
`FakeProviderReaderFactoryV1` (`:505`), `FakeArtifactVerifierV1` (`:532`),
`FakeProviderFamilyV1` (`:568`), scripted by `FakeEffectScriptV1` (`:65`).
`tests/execution/test_fake_provider_v1_conformance.py` is the conformance
ladder: it composes `TrainingOperationsV1` at `:506` and runs start, outcome,
logs, verify, artifacts, cancel, reconcile, concurrency and forgery cases
(`:548`-`:1144`). It is the closest existing analogue to a reference harness.
`tests/execution/fakes.py:9` covers the legacy lifecycle only.

**Canonical JSON** is enforced structurally by the contract types, not by a
dedicated `_contract.py` test module: `exact_fields` rejects unknown and missing
keys with a sorted diagnostic (`_contract.py:52-70`), `canonical_bytes` refuses
NaN/Infinity (`:73-85`), `contract_digest` prefixes a domain and a NUL byte
(`:88-90`). Digest stability is asserted through `plan_fingerprint` and
`basis_digest` equality checks in `coordinator_service.py:74-113`.

**Redaction** — `tuner/execution/providers/modal/redaction.py:67` returns one
bounded JSON string and never returns the input on failure (`:90-91` yields
`"[REDACTED:ERROR]"`). Two structural passes (`:72-85`) apply a key regex
(`:7-11`) and six text patterns (`:12-23`) under depth, item and byte bounds.
Host diagnostics use a closed phase/class/location vocabulary
(`examples/modal_chat/diagnostics.py:145`, path admission at `:106`);
`AGENTS.md:57-60` forbids emitting exception text, locals or tracebacks.

**Import closure** — `test_public_runs_api_v1.py:279`
(`test_public_v1_import_does_not_load_provider_or_database_modules`) spawns a
subprocess, imports `synaptic_tuner.api.v1`, and asserts that none of
`huggingface_hub`, `modal`, `runpod`, `sqlite3`, `tuner` appear in
`sys.modules`. That is why `api/v1/__init__.py:156-162` uses a lazy
`__getattr__`, and it is the hard constraint on where a reference
implementation may be exported from.
`test_public_training_api_v1.py:127` repeats the check for training symbols.
`test_modal_optional_dependency.py:46` asserts at AST level that no module under
`tuner/execution/providers/modal/` imports `modal` at module scope; `:18` pins
the `modal` extra to `modal==1.5.4` and the launcher lock to 37 hashed lines;
`:57` and `:64` assert the SDK is never materialized.

**Running the suite** — `pytest.ini` declares `testpaths = tests`, one marker
`integration`, and `addopts = -v --tb=short`. The "clean SDK-free selection"
referenced by `submodule-first-training-v1.md:82` is not encoded as a pytest
marker anywhere in the repo; it is a documented count (2,043 tests), not a
runnable selector.

---

## 6. Open questions for the architect

1. **Export site.** `CoordinatorTrainingService` and `TrainingOperationsV1` both
   end in `__all__: list[str] = []` (`coordinator_service.py:139`,
   `operations.py:624`), and `test_public_runs_api_v1.py:279` forbids `tuner.*`
   in `sys.modules` after importing `synaptic_tuner.api.v1`. Does the reference
   implementation live behind a lazily-imported module that mirrors
   `persistence.py` and `publication.py`, or behind a separate package?

2. **Naming.** `TrainingOperationsV1` implements `RunsOperations`, not
   `TrainingOperations`. Is it renamed here, and does the rename propagate to
   `coordinator_composition.py:61`, the Docker tests and the conformance suite
   in one commit?

3. **Persistence ownership.** `AGENTS.md:133-134` says the host owns
   coordinator and Foundation persistence and forbids an engine-owned database;
   the ruling says the engine ships reference implementations. Does an
   engine-shipped **in-memory** repository satisfy both, and is a file-backed
   one in scope at all, given the engine imports `sqlite3` nowhere?

4. **Planning store.** `PlanningStorePortV1` has no engine implementation, only
   `examples/modal_chat/authority.py:182`. Does the engine ship an in-memory
   planning store beside the four existing coordinator stores, or is planning
   state folded into an existing port?

5. **Composition shape.** `HostPorts` (`host.py:53-56`) carries only `runs` and
   `clock`, and `APIHost` has no `ArtifactsAPI`. Does the reference composition
   extend `HostPorts`, or return a record like `ModalCoordinatorComposition`
   (`coordinator_composition.py:58-64`)?

6. **Artifacts reachability.** `ArtifactsAPI` has a complete operations
   implementation but zero `DestinationPublicationPortV1` adapters on this
   branch. Is a local-filesystem destination adapter in scope now, or does
   `ArtifactsAPI` ship composition-only until Phase 5?

7. **Extraction order.** `compose_modal_coordinator`
   (`coordinator_composition.py:74`) hard-types four Modal classes at `:96-111`
   and duck-checks 24 ports at `:137-173`. Is the provider-neutral composition
   extracted from it, or written fresh against the fake provider with Modal
   retrofitted?

8. **Evidence bar.** Modal declares `cancel=False, reconcile=False`
   (`coordinator_adapter.py:50`) and the host denies reconciliation grants
   (`host.py:140-141`). Does clearing the `CONTRACT_ONLY` matrix row require
   flipping those flags and a real grant path, or is a fake-provider proof
   sufficient for this phase?

9. **Verb placement.** The roadmap puts `outcome`/`verify`/`reverify` on
   `TrainingAPI` (`submodule-first-training-product-roadmap-plan.md:116`); the
   code and a pinned test put them on `RunsAPI`. Which document is corrected?

10. **`host/v1` status.** Its eleven Protocols are imported by no engine module
    and do not match `coordinator_v1/ports.py`. Do the reference
    implementations consume them, which means adapting them to the
    coordinator's shapes, or is `host/v1` superseded and due for revision?

11. **Legacy record model.** `persistence.py` exposes a lifecycle/effect model
    the coordinator path does not use (§4.1). Must the reference implementation
    populate those records, or is that surface retired under Phase 10?
