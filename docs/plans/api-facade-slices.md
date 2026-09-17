# Public API facade v1 — implementation slices

Ordered slices for [`../architecture/api-facade-v1.md`](../architecture/api-facade-v1.md).
Each is one coder in one session. Every slice leaves the tree green and both
import-closure gates passing: the package gate
(`tests/contract/test_public_runs_api_v1.py:279`) and the module + source gates
(`tests/contract/test_provider_neutral_foundation_v1.py:450-473`).

**Standing rule for every slice that adds a contract module.** Add the new file to
*both* lists in `test_provider_neutral_foundation_v1.py` — the direct-import list at
`:450-462` and the AST path list at `:465-473`. A contract file absent from both is
ungated and the suite still passes. This is the most likely miss in the phase.

**Standing rule for every slice.** No live provider call, no Modal, Docker or vLLM
execution, no paid backend. Evidence is fake-provider plus the conformance ladder.

---

## Slice 1 — Neutral redaction + contract primitives for observations and usage

- **Scope.** Promote `providers/modal/redaction.py:67` to
  `tuner/execution/redaction.py` (provider-neutral; Modal imports it from there).
  Add `api/v1/observations.py` (`ObservationRecordV1`, `ObservationStreamRef`,
  closed per-family kinds, `ObservationsRequest`, `ObservationPage`) and
  `api/v1/usage.py` (`UsageRecordV1`, `UsageAvailability`, `SpendRef`).
- **Files.** `tuner/execution/redaction.py` (new), `providers/modal/redaction.py`
  (deleted, imports updated), `synaptic_tuner/api/v1/observations.py`,
  `synaptic_tuner/api/v1/usage.py`, `schemas/synaptic-observation-v1.schema.json`,
  `schemas/synaptic-usage-v1.schema.json`, `api/v1/__init__.py` lazy table.
- **Tests.** New `tests/contract/test_public_observations_v1.py`: closed kind
  vocabulary, unknown kind rejected, `exact_fields` round trip, page
  `next_cursor`/`truncated` agreement, sequence strictly increasing. Both gate
  lists updated. Existing Modal redaction tests repointed.
- **Gate.** Redaction behaviour byte-identical before and after the move; new
  contract modules import no `tuner.*`.
- **Depends on.** Nothing.

## Slice 2 — Public host ports and in-memory reference stores

- **Scope.** Add `api/v1/ports.py` (`ClockPort`, `SecretResolverPort`,
  `GrantAuthorityPort`, `DurableRecordStorePort`, `DurableStreamStorePort`,
  `StoredRecordV1`, `StoredPageV1`, `StoredStreamPageV1`, closed partition
  vocabulary). Add `api/v1/reference/stores.py` with in-memory implementations of
  both store ports. **Delete `synaptic_tuner/host/v1/`.**
- **Files.** `api/v1/ports.py`, `api/v1/reference/__init__.py`,
  `api/v1/reference/stores.py`, `synaptic_tuner/host/` (removed),
  `tests/contract/test_public_publication_v1.py:421-429` (delete the superseded
  `host_v1` test), `test_provider_neutral_foundation_v1.py:457,468` (drop the
  `host.v1` import and path).
- **Tests.** New `tests/contract/test_public_host_ports_v1.py`: compare-and-swap
  rejects a stale revision; `put_if_absent` is idempotent; stream append rejects a
  non-monotone sequence; `list_page` honours limit and cursor; no `sqlite3` in
  `sys.modules` after importing the reference stores.
- **Gate.** Grep proves no remaining reference to `synaptic_tuner.host`; both gates
  green with the `host.v1` entries removed.
- **Depends on.** Slice 1.

## Slice 3 — `HostPorts` / `APIHost` widening

- **Scope.** `HostPorts` becomes the eight-field record; `APIHost.__init__(ports)`
  exposes one property per family. Family operations not yet written are typed as
  their Protocols and may be `None` **only** until their slice lands — a `None`
  slot raises on property access rather than returning a broken facade.
- **Files.** `api/v1/host.py`, plus every construction site: `test_public_training_api_v1.py:65,100`,
  `test_public_runs_api_v1.py:272-275`,
  `tests/execution/providers/test_modal_coordinator_composition.py:164`,
  `tests/inference/test_run_chat_consumer.py:139`, `examples/modal_chat/host.py:329`.
- **Tests.** Update the exact field-tuple pin at `test_public_training_api_v1.py:65`
  to the new eight-name tuple, in this commit. Add: accessing an unfilled family
  property raises, and `APIHost` never silently returns a facade over `None`.
- **Gate.** Full suite green. This slice changes no behaviour, only shape.
- **Depends on.** Slice 2.

## Slice 4 — Provider-neutral composition + Training/Runs reference export

- **Scope.** Add `api/v1/reference/provider_family.py` (`ProviderFamilyV1`,
  `ReferenceHostPortsV1`) and `compose_reference_host` in
  `api/v1/reference/__init__.py`, extracted from
  `coordinator_composition.py:74-264`. Move the Modal exact-type and
  retained-binding checks (`:96-131`) into a Modal family builder. Add engine
  adapters satisfying `PlanningStorePortV1`, `WorkflowStorePortV1`,
  `PreparationStorePortV1`, `ExecutionGrantStorePortV1`,
  `ReconciliationGrantStorePortV1` over `DurableRecordStorePort`. Export
  `CoordinatorTrainingService` via `api/v1/reference/training.py` and the renamed
  `RunOperationsV1` via `api/v1/reference/runs.py`. Compose
  `PublicationOperationsV1` + `StrongInMemoryPublicationStoreV1` into the
  `artifacts` slot (composition only — no destination adapter).
- **Files.** `api/v1/reference/{__init__,provider_family,training,runs,artifacts}.py`,
  `api/v1/reference/repositories.py`, `coordinator_v1/operations.py` (rename),
  `coordinator_composition.py`, `tuner/training/coordinator_service.py` (`__all__`),
  `docs/plans/submodule-first-training-product-roadmap-plan.md:116-117` (verb
  placement correction), `docs/architecture/submodule-first-training-v1.md:85-86`
  (Modal flag correction).
- **Tests.** New `tests/contract/test_reference_composition_v1.py`: compose over the
  fake provider family and run the full ladder — start, outcome, logs, verify,
  artifacts, cancel, reconcile; `"reference" not in _LAZY_MODULE_ATTRIBUTES`; no
  module under `api/v1/reference/` imports `api/v1/persistence.py`; no module under
  `api/v1/reference/` imports `ProjectContext`. Conformance suite and Docker
  composition tests updated for the rename.
- **Gate.** `compose_reference_host(family=fake, …)` passes every case the Modal
  composition passes, and `test_modal_coordinator_composition.py` still passes
  unchanged in substance.
- **Depends on.** Slice 3. **This is the reference composition landing first.**

## Slice 5 — EvaluationAPI contract

- **Scope.** `api/v1/evaluation_facade.py`: `EvaluationRunRef`, request/plan/
  preflight/start/outcome/page/result types, `EvaluationRunState`, `JudgeVerdict`,
  `ScoreV1`, `EvaluationOperationCode`, `EvaluationOperationError`,
  `EvaluationOperations` Protocol, `EvaluationAPI` with the `RunsAPI._call`
  detach-and-revalidate discipline.
- **Files.** `api/v1/evaluation_facade.py`, two `schemas/` files, lazy table, both
  gate lists.
- **Tests.** `tests/contract/test_public_evaluation_api_v1.py`, mirroring
  `test_public_runs_api_v1.py`: exact verb set; result binds its request; input
  immutable across return **and** raise; `MappingProxyType`, `dict` subclass and
  `str` subclass field names rejected with no user callback run.
- **Gate.** Contract-only; no implementation, no `tuner.*` import.
- **Depends on.** Slice 1.

## Slice 6 — EvaluationAPI reference implementation (local backends only)

- **Scope.** `api/v1/reference/evaluation.py` over `Evaluator/` scenario loading,
  scoring and judging. Results and traces go to the host's artifact sink, never
  `Evaluator/results/`. Zero effects for a local backend. Cooperative cancel at
  scenario boundaries. Progress as `evaluation_*` observations. A paid backend is
  refused with `backend_unmetered` (§9 of the architecture).
- **Files.** `api/v1/reference/evaluation.py`, `Evaluator/reporting.py` (add
  `schema_version`, drop `raw_response`/`conversation_trace` from the public
  projection), `api/v1/reference/__init__.py` (fill the `evaluation` slot).
- **Tests.** `tests/contract/test_reference_evaluation_v1.py`: a local-backend run
  produces a digest-verified artifact and zero effects; a paid backend is refused
  with `backend_unmetered`; cancel mid-run yields `cancelled` with the partial
  artifact listed; durable event budget ≤12; the public result contains no
  attacker-influenced text.
- **Gate.** No network call in the test path; no write outside the injected sink.
- **Depends on.** Slices 4 and 5.

## Slice 7 — Chat module relocation

- **Scope.** Move `Evaluator/chat_session.py`, `Evaluator/vllm_runtime.py` and
  `Evaluator/owned_process.py` to `tuner/inference/`. Update every import directly.
  No re-export shim (CLAUDE.md). Behaviour unchanged.
- **Files.** The three moved modules plus every importer, including
  `tuner/inference/model_chat.py:18-27`, `run_chat.py:10`,
  `providers/modal/inference_channel.py`, `tests/evaluator/`, `tests/inference/`.
- **Tests.** Existing tests repointed, none rewritten. Add an AST test that no
  module under `tuner/inference/` imports `Evaluator`.
- **Gate.** Pure move: the suite passes with zero assertion changes.
- **Depends on.** Nothing. Can run in parallel with slices 5 and 6.

## Slice 8 — ChatAPI contract and reference implementation

- **Scope.** `api/v1/chat_facade.py` (`ChatSessionRef`, `ChatTurnRef`,
  `ChatSession`, `ChatTurn`, `ChatSessionState`, `ChatOperationCode`,
  `ChatOperations`, `ChatAPI`) and `api/v1/reference/chat.py` over
  `open_model_chat` / `open_run_chat`. No streaming. `cleanup_unresolved` surfaces
  the unresolved owned-process family; a non-Linux host returns `host_unsupported`.
- **Files.** `api/v1/chat_facade.py`, two `schemas/` files,
  `api/v1/reference/chat.py`, `tuner/inference/model_chat.py` (replace the
  attach-attribute-to-exception failure signal with a typed result), lazy table,
  both gate lists.
- **Tests.** `tests/contract/test_public_chat_api_v1.py` (facade discipline) and
  `tests/contract/test_reference_chat_v1.py`: `request_id` strictly consecutive; a
  second turn while one is in flight raises `session_busy`; an unresolved lease
  yields `cleanup_unresolved` and refuses further turns; a simulated non-POSIX host
  yields `host_unsupported`; a session creates no lifecycle record.
- **Gate.** No vLLM process started in tests — the runtime is faked at the
  `RunChatRuntime` boundary.
- **Depends on.** Slices 4, 5 (for the shared observation shape) and 7.

## Slice 9 — DataAPI contract and reference implementation

- **Scope.** `api/v1/data_facade.py` and `api/v1/reference/data.py` over SynthChat
  generation, improvement, dataset listing and validation. One run, one artifact.
  `partially_succeeded` is terminal. **The `_meta` header row moves out of the data
  JSONL into a `dataset_metadata` sidecar artifact.** Local backends only; paid
  refused with `backend_unmetered`.
- **Files.** `api/v1/data_facade.py`, two `schemas/` files,
  `api/v1/reference/data.py`, `SynthChat/result_writer.py:38-54` (non-truncating
  open, metadata sidecar), lazy table, both gate lists.
- **Tests.** `tests/contract/test_public_data_api_v1.py` and
  `test_reference_data_v1.py`: the produced JSONL is homogeneous and its SHA-256
  matches the returned `VerifiedArtifact`; a short run terminates
  `partially_succeeded` with the artifact listed; `datasets`/`validate` take no run
  and page correctly. Existing `tests/synthchat/` writer tests updated for the
  sidecar.
- **Gate.** No `.env` walk, no ambient key read, no `sys.exit` on any path the
  facade reaches.
- **Depends on.** Slices 4 and 5.

## Slice 10 — PipelinesAPI contract and reference implementation

- **Scope.** `api/v1/pipelines_facade.py` and `api/v1/reference/pipelines.py`.
  Pipelines **reference** child runs. `PipelineState` has 8 values including
  `partially_succeeded` and `reconcile_required`. Stage handoff by
  `VerifiedArtifact`. Resume by `attempt_key`. Only `train` and `evaluate` are
  admitted; `loss`, `analysis`, `recommendation` are refused with
  `stage_unsupported`.
- **Files.** `api/v1/pipelines_facade.py`, `schemas/synaptic-pipeline-record-v1.schema.json`,
  `api/v1/reference/pipelines.py`, lazy table, both gate lists.
- **Tests.** `tests/contract/test_public_pipelines_api_v1.py` and
  `test_reference_pipelines_v1.py`: a failed evaluate stage leaves the train child
  `succeeded` and the pipeline `partially_succeeded`; `resume` skips a stage whose
  `attempt_key` matches and re-runs one whose input digest changed; a stage whose
  input digest mismatches raises `stage_digest_mismatch`; a spec naming `loss`
  raises `stage_unsupported`; the pipeline writes no `.tracking`.
- **Gate.** Interrupted-then-resumed train → evaluate converges over the fake
  provider with no duplicate child run.
- **Depends on.** Slices 4, 6 and 9.

## Slice 11 — LLM usage metering, and the paid arm

- **Scope.** Change `BaseLLMClient.chat` and `structured_output` to return
  `LLMCompletionV1{text, usage}` / `LLMStructuredV1{value, usage}` and update every
  call site in one commit. No dual signature, no shim. Wire `UsageRecordV1` into
  the Evaluation and Data results, claim the single `spend` effect per run, and
  lift the `backend_unmetered` refusal for metered backends.
- **Files.** `shared/llm/base.py`, every `shared/llm/` adapter, `shared/judge/`,
  `Evaluator/`, `SynthChat/`, `api/v1/reference/{evaluation,data}.py`,
  `foundation_v2/identities.py` (add `EffectKind.SPEND`).
- **Tests.** `tests/contract/test_reference_spend_v1.py`: exactly one `spend` effect
  per run regardless of call count; an indeterminate spend fails the run with
  `spend_indeterminate`; usage absent implies no spend effect and no `usage` field;
  a metered paid backend is now admitted. Existing `shared/llm` and judge tests
  updated for the return type.
- **Gate.** Every call site migrated — grep proves no caller treats a completion as
  `str`. Largest blast radius in the phase; nothing else depends on it.
- **Depends on.** Slices 6, 9 and 10.

---

## Order and parallelism

Strictly sequential: 1 → 2 → 3 → 4. After slice 4, slices 5 and 7 may run in
parallel; 6 follows 5; 8 follows 6 and 7; 9 follows 5; 10 follows 6 and 9; 11 is
last. Slices 1–4 deliver the reference composition of Training, Runs and Artifacts
before any new family starts, as required.

## Phase exit

The phase is complete when `compose_reference_host` over the fake provider family
drives all seven facades through the conformance ladder, both import-closure gates
pass, `synaptic_tuner/host/v1` is gone, and no module under `api/v1/reference/`
imports `api/v1/persistence.py`. No live provider effect is claimed; the frozen
matrix rows for Modal cancel/reconcile and for publication are untouched.
