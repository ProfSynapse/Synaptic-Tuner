# Public API facade v1 — architecture

Status: ARCHITECT output for the public API facade phase. Design decisions, not
evidence; every code reference was read on `feat/modal-coordinator-adapter` and
carries a `file:line`. Binding inputs:
[`../plans/api-facade-decisions-20260917.md`](../plans/api-facade-decisions-20260917.md) (operator),
[`../plans/api-facade-preparation-reference-impls.md`](../plans/api-facade-preparation-reference-impls.md),
[`../plans/api-facade-preparation-new-families.md`](../plans/api-facade-preparation-new-families.md).
Implementation order: [`../plans/api-facade-slices.md`](../plans/api-facade-slices.md).

## 1. Executive summary

Seven facades, one composition function, one new port set. The engine already has
complete in-process implementations of Training, Runs and Artifacts; this phase
exports them behind a provider-neutral composition and adds four new families.

Three choices carry the design:

1. **Progress leaves the lifecycle record.** `LifecycleEvent` gains no payload
   slot. A parallel, append-only, typed-per-family **observation stream** keyed by
   entity id carries every progress payload, while the durable record stays coarse
   under a declared per-family event budget. That is the direct answer to the O(n²)
   re-serialisation limit.
2. **The host implements two storage ports, not eleven repositories.** A host
   supplies an opaque canonical-bytes record store with compare-and-swap and an
   append-only stream store; engine repositories are adapters over those. The
   engine ships in-memory defaults and opens no database, satisfying both the
   operator ruling and `AGENTS.md:134`.
3. **Each family gets its own exact reference type.** `EvaluationRunRef`,
   `DataRunRef`, `ChatSessionRef` and `PipelineRef` are structurally identical to
   `TrainingRunRef` but distinct exact types, so cross-family confusion becomes a
   `TypeError` at the boundary rather than a lookup miss. The store partition, not
   a discriminator field, is the run kind.

Nothing here performs a live provider effect. The evidence bar is the fake-provider
family (`tuner/execution/fake_provider_v1.py:568`) and the conformance ladder
(`tests/execution/test_fake_provider_v1_conformance.py`).

## 2. System context

```text
Syntunia (not a consumer in this phase)
  -> host service: HTTP/SSE, auth, credentials, durable storage
  -> synaptic_tuner.api.v1            contracts: no tuner.*, no SDK, no sqlite3
  -> synaptic_tuner.api.v1.reference  composition + operations
  -> tuner.execution.coordinator_v1 / foundation_v2   (internal)
  -> fake | docker | modal            (provider families)
```

The host service is an adapter, never a second engine: its route handlers take a
request type, call one facade method, and serialise the result with `to_dict()`. No
HTTP framework enters the engine dependency closure. Server-Sent Events, if the
host offers them, are the host replaying an observation page it polled — the engine
still has no push stream.

## 3. Package layout and export sites

### 3.1 The two import-closure gates

| Gate | Location | What it asserts |
|---|---|---|
| Package gate | `tests/contract/test_public_runs_api_v1.py:279` | after `import synaptic_tuner.api.v1`, none of `huggingface_hub`, `modal`, `runpod`, `sqlite3`, `tuner` is in `sys.modules` |
| Module gate | `tests/contract/test_provider_neutral_foundation_v1.py:450-462` | after importing seven **named contract submodules directly** plus `synaptic_tuner.host.v1`, the same set is absent |
| Source gate | `tests/contract/test_provider_neutral_foundation_v1.py:465-473` | an AST scan of those same files finds no import of `tuner`, `modal`, `sqlite3` |

Laziness satisfies the package gate (`api/v1/__init__.py:156-166`), which is why
`persistence.py` may import `tuner.execution.contracts` at module scope. Only a file
importing no `tuner.*` satisfies the module gate, so every new contract module must
be added to both lists in `test_provider_neutral_foundation_v1.py`.

### 3.2 Layout

```text
synaptic_tuner/api/v1/
  ports.py             NEW  host port Protocols, contract-typed only
  observations.py      NEW  ObservationRecordV1, ObservationPage, closed kinds
  usage.py             NEW  UsageRecordV1, UsageAvailability, SpendRef
  evaluation_facade.py chat_facade.py data_facade.py pipelines_facade.py   NEW
  host.py              REVISED  HostPorts, APIHost
  reference/           NEW SUBPACKAGE — implementations, never lazily exported
    __init__.py         compose_reference_host, ReferenceComposition
    provider_family.py  ProviderFamilyV1
    stores.py           in-memory DurableRecordStore / DurableStreamStore
    training.py runs.py artifacts.py evaluation.py chat.py data.py pipelines.py
```

**Export site ruling (open question 1).** The reference implementations are **not**
added to `_LAZY_MODULE_ATTRIBUTES` (`api/v1/__init__.py:15`). They follow the
precedent of `api/v1/docker.py` and `api/v1/modal.py`: physically inside the
package, importable only by explicit submodule path, invisible to `from
synaptic_tuner.api.v1 import *`. A host writes `from synaptic_tuner.api.v1.reference
import compose_reference_host`. A contract test pins the exclusion — `"reference"
not in _LAZY_MODULE_ATTRIBUTES`, no `compose_*` in `_FORMAL_EXPORTS` — so nobody can
quietly add it and break the package gate.

**Naming (open question 2).** The operations class in
`coordinator_v1/operations.py:205` implements `RunsOperations`, not
`TrainingOperations`, and its former `Training…OperationsV1` name was misleading.
It is renamed `RunOperationsV1` in the slice that exports it,
propagated in the same commit to `coordinator_composition.py:61,254`, the Docker
composition tests and the conformance suite. Exporting a misleadingly named class
is worse than a rename.

## 4. Progress, the lifecycle record, and effects

### 4.1 The constraint

`LifecycleEvent` (`tuner/execution/contracts.py:518-550`) accepts a code, a
timestamp, a mapped message code, and optionally exactly one of an `EffectRecord`
or a `GrantBinding`. `LifecycleRecord` requires `revision == len(events)`
(`:598-625`) and re-serialises the whole event tuple per revision. One event per
generated row, eval case or chat turn is quadratic and unbounded.

### 4.2 Ruling: a parallel, typed observation stream

`LifecycleEvent` is **not** extended. Progress lives in a separate record:

```json
{"schema_version": "synaptic-observation/v1",
 "stream": {"family": "evaluation", "project_ref": "acme", "entity_id": "ev-01J…"},
 "sequence": 41, "occurred_at": "2026-09-17T12:00:04Z",
 "kind": "evaluation_case_scored",
 "payload": {"case_ref": "tool-choice#3", "verdict": "passed", "score": 0.82}}
```

- **Append is O(1).** A stream is `(family, project_ref, entity_id) -> ordered
  sequence`; appending never rewrites earlier entries.
- **`kind` is closed per family and selects an exact payload type** — a frozen
  slotted dataclass round-tripped through `exact_fields`, not an open dict. An
  unknown kind is a hard error, so the record stays schema-pinned.
- **Observations carry no authority.** One can never authorize an effect, change a
  run state, or satisfy a verification, and losing the whole stream changes no
  outcome. That is what permits a host to truncate or discard it.
- **Read path is the settled one:** `ObservationsRequest{stream, after_sequence,
  limit}` (limit 1..200) and `ObservationPage`, reusing the
  `next_cursor`/`truncated` agreement rule from `runs_facade.py:148-170`.

Closed kinds by family — training: `training_phase_observed`. Evaluation:
`evaluation_case_started`, `evaluation_case_scored`, `evaluation_stage_completed`.
Chat: `chat_turn_started`, `chat_turn_completed`, `chat_token` (reserved, not
emitted here). Data: `data_row_written`, `data_stage_gate_evaluated`,
`data_scenario_completed`. Pipelines: `pipeline_stage_started`,
`pipeline_stage_completed`.

### 4.3 Durable event budget

The lifecycle record stays quadratic by construction, so each family declares a
fixed maximum of durable events per entity, asserted by a conformance test on the
fake-provider ladder: training 32 (`WorkflowRecordV1`, unchanged), evaluation 12,
data 12, pipelines 8 plus 6 per stage. Chat has no budget: a session is not a
lifecycle entity (§7.3).

### 4.4 EffectKind growth

The preparation doc cites `contracts.py:160-162` (`submit`, `cancel`), which is the
**legacy** enum the coordinator path does not use. The live enum is
`tuner/execution/foundation_v2/identities.py:6` and already has three members:
`stage`, `submit`, `cancel`. Growth is therefore not novel.

**Ruling: the ledger covers only effects whose occurrence can be indeterminate to
us.** It exists to solve one problem — a mutation may have happened at a party we
do not control and cannot observe atomically. So: writing a dataset JSONL is **not**
ledgered (we own the filesystem; the evidence is a `VerifiedArtifact` digest);
spawning a local vLLM server is **not** ledgered (an owned-process lease with its
own uncertainty vocabulary, §7.3); a paid third-party LLM call **is**.

`EffectKind` grows by exactly one member, **`spend`**, recording **one effect per
(entity, provider account)** — claimed before the first call, closed with
accumulated usage. Never one per call; that reintroduces unbounded growth.

`canonical_command` is the canonical bytes of the spend authorization (provider,
account, model set, cap), not a job submission. `provider_job_ref` once `found` is
the provider's usage correlation id where one exists, otherwise the engine-minted
spend claim id, so "found requires a ref" is preserved rather than weakened.
`ExecutionScope{provider, account_ref, namespace_ref}` is the LLM provider ref; a
fully local backend claims **no** spend effect, and a run with zero effects is legal
and is the normal local case.

## 5. Host ports and the composition function

### 5.1 `synaptic_tuner/host/v1` is superseded and deleted

Its eleven Protocols (`host/v1/ports.py`) are imported by no engine module. The
live composition uses `tuner/execution/coordinator_v1/ports.py`, whose 22 Protocols
traffic in `foundation_v2` types and therefore can never be public. Two port sets,
one dead, is the mismatch (open question 10). Resolution: **`coordinator_v1/ports.py`
is consumed and stays internal**, unrevised except that its store ports are now
satisfied by engine adapters over the new public storage ports;
**`synaptic_tuner/host/v1/` is deleted**, its two test consumers updated by the
deleting slice (§11); and **`synaptic_tuner/api/v1/ports.py` is the new public host
port set**.

### 5.2 The public host port set

A host implements five things, not eleven.

```python
# ClockPort            -> now() -> RFC3339 Z, now_epoch() -> int
# SecretResolverPort    -> resolve(SecretRef) -> str
# GrantAuthorityPort    -> authorize(requirements), bind(grant, operation, requirements)

class DurableRecordStorePort(Protocol):
    """Opaque canonical bytes under an engine-chosen key, with compare-and-swap."""
    def create(self, *, partition: str, key: str, canonical: bytes) -> bool: ...
    def read(self, *, partition: str, key: str) -> StoredRecordV1 | None: ...
    def compare_and_swap(self, *, partition: str, key: str,
                         expected_revision: int, canonical: bytes) -> bool: ...
    def put_if_absent(self, *, partition: str, key: str, canonical: bytes) -> bool: ...
    def list_page(self, *, partition: str, prefix: str,
                  after_key: str | None, limit: int) -> StoredPageV1: ...

class DurableStreamStorePort(Protocol):
    """Append-only, monotone per-stream sequence. Never rewrites."""
    def append(self, *, partition: str, stream_key: str,
               sequence: int, canonical: bytes) -> bool: ...
    def read_page(self, *, partition: str, stream_key: str,
                  after_sequence: int | None, limit: int) -> StoredStreamPageV1: ...
```

`StoredRecordV1{key, revision, canonical}` is the only record shape a host sees;
the engine serialises workflow records, plans, preparations, grants, publication,
evaluation and pipeline records to canonical bytes and hands them over opaquely.

- **The engine opens no database.** It hands bytes to the host. This reconciles the
  operator ruling with `AGENTS.md:133-134` explicitly: "the engine ships reference
  repositories" means *in-memory implementations of these two ports*, not a
  file-backed or SQLite store. No `sqlite3` import enters `tuner/`,
  `synaptic_tuner/` or `shared/`, and the package gate keeps it that way.
- Engine record shapes stay internal and may evolve behind `schema_version`, and
  the missing `PlanningStorePortV1` (open question 4) becomes an engine adapter
  over `DurableRecordStorePort` rather than a new host obligation.

Partitions are a closed engine-owned vocabulary: `workflow`, `plan`,
`plan_context`, `preparation`, `execution_grant`, `reconciliation_grant`,
`publication`, `evaluation`, `data`, `pipeline`, `chat_session`, `observation`.

### 5.3 The provider-neutral composition function

`compose_modal_coordinator` (`coordinator_composition.py:74-264`) hard-types four
Modal classes at `:96-111` and duck-checks 24 ports at `:137-173`. The neutral
function is **extracted from it** (open question 7), with the Modal exact-type and
retained-binding checks moved into a Modal family builder rather than deleted —
those checks are load-bearing Modal invariants, not composition boilerplate.

```python
@dataclass(frozen=True, slots=True)
class ProviderFamilyV1:
    descriptor: ProviderDescriptor
    planning: object       # describe, context, preflight
    preparation: object    # prepare, payload, resolve
    reader: object         # observe, logs, artifacts, iter_artifact_bytes
    effect_executor: object; reconciliation_adapter: object
    evidence_authority: object; artifact_verifier: object
    observation_authenticator: object; log_authenticator: object

@dataclass(frozen=True, slots=True)
class ReferenceHostPortsV1:
    records: DurableRecordStorePort; streams: DurableStreamStorePort
    clock: ClockPort; grants: GrantAuthorityPort; secrets: SecretResolverPort

def compose_reference_host(
    *, family: ProviderFamilyV1, ports: ReferenceHostPortsV1,
    recipes: RecipeRegistry, observed_at: str,
) -> ReferenceComposition: ...
```

`ReferenceComposition` carries `training, runs, artifacts, evaluation, chat, data,
pipelines, coordinator, foundation` and exposes `api() -> APIHost`. Port
completeness uses the existing `getattr_static` check
(`coordinator_composition.py:66-70`), promoted to the neutral module.
`compose_modal_coordinator` becomes: build the Modal `ProviderFamilyV1`, call
`compose_reference_host`. Its return type `ModalCoordinatorComposition` (`:58`) is
preserved, so `examples/modal_chat/host.py:329` and
`tests/execution/providers/test_modal_coordinator_composition.py:164` change only
in their `APIHost` construction.

### 5.4 Revised `HostPorts` and `APIHost`

`HostPorts` carries only `runs` and `clock` today (`host.py:53-56`) and `APIHost`
has no `ArtifactsAPI` slot (open question 5). Revised, `HostPorts` becomes an
eight-field frozen slotted dataclass — `training, runs, artifacts, evaluation,
chat, data, pipelines, clock` — and `APIHost.__init__(ports: HostPorts)` exposes
one property per family. This breaks an exact field-tuple pin and five
construction sites; see §11.

## 6. Reference composition of the existing three facades

### 6.1 TrainingAPI

Contract unchanged (`training_facade.py:199-250`). Implemented by
`CoordinatorTrainingService` (`tuner/training/coordinator_service.py:34`), which
already satisfies `TrainingOperations` exactly and is withheld only by `__all__:
list[str] = []` at `:139`. Exported through `api/v1/reference/training.py`.

Training has **no error-code enum** today; failures are `TypeError`/`ValueError`
(`training_facade.py:229-249`). This design adds `TrainingOperationCode`
(`request_invalid`, `resolution_failed`, `plan_unavailable`, `preflight_expired`,
`preflight_not_ready`, `plan_binding_mismatch`, `authorization_refused`,
`state_conflict`, `integrity_error`) and `TrainingOperationError` mirroring
`RunOperationError` (`runs_facade.py:337`). Wrong types keep raising `TypeError`.

### 6.2 RunsAPI

Contract unchanged. Implemented by `RunOperationsV1` (renamed, §3.2). Nine verbs,
13 closed codes.

**Verb placement (open question 9): code wins.** `outcome`, `verify` and `reverify`
stay on `RunsAPI` (`runs_facade.py:348,352,353`), pinned by
`tests/contract/test_public_runs_api_v1.py:49`. The roadmap text at
`submodule-first-training-product-roadmap-plan.md:116-117` is corrected in the same
slice. The roadmap is a plan; the test is a contract.

**Cancel/reconcile evidence (open question 8): fake-provider proof suffices here.**
The implementations exist and are exercised at
`tests/execution/test_fake_provider_v1_conformance.py:576,599,894,912`. Two things
block them in the reference host and both stay: the Modal descriptor declares
`cancel=False, reconcile=False` (`coordinator_adapter.py:50`), and
`_AuthorizationSlot.issue_reconciliation_grant` raises unconditionally
(`examples/modal_chat/host.py:140-141`). Flipping those needs a live Modal
cancel/reconcile smoke, which is later paid work. The frozen matrix row stays
`CONTRACT_ONLY` **for the reference host**; this phase claims only
`IMPLEMENTED_FAKE_TESTED` for the generic path. A matrix correction rides along:
the Modal flags are `ProviderCapabilities(True, False, False, False, True, False)`,
so `observe` and `artifact_streaming` are true, contradicting
`submodule-first-training-v1.md:85-86`.

### 6.3 ArtifactsAPI — composition only

Per operator ruling, no destination adapters ship. `PublicationOperationsV1`
(`coordinator_v1/publication.py:2128`) is already publicly re-exported
(`api/v1/publication.py:10-45`) and is composed with
`StrongInMemoryPublicationStoreV1` (`:1945`) into the `artifacts` slot.

Consequence: with no `DestinationPublicationPortV1` implementation on this branch,
the `Local final publication` and `Hugging Face final publication` matrix rows
**cannot be re-proved here**; they remain historical evidence for their named
commit. Destination adapters are Phase 5.

## 7. The four new facades

Every type below is a frozen slotted dataclass validating in `__post_init__`,
round-tripping through `exact_fields`, with canonical-JSON `to_dict()`/`from_dict()`.
Every facade rebuilds its request, passes a detached copy, and re-validates original
and presentation after the callback returns **or raises** — the `RunsAPI._call`
discipline (`runs_facade.py:402-418`) copied verbatim, not reinvented.

### 7.1 EvaluationAPI

One model × one backend × one scenario set = one evaluation run with its own
`EvaluationRunRef{run_id, project_ref}`. Open question 1 (shared lifecycle
repository, no run-kind discriminator) is answered by §1 choice 3: distinct exact
ref types, store partition as discriminator, so an `EvaluationRunRef` can never
reach `RunsAPI.cancel`.

```python
class EvaluationOperations(Protocol):
    def plan(self, request: EvaluationRequest) -> EvaluationPlan: ...
    def preflight(self, plan: EvaluationPlan) -> EvaluationPreflight: ...
    def start(self, plan: EvaluationPlan) -> EvaluationStart: ...
    def show(self, run: EvaluationRunRef) -> EvaluationOutcome: ...
    def cancel(self, run: EvaluationRunRef, reason: str) -> EvaluationOutcome: ...
    def list(self, r: EvaluationListRequest) -> EvaluationPage: ...
    def result(self, r: EvaluationResultRequest) -> EvaluationResult: ...
    def observations(self, r: ObservationsRequest) -> ObservationPage: ...
```

`EvaluationRunState` (closed): `planned, running, succeeded, partially_succeeded,
failed, cancel_requested, cancelled`. No `reconcile_required` — a spend whose
outcome is indeterminate fails the run with `spend_indeterminate` rather than
demanding provider reconciliation. `EvaluationOperationCode`: `run_missing`,
`cursor_invalid`, `scenario_invalid`, `backend_unavailable`, `backend_unmetered`,
`judge_failed`, `spend_indeterminate`, `cancel_ineligible`, `result_unavailable`,
`state_conflict`, `integrity_error`.

```json
{
  "schema_version": "synaptic-evaluation-result/v1",
  "run": {"run_id": "ev-01J...", "project_ref": "acme"},
  "state": "succeeded", "backend": "openrouter",
  "model": {"model_ref": "acme/qwen-sft", "model_revision": "a1b2…40hex"},
  "cases_total": 120, "cases_scored": 120,
  "scores": [{"name": "composite", "value": 0.812},
             {"name": "quality_gated", "value": 0.744}],
  "verdicts": {"passed": 97, "failed": 21, "inconclusive": 2},
  "usage": {"availability": "measured", "input_tokens": 412334,
            "output_tokens": 88120, "cost_minor_units": 1874, "currency": "USD"},
  "artifacts": [{"role": "evaluation_results", "sha256": "…", "size_bytes": 220114}],
  "diagnostic_code": null
}
```

- **Q4 — judge verdict is closed.** `JudgeVerdict{passed, failed, inconclusive}`
  plus `score` and `threshold` floats. `inconclusive` is new and necessary: today a
  judge failure silently becomes `passed=False` (`shared/judge/models.py:79-82`),
  conflating "the model was bad" with "the judge broke". Both composites are a
  `ScoreV1{name, value}` tuple, not two ad hoc fields.
- **Q5 — `raw_response` and `conversation_trace` do not enter the public record.**
  They are attacker-influenced text, against the standing redaction constraint. The
  record carries `response_digest` and `response_bytes`; the text is a
  `VerifiedArtifact` fetched through a bounded redacted stream.
- **Q6 — exact loss is a distinct capability, not an evaluation stage.**
  `EXPERIMENT_STAGES` already treats it as a peer (`experiment_spec.py:13`); the
  eval CLI bolting it on (`Evaluator/cli.py:219-343`) is legacy coupling. Loss gets
  its own facade later; `PipelinesAPI` refuses a spec naming it (§7.4).

Implemented by `Evaluator/` scenario loading, scoring and judging, driven from
`api/v1/reference/evaluation.py`: at most one `spend` effect, artifacts through the
host's sink, never to `Evaluator/results/`.

### 7.2 DataAPI

One generation or improvement run producing one dataset artifact,
`DataRunRef{run_id, project_ref}`. Open question 1: **one run, one artifact**, with
per-scenario progress as observations; one run per scenario would multiply spend
effects and defeat the single-spend rule.

```python
class DataOperations(Protocol):
    def plan(self, request: DataRequest) -> DataPlan: ...
    def preflight(self, plan: DataPlan) -> DataPreflight: ...
    def start(self, plan: DataPlan) -> DataStart: ...
    def show(self, run: DataRunRef) -> DataOutcome: ...
    def cancel(self, run: DataRunRef, reason: str) -> DataOutcome: ...
    def list(self, r: DataListRequest) -> DataPage: ...
    def datasets(self, r: DatasetListRequest) -> DatasetPage: ...
    def validate(self, r: DatasetValidateRequest) -> ValidationReport: ...
    def observations(self, r: ObservationsRequest) -> ObservationPage: ...
```

`DataRunState` (closed): `planned, running, succeeded, partially_succeeded, failed,
cancel_requested, cancelled`. **Q2 — partial success is a first-class terminal
state**: terminal, produced a verified artifact, did not reach the requested row
count. `RunOutcome` already permits artifacts on any state
(`runs_facade.py:91-112`); the vocabulary was the gap. A cancelled run also lists
the rows it produced.

**Q3 — the `_meta` header row leaves the data file. Mandatory.** Today
`result_writer.py:41-54` writes metadata as line 0 of the JSONL, so the file is not
homogeneous and its SHA-256 is not a dataset digest. Metadata moves to a sidecar
artifact with role `dataset_metadata`. This is a deliberate behaviour change to
`SynthChat/result_writer.py`, not a compatibility break to absorb.

**Q5, and cross-family Q7 — read-only queries live on their family's facade.**
`datasets` and `validate` take no run and no effects and page with the settled
contract; a separate query surface would need its own paging rules for no gain.

`DataOperationCode` (closed): `run_missing`, `cursor_invalid`, `scenario_invalid`,
`manifest_invalid`, `backend_unavailable`, `backend_unmetered`,
`spend_indeterminate`, `output_conflict`, `cancel_ineligible`, `dataset_missing`,
`state_conflict`, `integrity_error`. `synaptic-dataset-result/v1` mirrors the
evaluation record: run, state, `rows_requested`, `rows_written`, `scenarios`, usage,
artifacts (`dataset_jsonl`, `dataset_metadata`) and `diagnostic_code`.

### 7.3 ChatAPI

**Q1 — a session is not a run.** `ChatSessionRef{session_id, project_ref}` is a
first-class entity outside the lifecycle model, because `LifecycleRecord` demands
one event per revision and a 10,000-turn session (`Evaluator/chat_session.py:37-61`
bounds `max_turns` at 10000) is quadratic, and because a session has no paid
provider submission to reconcile. The code already made this choice —
`PreparedRunChat` binds a `TrainingRunRef` with no lifecycle record
(`tuner/inference/run_chat.py:39-70`).

```python
class ChatOperations(Protocol):
    def open(self, request: ChatOpenRequest) -> ChatSession: ...
    def turn(self, request: ChatTurnRequest) -> ChatTurn: ...
    def show(self, session: ChatSessionRef) -> ChatSession: ...
    def list(self, request: ChatListRequest) -> ChatSessionPage: ...
    def close(self, session: ChatSessionRef) -> ChatSession: ...
    def observations(self, request: ObservationsRequest) -> ObservationPage: ...
```

`ChatSessionState` (closed): `opening, ready, serving, closing, closed,
cleanup_unresolved`.

**Q5 — `cleanup_unresolved` is the public shape of** `OwnedProcessError("owned
process family cleanup remains unresolved")` (`Evaluator/owned_process.py:202`). It
is the chat analogue of `reconcile_required` and gets its own code, not a
`RunOperationCode`. A session in that state is terminal-uncertain: it serves no
further turn, and the host is told a process family may still hold a GPU.

**Q2 — streaming-ready identity without streaming.** A turn is
`ChatTurnRef{session, request_id}` with `request_id` a consecutive integer, already
the Modal frame shape (`inference_channel.py:24-62`), and `turn()` returns a
completed `ChatTurn{ref, state, content_digest, content, usage}`. Adding token
streaming later means emitting `chat_token` observations against **the same**
`ChatTurnRef`, read through the existing `observations` verb. No identity changes.

**Q3 — the `synaptic-modal-chat-channel/v1` frames stay provider-local.** Their
five kinds bind `launch_digest`, a Modal concept. The public vocabulary is the
session state enum plus `ChatOperationCode`.

**Q4 — the owned-process runtime arm is Linux-only, and the contract says so.**
`owned_process.py:207-216` requires `os.name == "posix"`, `/proc/self/stat` and
`os.killpg`. A non-Linux host receives `ChatOperationCode.HOST_UNSUPPORTED` from
the local runtime rather than a crash. The facade is platform-neutral and a remote
runtime (Modal) carries no such restriction. A Windows process-group owner is out
of scope and would be the weaker of the two implementations.

**Q6 — three modules move out of `Evaluator/`, and it is in scope.**
`tuner/inference/run_chat.py:10` imports `Evaluator.chat_session`, which would drag
15,630 lines including Rich UI, argparse CLIs and eight backend clients into any
reference chat implementation. `chat_session.py`, `vllm_runtime.py` and
`owned_process.py` move to `tuner/inference/`, imports updated directly, no
re-export shim (CLAUDE.md). Without it ChatAPI cannot ship cleanly.

`ChatOperationCode` (closed): `session_missing`, `session_closed`, `session_busy`,
`turn_bounds_invalid`, `model_ineligible`, `runtime_unavailable`,
`cleanup_unresolved`, `host_unsupported`, `integrity_error`.

### 7.4 PipelinesAPI

**Q1 — a pipeline references child runs; it does not own them.** `LifecycleRepository`
is keyed by `(project_ref, run_id)` with no parent concept (`contracts.py:776-781`)
and `WorkflowRecordV1` has no parent field, so ownership would force a nesting
change in the engine's most safety-critical record. **Q2 dissolves**: under
ownership the pipeline's phase is a function of concurrent children that disagree
(`experiment_orchestrator.py:112-126` runs evaluation and loss in a 2-worker pool),
whereas under reference each child keeps its own authoritative phase. And roadmap
Phase 9 requires that "evaluation or publication failure must not rewrite
successful training as failed" — reference satisfies that by construction.

```python
class PipelinesOperations(Protocol):
    def plan(self, request: PipelineRequest) -> PipelinePlan: ...
    def start(self, plan: PipelinePlan) -> PipelineStart: ...
    def show(self, pipeline: PipelineRef) -> PipelineRecord: ...
    def resume(self, pipeline: PipelineRef) -> PipelineRecord: ...
    def cancel(self, pipeline: PipelineRef, reason: str) -> PipelineRecord: ...
    def list(self, r: PipelineListRequest) -> PipelinePage: ...
    def observations(self, r: ObservationsRequest) -> ObservationPage: ...
```

```json
{
  "schema_version": "synaptic-pipeline-record/v1",
  "pipeline": {"pipeline_id": "pl-01J…", "project_ref": "acme"},
  "spec_digest": "…64hex", "state": "partially_succeeded", "revision": 6,
  "stages": [
    {"name": "train", "state": "succeeded", "attempt_key": "…64hex",
     "run": {"run_id": "modal-sft-…", "project_ref": "acme"},
     "inputs": [], "outputs": [{"role": "adapter", "sha256": "…", "size_bytes": 84}]},
    {"name": "evaluate", "state": "failed", "attempt_key": "…64hex",
     "run": {"run_id": "ev-01J…", "project_ref": "acme"},
     "inputs": [{"role": "adapter", "sha256": "…", "size_bytes": 84}], "outputs": []}
  ]
}
```

**Q3 — status vocabulary: v1 wins and `.tracking` is abandoned.** Legacy
`completed` becomes `succeeded`. Legacy `partial` (`experiment.py:111`) is the
legacy *default*, so most records carry it as noise rather than meaning; migrating
it would import that noise. Per Phase 10 ("delete rather than forward once parity
is proven"), `.tracking` is not migrated.

`PipelineState` (closed, 8 values): `planned, running, succeeded,
partially_succeeded, failed, reconcile_required, cancel_requested, cancelled`.
`partially_succeeded` is terminal with at least one succeeded and at least one
failed or skipped stage. `reconcile_required` means at least one stage's child run
is in `reconcile_required` and no stage can advance — calling that `running` would
be a lie. **These are pipeline-only states; no child run ever takes them.**

**Q4 — the seven SCREAMING_CASE HF state tuples (`experiment.py:14-48`) stay a
provider-local dialect behind the adapter.** `HF_TRAINING_RESULT_STATES` is
`VerificationStatus` in other casing, missing two members; promoting it would give
one concept two public vocabularies.

**Q5, Q6 — handoff by digest, resume by stage attempt key.** Stage inputs and
outputs are `VerifiedArtifact`, never `artifact_roots: dict[str,str]`
(`experiment.py:120`) — the Phase 9 "bind each stage to exact upstream artifact
hashes rather than `latest`" requirement. The resume key is `attempt_key =
contract_digest("synaptic-pipeline-stage-attempt/v1", {pipeline_id, stage,
spec_digest, input_digest})`: not the plan fingerprint (training-specific) and not
a mutable field. `resume` recomputes each key and skips a stage whose key matches a
succeeded stage.

**Stage admission here.** Only `train` and `evaluate` are admitted; `loss`,
`analysis` and `recommendation` are declared in the stage vocabulary and refused
with `stage_unsupported` until their facades exist. Declaring but refusing is
honest; silently accepting and ignoring is not.

`PipelineOperationCode` (closed): `pipeline_missing`, `cursor_invalid`,
`spec_invalid`, `stage_unsupported`, `stage_input_missing`, `stage_digest_mismatch`,
`resume_ineligible`, `cancel_ineligible`, `state_conflict`, `integrity_error`.

## 8. Schemas to add

Nine new JSON Schema files under `schemas/`, named as the existing ones are
(`schemas/synaptic-run-outcome-v1.schema.json`), one per new `schema_version`:
`synaptic-observation/v1`, `synaptic-usage/v1`, `synaptic-evaluation-plan/v1`,
`synaptic-evaluation-result/v1`, `synaptic-chat-session/v1`,
`synaptic-chat-turn/v1`, `synaptic-dataset-plan/v1`, `synaptic-dataset-result/v1`,
`synaptic-pipeline-record/v1`. Unlike `synaptic-event-v1.schema.json:9-14`, which
is deliberately open, every one is closed — `additionalProperties: false` at every
level, matching `exact_fields`.

## 9. Cross-cutting

**Redaction.** The only bounded redactor is Modal-specific
(`providers/modal/redaction.py:67`), though its behaviour — a key regex, six text
patterns, depth/item/byte bounds, `"[REDACTED:ERROR]"` on failure — is
provider-neutral. It is promoted to `tuner/execution/redaction.py` in the first
slice and Modal imports it from there. Every facade's machine-output path passes
through it, and no facade emits exception text, locals or tracebacks
(`AGENTS.md:57-60`).

**Secrets by name.** Every facade accepts `SecretRef` only and resolves through
`SecretResolverPort` at execution time. None of the ambient reads is inherited:
`Evaluator/config.py:156-159` (`VLLM_API_KEY` → `HF_TOKEN` fallback),
`shared_llm_adapters.py:251,347`, `cloud_eval_handler.py` (eight sites),
`SynthChat/services/rubric_runner.py:108-109`, and the `.env` walk up from `cwd`
(`shared/utilities/env.py:58-76`).

**`.synaptic` carve-out.** No facade constructs a path; `ProjectContext.host`
(`tuner/project/context.py:75-89`) and `_validate_context_roots`
(`manifest.py:240-268`) stay authoritative. A contract test asserts no module under
`api/v1/reference/` imports `ProjectContext` to build a default output path,
preserving the guarantee that discovery creates no `.synaptic`
(`test_embedded_host.py:38,57,87`).

**Cancellation.** `RunsAPI.cancel` is provider cancellation with a durable effect.
Evaluation and Data cancellation is **cooperative**, checked at scenario and row
boundaries, and a cancelled run still lists the artifact produced so far. Chat has
`close`, not cancel; pipeline cancel requests cancel on the in-flight child and
never touches completed children.

**LLM usage reporting: contract now, implementation last, paid arm gated.**
`shared/llm/base.py:16-37` returns a bare `str`, so no caller can count tokens, and
`SynthChat/config/settings.yaml:103-105` promises `cost_tracking` that nothing
implements. Three of four families are paid. `UsageRecordV1` and
`UsageAvailability{measured, unavailable}` are defined now. A nullable `usage` that
is always null would be a misleading default, so when availability is `unavailable`
the field is **absent** and no `spend` effect may be claimed. **The reference
Evaluation and Data implementations refuse an unmetered backend whenever
`AuthorizationRequirement.paid_effect` is true**, with `backend_unmetered`: local
backends work from the first slice, paid backends only after metering lands. The
metering slice changes `BaseLLMClient.chat` to return `LLMCompletionV1{text,
usage}` and updates every call site in one commit — no dual signature, no shim
(CLAUDE.md). It is the largest blast radius here and is scheduled last so nothing
else waits on it.

## 10. Legacy coupling not inherited, and Phase 10 consequences

| Facade | Not inherited |
|---|---|
| Training / Runs / Artifacts | `examples/host-project` hand-written operations; Modal-specific composition as the only entry point; fixed five-file publisher naming |
| Evaluation | engine-relative results default (`Evaluator/cli_utils.py:158`); engine-relative config dir (`cli.py:365-369`); ambient keys; `SystemExit` from library modules (six sites); Rich-only render with no structured return (`eval_handler.py:211-247`); subprocess re-invocation fallback (`:691-692, 820-822`) |
| Chat | failure signalled by attaching a mutable attribute to the raised exception (`model_chat.py:186-201`); `tuner/inference/` depending on `Evaluator/` |
| Data | `.env` discovery by walking up from `cwd`; `sys.exit` inside a service (`rubric_runner.py:615,620,630`); `print()` as the only progress channel; truncating output open defeating resume (`result_writer.py:38`); `_meta` header row inside the data JSONL; hardcoded output filename (`synthchat_handler.py:241,247-249`) |
| Pipelines | `.tracking` literal defaults (four sites); registry path walking module ancestors (`registry.py:264-271`); mutable 40-field durable record with no revision (`experiment.py:98-140`); whole process environment copied into an HF Job child (`cloud_loss_job.py:59`); `TrackingService` as the sole persistence port |

**Deleted in this phase:** `synaptic_tuner/host/v1/` (superseded, §5.1).

**Marked for Phase 10 deletion, frozen now (open question 11):**
`synaptic_tuner/api/v1/persistence.py` exposes a lifecycle/effect model the
coordinator path does not use — no import of `tuner.execution.contracts` exists
inside `coordinator_v1/` or `foundation_v2/`, and the only consumers are
`tuner/execution/service.py:9`, `broker.py:113` and test fakes. **No reference
implementation populates those records.** Rather than a deletion that would balloon
this phase, the discipline is wired now: a contract test asserts no module under
`api/v1/reference/` imports `synaptic_tuner.api.v1.persistence`. Phase 10 then
deletes `persistence.py`, `tuner/execution/service.py`, `broker.py`,
`lifecycle.py` and the lifecycle half of `contracts.py` together, alongside
`tuner/handlers/eval_handler.py`, `synthchat_handler.py`, `list_handler.py`,
`shared/experiment_tracking/service.py` and `.tracking`, `Evaluator/results/` as a
default, and the `sys.exit`-in-service sites. `cloud-eval` is kept until
EvaluationAPI has a proven replacement, per the roadmap.

## 11. Structural pins the implementation must update

Exact, currently-passing assertions this design breaks. Each is updated in the
same commit as the change that breaks it.

| Pin | Location | Broken by |
|---|---|---|
| `tuple(HostPorts.__dataclass_fields__) == ("runs", "clock")` | `tests/contract/test_public_training_api_v1.py:65` | §5.4 eight-field `HostPorts` |
| `APIHost(Operations(), HostPorts(runs=…, clock=…))` | `test_public_training_api_v1.py:100` | §5.4 `APIHost(ports)` |
| `APIHost(object(), ports)` | `test_public_runs_api_v1.py:272-275` | §5.4 |
| `APIHost(composed.training, HostPorts(composed.runs, clock))` | `tests/execution/providers/test_modal_coordinator_composition.py:164` | §5.4 |
| `APIHost(Training(), HostPorts(runs=…, clock=…))` | `tests/inference/test_run_chat_consumer.py:139` | §5.4 |
| `APIHost(composed.training, HostPorts(composed.runs, clock))` | `examples/modal_chat/host.py:329` | §5.4 |
| `"artifact_publisher" not in HostPorts.__dataclass_fields__` | `test_public_publication_v1.py:418` | survives unchanged — verify, do not edit |
| `import synaptic_tuner.host.v1` in the module gate | `test_provider_neutral_foundation_v1.py:457` | §5.1 deletion |
| `host/v1/ports.py` in the AST source gate path list | `test_provider_neutral_foundation_v1.py:468` | §5.1 deletion |
| `test_secondary_host_v1_publication_protocols_are_absent` | `test_public_publication_v1.py:421-429` | §5.1 deletion |
| former `Training…OperationsV1` name (now `RunOperationsV1`) | `coordinator_composition.py:61,254`; Docker composition tests; conformance suite `:506` | §3.2 rename |
| Module and AST gate file lists | `test_provider_neutral_foundation_v1.py:450-473` | every new contract module must be added to both |

The last row is the one most likely to be missed: adding a contract file without
adding it to **both** lists leaves it ungated and the suite still passes.

## 12. Risks

| Risk | Mitigation |
|---|---|
| The `HostPorts` change touches six sites across contract, execution, inference and example tests | One slice, enumerated in §11, tree green before the next slice starts |
| `compose_reference_host` extracted from a function with 24 duck-checked ports and four exact-type checks | Modal checks move into a Modal family builder rather than being dropped; `test_modal_coordinator_composition.py` is the regression oracle |
| Moving three modules out of `Evaluator/` breaks importers, and observation streams grow without bound on a long session | Direct import updates with no shims, `tests/evaluator/` and `tests/inference/` the oracle; the stream store is host-supplied and free to truncate, since observations carry no authority |
| Opaque-bytes storage hides record-shape bugs from the host, and the metering slice spans `Evaluator/`, `SynthChat/` and `shared/judge/` | Engine-side round-trip assertion on every read, with a canonical-bytes mismatch raising `integrity_error` rather than a silent accept; metering scheduled last with the paid arm refused as `backend_unmetered` until it lands |
