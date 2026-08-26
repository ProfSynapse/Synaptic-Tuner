# Submodule-first training product v1

Status: Phase 0 evidence and contract baseline
Approved roadmap: [`../plans/submodule-first-training-product-roadmap-plan.md`](../plans/submodule-first-training-product-roadmap-plan.md)

This document is the authoritative product boundary and evidence matrix for the
submodule-first training migration. Provider-specific references describe an
adapter; they do not redefine this contract.

## Product boundary

A host supplies the model and revision, dataset reference, method and
hyperparameters, provider profile, artifact destination, and authorization for
paid effects. The engine supplies immutable public contracts, planning,
lifecycle coordination, provider ports, semantic verification, and publication
ports. The host owns configuration, mutable state, credentials, provider
sessions, authorization, and final destinations.

```text
host project
  configuration / state / secrets / approvals / destinations
                         |
                         v
synaptic_tuner.api.v1
  TrainingAPI / RunsAPI / ArtifactsAPI / EvaluationAPI / PipelinesAPI
                         |
                         v
provider-neutral coordinator and provider registry
                         |
                         v
docker | modal | hf_jobs | runpod
```

The source submodule and host source tree are read-only inputs during a run.
Every mutable location is selected by the host. A provider adapter may stage
artifacts in provider-native storage, but verified final publication is a
separate host-selected operation.

## Evidence vocabulary

These are closed status values for this baseline:

| Status | Meaning |
|---|---|
| `LIVE_PROVEN` | A real provider or destination effect was observed and durable local evidence remains. |
| `IMPLEMENTED_FAKE_TESTED` | Code and hermetic tests exist; no live provider effect is claimed. |
| `CONTRACT_ONLY` | A public shape exists but no composed operational implementation is claimed. |
| `NOT_IMPLEMENTED` | The v1 product path does not provide the capability. |

Provider-free tests, authenticated read-only checks, paid execution, semantic
verification, and final publication are distinct proof levels. Success at one
level must never be used as evidence for a later level.

## Frozen capability matrix

| Capability | Status at engine `31d2683448919e1e694f36392fa4e40741226ae9` | Evidence or boundary |
|---|---|---|
| Modal SFT deploy/start/observe | `LIVE_PROVEN` | Run `modal-sft-20260826T144636Z-7aec224e893d`, provider call `fc-01M0Z8K9MCPN3P368V3CK94TV2`. |
| Modal semantic verification and reverify | `LIVE_PROVEN` | Host lifecycle revisions record invalid, reopened, then verified without a second submission. |
| Local final publication | `LIVE_PROVEN` | One durable receipt binds exactly five artifact roles and their SHA-256/size descriptors. |
| Local publication repeat convergence | `LIVE_PROVEN`, fixture does not independently replay it | The operation was observed as idempotent; the frozen fixture contains only the converged receipt. |
| Hugging Face final publication | `IMPLEMENTED_FAKE_TESTED` | No live repository mutation is claimed. |
| Operational `RunsAPI` cancel/reconcile in the reference host | `CONTRACT_ONLY` | Host composition still uses an unavailable implementation. |
| Local Docker through the public training API | `NOT_IMPLEMENTED` | Phase 2 gate. |
| HF Jobs through the public training API | `NOT_IMPLEMENTED` | Protected legacy lanes are not the product adapter. |
| RunPod through the public training API | `NOT_IMPLEMENTED` | New pinned adapter required. |
| KTO, DPO, embedding, GRPO through the public training API | `NOT_IMPLEMENTED` | SFT is the only method in the new path. |
| Evaluation and persisted pipelines through public APIs | `NOT_IMPLEMENTED` | Legacy evaluation/experiment paths remain outside v1. |

## Frozen Modal evidence

The repository fixture is
[`../../tests/fixtures/training_product/modal_live_v1/evidence-index.json`](../../tests/fixtures/training_product/modal_live_v1/evidence-index.json).
It is a closed, sanitized index derived only from already-local host evidence.
Its completeness fields are normative and machine-tested.

Captured evidence:

- exact run, project, provider-call, engine-commit, and host-commit identities;
- the canonical lifecycle-record SHA-256 and a sanitized projection of all 12
  durable events;
- the durable publication-receipt SHA-256 and a sanitized projection of the
  five roles, sizes, and content digests;
- bounded synthetic substitutes for contract tests, each separately hashed and
  explicitly marked as non-evidence.

Not captured:

- authenticated provider terminal-record bytes;
- authenticated provider completion-manifest bytes;
- lifecycle canonical bytes containing grant/command material;
- publication canonical bytes containing a private local path;
- model, tokenizer, or other artifact payload bytes;
- credentials, evidence keys/MAC material, raw provider responses, trainer
  output, exception text, or private absolute paths.

Absence is not failure of the live run. It is a boundary on what this portable
fixture can prove. Tests must use the typed completeness fields instead of
inferring that a digest implies captured bytes.

## Phase 1 boundary corrections

The next implementation may not ask other providers to emulate Modal. Before
Docker, HF Jobs, or RunPod becomes an adapter, the generic layer must remove:

- `HostPorts.modal_reads` in favor of host-owned provider sessions;
- Modal defaults and Volume/deployment fields from generic provider plans;
- `ModalStageTargetV1` from generic operation binding;
- `modal_preparations` from generic host persistence;
- direct Modal composition from the canonical host CLI.

The replacement is an import-light provider contract, capability declaration,
opaque prepared-operation record, lazy registry, and provider-neutral training
coordinator. `TrainingAPI`, `RunsAPI`, and `ArtifactsAPI` remain separate
semantic domains; there is no untyped universal job API.

## Phase 2 gate

The Phase 2 gate is satisfied only when fake and local-Docker providers pass
the same lifecycle suite for start, observe, logs, cancel, reconcile, verify,
publish, interruption, and restart. The same request and artifact contracts
must then be usable by Modal without provider fields entering public plans.
No paid cloud rerun is part of this gate.
