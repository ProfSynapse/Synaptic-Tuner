# Public API facade: operator decisions (2026-09-17)

Recorded before any facade work starts so the next phase begins from settled
constraints. Source: operator answers on 2026-09-17 after the standalone Modal
chat closeout. Survey basis: `synaptic_tuner/api/v1` today ships `TrainingAPI`,
`RunsAPI` and `ArtifactsAPI` as Protocols with no engine implementation;
`EvaluationAPI` and `PipelinesAPI` are named in
`docs/architecture/submodule-first-training-v1.md` but do not exist; chat,
SynthChat and dataset listing/validation have no facade.

| Decision | Ruling |
|----------|--------|
| HTTP transport | The engine stays transport-free (Python facades, `synaptic-result/v1` and `synaptic-event/v1` envelopes, JSONL CLI). A separate host service composes the engine and owns HTTP/SSE, state, credentials and auth for the frontend. No HTTP framework enters the engine's runtime dependency closure. |
| Facade families in scope | `EvaluationAPI` (roadmap Phase 9), `ChatAPI` (new), `DataAPI` for SynthChat generation/improvement plus dataset listing/validation (new), `PipelinesAPI` (after `EvaluationAPI`). |
| Foundation | v1 lifecycle only. Each facade is a v1 contract plus one engine implementation over lifecycle and effect records. Legacy handlers and `.tracking` are wrapped only where unavoidable and retired per Phase 10. No thin facades over legacy handlers. |
| Ownership | The engine ships reference in-process implementations of `TrainingAPI` and `RunsAPI` (and the new families) over host-supplied ports (repositories, clock, grant authority). Hosts keep storage and credentials; they no longer hand-write the operations as `examples/host-project` does. |
| Frontend | Syntunia is not wired until the facades exist and have been exercised end to end, to avoid rewiring the frontend after engine-side changes. |

Standing constraints that carry over unchanged: secrets by name only
(`SecretRef`, resolved at execution time), redaction on every machine-output
path, host-owned mutable state under `<project>/.synaptic`, closed status
vocabularies, cursor-paged polling for logs, no push stream in the engine.

Next step: PREPARE and ARCHITECT passes for the facade set, starting with the
contract shapes for `EvaluationAPI` and `ChatAPI` and the reference
implementations of `TrainingAPI`/`RunsAPI`.
