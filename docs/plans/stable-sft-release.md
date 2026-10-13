---
status: proposed
scope: OSS API v1 stable SFT release
baseline: 321a085f
updated: 2026-10-13
---

# Stable SFT release gate

This is the canonical preparation plan for the first broad stable OSS release.
The selected scope is public Python API v1 SFT parity across local Docker,
Modal and Hugging Face Jobs, with durable host composition, restart recovery,
cancellation and reconciliation. The baseline is main
`321a085f`; selecting this scope does not establish that its gates have passed.
No version bump, tag, package publication or provider execution is authorized by
this document. Live qualifications require reviewed finite scope and authority.

RunPod, additional training methods, Syntunia integration, browser transport,
managed multi-tenant serving and training-quality claims are deferred. Evaluation,
data, ingestion, chat, publication and pipelines must have truthful declared
status; their existence does not make every backend part of the stable SFT claim.
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
| Packaging/platforms | Distribution version `1.1.0`, Python metadata `>=3.10`, selected wheel assets and Python 3.12 CI; current wheel and sdist omit top-level schemas/scripts. | Complete supported distribution inventory, clean installed consumer, demonstrated Python/platform matrix and release artifacts. Metadata is not qualification. |
| Optional families | Public facades exist; some operations require host-supplied backends. | Per-operation backend status, filtering and explicit supported/experimental/deferred boundaries. |

All milestones below are **open**. Record terminal evidence against the exact
candidate source and artifact digest; no historical result automatically closes
a new gate. Retain failed attempts, consumed claims and uncertain submissions.

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
- [ ] Select and qualify explicit stable SFT recipes, model/tokenizer revisions,
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

Exit: clean installed consumer can perform provider-free planning and compose
all supported providers using only declared assets/dependencies.

## 3. Ship durable host composition and migrations

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

## 7. Audit optional facade and backend status

- [ ] Inventory every exported family and optional backend. Separate stable SFT
  integration needs from experimental reference implementations and deferred
  features; absence of a backend yields explicit unavailability.
- [ ] Pipelines, evaluation and chat cannot infer authority from a completed
  training run. Data and ingestion retain their own input/custody contracts.
- [ ] Examples and capability negotiation omit unsupported cancel/recovery,
  inference, model conversion or publication operations rather than advertising
  them and failing later. Additional methods and RunPod stay outside this tag.

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
