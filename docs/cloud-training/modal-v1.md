# Modal v1: submodule-first training

Correction (2026-09-09): the current Modal coordinator adapter is undergoing
provider-free qualification behind `synaptic_tuner.api.v1.TrainingAPI`.
Its six advertised read, lifecycle, artifact-streaming, and cost-quote flags
remain false. They are not a master start switch: exact consumer effect grants,
authenticated preflight, and Foundation lineage are required. The frozen live proof below
belongs to the earlier implementation, not this coordinator cutover.
The consuming application owns project configuration, its database, execution
grants, secrets, and persistence. The Toolset-Training submodule owns the
provider-neutral request/plan/outcome contracts and the reusable Modal
mechanisms. No engine SQLite database or legacy cloud-provider menu is part of
the design.

## Boundary

```text
host request/config
  -> TrainingAPI load / resolve / plan
  -> generic plan plus retained, authenticated Modal preparation
  -> TrainingAPI preflight (bounded and expiring)
  -> consumer-owned coordinator stores and Foundation authority
  -> Foundation effect broker / operation-scoped Volume staging
  -> exact Modal Function.spawn()
  -> fixed remote SFT runtime
  -> authenticated Volume evidence
  -> RunsAPI (provider reads require the corresponding advertised flag and qualification)
```

The fixed provider topology is SDK 1.5.4, an explicit authenticated client, a
deployed Function whose name derives from its deployment identity, A10,
retries disabled, and
two distinct existing Volume v1 objects. Every effect uses
`operations/{effect_id}/...`; there are no shared global input or output paths.

The remote worker independently clones the exact pushed host project and engine
commit, verifies the host project's engine gitlink, and invokes only the
canonical SFT runtime without a shell. The digest-pinned runtime lock is
`tuner/execution/providers/modal/modal-runtime-v1.lock.json`.

## Durable state

The generic plan and retained preparation bind the non-secret provider facts
required for restart: verified deployment evidence, explicit client scope,
exact Volume IDs, quote digest and expiry, and bounded cost. Immutable command
bindings additionally authenticate the complete operation identity and command.
The former Modal plan-context and training-repository contracts are no longer
the public composition surface.

`compose_modal_coordinator` assembles the service, Foundation effects,
retention, and authenticated reader from explicit consumer-owned stores and
authorities. `ModalCoordinatorStorePorts` and `ModalFoundationCompositionPorts`
describe these inputs. The consumer supplies durable implementations; the
engine does not supply a database. Retention authenticates commands before
publication and reuses exactly retained launch evidence on a fresh-wrapper
retry. Provider-free in-memory tests demonstrate this protocol, not database
crash durability. Terminal, log, and completion expectations derive from full
authenticated submit evidence, not a predicted provider job ID. An uncertain
effect must be reconciled read-only rather than repeated under the same authority.

Modal Volume is the source of truth for Modal artifacts. A completed run has
exactly five verified artifacts: workload record, training lineage, training
metrics, final model, and tokenizer. Control records are authenticated and
contain only structured, redacted status. Raw secrets, trainer output,
tracebacks, and provider response bodies do not enter the durable contract.

## Readiness

This implementation distinguishes:

- provider-free proof: tests, schemas, exact SDK object construction, image
  inspection, and a network-disabled runtime check;
- authenticated live preflight: account/workspace/environment, existing Volume
  IDs, deployed Function version/image, secret declarations, and quote;
- paid smoke: one explicitly granted `.spawn()` followed by observation and
  exact artifact verification.

Do not run a paid smoke until the host repository conformance tests and the
independent security/release barrier pass for the exact committed and pushed
tree.

## Frozen live proof

Historical evidence only: this section records the earlier implementation.
It does not qualify live execution of the current coordinator, its six
advertised read/lifecycle surfaces, or its updated source lock.

The first successful product run is
`modal-sft-20260826T144636Z-7aec224e893d`, bound to provider call
`fc-01M0Z8K9MCPN3P368V3CK94TV2`, engine commit
`31d2683448919e1e694f36392fa4e40741226ae9`, and host commit
`a4926a274b847e1a746ad4de53563b5976fc1574`. The durable host lifecycle records
one submit attempt, provider success, an initial invalid verification, a
read-only reopen, and final verification. Reverification did not submit a
second paid job.

The initial false negative was a location interpretation error: the plan's
logical run root was `/workspace/run`, while Modal exposed the same mounted
Volume under its resolved physical mount path. The corrected verifier accepts
only the evidence-bound relocation; it does not widen reads to an arbitrary
filesystem search.

The verified source was then published to a host-selected local destination as
exactly five roles. The resulting receipt is durable and a repeated publication
converged to the same receipt.

The closed sanitized fixture at
`tests/fixtures/training_product/modal_live_v1/` preserves the lifecycle and
publication digests, projections, and tiny synthetic substitutes. It does not
contain authenticated provider terminal-record or completion-manifest bytes,
artifact payloads, credential/evidence-authentication material, raw errors, or
private paths. Its typed completeness declaration is the limit of portable
fixture proof; it is not a reconstructed provider transcript.

The authoritative cross-provider status matrix is
[`../architecture/submodule-first-training-v1.md`](../architecture/submodule-first-training-v1.md).
