# Public environment execution through DataAPI

Status: effect-free contract/admission and a generic candidate-feedback seam
implemented and reviewed at `846f754b`, 2026-10-13. Actual environment execution
and backend qualification remain open. This is node
C of the [stable release plan](stable-v1-release.md); installation assets and
lightweight imports are parallel prerequisites for its installed acceptance.
Root owns release task `97d0f9df-6701-4887-a7a3-2993a117f90f`; design task
`27f8f9ac-04b8-4d73-b62c-a37d6c15d3f8` owns this proposal.

## Existing boundary and reason for a reusable change

At merged source `321a085f61c9ac031f89daddde0bf95633e58133`,
[`ReferenceDataPortsV1`](../../synaptic_tuner/api/v1/reference/data.py) supplies
configuration, scenario and rubric directories and an LLM factory. Its generator
construction explicitly passes `environment_validator=None`. Configuration alone
cannot enable environment execution through this public composition.

[`SynthChatGenerator`](../../SynthChat/generator.py) already carries environment
failures into response judging and improvement and then re-executes the revised
response. DataAPI improve mode constructs
[`ImprovementEngine`](../../SynthChat/engine.py) without environment feedback;
that engine accepts a static `prompt_context`, which cannot establish that a
revised candidate was executed. These are the two reusable seams to connect.

The existing DataPlan binds parsed scenarios for generation, and dataset bytes
plus rubric names for improvement. It does not bind all execution policy,
tool-schema or rubric bytes. Existing
[`EnvironmentValidator`](../../shared/environments/validator.py) also defaults
missing configuration to simulation. Requested execution must not inherit that
fallback. The local runtime is a temporary filesystem fixture, not a security
sandbox. E2B's ambient credential and SDK fallback behavior is not public API
authority.

## Contract and admission

Use the existing DataAPI lifecycle. Add contract types in an explicit,
provider-neutral `synaptic_tuner.api.v1.environment_execution` submodule. Do not
change root exports merely for convenience. Names below are proposed interfaces,
not existing exports.

- `EnvironmentSelectionV1`: a logical host profile reference and explicit
  `execute` or `simulate` mode. Add an optional selection to DataRequest.
- `EnvironmentBindingV1`: the resolved immutable execution identity, policy,
  dependency digests, scope and finite bounds persisted with an opt-in DataPlan.
- `EnvironmentExecutionPort`: resolve a selection without effects, report its
  authorization requirement, and open a bounded session after authorization and
  a durable attempt claim. Supply this optional port through ReferenceDataPortsV1.
- A session supports the existing validator's response/session use cases through
  a concrete bridge, plus bounded finalization and close. Generic interfaces must
  not import a provider SDK, `tuner.*`, or a current tool wrapper.

No environment selection means the existing canonical request and plan bytes
remain unchanged: omit optional fields rather than serialize null/default fields.
Use a separately versioned opt-in plan schema where the existing closed v1 shape
cannot represent the resolved binding. Pin both old and new canonical vectors.
An environment-required scenario must reject a missing selection or port;
environment generation without execution must be explicitly non-executing.

The binding records the host profile/backend revision, mode, scope/account
identity and exact tool-schema and execution-policy digests. Bind the actual
generation, validation and rubric dependencies that the selected path consumes,
including bytes rather than just rubric names. The dependency inventory must be
precise; do not hash unrelated configuration directories. Bind finite row,
turn, tool-call, fixture, output, snapshot and deadline limits. Revalidate inputs
at plan/preflight/start and immediately before claims or effects. Missing,
corrupt or changed dependencies reject before opening clients or environments.

Scenario text and generated fixtures are untrusted inputs. They may supply
bounded fixture contents and assertions, but cannot select a backend, credentials,
network permissions, broader actions or longer deadlines. Admit the effective
merged configuration against the host policy. Persist each generated fixture's
input digest before executing it. Tool-call shape and action interpretation stay
in declarative schemas/configuration, not Python constants for a current wrapper.

`execute` requires validated execution configuration and a qualified adapter.
Explicit `simulate` results must remain labeled simulation and cannot satisfy an
execution acceptance gate. Credentials and absolute scratch paths are absent
from public selection/binding documents. Launch authorization, generic port
conformance and observed backend execution are separate evidence.

The C1 admission contract adds the optional scenario declaration
`environment_execution: disabled|required`. `disabled` explicitly selects
generation of environment data without executing it; the existing
`environment: {enabled: false}` declaration is also explicit nonexecution.
`required` needs an environment selection, and a simulate selection cannot
satisfy that execution requirement. Unknown values, explicit null and nonstring
values reject. Either nonexecuting declaration requires no environment selection;
both execute and simulate selections conflict with it. `required` combined with
`enabled: false` is contradictory and rejects. Without either nonexecuting declaration, a scenario that requests
environment data needs an explicit selection. Scenarios without environment data
and requests without a selection retain the legacy request and plan documents.
There is no existing closed scenario-document JSON schema; C1 validates this
declaration during public reference admission. The opt-in closed dataset-plan v2
schema represents the selection and resolved environment binding.

C1 rejects external JSON Schema references and the legacy
`system: {template: <file>}` indirection rather than opening an unbound template.
It checks catalog ambiguity but commits only selected file bytes. Execution of
immutable admitted snapshots and validation of the effective fixture merge are
C2 requirements: nested fixture schema validity is not permission to override
host policy. Both execute and simulate starts remain unavailable, and their
preflight remains not ready, until a qualified adapter is connected.

## Session custody, usage and lifecycle

Open each episode with exact run, row, attempt, input and binding identities,
deadline and cancellation callback. The host owns private scratch and provider
handles. Limit fixture contents, files, snapshots and retained output. Contain
filesystem actions; retain descriptors where hostile path substitution matters.
Do not expose arbitrary host paths, shell execution or inherited network access
through the local fixture adapter.

Always close in `finally`. A partial creation failure must either clean up its
owned resources or retain an exact handle and uncertain outcome. A thread wait
timeout is not proof that an operation stopped; an advertised time bound needs
interruptible operations or a bounded execution process with demonstrated
termination. Avoid new effects after deadline or cancellation.

Persist the row attempt before creation through existing host record/stream
ports. Retain binding/input identity, exact effect identity and the later result
or uncertainty. Cancellation stops new episodes/tools/LLM calls, closes current
resources and preserves completed rows/traces. Recomposition can observe and
cancel retained state. Interrupted execution must not silently replay, renew
authority or infer absence. Resume requires exact termination/reconciliation
evidence; otherwise the attempt remains indeterminate. Full resumable data
workflow acceptance remains an open lifecycle gate.

LLM calls retain the existing MeteredLLMClient and Data spend ledger. Environment
usage is not token usage. A paid environment needs an explicit host grant and
existing effect/usage authority with measured resource receipts; reject an
unmetered paid backend before creation. Local/simulated modes declare no paid
environment effect. E2B will require explicit SecretRef resolution, pinned SDK,
retained handles, cancellation/reconciliation and its own bounded qualification.
Do not activate the legacy ambient `E2B_API_KEY` fallback through this port.

## Results and per-candidate feedback

Retain a bounded structured result with mode/backend, attempt/config/input
digests, finite execution status, action counts, assertion outcomes and safe
issue/output/state evidence. Distinguish assertion failure, infrastructure
failure, timeout, cancellation and simulation. Preserve useful structured
failure feedback for the judge and improver. Public diagnostics must not carry
credentials, host paths or raw provider exception text.

Generation reuses its existing environment-result feedback loop. Improvement
needs an optional generic per-candidate evaluation callback, absent by default.
Evaluate each changed response against the exact fresh fixture, update
`environment_result` context before judging/rejudging, and bind the final result
to the final candidate digest. Never reuse an earlier trace after a response
changes. The callback must not hardcode the dataset shape or current wrapper;
configured projections determine the response and fixture inputs.

This does not create a new environment workflow facade or an engine-owned
database. A generic factory's existence does not qualify local execution or E2B.

## Task DAG and acceptance

| Node | Dependency | Task-sized output and gate |
| --- | --- | --- |
| C1: contract and admission | Approved design | Explicit contract submodule, opt-in DataRequest/plan binding, precise dependency digests and host policy validation. Legacy bytes unchanged; missing/corrupt config, missing authority and scenario escalation reject before effects. |
| C2: local adapter and feedback | C1 | Host-injected local filesystem adapter and per-candidate generation/improvement feedback. Actual fixture actions, finite bounds, containment and cleanup tested; simulation is distinct. No security-sandbox or remote qualification claim. |
| C3: installed and lifecycle acceptance | C2, installed assets/import nodes A/B | Same deterministic fixture through checkout and installed sdist-built wheel from neutral CWD; actual public DataAPI execution, retained attempts, cancellation, uncertainty and recomposition checks. Remaining resume/recovery gaps explicitly open. |

C1 and the independent generic improvement callback passed 153 combined focused
tests at `846f754b`, plus the source-archive-built wheel resource/consumer gate.
The callback is not yet wired to a local environment adapter or generator.
Selected environment starts still reject before clients and claims. C2 must
enforce host authorization and prove bounded local execution and cleanup before
C3 can qualify the workflow.

The first C2 prerequisite retains private input captures from the same bounded
reads that establish dependency identity. Fresh re-admission must match the full
plan before returning a capture. Subsequent source mutation or deletion cannot
change those retained bytes. Public plans contain only dependency identities;
host policy, tool schemas, fixtures and dataset bytes remain reference-owned.
This process-local capture is not persisted recovery authority. Admission and
future workflow consumption share a bounded YAML parser that checks expanded
alias graphs before constructing mappings. The integrated capture/parser slice
passed 193 focused checks at `028fab5f`.

The snapshot workflow slice at `f7d2df31` passed 793 combined checks. Real
generation and improvement consume retained configuration, scenario, rubric and
dataset bytes after source deletion. Explicit wrapper parsing avoids ambient
registry lookup; optional parser/catalog injection preserves legacy defaults
when omitted. Pure validation rejects missing selections, unsupported client
overrides and malformed parser configuration before client creation. Snapshot
consumers preserve the injected metered client and select only admitted targets.
The executor catalog seam is available but is not yet an admitted execution
dependency. Host authorization, process termination, durable attempts and
execution remain separate C2 gates; execute and simulate starts remain disabled.

### Accepted local filesystem primitive


[`BoundedLocalEnvironmentRuntime`](../../shared/environments/bounded_local_runtime.py)
implements the existing runtime interface without selecting it as a default or
connecting it to DataAPI. The reviewed chain is `e41f764f`, `a7246526`,
`6e4c1db5`: 43 dedicated tests and 130 combined filesystem/validator/executor
checks passed. Lazy package exports preserve class identity while allowing this
primitive to import without attempting a provider SDK import.

The caller supplies an existing owner-private directory at mode 0700 and an
exact frozen `FilesystemLimits`. Unsupported Linux descriptor facilities reject
before scratch creation. Relative paths traverse retained directory descriptors;
absolute/traversal paths, symlinks, hardlinks and special entries reject. Parent,
scratch and traversed directory identities are checked, with owner/private-mode
checks on retained parent and scratch. This is filesystem custody, not a security
sandbox against arbitrary code or a hostile process running as the same user.

Limits separately bound source fixture entries, total files/directories, aggregate
UTF-8 file content, each returned result, snapshot JSON, path bytes/depth and
cooperative lifetime. Fixture validation precedes directory effects. Full bounded
inventory checks also cover recursive copy, search and snapshot work. A positive
snapshot entry request, including the interface default, is capped at the configured
entry bound; full inventory and snapshot byte limits still apply. Snapshots omit the absolute scratch path. Copy/move use bounded copy/delete, require capacity
for the temporary copy alongside its source, and are nontransactional: failure
may leave owned partial mutations; no atomicity or crash recovery is promised.

Cleanup has dedicated bounded descriptor traversal even after the cooperative
lifetime expires. Failed setup cleans only verified owned resources; unknown
creation identity, substituted names or changed private permissions preserve
ownership/uncertainty rather than deleting an unverified namespace or permitting
re-setup. This process-local retained state is not durable restart recovery.
Cooperative checks cannot interrupt blocked filesystem calls or establish process
termination, cancellation or an enforceable wall-clock resource limit.

The next gate is a host-authorized execution/session boundary with persisted
attempt authority, bounded process lifecycle and demonstrated cancellation,
termination, cleanup and uncertain-outcome handling. It must explicitly map
admitted public bounds to these independent filesystem limits and consume retained
inputs. The primitive alone supplies no host grant, policy allowlist, worker,
public execution enablement or provider qualification. Both selected execute and
simulate starts remain disabled until that separate gate is accepted.

The shared fixture starts with bounded files and a declared action/assertion
policy. A deterministic LLM reply first triggers a missing-file failure. That
structured result reaches the judge and improver; the revised reply performs the
declared filesystem action. Verify resulting bytes, assertion success and the
final evaluated candidate digest. Mock only LLM replies: use the real public
facade, adapter and filesystem execution in both installation forms.

Additional checks cover generated-fixture admission, explicit simulation,
configuration/schema/rubric/fixture mutation, denied scope escalation, traversal,
timeout/cancellation during execution, cleanup failure, private state custody,
persisted attempts without duplicate execution and retained uncertainty. Base
imports and local execution must not import E2B or another cloud SDK. A later
E2B live experiment needs its own concrete bounded authorization; this design
does not authorize one.

## Schema, exports and locked material

Update DataAPI serializers/guards, applicable dataset-plan/result schemas,
closed diagnostics, reference ports/composition and contract tests deliberately.
Keep optional result metadata outside legacy envelopes unless a versioned result
change is required. Update import-closure tests for the explicit submodule.

At `321a085f`, the inspected Data facade/reference implementation, SynthChat and
shared environment files are outside the four Modal runtime source inventories.
`synaptic_tuner/api/v1/__init__.py` is present in the training lock, inference
lock/closure and offline SFT closure. Avoid convenience root-export changes;
recheck every final changed path against inventories before any refresh. Moving
locked code must not remove it from its authenticated boundary.

Asset packaging proves installed bytes. Lightweight dependency fixes prove
imports. Contract tests prove the generic port. Actual local execution proves
the fixture path. None alone closes the stable release's provider parity,
cancellation, restart/recovery, publication or model-quality gates.
