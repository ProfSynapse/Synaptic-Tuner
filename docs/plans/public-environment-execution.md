# Public environment execution through DataAPI

Status: admission, immutable consumption and configured DataAPI session/worker
execution are implemented with installed CPU evidence. The acknowledged-custody
fixed guardian now covers owner death and stopped-child termination/cleanup;
pre-handoff owner loss, guardian/machine loss and unconfirmed cleanup retain their
limits. This is local supervision, not resumable generation or a full stable
release verdict. See the [current stable release status](stable-v1-release.md).
Root owns release task `97d0f9df-6701-4887-a7a3-2993a117f90f`.

## Existing boundary and reason for a reusable change

At merged source `321a085f61c9ac031f89daddde0bf95633e58133`,
[`ReferenceDataPortsV1`](../../synaptic_tuner/api/v1/reference/data.py) supplies
configuration, scenario and rubric directories and an LLM factory. Its generator
construction explicitly passed `environment_validator=None`. This historical
boundary motivated the optional host session-factory integration below;
configuration alone still cannot select or authorize an operational backend.

[`SynthChatGenerator`](../../SynthChat/generator.py) already carries environment
failures into response judging and improvement and then re-executes the revised
response. The optional integration also supplies
[`ImprovementEngine`](../../SynthChat/engine.py) with its generic per-candidate
evaluation callback. Both paths consume the admitted inputs and evaluate actual
changed candidates; a static prompt context alone cannot establish execution.

The legacy DataPlan binds parsed scenarios for generation, and dataset bytes
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
change root exports merely for convenience. The effect-free selection/binding types exist; session execution is supplied
by host composition rather than a new public facade or command family.

- `EnvironmentSelectionV1`: a logical host profile reference and explicit
  `execute` or `simulate` mode. Add an optional selection to DataRequest.
- `EnvironmentBindingV1`: the resolved immutable execution identity, policy,
  dependency digests, scope and finite bounds persisted with an opt-in DataPlan.
- `EnvironmentExecutionPort`: resolve a selection and dependency bytes without effects. It does not open
  sessions or mint permission. The optional `ReferenceDataPortsV1.environment_sessions`
  factory supplies the operational runtime after normal run claiming.
- `CapturedEnvironmentSessionFactory` prepares a validator from retained inputs
  and configured projections without opening a runtime/client. Its host-injected
  `runtime_factory(limits, boundary)` constructs the existing `EnvironmentRuntime`;
  setup and close occur only in the actual session. No provider credentials or
  current tool-wrapper constants are implicit in this seam.

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
host policy. Selected runs require the optional session factory; absent or
unsupported composition still refuses before client/session effects. The implemented
factory path has installed local workflow evidence; remote qualification and
durable restart recovery remain separate gates.

## Session custody, usage and lifecycle

The simplified composition uses normal DataAPI plan/preflight/start/show/cancel/
list and the existing host record/stream stores. Pure preparation validates the
retained workflow, host policy, tool schema/catalog and configured feedback
projection before runtime or LLM client creation. Start creates the normal
create-only run record before client/session effects, then uses existing CAS
transitions and observations. Recomposition finds that claimed run rather than
executing it again. Missing factories, invalid inputs and conflicting claims
remain refusals; no automatic child respawn or uncertain workflow replay occurs.
The proposed separate per-effect grant/episode DTO and parallel v2 lifecycle,
result and list framework were cancelled after user review. They are not required
for this composition.

The host supplies the private scratch parent, runtime factory and parent-only
boundary callback. The host supervisor retains scratch anchors and its actual
guardian `Popen` handle, transports only bounded whitelisted fixture operations,
and checks the original deadline/cancellation while polling IPC on the owning
thread.
After acknowledged custody, a fixed guardian owns the exact worker and scratch
descriptors. Normal close or owner-pipe EOF requests termination/escalation/wait;
confirmed child exit precedes guardian cleanup. Parent cleanup retry requires
confirmed exit and relinquished guardian custody. Lost replies, unknown startup,
exit or cleanup preserve consumed ownership/uncertainty instead of permitting execution again. Infrastructure
`WorkerLifecycleError` is fatal and must not become model-quality feedback.

Session closure occurs in `finally`. Cancellation prevents subsequent model/tool/
row work and requests owned-child termination, while preserving completed rows.
Unknown active attempts retain RUNNING or CANCEL_REQUESTED with safe diagnostics;
they are not falsely terminalized. Private controller/descriptors are process-local:
a cold composition cannot recreate kill authority from a stored PID or declare
cleanup merely because a receipt is absent. Acknowledged-custody owner-death
cleanup is now tested with real SIGKILL and a stopped worker. Pre-handoff death, guardian/machine loss and unconfirmed cleanup
remain outside that guarantee. No PID-based adoption, regenerated lease or row
continuation is claimed here; advertised cancellation still needs actual backend
qualification.

Worker IPC and termination waits are finite. `Popen` construction and parent
filesystem setup/cleanup remain synchronous OS calls; this is not a universal
wall-clock guarantee or a security sandbox. Killing the fixture child cannot
cancel an in-flight parent LLM/provider request or establish its measured spend.
Boundary cancellation and host-client transport deadlines remain distinct.

LLM calls retain the injected metered client and existing Data spend ledger.
Local/simulated environments declare no paid environment effect; explicit
simulation cannot satisfy execution acceptance. No paid remote environment or
ambient E2B credential fallback is enabled by this local composition. Remote
backends and measured environment usage need their own supported host adapters
and bounded qualification.

## Results and per-candidate feedback

Retain a bounded structured result with mode/backend, attempt/config/input
digests, finite execution status, action counts, assertion outcomes and safe
issue/output/state evidence. Distinguish assertion failure, infrastructure
failure, timeout, cancellation and simulation. Preserve useful structured
failure feedback for the judge and improver. Public diagnostics must not carry
credentials, host paths or raw provider exception text.

Generation reuses its existing environment-result feedback loop. Improvement
uses the optional generic per-candidate evaluation callback, absent by default.
Evaluate each changed response against the exact fresh fixture, update
the configured feedback context before judging/rejudging, and bind the final result
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
These are historical prerequisite checks. The current optional factory connects
local execution and per-candidate feedback through DataAPI. Runtime `8de9e430`
passed 916 combined tests with one intentional skip. Independent review and a
fresh sdist-built installed wheel each passed all 51 session/public cases, with
only the non-worker SIGSTOP variant skipped. The installed check uses isolated
Python, neutral CWD and installed fixture assets, and now runs in CI. Tests cover
actual repair, stopped-child termination/cleanup, cancellation races, retained
uncertainty and no replay after a failed final CAS. They do not qualify durable
process restoration, resumable continuation or multi-host transition recovery.

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
The session integration now consumes retained tool schema/catalog and configured
execution inputs. Existing run claims and fixed-worker process supervision supply
the selected local path, including installed public acceptance. Complete durable
recovery remains a separate gate. Missing optional factories still refuse selected execution.

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

### Fixed worker and public integration acceptance

[`WorkerEnvironmentRuntime`](../../shared/environments/local_worker.py) adds the
accepted fixed-child supervisor at `27dde56e`. Real-child tests cover stalled
termination, cancellation on the parent thread, backpressure, lost replies and
retained cleanup uncertainty. Its fixed `-I` entrypoint works from checkout and
an sdist-built installed wheel without CWD/PYTHONPATH lookup or inherited
credentials. The existing neutral-wheel CI smoke verifies fixture writes/reads,
default snapshot, confirmed child exit and empty owned scratch. This establishes
the primitive and packaging; it does not alone qualify the composed public path.

The optional session integration maps the admitted bounds into independent
`FilesystemLimits`, uses configured action/projection data and carries structured
assertion failures into judging/improvement. Changed candidates use fresh fixtures;
supervisor/cancellation/cleanup failures do not masquerade as assertion success
or ordinary model quality. Installed public tests exercise these facts through
normal DataAPI, including retained claims, final CAS failures and cancellation
of a stalled owned worker. Restart no-replay evidence is distinct
from restored process ownership, safe continuation or complete durable recovery.

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
the fixture path. No primitive alone closes the stable release's provider parity,
cancellation, restart/recovery, publication or model-quality gates.
