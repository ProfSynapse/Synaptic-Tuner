# Modal adapter workstreams

Checkpoint: 2026-09-09. Engine-only implementation; no EHR changes.

The common base is local commit
`af71ffee2540a51d573f5cd5365772a53b6d3f0e`. Its preparation adapter is
intentionally non-operational. Existing tests establish provider-free
regressions, not live cloud or trained-model readiness.

## Complete slice map

| Slice | Deliverable | Dependency / ownership |
| --- | --- | --- |
| 0. Preparation | Generic planning, binding, canonical preparation; executable preflight refuses | Complete locally at the common base |
| 1. Effects | Foundation stage, submit, cancel and conservative lookup/reconciliation | Sol effects lane; production transport depends on 3 |
| 2. Authenticated reads | Full submit-proof validation before status/log/inventory/byte reads | Sol reader lane; common command binding and evidence depend on 3 |
| 3. Remote wire and shared authority | One canonical command lineage through staging, remote admission and worker evidence; packaged source/runtime locks updated together | Lead; coordinate signatures before agents depend on them |
| 4. Real preflight and restart | Authenticated deployment/client/Volume/quote checks; exact configuration retained through consumer-owned persistence | Lead / subsequent bounded lane after 3; no new engine database |
| 5. Public cutover and consumer proof | Existing lazy registry and generic coordinator composition; remove old lifecycle and Modal-specific host field; minimal consumer fixture inside engine | Integrate 1–4; atomic cutover, no legacy fallback |
| 6. Retrieved-model usability | Verified local materialization and explicit model-load/inference check; no claim that training completion alone proves usable weights | After reader and public cutover; separate from model quality evaluation |
| 7. Bounded chat | Easy local/Modal conversation using shared inference/evaluation plumbing, explicit stop and bounded GPU lifetime | After 5–6; no perpetual endpoint by default |
| 8. Qualification and release | Repeatable provider-free CI now; integrated conformance, independent audit, scoped live proof, docs and feature-branch integration later | Sol conformance lane now; lead owns final activation/release decisions |

This map is for the agreed Modal/submodule-first scope, not a requirement to
implement every cloud provider before the first release. Main-branch merge,
publication and release are not authorized by assigning these tasks.

## First-wave ownership

All three agents use `gpt-5.6-sol`. Worktrees live under
`/home/profsynapse/code/synaptic-tuner-worktrees/`.

| Agent | Worktree / branch | Exclusive writable paths | Checkpoint |
| --- | --- | --- | --- |
| `/root/modal_effects_slice` | `modal-effects` / `feat/modal-effects` | `tuner/execution/providers/modal/coordinator_effects.py`; `tests/execution/providers/test_modal_coordinator_effects.py`; `docs/review/modal-effects-slice.md` | Exact immutable binding integrated; scope fix reviewed; 64 focused tests independently passed under pytest 8.4.2 |
| `/root/modal_reader_slice` | `modal-reader` / `feat/modal-reader` | `tuner/execution/providers/modal/coordinator_reader.py`; `tests/execution/providers/test_modal_coordinator_reader.py`; `docs/review/modal-reader-slice.md` | Complete authenticated Foundation derivation and adversarial reader checks integrated; independent final review passed |
| `/root/modal_conformance_slice` | `modal-conformance` / `feat/modal-conformance` | `.github/workflows/provider-free-conformance.yml`; `docs/review/modal-conformance-slice.md` | Includes binding/effects/reader tests, verified immutable Actions, neutral-CWD wheel/resource checks; CI not dispatched |

Lead worktree: `modal-coordinator-adapter`, branch
`feat/modal-coordinator-adapter`. Lead owns this record, shared contracts,
preparation, registration/composition, existing runtime files, public exports,
integration and review. Agents must request path expansion rather than edit
another lane. No agent commits, pushes, merges, installs dependencies, accesses
credentials or makes cloud calls in this wave.

Approved next-wave extensions: effects owns new `coordinator_staging.py`, its
test and `modal-staging-slice.md` only; conformance owns the new runtime-lock
regeneration script/test/review note plus a scoped AGENTS maintenance rule and
canonical Modal skill-reference section with synchronized mirrors. Lead copied
the shared preparation adapter, command-binding module and binding tests into
both runtime worktrees as read-only dependencies for those agents.

Subsequent bounded lanes: effects owns new `coordinator_launch.py`, its test
and review note, plus a scoped stage-reference identity helper/test correction;
conformance independently reviews launch and the integrated candidate. Reader
owns new `coordinator_bundle.py`, its test and review note after agreeing its
Foundation-native member/digest contract. Existing reader implementation is
frozen. These are local wire-preparation changes, not production activation.

Measured integration checkpoints: 923 tests passed on pytest 9.0.2 before the
final reader changes; 926 passed on pytest 8.4.2 with runtime-lock checks and
the exact effects binding. Neither count is a final all-patch qualification.
The pytest 8 environment is a temporary venv using existing system packages;
it does not change the training environment and is not a clean CI runner.

Correction (2026-09-09): a separately created clean temporary environment,
without system-site packages or the Modal SDK, passed the integrated 1,070-test
selection in 166.20 seconds using CPython 3.12.9 / pytest 8.4.2. It includes
final effects, reader, staging and Volume identity checks, but predates launch
and lock-regenerator integration. The latter module separately passed 13 tests
in 0.29 seconds. The installed non-editable wheel imported all five integrated
adapter modules and both packaged runtime resources from a neutral working
directory. These are local checks; CI and live Modal execution remain unrun.

Launch review found a cross-material substitution gap: authenticating a stage
record and stage material separately does not prove the record established
that material. Correction (2026-09-09): launch now requires the authenticated
provider stage reference to equal `modal-stage-claim:<sha256(claim)>`; the
cross-material substitution regression passes and independent review is
closed. The exact 68-test launch/staging/binding/preparation selection passed
independently.

The subsequently combined lead selection passed **1,091 tests in 164.73
seconds**, including launch and lock regeneration, in the same clean pytest 8
environment. A test-only improvement then made the launch fixture's tag cover
the complete payload rather than its length; all eight launch tests passed
again in 5.57 seconds. The launch-integrated installed wheel and both packaged
resources passed neutral-directory checks without the Modal SDK. No CI run or
production activation is implied.

Current next-wave work: effects owns `tuner/training/coordinator_material.py`,
its training test and review note; reader owns the Foundation bundle codec,
its provider test and review note; conformance owns `coordinator_wire.py`, its
provider test and review note. Lead supplies shared files as read-only inputs
and reviews boundary derivations before integration. The material lane must
connect the generic planning DTO to exact recompiled source/workload bytes,
keep domain fingerprints distinct from raw SHA-256, and retain no mutable
caller input. The wire lane receives no host grant/signing services.

Correction (2026-09-09, subsequent integration): the first bounded adapter
checkpoint is committed locally as `eecdb93f1f614f8458b5dba9ae90cb24fbe3ddc8`.
Resolved material, the eight-member bundle, remote wire admission and the
single-argument dispatch codec have since passed their independent reviews.
The lead measured 1,177 tests passing in 167.59 seconds before dispatch
integration, then 19 dispatch/wire tests in 11.78 seconds. These are separate
runs, not one combined result. The candidate wheel containing those four new
modules imports from its installation in a neutral directory without Modal;
its SHA-256 is
`5404e0972a607433499fe38d2937fb966b82d9a9d0419e473a04b568fec52760`.

The current bounded lanes are now worker mechanics extraction (effects),
Foundation-native worker admission/orchestration (reader), and host semantic
submit preparation plus a real coordinator/bundle test fixture (conformance).
The latter positive path uses real stage and submit records, semantic bundle
compilation, dispatch encoding and wire admission; only the external authority
ports are provider-free fakes. Lead review and combined tests precede local
integration. Worker extraction changes locked runtime sources, so the lock
inventory and hashes must be updated together before qualification.

Worker review correction (2026-09-09): checking a dispatch's shape and static
deployment identity is not launch authentication. The mounted worker must
verify its embedded launch claim before any Volume-file read, then perform
full stage/bundle admission before source or process effects. The admitted
invocation is immutable reconstructed data, not a new authority service or
permission to execute independently of that admission boundary.

The next combined lead run passed **1,192 tests in 175.25 seconds**, including
dispatch and real host submit preparation, under the same clean pytest 8
environment. This checkpoint predates worker extraction and its runtime-lock
refresh. No cloud call, CI dispatch, push or merge has occurred.

This material/dispatch checkpoint is local commit
`353cdeeca93b07ea638e3f5508e71686b96ca8e8`. The subsequent worker-mechanics
extraction passed independent review, and the lead passed 166 focused tests in
1.62 seconds after updating its lock. The lock schema, runtime policy and
maintenance script explicitly require ten members, including the extracted
`worker_ports.py` and `worker_source.py`; its check reports `CURRENT` and both
skill mirrors are synchronized. Image, dependency, SDK, Python and ML-stack
pins are unchanged. The unchanged 66-member offline trainer closure does not
contain these provider bootstrap files.

Import correction (2026-09-09): previous adapter checks proved absence of the
SDK and old training lifecycle, not absence of every legacy provider import.
The internal package initializer still eagerly imported the old bundle then.
It now has no reexports, and a fresh-process runtime import loads neither the
legacy remote module, bundle nor broker. The public Modal API files remain
unchanged and its import still needs no optional SDK.

The corrected Foundation worker has now been integrated for combined testing;
its author measured 12 focused tests passing after fixing pre-read launch
authentication and source/workload digest reconstruction. It is still not
installed in the production wrapper. The effects lane owns the new completion
producer; conformance owns a Foundation host transport with one-attempt
spawn/cancel and conservative indeterminate reconciliation. Consumer-owned
retained stage and launch facts must be connected during composition; these
new source protocols grant no authority and add no engine database.

The worker/extraction combined checkpoint passed **1,210 tests in 181.88
seconds**. Subsequent review strengthened mounted artifact I/O with streaming
hash verification, exclusive output-directory claims and bounded exact regular
file inventories; malformed bounds/content now fail before creating output
paths. Those helper and existing producer regressions passed 43 tests in 0.80
seconds. The lock's mounted-I/O source hash was deliberately refreshed; this
postdates the extraction-only hash-delta audit above.

The Foundation producer passed 15 focused tests, including actual temporary
filesystem publication and verification of all three evidence MACs; independent
review also passed. The host transport passed independent review and then 12
focused tests after added malformed-result/deployment/verifier cases. These
modules are integrated locally, not installed in production composition.

Read integration exposed a timestamp gap: the old log wire has only code and
message, while the public log entry requires a timestamp. The conformance lane
now owns a separate timestamped Foundation log codec, leaving the old parser
unchanged; reader owns authenticated metadata-only inventory and separate bounded
artifact streams. Effects owns operational preflight design using existing Host
evidence authority, exact Volume/Secret hydration and an explicitly authenticated
price-policy fact. No invented live pricing API or secret-value read is required.

The combined worker/producer/transport checkpoint passed **1,264 tests in
199.55 seconds**. Its installed wheel passed SDK-free imports of 16 modules
and both packaged resources from a neutral directory; wheel SHA-256:
`34d74b732b66b4f55fefce38f891085465e68d68bc0ecbda39d5f3a7b609b18f`.
That wheel and full-suite result predate the timestamped log revision. After
integrating that revision, the lead separately measured **97 tests passing in
26.74 seconds** across logs, producer, worker, transport, facade and mounted
I/O. Independent review closed the mounted-input and pending-poll fixes.
Authenticated read transport and operational preflight remain under review;
none of these local results activates the public cloud path.

The next combined run, including timestamped logs, passed **1,279 tests in
201.16 seconds**. The ten-member runtime lock reports `CURRENT`, skill mirrors
are synchronized, and the working diff passes whitespace checks. Read transport
and preflight are not included in this result. Review corrected an earlier
nonterminal assumption: a provider poll timeout cannot prove either queued or
running, so the read transport must report unavailable rather than invent a
phase until authenticated evidence establishes one.

That reviewed checkpoint is committed locally as
`2c42bef67a96283568e6834f78d88744df20308d`. Operational preflight was then
independently reviewed and integrated; the lead measured 112 preflight,
preparation, bundle and facade tests passing in 3.29 seconds. Its quote policy
requires a maximum five-minute age/lifetime, with full source/deployment/quote
authentication and trusted identities checked before provider reads. The
factories lane and consumer-owned retention/delegation lane are now active;
the reader lane is adding unpatched-facade fake-Volume integration coverage.

The reader subsequently passed final independent review with the unpatched
explicit-client facade and fake SDK Volumes. Its lead integration, preflight
and inactive factories passed **1,326 tests in 223.91 seconds** together. The
installed wheel passed neutral-directory imports of 20 modules and both
packaged resources with no Modal SDK; SHA-256:
`af7724f83af1a8c3c95c5893e75f177b32ef267550c2f257121e1cdba18c05b4`.
Generic load/resolve/plan service composition and the candidate remote wrapper
are now separately in progress. An import probe measured 86 engine source
files loaded by the current worker/producer/runtime entrypoints: the current
ten-member lock is not a complete new-bootstrap inventory. That inventory
must be explicitly qualified before remote activation; the 66-member trainer
closure is a separate artifact.

## Shared contract checkpoint

The existing `ExecutionResolutionRequestV2` remains unchanged. A digest is an
identity, not authentication: resolution must recover and authenticate the full
canonical command and exact retained Modal configuration before I/O. Use the
existing command-catalog pattern, not another registry or authority system.
Both adapter lanes agreed provisionally on `catalog.resolve(command_digest)`
and a separate `binding_authority.authenticate(value)` boundary. The immutable
value retains exact command bytes, command digest, provider/profile/account/
namespace, the explicit Modal client binding and verified deployment. Its
authentication must cover all retained content, not merely a supplied digest.
Adapters must also recheck configuration-derived commitments; missing fields
needed to recompute the preparation adapter's profile digest remain an explicit
integration requirement. The lead owns the concrete common type. Private
structural typing protocols let the lanes proceed without importing an
unimplemented shared module or changing generic Foundation contracts.

The pre-submit binding cannot claim a provider job reference. Readers obtain
that reference from the complete authenticated submit outcome. Remote read
transports must retain authenticated identity and terminal/log metadata rather
than returning unbound chunks or interpreting an unknown provider state as
success. These provisional interfaces are not production-composition approval.

Measured at the common base: `modal/mutation.py` parses legacy
`MutationCommandV1`; `modal/remote.py` also imports that command;
`modal/staging.py` requires `StageMaterialV1` whose expectation contains a
legacy `OperationBindingV1`. These cannot be treated as Foundation-native
transports merely by relabeling arguments. Port implementations tested with
injected transports are intermediate work, not evidence of working production
transport. Public readiness stays disabled until that dependency is resolved.

Artifact inventory should not bulk-read model weights. Inventory authentication
and bounded verified streaming are separate reader operations.

## Monitoring and evidence

Active-session polling correction (2026-09-09): the lead must not end a turn
merely because an agent is still working or just reported completion. Check
agent status against a 60-second clock deadline while doing local work; use
short bounded waits when otherwise idle. Drain completed handoffs into review,
fixes and integration immediately. A stopped session still cannot promise a
background wakeup. Check the deadline between bounded work chunks, not only
when messages arrive.

Agents report directly at interface agreement, first tests, blockers and final
handoff, with periodic updates during active work. The lead checks agent
status/messages and reviews diffs and tests before integration. Handoffs must
list exact paths, measured test results, unexecuted checks and remaining seams.

PACT is not exposed among this session's tools or available skills. This is a
repository checkpoint plus built-in agent coordination, not a PACT installation
or a recurring background monitor. This file survives a restart; agent-process
continuity and timed monitoring while the session is inactive are not promised.

The baseline measurements used CPython 3.12.9 / pytest 9.0.2; the declared test
extra requires pytest below 9. Preserve that distinction when reporting local
regressions versus pinned-environment qualification. Candidate tests run with
an empty credential environment and plugin autoload disabled, without
`PYTHONPATH`, from the clean standalone engine source directory.
