# Modal adapter cutover

Status: preparation, complete-command configuration binding and injected
effect/read boundaries implemented locally; production execution cutover incomplete.

This work belongs to the engine. Consumers supply configuration, credentials,
approval, and durable stores. No research-project implementation is required
by the adapter or its tests.

## Existing boundaries

The engine contains two execution paths. The existing Modal implementation in
`tuner/execution/providers/modal/training.py` still owns its lifecycle. The newer
`TrainingCoordinatorV1` in `tuner/execution/coordinator_v1` owns provider-neutral
stage, submit, cancellation, reconciliation, and durable transition ordering.
Docker already implements ports for that coordinator.

Connecting Modal means implementing those same ports, not wrapping the old
Modal lifecycle behind a renamed service. The existing Foundation registry,
broker, coordinator, public contracts, and consumer-owned persistence remain
the integration points. This slice does not add another registry or database.

## Implemented preparation adapter

`tuner/execution/providers/modal/coordinator_adapter.py` defines the internal
`ModalPreparationAdapter`. It implements:

| Existing port | Adapter behavior |
| --- | --- |
| `PlanningPortV1` | Describes the preparation-only capability set, binds one resolved request to a provider context, and refuses executable preflight. |
| `ProviderBindingResolverPortV1` | Resolves only the exact configured provider, profile, and plan context. |
| `PreparationMaterializerPortV1` | Builds and rechecks `CanonicalPreparationV2` and canonical effect payloads for the existing coordinator. |

Configuration is snapshotted into bounded canonical bytes. The adapter reuses
the checked-in Modal profile parser, runtime lock, deployment selection builder,
and environment/Secret policy. Workspace and environment form a canonical tuple
digest, not an ambiguous concatenated namespace. Configuration, session scope,
resources, and the supplied quote digest contribute to the execution binding.

Only generic references and digests enter plans and preparations. Function
names, Volume names, runtime environment values, and Secret declarations remain
inside the adapter. A supplied quote digest and a locally constructed deployment
selection are commitments, not authenticated provider observations.

Preparation does not grant permission to stage or submit. The adapter advertises
no operational capabilities and returns `ready=False` with the closed diagnostic
`modal_coordinator_execution_unavailable`. It is not registered as an executable
provider. There is no fallback to the older Modal lifecycle.

The tests drive the real generic coordinator through stage and submit using its
existing synthetic Foundation backend. They check one-shot/restart behavior,
preparation and command lineage, substituted inputs, consumer-input mutation,
scope binding, and the refusal of the adapter's production preflight. Synthetic
success does not prove a Modal API call, deployment, training run, or artifact.

## Import boundary

The provider package no longer eagerly imports Modal. The internal Modal package
no longer re-exports its older composition functions. Import those from their
defining module, or use the existing public `synaptic_tuner.api.v1.modal` surface.
No compatibility aliases were added. The public API exports remain unchanged.
Importing the new adapter is tested without loading a provider SDK, consumer
code, SQLite, or the old Modal training lifecycle.

## Remaining work before activation

1. Connect the Foundation executor and lookup adapter to a Foundation-native
   staging, submit, cancellation and evidence transport. Their injected-port
   tests do not establish production transport. The older stage material and
   worker command still use different contracts; do not convert a new command
   into old authority or send it to a parser expecting the old wire format.
2. Connect the authenticated run-reader boundary to real terminal/log/inventory
   reads. Inventory authentication must not download model bodies; byte streams
   independently verify exact membership, identity, bounds and hashes.
3. Connect authenticated preflight, including deployment/Volume identities,
   quote expiry, runtime/source locks, and explicit client selection. A digest
   alone must never become authority. Persist the exact resolved configuration
   through consumer-owned ports before allowing restart.
4. Register only the implemented factories and capabilities in the existing
   lazy registry. Test interruption, ambiguity, restart, cancellation, and
   artifact retrieval through the same coordinator as other adapters.
5. Cut over the public composition atomically, remove the old Modal lifecycle
   path and `HostPorts.modal_reads`, and verify a clean minimal consumer. Do not
   migrate historical run authority implicitly. Live proof remains a separate
   gate after provider-free conformance and review.

Internal adapter/service names do not need numeric suffixes. Existing public
namespaces and persisted or signed schema versions remain explicit; changing
an implementation does not itself require a new schema version.

## Complete-command configuration binding

`ModalPreparationAdapter.snapshot()` emits canonical non-secret configuration
and plan-basis bytes for consumer-owned retention. `restore()` runs the normal
profile/runtime/environment policy again and requires a byte-identical rebuilt
snapshot. It does not simply accept the retained digest fields as correct.

`ModalCommandBinding` retains only immutable command, preparation-snapshot and
deployment bytes. It reconstructs the complete expected preparation, payload
and executor from that configuration and requires exact deployment-selection
equivalence. A changed source/workload/runtime/resource/quote/secret commitment
cannot reuse an old command. Parsed nested objects are detached from retained
bytes. The catalog's separate authority must authenticate the full canonical
content; the binding digest is an identifier, not a signature or permission.

This establishes equivalence to the retained configuration, not evidence that
source is pushed/available, a quote remains fresh, or a deployment attestation
is authentic. Those are explicit preflight and composition responsibilities.
The preparation adapter remains non-operational after snapshot restoration.

## Remote cutover constraint

The existing `remote.py` admits `MutationCommandV1` and a v1 stage claim;
`bundle.py` retains `OperationBindingV1`. Foundation stage and submit are
different effects, with submit carrying the exact authenticated stage
predecessor. Relabeling the older bundle or launching it through a wrapper
would not preserve that lineage.

The new stage writer therefore writes its bounded bundle and v2 claim below
the stage effect. Host-local launch preparation must derive the authenticated
Foundation stage result, prove that result established the exact staged
material, and bind the subsequent submit. A valid signature on each of two
unrelated objects does not prove their relationship.

Remote admission must receive a bounded authenticated wire record, not host
grant/receipt/assessment authenticators, catalogs, repositories, or signing
authority. It must independently check the exact stage and submit commands,
stage material and deployment identities before source or process I/O.
Worker logs, output artifacts and terminal evidence belong to the submit
effect, not the staging effect. A stage bundle cannot contain an as-yet
unknown submit effect or accept caller-supplied runtime argv as authority.

The atomic activation set is the Foundation-native bundle and launch wire,
durable submit transport and reconciliation, remote admission/invocation,
worker evidence production, authenticated reader transport, and public
composition/registry cutover with old lifecycle removal. Local wire codecs
and injected transports may be reviewed before activation, but do not make
the provider executable. The engine continues to define ports rather than
selecting a consumer database.

## Verification checkpoint (2026-09-09)

The unchanged baseline passed 518 coordinator, Foundation, selected Modal, and
provider-neutral contract tests. Its tested tree at `f00cef3` equals feature
merge `68f3bea9`. The candidate passed 40 new adapter tests, then 896 tests in
the broader coordinator, Foundation, mutation-broker, public Runs/Training,
optional-dependency, and complete `test_modal_*.py` provider set. An independent
source review found no remaining blocker for this preparation-only slice.

Tests ran from the clean engine release directory with explicit candidate
imports, an empty credential environment, plugin autoload disabled, and no
`PYTHONPATH`. These measurements used CPython 3.12.9 and pytest 9.0.2. The
declared test extra currently specifies pytest below 9; this is a same-environment
regression comparison, not a pinned release-environment qualification. No
provider lookup, cloud mutation, paid training, publication, or chat serving
was performed. EHR was not changed.

Correction (2026-09-09): after complete-command binding, effects, reader and
stage-writer integration, the expanded provider-free selection passed 1,070
tests in 166.20 seconds under clean CPython 3.12.9 / pytest 8.4.2 without the
Modal SDK or system-site packages. Installed-wheel imports of those adapter
modules and both packaged runtime resources passed from a neutral working
directory. The new runtime-lock maintenance module separately passed 13 tests
in 0.29 seconds. These measurements precede launch/bundle integration and
do not replace pending production cutover, CI execution or live qualification.

Correction (2026-09-09, later local integration): the host can now derive an
immutable resolved-material record, recompile it into an exact eight-member
stage bundle, authenticate Foundation stage/submit lineage, and encode one
bounded dispatch argument. Remote wire admission receives no Host grant,
receipt repository or signing authority. The fixed worker must independently
match its static deployment selection and authenticate the launch before
mounted reads, then authenticate stage evidence and recompile the bundle
before source or process effects. These contracts do not activate the public
adapter: production transport, worker completion, preflight and atomic legacy
lifecycle removal remain outstanding.

The expanded material/bundle/wire selection passed 1,177 tests in 167.59
seconds; dispatch/wire subsequently passed 19 tests in 11.78 seconds. Both
used the clean CPython 3.12.9 / pytest 8.4.2 environment without Modal. These
are distinct checkpoints, not a live execution or a combined final gate.

Correction (2026-09-09, subsequent local implementation): the Foundation worker,
timestamped evidence producer, one-attempt host transport, authenticated read
transport, current-fact preflight and inactive registry factories are now
integrated. The combined provider-free selection passed 1,326 tests in 223.91
seconds in the clean pytest 8 environment; the corresponding installed wheel
passed SDK-free imports and packaged-resource checks. This closes the earlier
individual transport/preflight implementation gaps, not the activation gate.
Consumer-owned restart retention, the generic five-method training service,
candidate deployment/bootstrap qualification, and atomic public legacy removal
remain in progress. Unknown live provider phase is reported unavailable rather
than inferred from a poll timeout.
