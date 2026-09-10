# Modal adapter cutover

Status: Foundation-native engine cutover implemented locally; public execution
capabilities remain disabled pending consumer integration and live qualification.

This work belongs to the engine. Consumers supply configuration, credentials,
approval, and durable stores. No research-project implementation is required
by the adapter or its tests.

## Current boundaries

`TrainingCoordinatorV1` and the Foundation V2 broker own provider-neutral stage,
submit, cancellation, reconciliation, and durable transition ordering. The
Modal implementation now composes those existing engine contracts with exact
command binding, consumer-retained stage and launch facts, one-attempt provider
effects, authenticated reads, and the fixed remote worker. The former Modal
lifecycle modules were removed rather than wrapped or converted into a second
authority path.

The consuming host still owns authorization, evidence keys, configuration,
durable Foundation state, retained immutable catalogs, and explicit Modal SDK
client construction. The engine supplies ports and composition; it does not add
a provider database, adopt historical lifecycle records, or infer authority
from digests.

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

Preparation alone does not grant permission to stage or submit. The preparation
adapter still advertises no operational capabilities and its own preflight
refuses execution. Production composition instead requires the separately
authenticated operational preflight, the exact shared preparation, deployment,
explicit client, quote, source, clock, authorities, and consumer stores. The
checked-in registration remains capability-disabled, so this local cutover does
not by itself authorize provider execution. There is no fallback to the former
Modal lifecycle.

Provider-free tests drive the real generic coordinator, Foundation broker,
retention delegate, Modal host transport, worker admission, producer, and reader
against consumer-owned fakes. They check one-shot/restart behavior, complete
stage-to-submit lineage, cancellation targeting, substituted inputs, source and
deployment binding, bounded evidence, and zero provider I/O on denial. Synthetic
success does not prove a Modal API call, deployment, training run, artifact,
model quality, or spending approval.

## Import boundary

The provider package does not eagerly import the optional Modal SDK. The internal
Modal package has no compatibility re-exports for the removed lifecycle. The
explicit `synaptic_tuner.api.v1.modal` surface now exports the reviewed
Foundation-native composition types and functions; removed public lifecycle
names have no aliases. Importing the provider-neutral public API root remains
lazy and does not load `tuner`, Modal, SQLite, or host implementation code. SDK-free source and
installed-wheel checks cover the new Modal modules and packaged lock resources.

## Remaining work before activation

1. A consumer must bind the public composition to its reviewed durable stores,
   authorities, explicit Modal client, retained input source, configuration, and
   secret declarations. The engine intentionally provides no default database,
   credential source, signing key, or ambient-client fallback.
2. The capability-disabled registration must remain disabled until the exact
   committed and pushed tree passes the security/release barrier and an
   authorized live preflight. Provider-free tests and fake SDK objects do not
   establish current account, deployment, Volume, Secret, quote, or price facts.
3. Any paid smoke requires separate explicit approval and must preserve the
   existing one-attempt/indeterminate semantics. Historical legacy records are
   not migrated or admitted as Foundation authority.
4. A successful live run must verify authenticated terminal/log/completion
   evidence and exact bounded artifacts through the generic reader before any
   capability or release-readiness claim changes.

Internal adapter/service names do not need numeric suffixes. Existing public
namespaces and persisted or signed schema versions remain explicit; changing
an implementation does not itself require a new schema version.

Correction (2026-09-09): the earlier sections described the preparation-only
checkpoint and the now-removed Modal lifecycle as current. The atomic local
cutover now uses the Foundation-native coordinator composition, removes the old
runtime/producer/training/read path and `HostPorts.modal_reads`, and exposes the
new explicit Modal surface without compatibility aliases. This correction does
not turn provider-free conformance into live-provider or release qualification;
the registration remains capability-disabled.

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

The removed `remote.py` admitted `MutationCommandV1` and a v1 stage claim;
the historical `bundle.py` retains `OperationBindingV1` but is not used by the
new coordinator worker. Foundation stage and submit are
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

Correction (2026-09-09, public cutover): the retention, generic service,
candidate deployment builder, explicit bootstrap inventory, and public legacy
removal described above are now integrated and independently reviewed locally.
The public composition supplies both training and the existing generic runs
service to `APIHost`; list/show are exercised through the host facade, while
provider outcome/log/artifact reads remain capability-denied. The 97-member
Modal inventory and separate 66-member trainer closure both report `CURRENT`.
CI triggers cover every member of both inventories. These checks do not enable
live execution, prove consumer database durability, or qualify a release.
