# Modal adapter cutover

Status: preparation slice implemented; execution cutover incomplete.

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

1. Implement the Foundation executor and lookup adapter. Bind each command to
   the exact resolved session, prepared inputs, and durable grant. Reuse the
   existing Modal staging and worker mechanisms without nesting the old broker
   or minting a second authority from a new-format command.
2. Implement the generic authenticated run-reader port. Require the complete
   Foundation submit record and assessment before provider reads. Preserve
   authenticated terminal, log, artifact inventory, and byte verification.
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
