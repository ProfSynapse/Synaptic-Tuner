# Modal Foundation effects slice review

Status: bounded adapter implemented; production transport intentionally absent.

## What this slice establishes

`ModalFoundationEffectExecutor` and `ModalFoundationReconciliationAdapter`
implement the Foundation executor and reconciliation shapes without importing
or invoking the old Modal training lifecycle. Both resolve an immutable
authenticated command binding by the exact Foundation command digest, require
the injected authority to authenticate it, reparse the retained canonical
command, and compare its provider, profile, account, namespace, command, and
preparation scope before provider transport is entered.

The operational transport is deliberately injected. Dispatch makes exactly
one transport call. Reconciliation makes exactly one lookup call. A transport
can return only a closed Foundation disposition plus the one typed provider
reference appropriate to stage, submit, or cancel. Non-found results cannot
carry references. Definite absence additionally requires evidence acceptable
to Foundation; weak Modal lookup failures must therefore remain indeterminate.

The command-binding boundary now consumes the exact immutable
`ModalCommandBinding`:

```text
catalog.resolve(command_digest) -> authenticated binding
authority.authenticate(binding) -> bool

binding byte snapshots:
  command_bytes
  preparation_snapshot
  deployment_bytes
```

The effects adapter reconstructs a fresh binding from all three byte snapshots
before authentication and then uses the reconstructed snapshot. The
authentication implementation must bind its complete `canonical_bytes`; the
derived digest is an identifier, not authentication by itself.

## Honest remaining seams

The checked-in `_ExplicitModal154VolumeWriter` requires `StageMaterialV1`,
whose expectation is built from the legacy `OperationBindingV1`. The existing
remote worker and function mutator likewise parse the legacy
`MutationCommandV1`. Foundation commands must not be converted into those
types or submitted to that worker under a false compatibility claim. A real
transport therefore still needs a Foundation-native stage claim, bundle, and
remote worker wire format before registration.

The common binding reconstructs the declared retained preparation commitments,
including the profile, execution binding, resources, source/runtime digests,
artifact policy, quote digest, and secret requirements. This proves internal
configuration equivalence; it does not authenticate provider/source state or
prove quote freshness.

Cancellation needs a checked-in one-attempt transport that resolves only the
exact submitted `FunctionCall` and preserves an ambiguous provider boundary as
indeterminate. Lookup needs authenticated stage evidence and exact provider
job/cancellation evidence; `FunctionCall` unknown/not-found alone is not proof
of definite absence.

Consequently this module is not exported, registered, composed, or advertised
as ready. No cloud calls, credentials, deployments, or paid jobs were used by
its tests.
