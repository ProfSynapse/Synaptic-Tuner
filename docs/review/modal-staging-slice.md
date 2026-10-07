# Modal Foundation staging slice

Status: provider-free staging mechanics implemented; not registered or ready.

`coordinator_staging.py` introduces a Foundation-native stage claim and Volume
writer without constructing `OperationBindingV1`, `StageMaterialV1`, a legacy
mutation command, or another grant. The existing Foundation grant remains the
sole mutation authority.

The preparation function accepts an exact immutable `ModalCommandBinding`,
reconstructs it from its three canonical byte snapshots, requires an injected
authority to authenticate the reconstructed complete content, and accepts an
already-materialized bounded bundle as opaque bytes. It requires an exact
`StageCommandV2` and signs a `synaptic.modal-stage-claim/v2` claim for purpose
`modal-stage-claim/v2`. The claim embeds the full canonical stage command and
binds its command and binding digests, provider scope, project/run, stage
effect, nonce, plan and preparation, exact Volume IDs and key reference, and
bundle SHA-256 and size.

Before its first provider observation, `ModalFoundationVolumeWriter`:

1. reconstructs the command binding and stage command;
2. reconstructs and byte-compares the entire claim;
3. reauthenticates the complete binding;
4. verifies the existing claim signature;
5. checks that the facade's cached ID-to-name mapping resolves exactly to the
   retained profile's configured control and artifact Volume names; and
6. verifies the explicit client session through the facade.

Every subsequent facade Volume resolution also hydrates the named object and
requires its provider `object_id` to equal the requested retained Volume ID.
This closes name-reuse or replacement races across listing, readback, and
upload; a successful name lookup alone is never treated as object identity.

It then lists only the exact `operations/{stage_effect_id}/` prefix, with a
bound of two expected control entries plus one collision detector and one
expected artifact entry plus one collision detector. It writes only:

- `input/bundle.bin` on the artifact Volume;
- `control/stage-claim.v2.json` on the control Volume; and
- `control/stage-claim.v2.mac` on the control Volume.

Uploads use `force=False`. Existing bytes must match exactly; unknown entries,
duplicates, size/content changes, partial writes with changed bytes, or final
readback mismatch fail closed. Exact partial writes may resume by adding only
missing files. No Volume, namespace, object, or directory is created by name.

## Remaining activation dependency

This slice intentionally does not interpret or validate the bundle's semantic
contents. The current `ModalExecutionBundleV1`, mounted worker, and remote
admission path remain based on the legacy operation and mutation command. A
separate Foundation-native bundle, submit-launch evidence, and remote admission
migration must bind the actual source, workload, artifact contract, runtime
closure, deployment, and stage predecessor receipt before this staging writer
can be composed or registered. Its successful provider-free tests are not a
claim of a runnable Modal deployment or training job.
