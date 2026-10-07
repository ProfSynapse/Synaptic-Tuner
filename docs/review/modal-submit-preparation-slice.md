# Modal host submit-preparation slice

Status: internal provider-free semantic gate implemented; no SDK spawn,
registration, public API, cloud call, or runtime execution.

`prepare_modal_submit_dispatch` accepts one exact `ModalLaunchEnvelope` and the
existing host verification ports. It snapshots the immutable claim, tag,
command, preparation, deployment, stage claim, bundle, Volume, and key values;
then calls `admit_modal_foundation_launch` to authenticate the complete existing
Foundation stage-to-submit lineage. Digests are never converted into authority.

After admission, it reconstructs the exact stage command from the admitted
stage claim and combines it with the shared retained snapshot and deployment to
rebuild `ModalCommandBinding`. It then invokes the eight-member
`ModalCoordinatorBundle.parse_transport` with an exact local `RecipeRegistry`.
That mandatory parse rechecks all bundle members and independently recompiles
the resolved material before any provider dispatch is prepared. Opaque legacy
or mismatched bundle bytes fail here.

Only after those gates pass does the function derive a complete
`ModalWorkerLaunchExpectation` from admitted/reconstructed bytes, retained
profile Volume names, and verified deployment selection, and encode the private
one-argument `ModalWorkerDispatch`. It does not accept a caller-provided worker
expectation or infer trust from a digest.

The result remains host preparation data. It owns no provider client, SDK,
filesystem, process, credential, signing key, grant, catalog, repository, or
additional mutation authority. A future composition must persist and dispatch
it under the already authenticated Foundation submit authority.

The provider-free integration fixture exercises the real generic coordinator,
claim-derived stage result, eight-member bundle builder and parser, material
recompilation, launch admission, submit preparation, dispatch round trip, wire
admission, and bundle reopening in one positive path. It uses in-memory trusted
host authenticators and performs no provider call.
