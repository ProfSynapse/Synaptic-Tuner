# Foundation-native Modal bundle slice

Status: internal codec only; not registered, runnable, or production-ready.

## Boundary

`coordinator_bundle.py` defines a bounded stage bundle for one exact
`ModalCommandBinding` containing an exact `StageCommandV2`. It imports neither
the legacy broker command nor `OperationBindingV1`. The bundle has no provider
SDK, volume, materialization, credential, signing, or network behavior.

The exact members are:

- `deployment.json`: byte-identical to the verified deployment in the complete
  Modal command binding.
- `execution-source.json`: a canonical `ExecutionSourceV1`, cross-checked with
  the deployment interpreter, runtime environment, secret requirements, and
  deployment-member SHA.
- `workload.json`: the fixed canonical SFT workload, embedding the exact source.
- `artifact-contract.json`: the exact five singleton artifact roles embedded in
  the workload.
- `log-terminal-policy.json`: bounded policy only. It intentionally contains no
  effect, run, provider-job, or path identity because those outputs belong to a
  future submit effect which does not exist at stage time.
- `resolved-material.json`: the exact reconstructable coordinator resolution;
  remote parsing recompiles it with the local fixed recipe registry and rejects
  secret-bearing request, context, configuration, or workload content.
- `worker-closure-manifest.json`: an exact parsed offline worker closure.
- `stage-plan.json`: a deterministic manifest binding the stage command,
  preparation commitments, typed material fingerprints, closure digest, and
  every predecessor member's raw SHA-256 and size.

There is no invocation-intent member. Executable, argv, cwd, environment, and
operation-scoped output paths must be derived at remote admission from the
authenticated source, deployment, fixed worker entrypoint, closure, policy, and
future authenticated submit launch. Caller-supplied invocation fields are not
authority.

## Digest semantics

Raw member SHA-256 values, `ExecutionSourceV1.fingerprint`, canonical workload
fingerprint, Foundation preparation digests, and the authenticated Modal binding
digest remain explicitly distinct. The codec does not relabel one digest domain
as another.

The earlier gap between rich resolved inputs and the digest-only planning DTO is
closed here by requiring `CoordinatorResolvedMaterial` and its exact recipe
registry at both build and parse boundaries. The material is reparsed from its
canonical bytes and the workload is deterministically recompiled before any
cached properties are used. Its planning source, workload, runtime, and artifact
policy digests must equal the complete stage preparation; its exact
source/workload/artifact bytes must equal bundle members. Raw member SHA-256
values remain separate byte identities. The resolved-input digest participates
in the training plan fingerprint through the authenticated preparation rather
than being relabeled as a member digest.

## Security properties covered

- exact immutable Modal binding reconstruction and exact stage-command kind;
- strict canonical transport, canonical padded Base64, exact field/member sets,
  per-member, aggregate, decoded, and encoded bounds;
- deployment and client/runtime selection equality;
- resolved image, dependency lock, Python, accelerator, count, and timeout
  equality with the verified deployment selection;
- source/workload/artifact/closure semantic cross-links;
- mandatory reconstructed resolved material and recipe compilation;
- deterministic stage plan and tamper rejection;
- no future submit identity, provider job reference, arbitrary invocation
  authority, credentials, or literal secret fields introduced by this codec.

Authentication remains outside this codec. The stage coordinator must
authenticate the complete binding and sign the claim only after the resolved
material comparison succeeds. A bundle digest or binding digest is content
identity, not authentication.
