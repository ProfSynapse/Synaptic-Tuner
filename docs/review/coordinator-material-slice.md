# Coordinator resolved-material derivation

Status: canonical provider-neutral derivation implemented; no public composition
cutover or provider operation is included.

`tuner/training/coordinator_material.py` converts one exact rich
`api.v1.training.ResolvedTrainingRequest` into an immutable canonical snapshot
and the existing digest-only `api.v1.planning.ResolvedTrainingRequest`. It
recompiles the workload through the supplied exact `RecipeRegistry`, requires
byte-equivalence with the resolved workload, extracts the exact compiled
artifact contract, and commits every retained input in the snapshot. The
material retains only canonical bytes: accessors reconstruct their values from
that snapshot and never consult the caller's nested objects or a previously
returned planning DTO. Exact rich nested contract types and an exact
`CompiledWorkload` result are required; subclasses are rejected. Parsing
reconstructs independent typed values and repeats compilation and all digest
checks; there is no unchecked bytes constructor.

The planning digest mapping is:

- `source_digest`: existing `ExecutionSourceV1.fingerprint`, domain
  `synaptic-execution-source/v1`;
- `workload_digest`: existing `CompiledWorkload.fingerprint`, domain
  `synaptic-training-workload/v1`;
- `resolved_config_digest`: `synaptic-coordinator-resolved-input/v1` over the
  exact request, execution context, resolved config, and resources;
- `runtime_digest`: `synaptic-coordinator-runtime/v1` over the exact
  `RuntimeSpec` mapping; and
- `artifact_policy_digest`: `synaptic-coordinator-artifact-policy/v1` over the
  exact policy mapping.

The compiled artifact contract is distinct from artifact policy. Material
exposes its exact canonical bytes and raw SHA-256 for future bundle-member
matching, but does not invent a second artifact-contract domain. Likewise,
execution-source fingerprint is distinct from the raw SHA-256 of a serialized
bundle member; a bundle consumer must check both the domain commitment and
exact member bytes.

The execution source already contains the allocated `run_id` and verified
deployment-member commitment. Derivation requires the caller's allocated run
to equal that identity and never rewrites it. Composition must therefore
allocate the coordinator run first, finalize the deployment-bound execution
source for that run, derive this material, and have
`CoordinatorRunIdentityPortV1.for_plan` return the same run and project.

Current Modal preparation maps planning `artifact_policy_digest` into its
Foundation `artifact_contract_digest`. That existing mapping describes policy,
not the compiled contract bytes, and must not be treated as bundle-contract
proof. The later bundle/material integration must enforce the exact compiled
contract independently. No legacy Modal lifecycle, authority callback, cloud
call, registration, or compatibility signature is introduced here.

As in `TrainingService.resolve`, every required artifact-policy role must be
present in the deterministically recompiled artifact contract before material
derivation succeeds.
