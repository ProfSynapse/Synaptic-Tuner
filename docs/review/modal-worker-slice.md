# Foundation-native Modal worker slice

Status: private provider-free implementation; no deployment wrapper,
registration, public export, provider call, or runnable capability is enabled.

## Admission boundary

The worker accepts one canonical host dispatch byte argument. Before its first
mounted-volume read, the mounted wrapper parses that argument and compares every
field of the verified deployment selection plus provider, profile, executor,
physical Volume IDs, configured Volume names, key reference, and three mount
roots against `ModalWorkerStaticExpectation`. This expectation is supplied by
the deployment constructor. It is never learned from a claim or Volume.

Only the stage effect ID is taken from the parsed submit command's authenticated
stage-predecessor structure. The wrapper then performs three bounded regular-file
reads beneath that operation root: the v2 stage claim, its tag, and the eight-
member coordinator bundle. Pure admission verifies the complete embedded launch
claim and exact stage material through `admit_modal_launch_wire`, reconstructs
the stage `ModalCommandBinding`, and parses the Foundation-native bundle with an
exact local `RecipeRegistry`. The production wrapper constructs that registry
with only `SFTRecipe`; no host registry or compiler authority crosses the remote
boundary.

The dispatch parser establishes canonical structure, not authentication. The
wire verifier authenticates the complete launch and stage claims. No Foundation
grant, record, assessment, catalog, binding authority, host signer, credential,
or provider client is accepted by this worker.

The mounted wrapper authenticates the complete embedded launch claim with its
constructor-owned key reference before reading any Volume member. Full wire
admission deliberately repeats that verification while cross-binding the stage
claim and bundle.

## Invocation contract

`ModalWorkerInvocation` is admission-minted, immutable data rather than a new
cryptographic authority. Its private construction path revalidates all retained
bytes and fixed derivations, and execution is a private helper used immediately
after full admission by the mounted workflow. It retains canonical
submit/stage command bytes, preparation/deployment/source bytes, workload,
identity-free log policy, packaged closure, immutable sorted environment pairs,
fixed argv/cwd/closure path, Volume/key identities, launch/stage/bundle digests,
and the complete log policy. Typed command, deployment, source, environment, and log
policy views are reconstructed or freshly copied on every access.

Reconstruction requires the source run and source fingerprint and the canonical
workload fingerprint to equal the submit preparation. It validates the complete
closed log-policy field set with strict integer bounds and requires the raw
closure path spelling to be canonical and submit-scoped. Launch-claim,
stage-claim, and bundle digest fields are format-checked during later invocation
validation; their original bytes are not all retained in the invocation. Their
content equality and authentication are established by wire and bundle admission,
not re-claimed by the data validator.

The separate completion producer receives:

```text
finalize(invocation: ModalWorkerInvocation,
         result: ModalProcessResult,
         *, job_ref: str) -> object
```

It can derive the submit effect, command and plan digests, deployment
attestation, invocation nonce, workload identity, log bounds/generation from
the single retained policy,
Volume/key identity, and artifact roots without importing a legacy lifecycle
command. The producer remains a separate cutover slice and is not implemented
or adapted here.

## Fixed execution

No invocation member supplies executable authority. Admission derives the
three-element argv from the authenticated execution source and the fixed
`Trainers/sft/runtime_v1.py --canonical-workload-stdin` entrypoint. It derives
cwd, offline model snapshot, workload fingerprint, isolation variables, and the
closure path under the **submit** effect. The log policy supplies bounded policy,
not an independent generation claim.

Execution reuses the low-level source materializer, process runner, and source
closure helpers. It verifies the dual clone, compares the checked-in closure to
the exact packaged bundle closure, exclusively stages the runtime closure, and
passes the preparation-commit callback to the fixed process runner. Closed phase
errors become `ModalProcessResult`; unauthorized admission errors occur before
clone, process, or completion behavior.

The mounted wrapper uses only constructor-owned roots and descriptor-relative
bounded reads. Job identity and callback shape are validated before remote work.
No compatibility conversion to `MutationCommandV1`, `OperationBindingV1`, the
legacy bundle, broker fallback, or old lifecycle is present.
