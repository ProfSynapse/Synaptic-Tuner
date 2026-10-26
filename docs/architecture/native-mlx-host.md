# Native MLX API v1 host integration

Native SFT uses the existing `TrainingAPI` and `RunsAPI`. The optional
`synaptic_tuner.api.v1.native_mlx` module exposes its explicit host composition
types, compiler, command catalog and `native_mlx_family` factory, following the
same opt-in pattern as `api.v1.modal`. The base API remains provider-neutral.
Importing the native module installs nothing, loads no ML framework, opens no
database and grants no execution authority.

This is the open-source engine contract for a desktop or other trusted local
host. Application installation/signing, UI, HTTP/MCP transport and model-library
UX belong to the consuming application. The current worker requires captured
immutable executable source assets as well as the installed Python library;
installing a wheel alone does not prepare a model or authenticate that source.

## Inputs and composition

Use these public imports for the native provider boundary:

```python
from synaptic_tuner.api.v1.native_mlx import (
    NativeCommandCatalogV1, NativeEvidenceAuthority, NativeMaterialAuthority,
    NativePlanningV1, compile_native_sft, native_mlx_family,
)
from synaptic_tuner.api.v1.reference import (
    ReferenceHostPortsV1, ReferenceRequestPortsV1, compose_reference_host,
)
```

The consuming host supplies the actual arguments; no permissive default host or
qualification authority is shipped. The checked-in
[restricted qualification consumer](../../examples/training_qualification/native-host.md)
shows a complete composition. Its external signature protocol qualifies a
particular attempt; it is not a required UI or an ordinary qualification issuer.

Before composition, the host must prepare and authenticate:

1. Exact engine source, private CPython/dependency/native-library capture and OS
   trust boundary. The separate capture CLI observes bytes; independent review
   admits them. Keep mutable output/cache paths outside immutable runtime code.
2. Complete private model and tokenizer snapshots with pinned repository,
   revision, size and SHA256 inventories. The engine does not install or download
   during training. Model preparation is a separate bounded checked-in workflow.
3. An immutable prepared publication and `PreparedTrainingInputIdentity`, with
   explicit train/validation rows and shared preprocessing/token labels. Compile
   the public `TrainingInputV1` with the declarative `NativeMLXProfile`; reject
   unsupported controls, truncation, missing supervision and layout changes.
4. Either independently authenticated qualification for the exact selected
   profile/source/runtime/hardware, or a fresh bounded restricted start
   admission. Two report hashes alone cannot issue qualification. Restricted
   admissions must expire within one hour of the execution host's clock.
5. Caller-owned grants, signing keys, command catalog, clock, record and stream
   ports. Supply the same thread-affine SQLite store to both ports when using
   the reference durable implementation. Keep secrets out of the worker.

`NativePlanningV1` binds the resolved request, compiled workload, captured runtime,
process limits, provider scope and exact run. `native_mlx_family` takes that
planning object plus the catalog, material/evidence authorities, records,
Foundation/assessment authenticators and clock. Pass its `ProviderFamilyV1` to
`compose_reference_host` with the host's existing request and storage ports.
The host's grant-binding hook authenticates and publishes the exact
`NativeCommandMaterialV1` before effects. A dictionary, matching checksum or
catalog row cannot substitute for those authorities.

## Submission and lifecycle

Call the usual public sequence: `training.load`, `resolve`, `plan`, `preflight`
and one `start`. Native submission is bounded and **synchronous**. Keep it on
the thread owning the host store; a desktop can supervise a separate host
process so the UI remains responsive. Do not move SQLite operations into a
background thread or assume start returns before the worker exits.

The native guardian enforces the admitted deadline, output and file limits. Its
live owner controls an anonymous liveness pipe. Owner loss requests cleanup of
the owned worker group; stored numeric PIDs grant no future signal authority.
Unconfirmed effects remain uncertain and consumed. Never start again because a
caller timed out, a result was lost, a read failed or a local download failed.

After start, `runs.outcome` obtains authenticated terminal observation;
`runs.show` is retained state. Only `runs.verify` plus the verified manifest
establishes artifact availability. Public logs, cancel and reconcile remain
unavailable. Applications must query capabilities and display those limits;
they must not expose a functional cancel button or claim resumable training
based on the guardian's internal cleanup behavior.

For fresh-process readback, reopen the original private store and retain the
original catalog, source/model/runtime inputs and signing authorities. The
reference consumer's `show`, `observe`, `verify`, `reverify` and `download` actions rebuild
only the exact selected read binding. Expiry blocks a new restricted start,
not authenticated retained reads. Missing keys, modified material or an
uncertain submission cannot be repaired by adopting another run or issuing a
new submission grant. Preserve diagnostic history even when later native
verification succeeds.

Use `runs.verify` for initial artifact admission and `runs.reverify` for an
already verified run. These are distinct authenticated transitions; repeating
initial verification is refused. The reference consumer exposes both actions.

## Artifacts and support claims

The five public roles are `final_model`, `tokenizer`, `training_lineage`,
`training_metrics` and `workload_record`. Require successful public verification
and bounded, hash-checked streaming into fresh private destinations. The model
role is an MLX LoRA archive; it is not PEFT, a merged model, or GGUF. Base model,
tokenizer/template, runtime and workload identities travel in the manifests.
Desktop serving and format conversion need their own qualified engine path.

The native verifier checks actual optimizer count, finite nonzero saved adapter
delta, exact tensor/checkpoint inventory and held-out evaluation after loading
the saved final adapter. An exit code or the upstream in-loop loss is not this
evidence. See the [real M4 qualification report](../review/native-mlx-sft-qualification.md)
for exact sources and narrow hardware/model limits. Native GRPO, arbitrary
models/OS versions, training quality, checkpoint resume and public lifecycle
capabilities are not implied by those checks.
