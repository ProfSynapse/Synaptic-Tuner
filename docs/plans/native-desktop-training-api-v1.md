# Native desktop training through API v1

Status: implementation preparation, 2026-10-20. Required product scope is local
training on NVIDIA and Apple silicon, alongside Modal SFT and GRPO. HF Jobs
remains paused. This plan complements the
[stable release gate](stable-v1-release.md); it does not qualify a runtime.

## Existing code and evidence

| Boundary | Reuse | Gap |
| --- | --- | --- |
| Public training and runs | `TrainingAPI`, `RunsAPI`, `ProviderFamilyV1`, reference composition and caller-owned stores | Native Mac provider operations and lifecycle evidence |
| Prepared input | Existing immutable dataset identities, source lineage and split contracts | An MLX projection that preserves exact prepared splits, target boundaries and configured tokenizer/template |
| Mac training | `tuner/backends/training/mac_backend.py`, `Trainers/mlx_sft_mac` | Legacy SFT subprocess only; no public v1 provider, native artifact verification or durable known-run reader |
| Runtime selection | Named declarative profiles and exact source/model bindings | Current `RuntimeSpec` admits only `published_oci` and `modal_build`; native execution needs an explicit reviewed representation |
| Artifact inventory | Five public roles and bounded hash/size streams | MLX adapter/tokenizer formats must be independently verified, not labeled PEFT without conversion evidence |
| Local NVIDIA | Existing Docker public consumer; writable-root correction in PR #259 | Fresh live completion, cancellation and recovery |
| GRPO | Existing selected Modal implementation | Genuine Modal update/reward qualification; no native Mac GRPO implementation or qualification is established |

The audit found concrete hazards in the legacy Mac path:

- Dataset preparation loads `mlx-community/Qwen3-0.6B-4bit` regardless of the
  configured training model.
- It filters labels, reshuffles and splits rows itself, rather than consuming
  an admitted prepared dataset with its existing split identities.
- Template failure can fall back to manually rendered ChatML.
- Layer count is fixed at 16, and reported LoRA settings do not by themselves
  prove that the invoked MLX command applied those settings.
- `--dry-run` reaches model loading and output writes before returning;
  `--resume` is parsed without a corresponding execution branch.
- Dependencies use lower bounds. Timestamped output and ad hoc lineage do not
  implement verified workload/metrics/model/tokenizer roles.

Do not expose this legacy subprocess as successful API v1 execution. Preserve
its existing entrypoint while extracting only behavior that satisfies the
reviewed input/runtime/output contract. Do not silently repair a user's data.

## Required design

Use one public workflow vocabulary and separate execution implementations:
Docker/CUDA on supported NVIDIA hosts, native macOS GPU execution on Apple
silicon, and the existing Modal backend. A frontend's “local” choice does not
mean all platforms use the same container or support identical methods.

Windows is explicitly in desktop scope. Qualify the Windows host, WSL2 execution
environment and pinned NVIDIA Docker runtime together. Keep native Windows
metadata and WSL Linux metadata separate; neither confirms GPU passthrough.
Exercise Windows AMD64/ARM64 and WSL reporting in unit tests and the installed
diagnostic in the existing Windows/macOS/Linux CI matrix.

Native execution must bind the exact interpreter, dependency closure,
platform/architecture, source, model/tokenizer revisions, method and prepared
input. Do not invent an OCI image for a native environment or add a fallback
that weakens existing CUDA/Modal admission. Decide the native material schema
and compiler/worker boundary before adding it to `RuntimeSpec`; preserve
canonical bytes for existing admitted documents.

The existing MLX implementation is the first reuse candidate. Choosing MLX does
not establish SFT loss-mask equivalence, GRPO availability or adapter portability.
Those require explicit method/configuration and artifact tests against pinned
upstream behavior. An alternate framework is a reviewed decision, not a fallback
after a failed check.

The host owns private state, source/configuration, current permissions and
one-shot claims. Reuse generic records/streams and coordinator/Foundation ports;
do not introduce a second local lifecycle database or parse console text into
authoritative outcomes. Retain process identity and exact run bindings so a
desktop restart cannot submit the same uncertain attempt again. Cancellation
requests, confirmed process termination and artifact verification remain
separate facts.

## Implementation sequence

1. **Observe the candidate machine and environment.** Extend the existing doctor
   surface with a bounded, read-only native-runtime diagnostic. Record platform,
   architecture, Python and selected package metadata without loading ML
   frameworks, installing dependencies or contacting providers. Missing and
   failed observations differ. This is not a dependency lock, GPU probe,
   execution authority or supported-provider declaration.
2. **Specify and pin native execution.** Capture an isolated Mac runtime and
   exact package artifacts after selecting the reviewed dependency set. Define
   the native material/profile representation and refusal rules. Review the
   compatibility effect on existing compilers and providers before enabling it.
3. **Implement prepared SFT execution and verification.** Reuse the prepared
   dataset service, preserve declared splits and target loss semantics, project
   explicitly configured controls, and produce genuine bounded native artifact
   roles. CPU fixtures exercise admission/serialization; real Apple hardware
   establishes MLX behavior and an actual training update.
4. **Join the durable provider.** Connect the worker to existing public ports,
   exact-run observation, retained readback and owned cancellation. Test process
   loss before/after claim, spawn, identifier retention and artifact publication.
   Unknown effects remain uncertain without automatic replay.
5. **Qualify GRPO explicitly.** Pin the native implementation, reward/configuration
   semantics and supported models. Demonstrate changed parameters and grouped
   reward variation; constant output remains inconclusive. Until then, expose
   native GRPO as unavailable rather than borrowing Modal's capability.
6. **Qualify desktop integration.** Exercise SFT/GRPO availability, memory limits,
   cancellation, restart and artifact consumption on both real hardware
   families. Test model/adapter conversions separately from raw file transport.

## Current preparation evidence and remaining limits

The owner-selected Mac is reachable over authenticated SSH and reports Apple M4,
24 GiB unified memory and arm64. Its shared Homebrew CPython 3.11.14 had no
MLX training dependencies installed at initial inspection. The existing Mac repository contains local
changes and is not an execution source for this qualification; preserve it.
Use isolated, explicitly selected source and environment paths for later work.
The subsequent isolated candidate install used a new private virtual environment;
the shared interpreter and existing checkout were not changed. All 50 installed
package versions match the candidate hash lock, and `uv pip check` passed.
No Mac ML framework import, model load or GPU training has been performed.

The next isolated dependency candidate is retained under
[`Trainers/mlx_sft_mac/runtime-candidate`](../../Trainers/mlx_sft_mac/runtime-candidate/README.md).
It pins MLX/MLX-LM and the transitive wheel dependency closure separately from
the CUDA/Modal environments. It is not yet a native runtime material or model
profile; worker token semantics, interpreter/native library capture and genuine
training updates remain unqualified.

The NVIDIA desktop's RTX 3090 and exact pinned Docker image/daemon were observed
through the configured endpoint. PR #259 passed all 18 checks and merged with a
tree identical to its reviewed head. The previous failed container and expired
source proof remain preserved. A fresh host source binding to the reviewed engine
and a config-only plan now exist with unused run/state/artifact paths. Just-in-time
source proof, execution material and reviewed model preparation are still needed
before the next live run.

Both these observations are preparation evidence. Neither a reachable machine,
installed package metadata, macOS base-import CI nor a successful container
inspection qualifies local training.

## Prepared SFT increment — 2026-10-20

Internal modules `tuner/runtime/native_mlx_sft.py` and
`tuner/runtime/native_mlx_runtime.py` implement a candidate preparation/execution
boundary without changing `RuntimeSpec`, `ProviderFamilyV1`, the legacy trainer,
or the public API. The worker consumes the existing shared tokenized examples,
retains host-provided row/split/source/tokenizer identities, and copies their
tokens and labels. It never loads text through MLX-LM's dataset wrappers. A
canonical projection digest identifies the copied data and worker controls;
it is not proof of source derivation or a replacement prepared publication.

The custom batch iterator retains every row, including small and partial
batches, and validation order. The custom causal loss uses each shifted label,
substitutes a safe gather target for ignored labels, and divides by the number
of supervised tokens. Explicit finite controls cover iteration count, sequence
and batch-token bounds, shuffle seed, and report/evaluate/save intervals. This
first boundary permits one batch per update, no distributed execution, no
packing, no gradient checkpointing, and unpadded causal examples only. It rejects
failed preprocessing diagnostics and truncation instead of repairing rows.

Candidate runtime preparation compares the caller's reviewed complete package
versions, dependency-lock digest, CPython version and interpreter bytes before
the worker imports MLX. It is read-only and returns `CANDIDATE_MATCH_ONLY`.
Package metadata equality is not wheel-byte or native-library attestation.
The checked-in requirements lock and its pins are unchanged.

### Remaining joins, in dependency order

1. **Native material and cold preparation.** Specify a versioned native runtime
   material/profile with exact source closure, interpreter plus linked native
   libraries, selected wheel bytes and the configured model/tokenizer revisions.
   Revalidate in the child before ML/model imports. Preserve existing canonical
   documents. A caller-supplied expectation or a matching report is not authority.
2. **Prepared-input host join.** Reuse the existing retained prepared-input source
   and shared tokenizer materialization under authenticated model/tokenizer
   pins. Bind the exact row IDs, splits, template arguments and projection
   digest to the compiled workload. The new projection accepts already prepared
   objects; it cannot independently prove their association with caller-provided
   source identity. Source lineage/held-out derivative checks remain upstream.
3. **Model/optimizer child composition.** Prepare the pinned offline model,
   explicitly configured LoRA targets/layers/rank/scale and optimizer/schedule.
   Verify trainable parameters and the applicable model's causal behavior.
   Never translate unsupported training knobs into silent MLX defaults.
4. **Bounded native child lifecycle.** The worker's finite iterations do not bound
   a hung Metal call, model load, validation, save, or stdout. The host needs an
   owned process group and identity, clean environment, bounded private I/O,
   wall-clock limit, stdout/stderr limits, explicit termination/wait escalation,
   and durable claim/spawn/identity retention. Test owner death and restart at
   each boundary. Retain uncertainty after lost spawn acknowledgement; never
   replay automatically. Existing `shared/environments/local_worker.py` and
   `bounded_local_runtime.py` are Linux-specific fixture workflows, not a
   qualified macOS training supervisor. Do not repurpose them without review.
5. **Native artifacts and API provider.** Bind authentic metrics/workload/lineage,
   MLX adapter and tokenizer roles to the exact run, verify bounded streams, and
   connect the existing `TrainingAPI`/`RunsAPI` ports to host-owned durable stores.
   Upstream saves and console text are not verification or durable completion.
   Keep known-run reads, cancellation request, confirmed termination, and
   verified artifact publication distinct. No engine database is needed.
6. **Real Mac/Syntunia qualification.** Independently compare raw-text,
   prompt/completion and noncontiguous mask fixtures through the real pinned
   tokenizer and MLX loss/autograd; demonstrate changed adapter parameters,
   held-out evaluation and verified artifacts on the M4. Then exercise bounded
   stop, failure/restart and consumer readback. Syntunia should keep native
   training unavailable until these joins pass; native GRPO remains separate.

CPU contract tests prove projection/control behavior and scalar loss algebra,
not MLX graph compilation, Metal memory capacity, model compatibility, native
artifact validity, process lifecycle, or successful training.
