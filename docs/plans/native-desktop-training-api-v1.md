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
