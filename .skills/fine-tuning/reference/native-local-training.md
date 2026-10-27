# Native local training preparation

For representative public conversation and explicitly positive local tool subsets,
use the checked-in config-first host workflow in
`examples/training_qualification/native_sft_datasets/README.md` (repo root).
It fills the source acquisition/selection CLI gap and delegates immutable publication
to the existing conversation normalizer; it never trains, installs a model/runtime,
or creates admission. Retain source hashes and original split provenance, audit
exclusions and complete examples, and run the existing tokenizer-only mask doctor
on every selected row. Legacy `tool_call:` text is an explicit projection and does
not establish compatibility with the current Nexus API or native structured calls.
Canonical publication members remain sealed 0400. Native staging creates separate
byte-identical 0600 members under a new 0700 content-addressed directory, then
revalidates their identity; never chmod canonical originals to satisfy the reader.

## Current evidence — 2026-10-26

The M4 now has successful bounded raw-text and assistant-only prompt/completion
MLX SFT runs through the public `TrainingAPI` and `RunsAPI`, including native
verification and five-artifact delivery for each. See
`docs/review/native-mlx-sft-qualification.md` (repo root) for the
exact execution source, run, numeric adapter-change evidence and remaining
desktop integration gates. The preparation chronology below retains historical
blockers; it does not supersede that dated evidence or enable ordinary native
training, GRPO, public cancellation, or product support.

Restricted signed start admissions must satisfy
`now < expires_at <= now + one hour` on the execution host. `admission-candidate`
constructs fields; it does not establish a valid current start window. Review
the independent signature, exact selection and current clock before the sole
start. Preserve rejected records; use fresh identities for a separately reviewed
attempt. Never renew a consumed or uncertain submission.

## Native GRPO implementation candidate — 2026-10-27

The optional public native module also exposes `NativeMLXGRPOProfile` and
`compile_native_grpo`. This is a CPU-tested source candidate, not MLX hardware
qualification. See `docs/architecture/native-mlx-grpo.md` and the separate
`Trainers/mlx_grpo_mac/runtime-candidate` closure. Never install into or relabel
the qualified SFT environment. The original SFT 50/54-package locks stay intact.

The candidate consumes the existing immutable `synaptic-grpo-prompts/v1`
publication and public response-reward declaration through the same planner,
family, host grants/catalogs, owned process and five verified artifact roles.
Profile data binds the exact model/source/runtime, pinned attributed objective,
quantization, rollout/reward/evidence budgets, and timeout-bounded regex-V0
reward semantics. Only selected prompt messages reach generation; generated IDs
and completion masks feed a detached-old/base-reference, token-mean unclipped
on-policy objective. Unsupported controls reject rather than silently default.
Constant grouped rewards or unchanged saved parameters cannot verify success.
Final saved-adapter rollout replay is not held-out writing-quality evaluation.

Evidence publication must pass the profile's exact `maximum_evidence_bytes` to
both the exclusive write and its same-tree readback. `NativeTree.write_exclusive`
retains its 1 MiB default for SFT/control documents; larger writes require an
explicit finite limit within the existing native per-file ceiling. An admitted
GRPO record above 1 MiB must round-trip through actual publication and five-role
verification in CPU controls. Reject over-budget bytes before creating the file;
do not fix mismatched limits by silently increasing every caller's default.

Native-only input preparation has a separate source-checkout host example:
`python -B examples/training_qualification/native_grpo_preparation/prepare.py
inspect <reviewed-config.json>`, then `prepare` after reviewing it. It composes
the existing GRPO preparation service on Linux with fresh create-only private
roots, exact seed hash/size and actual prepared identity/request outputs. Read
its README for safe-parent and retained-failure rules. It does not consume a
Modal profile, construct a native profile/admission, import ML or run on the Mac.

The existing restricted `native_host.py` consumer accepts proposal `method:
grpo`, exact `grpo_profile`/`grpo_controls`, attempt kind `grpo_on_policy`, and
reviewed `input_review.prompt_rows`/`publication_sha256`. Its inspect, fresh root
admission, synchronous start, original-authority retained reads and no-replay
rules are unchanged. The separate native source capture must include the new
worker/math/verifier bytes. No new provider verb or accepting authority exists.

Linux CPU controls must not import MLX. The real tiny numerical test additionally
requires `SYNAPTIC_TEST_NATIVE_GRPO_MLX=1` on Darwin arm64; root/source/runtime
review and a bounded authorized command must precede it. Mac execution, GPU fit,
real grouped reward variation, saved adapter changes and public artifact
readback remain separate gates. Public logs/cancel/reconcile remain disabled.

## Read-only host diagnosis (commands)

Consumer integration uses the optional public
`synaptic_tuner.api.v1.native_mlx` composition module with existing
`TrainingAPI`/`RunsAPI` and host-owned ports. See
`docs/architecture/native-mlx-host.md` (repo root) for installation/source assets,
authority, synchronous submission, retained reads and explicit capability limits.

Run `python tuner.py doctor native-training --json` on the target machine to
record a bounded, versioned `DIAGNOSTIC_ONLY` report. The command observes OS,
architecture, macOS version when applicable, CPython implementation/version,
and installed distribution versions for `mlx`, `mlx-lm`, `numpy`, `transformers`,
and `torch`. `absent`, `malformed`, and `observation_failed` remain distinct.

`native_apple_candidate` is true only for observed Darwin `arm64`. The report
does not establish Metal access, GPU capacity, package provenance, runtime
compatibility, or SFT/GRPO training support. It does not import the ML packages,
load a model, access credentials, initialize a GPU, install packages, or make
network calls. `--fix` and `--env-file` are rejected. Existing legacy MLX SFT
is separate from an API v1 native Mac adapter and training qualification.

Run the same command on Windows and inside WSL separately: each report describes
the OS running that Python process. Windows AMD64/ARM64 and Linux/WSL metadata
do not establish NVIDIA hardware, CUDA, WSL GPU passthrough or Docker readiness.
The Windows desktop qualification path remains WSL2 plus the pinned NVIDIA
Docker runtime. Installed-base CI exercises this diagnostic on Windows, macOS
and Linux; those checks remain separate from real GPU training evidence.

For the next Apple dependency-preparation step, use the checked-in
`Trainers/mlx_sft_mac/runtime-candidate/README.md` and its hash-pinned lock.
Resolve with the documented platform/interpreter/index and install only in a
fresh private environment with hashes required and source builds disabled.
This candidate does not select a model, enable a provider, or qualify Metal,
SFT or GRPO. Keep CUDA/Modal environments and the legacy Mac checkout unchanged.

## Prepared SFT worker candidate

`tuner/runtime/native_mlx_sft.py` is an internal CPU-tested boundary. It consumes
the existing shared `PreparedSFTExample` objects after host verification and
tokenization; it does not introduce a dataset format or public execution API.
Pass the exact prepared-input identity, tokenizer revision, retained row IDs,
declared train/validation tuples and explicit `MLXSFTConfig` controls. The
projection copies all tokens/labels immutably and records a deterministic digest.
The host must still bind those rows to the authenticated source and tokenizer.

No MLX text/chat loader, extra EOS, resplit, truncation, packing, tail dropping,
or single-offset masking is allowed. Failed masking diagnostics, already
truncated rows, attention masks other than all-ones unpadded causal rows, and
rows without shifted target supervision reject. Arbitrary ignored token spans
remain ignored. Training shuffles deterministically within its own split;
validation retains every row in declared order, including partial batches.

The custom hooks target pinned MLX-LM 0.32.2. One batch is one optimizer update;
gradient accumulation and activation checkpointing are deliberately unavailable in this
increment. Upstream accumulation averages microbatch means, which needs a
separate objective review for unequal supervised-token counts. Model, LoRA,
optimizer and learning-rate policy are prepared explicitly by the isolated
child composition; this worker does not infer those controls.

`tuner/runtime/native_mlx_runtime.py` compares host-supplied exact interpreter
and complete distribution pins against the running candidate before worker ML
imports. `CANDIDATE_MATCH_ONLY` means metadata and executable bytes match.
It does not authenticate wheel files, imported module origins, Homebrew's linked
Python framework, other native libraries, Metal, or launch authority. Retain the
original hash lock and use the existing `uv pip freeze`, `uv pip check`,
`shasum -a 256` and `otool -L` tools for review evidence; do not install packages
into the shared interpreter or treat a newly observed inventory as approved.

The internal worker may run only in an independently admitted, host-owned
bounded child. Use the checked-in restricted host consumer described below;
do not call the worker directly as an ad hoc substitute.

## Execution-boundary continuation

The internal runtime modules now include a read-only native material capture,
retained-source/tokenizer join, explicit LoRA/constant-AdamW composition,
independent safetensors delta verifier, and owned subprocess guardian. These are
reviewable runtime boundaries. The gated family described below composes them
through the existing public coordinator; no global provider registration is
supplied.

`scripts/capture_native_mlx_runtime.py` takes a JSON object with `capture` and
`expectation`. `expectation` uses the existing `MLXRuntimeExpectation` fields.
`capture` contains absolute `stdlib`/`packages` roots, sorted `sources` trees,
sorted explicit external `files`, sorted `system_libraries`, and sorted
`aliases` pairs of absolute alias path and exact symlink spelling. Every tree
file, including startup `.pth` and bytecode, is captured. Unlisted aliases reject.
Homebrew's framework aliases must resolve to explicit captured bytes. Its
stdlib `site-packages` alias is recorded but excluded: the isolated child uses
`-I -S -B` and explicitly binds only its captured package search path.

Run the selected interpreter with `-B`, from an independently reviewed source
checkout, to avoid creating bytecode while capturing. Output requires an
existing owner-private parent and is exclusive; no environment repair or
installation occurs. `--check <capture> --expected-digest <reviewed-sha256>`
revalidates the exact original configuration and capture. Unknown/ambiguous
Mach-O dependencies, uncaptured external files and extra declared OS libraries
reject. OS shared-cache dependencies are explicitly trusted through the recorded
OS/kernel build, not falsely described as hashed ordinary files.

For `@rpath` dependencies, inspect the complete captured load chain. Apple's
[run-path guidance](https://developer.apple.com/library/archive/documentation/DeveloperTools/Conceptual/DynamicLibraries/100-Articles/RunpathDependentLibraries.html)
and `dyld(1)` contract describe `LC_RPATH` entries from the chain leading to the
dependent image; expand `@loader_path` relative to the image declaring each
entry. The static checker requires one captured Mach-O target across every
proven context and rejects ambiguous targets, paths outside the capture,
unreachable cycles and excessive contexts. Do not insert an ambient search
directory or a package-specific fallback to make a candidate pass. A static
capture remains an observation; repeat it on the final independently reviewed
pushed source after any checker change.

This captures installed bytes and static native dependencies. It does **not**
prove selected wheel archive provenance, attest dynamic `dlopen` behavior,
freeze the filesystem against a malicious same-uid process, or independently
grant execution. Root review must reconcile the original hash-required install
evidence and selected archives; public capability remains gated.

The v2 capture format separately records selected literal `ctypes.CDLL` OS
shared-cache names. The config declares each exact captured `.py` source member,
its SHA-256, the direct `ctypes.CDLL` call and the absolute OS library name.
Capture parses bounded source bytes without executing them and requires the
declared names to equal all such calls in each selected member. It rejects
aliases, computed paths, extra or missing calls, source hash drift and duplicate
declarations. The current Darwin guardian selects both `/usr/lib/libSystem.B.dylib`
and `/usr/lib/libproc.dylib`. These are OS shared-cache names on the observed
Mac, not ordinary files with invented hashes or static Mach-O edges. The v2
document records the exact macOS/kernel build and selected `_ctypes` built-in
against the interpreter's full `sys.builtin_module_names` inventory digest;
the config must select the independently reviewed macOS and full kernel build
before capture, and capture rejects a different observed build.
`ctypes` Python files remain inside the captured stdlib. Keep the existing
strict Mach-O graph and its exact used/static OS-library set unchanged.
Across the selected project source trees, any explicit absolute `ctypes.CDLL`
call requires a matching v2 declaration; omitting the v2 fields cannot produce
a v1 capture of the current guardian. Existing v1 sources without such calls
retain byte-identical canonical captures.
Source-observed load declarations and a same-build recheck do not attest an
actual later `dlopen`, the loaded image's origin, or all dynamic behavior.
Legacy configurations without these selections still serialize as exact v1
captures. See the deliberately incomplete
`examples/training_qualification/native-capture-v2.example.json`; replace its
paths, source/executable hashes, complete 54-distribution expectation, all
static system libraries and aliases from the independently reviewed selection
before a new capture. Never add a dynamic name to `system_libraries` to quiet
an unused or missing static dependency.

`native_mlx_input.py` consumes the existing retained stream lease, verifies
identity/hash/size, and feeds the established JSONL publication rows to shared
`materialize_sft_example`. It retains row IDs and splits without filtering or
repair. The tokenizer loads only from a verified private local snapshot with
`trust_remote_code=False` and `local_files_only=True`. Model and tokenizer
snapshots, exact source, rendering arguments and worker controls are committed
in the bound input. The host still owns publication semantic/lineage validation
and binding to an actual compiled workload.

`native_mlx_composition.py` loads a configured local MLX snapshot, selects every
LoRA target/layer explicitly, checks the realized trainable names/shapes and uses
explicit AdamW parameters with constant schedule. Unsupported schedules reject.
It checks the optimizer's actual step count. The native adapter verifier checks
tensor headers, bounds, names/shapes/dtypes, finite values and a nonzero numeric
parameter delta. Final validation loads a second clean pinned base, attaches the
same configured LoRA, loads the saved final adapter and compares every trainable
tensor against the saved bytes before evaluating that second model. A no-op or
partial reload rejects. Upstream in-loop validation occurs before the last
optimizer update.
Pinned MLX `mx.save_safetensors` may emit a present `__metadata__: null` header
when no metadata is supplied. The CPU-only adapter parser accepts that exact
absence marker or a string-to-string metadata object, while still rejecting
other metadata shapes and checking every tensor byte range. Check a retained
adapter with this parser before attributing a closed worker phase to training:
an initial adapter file alone proves neither parser success nor an optimizer
update. A source fix requires fresh captured bytes and independent admission
before another live attempt; a phase code alone does not establish the exact
historical exception.
It saves native `adapter_config.json`, `input_binding.json` and
`native_result.json` exclusively. These worker files become authenticated public
artifact roles only through the family verifier described below.

`native_process.py` records claim/spawn/outcome through the existing
`DurableRecordStorePort` in the effects partition. There is no engine database.
The stdlib guardian owns the worker group, drains bounded output without
exposing its contents, enforces time/file bounds, and terminates on owner-pipe
loss or cancellation. Darwin uses kqueue process notifications because the
qualification CPython does not expose `os.waitid`. Linux tests use waitid without
reaping. Persisted PIDs are diagnostic only; another process cannot acquire
signal authority by reading them. Interrupted submission/publication retains
uncertainty and never auto-replays. This is trusted-worker supervision, not an
OS sandbox; escaped hostile descendants and guardian failure remain outside the
current evidence. Real Darwin lifecycle tests remain required.
If an owned guardian exits before an outcome is retained, the transport records
only a closed diagnostic category and guardian exit code. Exit codes 120-127
identify protocol, limits, spawn, exit watch, pipes, monitor, termination, and
cleanup phases respectively; they contain no exception, worker output, path,
argument, or environment data. A missing outcome remains uncertain and cannot
authorize replay or signals to a persisted PID. These diagnostics describe a
future attempt only; they cannot assign a phase to an earlier consumed claim.
After the r23 CPU-only Mac probe returned the broad termination code 126 with
claim/spawn/diagnostic but no outcome, the guardian added closed subphase codes
128-131 for SIGTERM, grace observation, SIGKILL, and child wait. The fixed CPU
worker's result file existed, but that does not prove guardian cleanup or a
successful process outcome. The r23 cause is still unknown, and its consumed
state must not be replayed. The new codes describe only a future, separately
admitted CPU-only attempt; they do not broaden native training authority.
The r25 CPU observation stopped at SIGTERM code 128 without an errno. The
guardian now assigns codes 132-135 to SIGTERM and 136-139 to SIGKILL failures:
EPERM, other OS error, missing API, and non-OS failure in that order. These
fixed classes expose no exception text or errno value. Darwin's XNU source can
return EPERM when an existing group has no counted signalable members, while
EPERM may also mean denied access to a live member. Neither a worker result nor
an EPERM class proves the group is empty. Keep the leader unreaped until group
cleanup and preserve the consumed claim on any missing outcome. The r27
CPU-only probe subsequently classified SIGTERM `PermissionError`/`EPERM`,
without a process outcome. It does not establish that the group was empty.
For the reviewed Darwin arm64/macOS 27.0.1 candidate only, an owned guardian
may recover from this exact EPERM when its kqueue observer confirms the still
unreaped child exited and two bounded `libproc` `PROC_PGRP_ONLY` selections
include the owned leader, are complete and identical, and every selected PID
has a complete `PROC_PIDT_SHORTBSDINFO` record with the same group and zombie
status. The leader record must identify this guardian as parent. A zero/full
selection, denied or short PID record, changed membership, live descendant,
unknown OS build, or any other uncertainty preserves EPERM and the claim.
Each selection also brackets the fixed buffer with the kernel's no-buffer
capacity estimate, because XNU can silently cap a short result at its own
`nprocs + 20` allocation. No system process list is copied to userspace.
The guardian signals the group again at SIGKILL, repeating proof if that also
returns EPERM, and reaps the child only after the final signal/query. The
system reads use pinned absolute `libproc.dylib` and `libSystem.B.dylib` paths
and require `_ctypes`; a fresh runtime capture must include this newly loaded
stdlib extension and library graph before any further ML qualification. This
is a group-scoped observation for trusted workers, not proof about descendants
which deliberately escaped the group or future OS builds.
The r31 CPU-only Mac attempt retained claim, spawn and outcome records for one
fixed nonforking worker: `worker_exited`, exit 0, confirmed termination, zero
stream bytes. It did not qualify ML, deadline or owner-loss behavior. Historical
r20/r23/r25/r27/r30 attempts remain consumed, with r30 SSH transport uncertain.
The current guardian source needs a new independently reviewed v2 runtime
capture and raw host selection before any new ML execution admission.

For a provider-free Darwin guardian check, use the private
`examples/training_qualification/native_cpu_process.py` config-first
`inspect`/one-shot `run` flow and its fixed stdlib worker. Bind the exact clean
source commit, actual private launcher and regular interpreter target, all
executable/guardian/worker/request hashes, unused 0700 state root, attempt and
finite limits. It reuses `NativeOwnedProcess` and the host SQLite record port;
no model, tokenizer, ML package, dataset or credential is needed. Review the
local source/ref and config before effect, separately verify the pushed commit,
and never adopt or replay a consumed root. This CPU observation is not a signed
training admission or evidence of MLX updates. See
`examples/training_qualification/native-cpu-process.md`.

Before ML imports/training, independently review the isolated child bootstrap,
authenticated compiled workload/native material, public provider-family
composition and native artifact roles. The bounded next proposal is
`Trainers/mlx_sft_mac/runtime-candidate/sft-qualification-proposal.json`; its
status is review-required and it is not executable admission.

## Public coordinator continuation

`native_mlx_compilation.py` recompiles `TrainingInputV1` plus the retained
`PreparedTrainingInputIdentity` and explicit `NativeMLXProfile` into a normal
`CompiledWorkload`. Parsing re-derives the complete document. Model/tokenizer
pins, loss semantics, chat arguments, batch/sequence/update/seed/checkpoint
controls, LoRA and constant AdamW are explicit. Accumulation, epochs, DoRA,
RSLoRA, implicit splits, packing and unsupported checkpoint-pruning policies
reject. Profiles declare the alpha/rank scale conversion; no model switch is
embedded in a launcher.

`native_mlx_child.py` is an internal `-I -S -B` entrypoint. It restricts import
roots, checks imported source/extension origins against capture members and
compiles pinned Python source rather than ambient bytecode. It joins the exact
staged dataset, capture, lock and snapshots to the recompiled workload. Before
optimizer work, it builds the exact complete tokenizer role archive, extracts
those bytes into a private temporary directory and loads the tokenizer locally
with remote code disabled. The host later publishes that same deterministic
archive. It is
not a public invocation or an independent source/qualification authority.
`native_runtime_spec` represents the capture with the existing `RuntimeSpec`
`native_environment` discriminator; it invents no OCI image. Legacy OCI
canonical bytes are unchanged.

`native_mlx_planning.py` supplies immutable planning/preparation adapters.
`native_mlx_family.py` supplies the existing `ProviderFamilyV1`: stage, bounded
synchronous submit, authenticated known-run observation and artifact streaming.
Use it with `compose_reference_host`; Syntunia's existing loader/resolver,
grant, record, stream, secret and clock ports remain authoritative. The consumer
must authenticate actual source evidence and current hardware before signing
command material. `NativeCommandCatalogV1` retains create-only signed material
through the existing record port. Signed terminal evidence must also match the
retained process outcome on every read. It is not an engine database or replay grant.
Immediately before a submit process claim, the family recaptures the admitted
runtime/source closure and verifies complete model/tokenizer inventories. Drift
at this boundary rejects before `run_once`; later dispatch remains uncertain.
The CPU artifact verifier requires every configured target in the exact final
model layer set and complete tokenizer snapshot selection.

No accepting **ordinary** hardware authority is shipped. An unadmitted
candidate has observe/artifact capabilities disabled and preflight is not
ready. A separately verified `NativeRestrictedAdmissionV1` enables only its
exact compiled workload and host selection, including project, request, run,
attempt, private state root, process limits, artifact budget, source, runtime,
prepared bytes and model/tokenizer snapshots, subject to a finite start expiry;
it is not qualification. Retained public reads authenticate that signed selection
after expiry and remote branch advancement without new dispatch authority. The host
consumer in `examples/training_qualification/native_host.py` verifies the
external signature using a separately supplied reviewed public-key SHA-256.
It has no signing key or acceptance switch. The same host checks the actual
clean pushed Git ref, full runtime recapture, private prepared publication,
model snapshot inventories and M4 hardware before issuing command material.
The proposal remains parameters, not an admission. See
`examples/training_qualification/native-host.md` for exact config fields,
preparation/readback commands, and the root signing boundary. The first signed
raw Mac attempt was consumed and returned `reconcile_required`; it produced an
initial adapter but no retained process outcome or verified final artifact.
Its historical exit cause and update count remain unknown. It grants no native
qualification or replay authority.

`NativeQualificationV1` contains exact selected profile/source/runtime/hardware
and two distinct retained attempt-evidence references; its independent host
authority must authenticate the supporting real runs, selected wheel/install
provenance and final-adapter/public-readback evidence. Reference hashes alone
are not qualification. Public logs, cancellation and reconciliation remain
disabled even with this narrow qualification. The guardian's internal stop
path does not promote public cancellation.

`native_mlx_publication.py` independently checks the exact output inventory,
workload/input binding, optimizer count, finite saved tensor delta, retained
checkpoint tensors and final-adapter validation identity. It emits the five
existing roles. `final_model` is a deterministic POSIX tar containing the MLX
adapter/config/checkpoints and pinned base manifest; `tokenizer` is an explicit
configured member archive from the verified snapshot. Lineage/metrics retain
the exact worker bindings. Finite file/aggregate budgets and public 1 MiB
stream chunks apply. CPU verification does not rerun the final evaluation.

R5 Mac preparation retained a clean pushed `aba3c576` source clone and two
genuine synthetic prepared publications under a fresh 0700 private stage.
The selected Homebrew stdlib cannot currently pass static capture: its Cellar
ancestor is group-writable. The candidate venv lacks `jsonschema` required by
the current host composition. Preserve both as blockers rather than changing
shared Homebrew permissions, weakening native tree traversal, or claiming a
capture. Existing wheel-cache metadata does not attest absent wheel archives.
The current child also imports `jsonschema` through
`tuner.training.contracts` → `tuner.project.execution_source` → the eager
`tuner.project` initializer. The host and child need the same exact inventory
under the current `verify_native_material` contract. The r7 candidate keeps
the original 50 lock unchanged and adds four reviewed Mac wheel hashes in
`requirements.host-additions.lock`. Its flat `requirements.runtime.lock` is
exactly the original bytes plus those four lines (SHA-256
`7c167018ee6bd61a3609971e3f834082c798e7bd2a3fa0c5c2564887da4fde4f`).
The candidate proposal selects this 54-package lock; it grants no runtime or
hardware authority.

Use `examples/training_qualification/native_runtime_preparation.py plan` with
the exact checked-in `native-private-runtime.example.json` to inspect the
166,269,790-byte archive selection without effects. Only after independent
review, copy that config to a new 0600 file in an existing 0700 private parent
and use its `acquire`, `install`, and `verify` phases from a clean pushed source.
The CLI retains the exact Astral 3.11.14+20260211 interpreter archive and 54
wheel archives, checks hashes before an offline uv local-mirror install, binds
the uv executable hash/version, confines caches and temporary paths to the
private root, and retains consumed claims. HTTPS transfers stop before writing
an excess byte; finite expansion checks are pre-install inventory bounds, not
a hard aggregate filesystem quota for uv temporary files. A 2 GiB retained
private-tree guard runs after each phase. Include narrow
`source/shared` in the subsequent final-source static capture alongside
`source/tuner` and `source/synaptic_tuner`. Do not chmod Homebrew or reuse the
old venv/capture attempt.

Pinned uv 0.12.23 was observed creating empty owner-owned `uv-python/.lock`
and `venv/.lock` at mode 0666 despite umask 077 inside the private 0700
runtime. The checked-in fresh `install` seals only those exact regular,
zero-byte, single-link files to 0600 after the last uv call and checks the
installed tree before its result. For the already completed r9 install,
`seal-uv-locks <same-absolute-config>` is a separate, one-use repair action
requiring independent review and exact retained source/config/archive/uv/
install-record admission. It writes a consumed repair claim before either
descriptor mode change; interruption or identity drift retains the claim and
partial evidence without automatic replay. It preserves the original install
claim/result. `verify` rejects incomplete repair records and writable members.
Do not run static capture until custody passes; the capture must bind the
final pushed source, including `shared`, and the combined 54-package lock.

The checked-in `examples/training_qualification/native_model_preparation.py`
is the later reviewed model-acquisition action: `inspect` is local,
`metadata-candidate` reads at most 2 MiB of the exact Hub metadata response
before JSON/SDK `ModelInfo` parsing, and `acquire` requires a separately
reviewed complete member/tokenizer selection. Model files stream from the
exact pinned Hub resolve URLs into precreated private 0700 nested parents;
per-file and aggregate caps reject before excess writes. The SDK's unbounded
snapshot/cache download path is not used. Unexpected redirect hosts fail
closed. It verifies upstream hashes and publishes private create-only
snapshots. Its receipt is
`OBSERVED_NOT_ADMITTED`. No complete Mac model snapshot or tokenizer labels
have yet been acquired/reviewed. See `examples/training_qualification/native-host.md` for
exact staging paths, publication digests and commands.

The first bounded model acquisition left a consumed partial workspace with
five verified small members and a zero-byte `model.safetensors`. Preserve it;
the byte count alone does not diagnose the transport. A separately admitted
`probe-transfer` uses the existing helper and a fresh private record root,
binding the exact reviewed acquisition config hash and selected member. It
performs at most one HEAD chain and one GET chain, up to three redirects each,
20 seconds per request, 120 seconds for the owned child, and reads no more
than one GET body byte without a Range header. Unknown redirect origins stop
before follow. Closed observations contain no URL, query, header value,
credential or exception text. Its consumed claim and result cannot repair or
authorize replay of the old acquisition. See the host example for the config.
The r13 one-use probe observed HTTP 302 on both methods to the exact
`us.aws.cdn.hf.co` origin documented by Hugging Face; the preexisting
acquisition allowlist rejected it before follow. Only that exact observed
origin was added to the helper allowlist. Preserve the r12 consumed workspace;
the probe cannot retroactively establish its failure cause or admit a retry.

See `docs/plans/native-desktop-training-api-v1.md` for remaining review and
real-Mac acceptance. The CPU facade tests substitute the ML child;
they are not a hardware result and must never unlock the candidate.

### Verification and numeric checks — 2026-10-26

The earlier staging observations above are historical. Exact successful Mac
attempts and their limits are recorded in
`docs/review/native-mlx-sft-qualification.md`; use that report's source bindings.
The engine's `runs.verify` admits an initially unverified artifact set.
For an already verified run, use `runs.reverify`; the restricted host consumer
exposes separate `verify` and `reverify` actions. A repeated initial verification
can refuse by design and is never a reason to rerun training or renew admission.
Retained reads use the original keys, catalog and material after start expiry.

On the admitted Apple runtime, the checked-in
`tests/runtime/test_native_mlx_numeric.py` compares the production loss hook and
real MLX autograd against independent scalar softmax/cross-entropy gradients for
full-sequence, prompt-masked and noncontiguous labels. Run with the separate test
tools and a bounded process; it loads MLX but no model or training publication.
Its three cases skip on non-Apple hosts. A skip or CPU algebra double is not
real MLX numerical evidence and does not issue training authority.
