# Docker r2 Torch import boundary — CPU diagnosis, 2026-10-20

## Retained failure

The one authorized r2 public start on engine `0d515ae1ae235d0858802228ce4af8dba8fccd03`
and host `85ea6b56af89f6dcb7c0a2c7644e95c511762486` was accepted at
13:47:50Z and failed at 13:47:52Z. Container
`5256d2446d3751e423a6ff535c3ed9a13a2bc5df92765a61db5da700bf7e601d`
remains retained with exit 23. Its 81-byte trainer stderr, SHA-256
`95d34d74eae9cb55a223eeda8428ee4387ee79095e5c0f39a0ceb60da9f45513`,
contains only `OFFLINE_SFT_WORKER_REJECTED phase=TRAINER category=IMPORT
milestone=TORCH_IMPORT`. This locates the first Torch import boundary in
`Trainers/sft/train_sft.py`; it does not identify an import dependency or a
remedy. The earlier bare marker on container
`07ecde6d5b798abf51eb85595be165039cdc1043295cbd0a5b0d3d526064bfd7` remains historical
and uninterpreted.

The release file SHA-256 is
`5f65824ceb5e250a58314da0233b9c07f1d63f1b0a39892d9b36af694938bfe6`;
its immutable image ID is
`sha256:e737a0a091474c4892690690e712bc1f932f108856a3d5df60383e1cd9436e1e`.
The retained final runtime capture records CPython 3.12.3 and Torch
`2.11.0+cu128`. Source, image, release and installed wheel were not changed.

## Targeted CPU experiments

The r4 version of `scripts/diagnose_docker_torch_import.py` bound that release
file hash, daemon ID and exact local image ID. It requested at most 2 CPUs,
4 GiB, no network or GPU devices, and had no credentials, model mount or
training entrypoint. Its `subprocess.run(timeout=120)` bounded the Docker CLI
call but **did not enforce a 120-second container lifetime**. Its
`capture_output=True` also buffered CLI output without a preallocation limit.
The earlier 120-second container claim was incorrect. The retained r4 results
are observations, not proof of a bounded lifecycle.

The bare image probe returned `PASS` in container
`33af62c4fdda5d855b3b154c2937912b5266a88a48f2c0757b3664fe42a45917`.
The final guarded probe mounted the r2 staged source read-only, validated a
manifest against the digest in that same mount, installed its owned-module guard, ran
`init_trainer_env()`, and imported Torch successfully. Its retained container
is `2e5b6d06a4cc946541a475da72b7aa7e851b5840f86fd675c2099540b6fc1045`;
`evidence-guarded-final2/result.json` has SHA-256
`326b59ebf3896a737732661895ef67b3efcf2527d65ce912b02ff20d9437ee33`.
The exact final checked-in helper source has SHA-256
`c214fc8b9d617fe7171e7eac987d93187169b8a637fac9ef24b4323a70bf6c53`.
These artifacts are under the private Windows WSL root
`/home/profsynapse/.local/share/synaptic-qualification/torch-import-r4-20261020/`.
The r4 helper had no independently admitted source or closure commitment, so
this `PASS` cannot establish that it executed the exact r2 closure. The result
remains historical and is not relabeled after the helper repair.

Earlier guarded diagnostic containers returned `READ_MANIFEST/OS` because
the probe dropped all capabilities while running as root against mode-0600
staged files owned by the WSL user. Running the diagnostic as that owner
resolved this probe setup failure. Those containers and reports are retained;
their failures do not reinterpret r2.

## Interpretation and next boundary

Torch is importable in the exact image both bare and after the retained worker
closure, guard and trainer bootstrap under the finite CPU diagnostic. The r2
container requested GPU 0 and used root, default capabilities and a writable
root filesystem; the CPU probes deliberately did not reproduce that full
runtime. The cause of r2's Torch import failure remains unresolved. In
particular, the evidence does not support changing packages, image, recipe,
model, GPU settings or SFT runtime code yet.

Read-only inspection of the retained r2 container on 2026-10-20 confirmed
entrypoint `/opt/unsloth-venv/bin/python3`, command prefix
`/source/engine/Trainers/sft/runtime_v1.py --canonical-workload-file
/source/control/workload.json`, user `root`, workdir `/workspace`, runtime
`runc`, writable root, default capabilities, network `none`, and Docker GPU
device request `0`. The parent runtime creates a separate offline worker with
an explicit child environment (`Trainers/sft/runtime_v1.py:1480-1497`) and
executes the trainer through `runpy.run_path`; the trainer prepends its `src`
and engine roots to `sys.path` immediately before `init_trainer_env()` and
`import torch` (`Trainers/sft/train_sft.py:217-227`). The r4 guarded probe
cleared/replaced its environment, disabled CUDA/NVIDIA visibility, dropped
capabilities, used the staged-source owner, and skipped the parent runtime.
Its path setup also lacked the trainer's exact `src` insertion. The repaired
guarded probe mirrors that path order, but it still does not reproduce the
parent's environment, user, capabilities, writable root or GPU request.

The r2 claim, container and all older failure evidence remain consumed and
preserved. No training start, image build, pull, model load, package install or
HF action occurred during this diagnosis. A further comparison of the
original GPU/runtime environment needs a separately reviewed exact scope;
this diagnostic does not authorize another training attempt. No runtime
closure or Modal lock refresh is needed for the diagnostic script and skill
changes, because no locked runtime member changed.

## Repaired helper and next review boundary

The successor helper claims a fresh unique diagnostic name before create,
caps aggregate Docker CLI output at 8 KiB, requests JSON log rotation at
1 MiB with one file, and
separates create, authenticated start, terminal inspection and one owned stop
request. Its 120 seconds are an execution deadline followed by a bounded
cleanup and inspection tail; a Docker stop response alone never proves physical
quiescence. If create/start/stop or identity is uncertain, the claim and events
remain and no automatic replay occurs. For guarded mode, independently supplied
r2 source-lock SHA-256 `d19aecb2d3ca1b7b6fa32a2445c45251e59d8cac2332293c68b73a61cbff4665`,
manifest SHA-256 `3842689403abc5b42e28d58545b88616b81cd8fc738ee5557e97159df65d00ef`,
and closure digest `8465ff2b7cedb352337e7cedc5bb17cc169892c2eedae0b280569f4a0cde5002`
are checked before mounted code executes and included in the result. These
values were read from the retained r2 material on 2026-10-20; live rechecking
before any new effect remains required. The reviewed successor CPU probe ran
once as r6 below. A GPU-exposed probe still needs separate scope review.

The cleanup path attempts its single authenticated stop even if evidence
appends fail, then inspects terminal state independently even if the stop CLI
acknowledgement fails. Failed evidence writes retain an uncertain result; an
in-memory observation does not establish durable proof. Source verification
and a read-only container mount do not freeze files on the host. Before this
probe, verify private ownership/permissions of the retained source and its
ancestors, and ensure that no preparation or other writer is using that source.
This probe assumes that private host custody remains in force during execution.

The reviewed r6 guarded CPU command is retained here as the exact executed
invocation. It ran from a clean WSL checkout of
`c48fa8299333d6057bc876899d484386f2059ddc` after confirming that its
helper and trusted closure verifier matched the reviewed source. Its parent
`/home/profsynapse/.local/share/synaptic-qualification/torch-import-r5-20261020`
was private mode 0700; the `guarded-cpu-1` child was absent before the run.

```sh
python3 -I scripts/diagnose_docker_torch_import.py \
  --docker /usr/bin/docker \
  --docker-config /home/profsynapse/.local/share/synaptic-qualification/docker-sft-stable-image-c/docker-config \
  --release /home/profsynapse/.local/share/synaptic-qualification/docker-sft-stable-image-c/evidence/runtime-release.json \
  --expected-release-sha256 5f65824ceb5e250a58314da0233b9c07f1d63f1b0a39892d9b36af694938bfe6 \
  --daemon-id 24805003-b125-469b-8080-9da4daacd364 \
  --source-root /home/profsynapse/.local/share/synaptic-qualification/docker-public-plan-0d515-20261020/attempt-r2/source/engine \
  --source-lock /home/profsynapse/.local/share/synaptic-qualification/docker-public-plan-0d515-20261020/attempt-r2/material/source-lock.json \
  --expected-source-lock-sha256 d19aecb2d3ca1b7b6fa32a2445c45251e59d8cac2332293c68b73a61cbff4665 \
  --expected-manifest-sha256 3842689403abc5b42e28d58545b88616b81cd8fc738ee5557e97159df65d00ef \
  --expected-closure-digest 8465ff2b7cedb352337e7cedc5bb17cc169892c2eedae0b280569f4a0cde5002 \
  --output-dir /home/profsynapse/.local/share/synaptic-qualification/torch-import-r5-20261020/guarded-cpu-1
```

This repeated the reduced CPU import boundary with repaired ownership and
source binding. It was **not** an exact r2 invocation, did not expose a GPU,
and did not authorize another SFT start. The original r2 claim remains consumed.

The next comparison should hold the independently authenticated r2 source,
daemon and image fixed, then compare the offline child environment produced by
`runtime_v1.py` with the CPU diagnostic at the Torch import boundary. In
particular compare Python executable/`-I` invocation, worker closure and
module guard, final `sys.path`, `PATH`, `LD_LIBRARY_PATH`, `HF_HOME`,
`TRANSFORMERS_CACHE`, `TMPDIR`, and CUDA/NVIDIA visibility by key and approved
value or digest without printing secrets. Vary one of user/capabilities,
writable root and GPU visibility at a time in import-only containers under
reviewed bounds. A GPU-visible import-only experiment is a separate reviewed
scope; no trainer, model load or training is proposed. Treat any changed result
as evidence for a narrower hypothesis, not retrospective proof of r2's cause.

## r6 result and read-only r2 context comparison

The one authorized r6 run used the unchanged command above after read-only
preflight and a saved/read-back Nexus admission. The clean WSL checkout was
`c48fa8299333d6057bc876899d484386f2059ddc`; helper SHA-256 was
`290bf1aa8e10ce3cf8a7eb0f1ff8e5a394a7ac726495330eaf0ff24fe9a33ac1`
and trusted closure-verifier SHA-256 was
`2304eeb00667954510c34cc4aacaa818532a301fbeabf04c2fb165d6cd1d70b5`.
The retained r2 manifest, source lock and release matched the independently
approved hashes in the command. No matching preparer or worker process was
observed before start; the source and parent were privately owned. This
custody check does not freeze host files against a later writer.

The r6 child returned `PASS` at `TORCH_IMPORT`, with `training_executed=false`
and `gpu_used=false`. Container
`8c578b6ec0930e9c7668f237672b9d11f4122524befe601d077504966ff203e0`
is retained and independently inspected as exited, code 0. Its claim label is
`f07f5348f19fc6bd7210929c78a1b7c2b900361da249ac5755ce6eabb8b2330d`;
the daemon and image IDs match r2. It started at
`2026-10-20T14:51:44.979964328Z` and finished at
`2026-10-20T14:51:48.957092702Z`. The claim, events and result under the
private `torch-import-r5-20261020/guarded-cpu-1` directory have SHA-256
`a62c0bb6a610c2a27a7951fe7e7c95038720b9ed882a6c6044fe1aeffaf6c8dc`,
`0bab6899f60dd05b885e65fa9a4c90ee8f65d5c751357bf4fad8752943a2446d`,
and `c22892913e1c0b3aaff1bfc99b876c80a653000ae66c8f29d3e83023c00a1308`,
respectively. Events show one claimed create, one start and terminal exit 0;
no stop or deletion was needed. The r2/r4 claims and containers remain preserved.

The retained r2 workload SHA-256 is
`1534308c977c2c1cb2b1d80d9cc5c07e31303db63b336686e1c5ccc2644b1064`;
its Docker runtime declaration SHA-256 is
`947809449ffce1fde2cdb6a928c2cec60444cc3d841d173405080b7f545e4aae`.
The staged r2 `runtime_v1.py`, `train_sft.py` and `offline_sft_worker.py` bytes
match engine commit `0d515ae1ae235d0858802228ce4af8dba8fccd03`.
The pinned release records CPython 3.12.3 and manifest digest
`c5d6c01fb1dd62276b033234982cd14846865500dd07eaed2a2102dea97a1535`.
Read-only inspection of both retained Docker container configs and these
sources gives the comparison below.

| Boundary | Retained r2 | Guarded r6 CPU |
| --- | --- | --- |
| Image, daemon, source | Pinned image and daemon; staged engine closure `8465ff2b...` | Same image and daemon; independently authenticated same staged closure |
| Host container | Root, `/workspace`, default caps, writable root, GPU device request `0`, 16 GiB | UID/GID 1000, `/tmp`, all caps dropped, read-only root, no device request, 4 GiB |
| Common host settings | 2 CPUs, network `none`, `runc` | 2 CPUs, network `none`, `runc`; 120-second execution deadline plus bounded cleanup tail |
| Python entry | Image Python starts parent `runtime_v1.py`; parent starts separate `-I offline_sft_worker.py -- ...` child | Same image Python runs `-I -c` fixed diagnostic; no parent or worker entrypoint executes |
| Closure and import order | Worker loads closure, verifies origins, installs guard, adds trainer directory and engine root, then `runpy.run_path` enters `train_sft.py`; trainer adds engine and `src`, bootstraps, imports Torch | Probe loads same closure, verifies origins, installs guard, arranges the same leading engine/`src`/trainer/engine path order, bootstraps and imports Torch directly; later path tail and loaded-module history differ |
| Offline child environment | Source builds `child_env` from the planned environment with `clear_inherited=true`, removes `PYTHONPATH`, adds closure/root bindings and overrides offline/HF/cache flags; actual process environment was not separately captured | Probe clears inherited environment and installs fixed import-only values. It lacks r2's closure/root environment bindings and HF/cache paths; expected closure digest is embedded in fixed probe code |
| Visibility inside child | Planned environment has no `CUDA_VISIBLE_DEVICES`, `NVIDIA_VISIBLE_DEVICES` or `LD_LIBRARY_PATH`; parent Docker config has NVIDIA/LD entries, but `env=` prevents child inheritance | Probe sets CUDA visibility empty and NVIDIA visibility `void`; it also omits `LD_LIBRARY_PATH` |

The r2 planned environment's sorted compact `jq -cS` serialization
(including its trailing LF) has SHA-256
`e621be06bb27b0c258e8ef8ecfd35eccaa362236f9a194eb122bfe4445a798ed`.
Its `PATH` value SHA-256 is
`ff201961161ffc598d11b7e168870c3fdbe7485141fa414c3f67f2ce89244338`;
the r6 fixed probe `PATH` value SHA-256 is
`05f89650ff263293ba277e3c910f4343e0f9b1a37c6d27e0aab9cb3b772e0896`.
The r2 plan's `PYTHONPATH` equals the engine root but is explicitly removed
from `child_env`. Its declared `HF_HOME` and `TRANSFORMERS_CACHE` equal the
cache root plus their respective subdirectories, matching the child builder's
overrides. These are source-and-plan deductions, not a captured r2 child
environment. No environment value, raw argv or exception text is published.

The strongest next single-factor import-only comparison is GPU visibility in
the otherwise unchanged guarded diagnostic. This holds the authenticated
image/source, user, capabilities, read-only root, fixed probe and deadline
constant while changing the device request and removing the two explicit
visibility suppressors. Docker documents `--gpus device=0` for exposing only
GPU index 0 in its [GPU access guide](https://docs.docker.com/engine/containers/gpu/).
A changed result would implicate GPU exposure under
these reduced conditions; either result would leave r2's exact parent/child
environment, user/caps and filesystem differences unresolved. The existing
helper now has a fixed guarded-only `--gpu-device 0` option. It requests only
device 0, checks the resulting Docker device request before start, includes
this scope in its claim, and reports `gpu_used=null` because device use is not
measured. It retains the 8-KiB CLI output budget, 120-second execution
deadline, single owned stop and independent terminal inspection. The fixed
child still only installs the source guard, bootstraps and imports Torch; it
does not invoke the trainer, load a model or run training.

Root separately reviewed and authorized the following command for one r8
import-only run. It required a clean checkout of `98e4a49ee175102f645e8c497acaff90fd5c4248`,
fresh read-only daemon/image/source/custody checks, a new private mode-0700
`/home/profsynapse/.local/share/synaptic-qualification/torch-import-r7-20261020`
parent, an absent `guarded-gpu0-1` child, and a saved/read-back one-use
admission. This exact command ran once and is consumed:

```sh
python3 -I scripts/diagnose_docker_torch_import.py \
  --docker /usr/bin/docker \
  --docker-config /home/profsynapse/.local/share/synaptic-qualification/docker-sft-stable-image-c/docker-config \
  --release /home/profsynapse/.local/share/synaptic-qualification/docker-sft-stable-image-c/evidence/runtime-release.json \
  --expected-release-sha256 5f65824ceb5e250a58314da0233b9c07f1d63f1b0a39892d9b36af694938bfe6 \
  --daemon-id 24805003-b125-469b-8080-9da4daacd364 \
  --source-root /home/profsynapse/.local/share/synaptic-qualification/docker-public-plan-0d515-20261020/attempt-r2/source/engine \
  --source-lock /home/profsynapse/.local/share/synaptic-qualification/docker-public-plan-0d515-20261020/attempt-r2/material/source-lock.json \
  --expected-source-lock-sha256 d19aecb2d3ca1b7b6fa32a2445c45251e59d8cac2332293c68b73a61cbff4665 \
  --expected-manifest-sha256 3842689403abc5b42e28d58545b88616b81cd8fc738ee5557e97159df65d00ef \
  --expected-closure-digest 8465ff2b7cedb352337e7cedc5bb17cc169892c2eedae0b280569f4a0cde5002 \
  --gpu-device 0 \
  --output-dir /home/profsynapse/.local/share/synaptic-qualification/torch-import-r7-20261020/guarded-gpu0-1
```

## r8 single GPU-visible import result

Before the single r8 start, a fresh clean WSL checkout and the reviewed helper
SHA-256 `95849f03b1133302cef1bc7ceec1e20674897877152c9b35b64af2b86cf718c3`
were authenticated. The trusted closure verifier SHA-256 was
`2304eeb00667954510c34cc4aacaa818532a301fbeabf04c2fb165d6cd1d70b5`.
Release, image, daemon, retained source lock, manifest and closure identities
matched those above. The source and ancestors passed private-custody checks,
no matching writer was observed, and the private mode-0700 output parent had
no `guarded-gpu0-1` child. Windows and WSL GPU queries both identified device
0 as `GPU-ac4398bb-fc15-d722-b39c-2790b4b5f9cc` (RTX 3090) and returned
no active compute-process row. The one-use Nexus admission was saved and read
back before Docker create.

The closed r8 result is `PASS` at `TORCH_IMPORT`, `gpu_device="0"`,
`gpu_used=null` (device use not measured), and `training_executed=false`.
Container `d41a2c26ee089777b7583e6100e67c5a899eedc7ff2c72b9beddae22cfe669c2`
is retained. Independent Docker inspection found the exact claim label
`e076d416d5d8ea04d97ae245d989db7d1d41d4ef74efe036c3aec723dff57dd2`,
image ID, one device request with `DeviceIDs=["0"]` and `Capabilities=[["gpu"]]`,
2 CPUs, 4 GiB, network `none`, read-only root, UID/GID 1000, all capabilities
dropped, and terminal `exited` with code 0. It ran from
`2026-10-20T15:11:08.249437944Z` to `2026-10-20T15:11:11.665541884Z`.
Events record one claimed create, one start and terminal exit 0; no stop or
deletion occurred. The retained `claim.json`, `events.jsonl` and `result.json`
under `torch-import-r7-20261020/guarded-gpu0-1` have SHA-256
`a7a85de97044ed8b2381f20be1c5e0846fc9ddb1b0217bde6b1da7b3d95ae89c`,
`05b15a0b21e2dc653572f5f4f51f12e5b98371a65fe216a5b76969efecdc158d`,
and `439d5a19c1b838840ee8cd49085a95b68acfa50e8182e3dceca43b52722794f1`,
respectively. The r2/r4/r6 claims and containers remain unchanged.

Both guarded r6 CPU and r8 device-0 imports passed in the reduced diagnostic.
Adding device-0 exposure and removing the two visibility suppressors did not
reproduce r2's closed import rejection under those conditions. It does not
recreate the r2 parent/isolated-child environment, Python entry, root/default
capabilities, writable filesystem or 16-GiB limit, and cannot establish or
rule out an interaction with them. The cause of r2 remains unproven. This
consumed diagnostic grants no replay, broader GPU comparison or training
authority.

## r9 prepared worker import boundary — no live execution

The r8 result leaves the original isolated child invocation and trainer
preamble untested. The next proposed **single** import-only comparison uses
the retained r2 workload, source lock, closure and image. This is preparation
for root review; no new container, device request or probe was started in r9.
It supersedes the earlier idea of varying incidental host settings one at a
time.

The retained r2 workload is SHA-256
`1534308c977c2c1cb2b1d80d9cc5c07e31303db63b336686e1c5ccc2644b1064`.
Its pinned `runtime_v1.py` and `train_sft.py` are SHA-256
`460121aaee3eb83764f67d9307cfb57a3b25c117611d987577b9631b6956a1b3`
and `75c2ab22fbc5dd80b8ddd5e6d0da816d1451cda0cd620c34717336a39a8a7644`.
The helper verifies all three against independently supplied expected hashes
before Docker create and again before start, in addition to the existing
release, daemon, image, source lock, manifest and closure checks. The workload
file is included in source custody checks. The source mount remains read-only;
private host custody and no concurrent writer remain necessary during a run.

`--worker-context` derives the child environment from the pinned workload
using the exact `runtime_v1.py:1479-1497` transform: start from its declared
variables, remove `PYTHONPATH`, bind the closure and runtime roots, then
override the workload fingerprint, isolated Python flags, HF/cache/offline
values and model-snapshot **path**. It starts a separate `-I` interpreter
with that replacement environment and cwd `/artifacts/run/tmp`. The
model-snapshot path is a string only; no model weights are mounted or loaded.
The child evaluates the authenticated worker module without its `__main__`
entry, loads the closure, verifies origins, installs the owned-module guard
and inserts the worker's trainer-directory/engine path pair. It then compiles
the authenticated `train_sft.py` prefix ending with `import torch`, with
the real trainer `__file__`, `__main__` name and phase callback. The
`UNSLOTH_IMPORT` marker and every later statement are excluded from the
compiled prefix. Marker order, exact Torch/Unsloth adjacency, source hashes
and the last AST statement fail closed on drift. The child returns only
closed category/stage and SHA-256 digests of its preamble environment, cwd,
trainer file path and `sys.path`; the helper checks the environment digest
against its own reconstruction. No raw environment, argv, path, exception or
traceback enters the report.

The proposed host stays within 2 CPUs, 4 GiB and a 120-second execution
deadline plus bounded owned cleanup. It keeps r8's UID/GID of the staged
source owner, dropped capabilities, read-only root, network `none`, and
device-0 request; it adds only a 64-MiB `/artifacts` tmpfs for private
runtime roots and no model mount. The separately launched child's cwd
matches r2; the outer container still uses `/tmp`. This is closer to r2
at the child import boundary, but root/default capabilities, 16-GiB memory,
writable root, actual model cache, and the full parent admission/preparation
remain different. The original child environment was not captured, so this
is a source-and-plan reconstruction rather than a historical byte capture.
Even a changed import outcome would narrow a hypothesis, not prove r2's
cause or authorize training.

The exact **unexecuted** proposal for root review is:

```sh
python3 -I scripts/diagnose_docker_torch_import.py \
  --docker /usr/bin/docker \
  --docker-config /home/profsynapse/.local/share/synaptic-qualification/docker-sft-stable-image-c/docker-config \
  --release /home/profsynapse/.local/share/synaptic-qualification/docker-sft-stable-image-c/evidence/runtime-release.json \
  --expected-release-sha256 5f65824ceb5e250a58314da0233b9c07f1d63f1b0a39892d9b36af694938bfe6 \
  --daemon-id 24805003-b125-469b-8080-9da4daacd364 \
  --source-root /home/profsynapse/.local/share/synaptic-qualification/docker-public-plan-0d515-20261020/attempt-r2/source/engine \
  --source-lock /home/profsynapse/.local/share/synaptic-qualification/docker-public-plan-0d515-20261020/attempt-r2/material/source-lock.json \
  --expected-source-lock-sha256 d19aecb2d3ca1b7b6fa32a2445c45251e59d8cac2332293c68b73a61cbff4665 \
  --expected-manifest-sha256 3842689403abc5b42e28d58545b88616b81cd8fc738ee5557e97159df65d00ef \
  --expected-closure-digest 8465ff2b7cedb352337e7cedc5bb17cc169892c2eedae0b280569f4a0cde5002 \
  --worker-context \
  --expected-workload-sha256 1534308c977c2c1cb2b1d80d9cc5c07e31303db63b336686e1c5ccc2644b1064 \
  --expected-runtime-sha256 460121aaee3eb83764f67d9307cfb57a3b25c117611d987577b9631b6956a1b3 \
  --expected-trainer-sha256 75c2ab22fbc5dd80b8ddd5e6d0da816d1451cda0cd620c34717336a39a8a7644 \
  --gpu-device 0 \
  --output-dir /home/profsynapse/.local/share/synaptic-qualification/torch-import-r9-20261020/worker-context-gpu0-1
```

The output parent must first be created privately at mode 0700 and the named
child must be absent. Root review must confirm a clean pushed helper/verifier,
the exact remote identities and source custody, no active device-0 compute
job, the proposed UID/caps/tmpfs/readonly filesystem and bounds, then save
and read back a fresh one-use admission before any start. A collision or drift
rejects without renaming or retrying; r2/r4/r6/r8 claims remain consumed.

## r10 startup-environment evidence repair — no live execution

Independent review found that the r9 helper compared a digest of the
environment supplied to the isolated child with a digest observed *after*
CPython startup. The retained r2 child plan supplies no locale keys. CPython
may add `LC_CTYPE=C.UTF-8` when coercing the default C locale; its
[3.12 command-line documentation](https://docs.python.org/3.12/using/cmdline.html#envvar-PYTHONCOERCECLOCALE)
describes the insertion, and the
[v3.12.3 preconfiguration source](https://github.com/python/cpython/blob/v3.12.3/Python/preconfig.c)
contains that coercion path. A real local isolated CPython 3.12.15 subprocess
with the fixed no-locale input added exactly this key/value. The pinned image's
CPython 3.12.3 behavior has **not** been observed by executing this proposal.

The repaired helper passes the exact supplied mapping to the child through a
bounded private stdin pipe as well as `env=`. Before mounted worker code, the
child checks the supplied digest against its independently bound expectation.
At the existing preamble observation point it requires every supplied key
and value unchanged. It accepts either no added keys or exactly
`LC_CTYPE=C.UTF-8`; any other addition or changed supplied value rejects.
It reports separate `environment_supplied_sha256` and
`environment_observed_sha256` plus the closed delta `NONE` or
`LC_CTYPE_C_UTF8`. The host verifies both digests against its own
reconstruction, including the exact allowed locale variant. It never
prints environment values, resets them, or claims to have captured r2's
historical child. The owned lifecycle, 120-second execution deadline,
8-KiB Docker CLI output limit, source commitments and stop-before-Unsloth
boundary are unchanged.

The proposed command in the r9 section above is unchanged and remains
**unexecuted**. For root delta review, the repaired helper SHA-256 is
`c7543b50c4aa41b91ed8293474b4f1719c69342550515a8d367b06dd3f1fd229`;
the trusted closure verifier remains
`2304eeb00667954510c34cc4aacaa818532a301fbeabf04c2fb165d6cd1d70b5`.
Root must review the pushed commit and exact command, repeat the current
identity/custody/device preflight, and save/read a fresh one-use admission
before any container start. No r10 container, GPU request, ML import or
training effect occurred.

## r11 single worker-context import result

Root accepted the r10 repair and authorized exactly one run of the unchanged
r9 command above. Before Docker effects, the fresh private mode-0700 remote
parent `torch-import-r9-20261020` had no `worker-context-gpu0-1` child.
Its clean detached checkout matched pushed
`4b56cdcdb6c4d5f35d5e7af2a03e322f39c50622`; helper SHA-256 was
`c7543b50c4aa41b91ed8293474b4f1719c69342550515a8d367b06dd3f1fd229`
and trusted closure verifier SHA-256 was
`2304eeb00667954510c34cc4aacaa818532a301fbeabf04c2fb165d6cd1d70b5`.
The exact release, source lock, manifest, closure, workload, runtime and trainer
hashes in the command matched read-only remote validation. The explicit
Docker daemon and image IDs matched. Source files and ancestors had private
owner/root custody, no symlink or group/world write; a process/descriptor
scan found no matching active preparer or open writer. This does not
atomically freeze host files, so private custody remains a scope assumption.
Windows and WSL reported no active compute-process rows for device 0
(`GPU-ac4398bb-fc15-d722-b39c-2790b4b5f9cc`, RTX 3090). The exact
command and preflight were saved and read back in the Nexus one-use state
`Docker Torch r11 worker-context one-use admission — 2026-10-20` before
start. An immediate checkout/source/output/GPU recheck passed.

The single helper invocation returned closed `PASS` at `TORCH_IMPORT` with
`boundary=BEFORE_UNSLOTH`, `scope=worker_context`,
`training_executed=false` and `gpu_used=null` (actual device use was not
measured). The supplied child environment SHA-256 was
`1bb0e06d9708dd6fd637f59eaf134dcbe8165f43e120a0b94daad6c79a63699c`.
The observed preamble environment SHA-256 was
`a1eeabbf01741b374ca4e410d62119cf8d882b97d33da962257ec7570bdcb08c`,
with the only accepted startup delta `LC_CTYPE_C_UTF8`. The reported
cwd, trainer `__file__` and `sys.path` digests were respectively
`55846bb91af8c8ccb1548b812885d0ca9d413863279b9030c6a6a1b2327d00a6`,
`4863bf5f38bf9e9cf9b2c8d44473326e77b6aa2c55d9fe039fd20092bb68ad9a`,
and `fbb8a8e665063c6fa0c114cb5edc272b4ecdbbab8c521dac9653a2803c1c0a0d`.
No raw environment, argv or exception text was published.

Container
`d829fc00a2f9a48437bae1ccfd562d7846b97228b4c526759928fe98c9a618cd`
is retained with exact claim label
`db3afc2a9434fadb65ecaf38d97f26b7e1f7039ea66265554ab5615071209bc4`.
Independent Docker inspection confirmed image/daemon ownership, the single
GPU device-0 request, 2 CPUs, 4 GiB, 64 pids, network `none`, read-only root,
all capabilities dropped, UID/GID 1000, a read-only `/source` mount, private
64-MiB `/artifacts` tmpfs, and no model mount. It started at
`2026-10-20T15:58:15.445432531Z` and finished at
`2026-10-20T15:58:19.681177983Z`, independently observed `exited` with
code 0. Events contain one `CREATE_CLAIMED`, `CREATED`,
`START_CLAIMED`, `STARTED` and `TERMINAL_OBSERVED`; no stop or deletion
was needed. The private `worker-context-gpu0-1` evidence has SHA-256:

| File | SHA-256 |
| --- | --- |
| `claim.json` | `e65eed000fa69e18d387d18484b0055eff1cf4a9dbb349d645e311a25b5fe99a` |
| `events.jsonl` | `e5a7460c94e94b90f1c3142ed505e34c5bf2c8854447c26ad74b0b840b9bddb1` |
| `result.json` | `1d4e4a9d2e2d9f43a2250b6634c63c270852e0b2babd027d61779f30306e054d` |

Read-only inspection still found the earlier r2/r4/r6/r8 containers retained
in their previous terminal states. The r11 claim is consumed; no second probe
or training followed. This PASS establishes Torch import through the
authenticated, source-derived child environment and trainer prefix under the
specified reduced host context. It is not a capture of r2's historical child
and leaves root/default capabilities, 16-GiB memory, writable root, actual
model cache and full parent preparation different. The r2 failure cause
remains unproven.

## r12 worker-entry comparison and proposed import-only probe — unexecuted

Read-only inspection of the retained r2 workload and containers on
2026-10-20 found a prepared dataset reference; the r2 parent source therefore
passes one sealed prepared-dataset descriptor to its isolated worker. The
retained workload remains SHA-256
`1534308c977c2c1cb2b1d80d9cc5c07e31303db63b336686e1c5ccc2644b1064`.
Its planned environment is bound by that workload, but no historical child
environment or complete child argv was captured. Source inspection establishes
how `runtime_v1.py` builds them, not their actual r2 process bytes.
Read-only execution of the hash-pinned runtime's argument appender against
the hash-pinned workload produced 34 appended tokens with SHA-256
`7d2de6641ff52cd26c4515b2c0f65bed94b9eb11ee8ae02b0910596b26fd5937`;
this is a source/plan reconstruction, not a retained child argv digest.

| Boundary | Retained r2 source/container | Consumed r11 probe | Proposed r12 worker entry |
| --- | --- | --- | --- |
| Isolated launch | Parent calls `/opt/unsloth-venv/bin/python3 -I` on the worker script with trainer arguments | Separate `-I -c` child; evaluates worker under a diagnostic name, skipping its main and argument validation | Separate `-I -c` child enters the authenticated worker with `runpy.run_path(..., run_name="__main__")`; worker runs its real admission and nested trainer `runpy.run_path` |
| Trainer module | Worker sets full trainer argv and `runpy` temporarily binds trainer as `sys.modules['__main__']` | Compiled AST prefix executes in an unattached dictionary with only the trainer filename in argv | Trace checks trainer filename, reconstructed argv and actual `sys.modules['__main__']` namespace, then stops at the authenticated `UNSLOTH_IMPORT` marker |
| Environment | Parent replaces inherited environment after removing `PYTHONPATH`; r2 process bytes were not retained | Source/plan reconstruction, supplied through a private stdin pipe | Same source/plan reconstruction, with separate supplied and observed digests; stdin is `/dev/null` |
| Descriptors and stdio | Prepared input gives one sealed `pass_fds` descriptor; stdin `/dev/null`, stdout/stderr exclusive regular files | No prepared descriptor; stdout pipe and stderr `/dev/null` | One sealed one-byte **synthetic** descriptor and `pass_fds`; stdout/stderr private exclusive files under the 64-MiB artifacts tmpfs |
| Container and data | Root, default capabilities, writable root, 16 GiB; `/source` read-only, `/artifacts` writable and populated with model/cache/prepared state; device 0 | Source-owner UID/GID, caps dropped, read-only root, 4 GiB, private artifacts tmpfs, no model or dataset bytes; device 0 | Keeps those reduced r11 host settings and device 0; no model or historical dataset bytes mounted |

The new `--worker-context --worker-entry` mode uses the pinned runtime source's
`_append_sft_arguments` and retained workload fields to reconstruct the
trainer flags and values, while substituting a sealed one-byte descriptor for
the historical prepared data. The worker's own argument validator, closure
check, origin verification and import guard execute. The complete authenticated
trainer source runs through `import torch`; a line trace raises at the unique
`UNSLOTH_IMPORT` marker, before `from unsloth ...`. A narrow import-call and
module-finder barrier prevents Unsloth entry if tracing is disabled, even if
the module name was cached.
The worker's closed `phase=TRAINER category=OTHER milestone=TORCH_IMPORT` stop
label, real trainer `__main__` binding and no barrier activation are required
for `PASS`. It records environment digests immediately before Torch import and
after it returns, alongside the separate supplied and interpreter-startup
digests; no environment values are published. Any worker failure reports only
fixed stage/category labels. The
helper retains source, release, image and daemon checks, claim-before-create,
8-KiB Docker CLI output budget, 120-second execution deadline and exactly-once
owned cleanup. It does not call trainer main, load weights or train.

The wrapper and trace/import barrier are diagnostic additions that can affect
timing or a transitive Unsloth import; r2 used direct
`-I` worker script startup. The synthetic dataset content, reduced host
privileges/filesystem and absent model cache remain differences. A PASS or
failure would test this narrower worker-entry hypothesis under those reduced
conditions and could not retroactively identify r2's cause. No container,
Docker start/exec, device request or ML import occurred while preparing r12.

The **unexecuted** proposed single command, requiring root review and a new
one-use admission, is:

```sh
python3 -I scripts/diagnose_docker_torch_import.py \
  --docker /usr/bin/docker \
  --docker-config /home/profsynapse/.local/share/synaptic-qualification/docker-sft-stable-image-c/docker-config \
  --release /home/profsynapse/.local/share/synaptic-qualification/docker-sft-stable-image-c/evidence/runtime-release.json \
  --expected-release-sha256 5f65824ceb5e250a58314da0233b9c07f1d63f1b0a39892d9b36af694938bfe6 \
  --daemon-id 24805003-b125-469b-8080-9da4daacd364 \
  --source-root /home/profsynapse/.local/share/synaptic-qualification/docker-public-plan-0d515-20261020/attempt-r2/source/engine \
  --source-lock /home/profsynapse/.local/share/synaptic-qualification/docker-public-plan-0d515-20261020/attempt-r2/material/source-lock.json \
  --expected-source-lock-sha256 d19aecb2d3ca1b7b6fa32a2445c45251e59d8cac2332293c68b73a61cbff4665 \
  --expected-manifest-sha256 3842689403abc5b42e28d58545b88616b81cd8fc738ee5557e97159df65d00ef \
  --expected-closure-digest 8465ff2b7cedb352337e7cedc5bb17cc169892c2eedae0b280569f4a0cde5002 \
  --worker-context --worker-entry \
  --expected-workload-sha256 1534308c977c2c1cb2b1d80d9cc5c07e31303db63b336686e1c5ccc2644b1064 \
  --expected-runtime-sha256 460121aaee3eb83764f67d9307cfb57a3b25c117611d987577b9631b6956a1b3 \
  --expected-trainer-sha256 75c2ab22fbc5dd80b8ddd5e6d0da816d1451cda0cd620c34717336a39a8a7644 \
  --gpu-device 0 \
  --output-dir /home/profsynapse/.local/share/synaptic-qualification/torch-import-r12-20261020/worker-entry-gpu0-1
```

Before any effect, review the clean pushed checkout and helper SHA-256
`67224904f0124d6defb38003a9da23406848957ec2a5b2935d5c971781485daa`,
trusted verifier SHA-256
`2304eeb00667954510c34cc4aacaa818532a301fbeabf04c2fb165d6cd1d70b5`,
all pins above, source/ancestor private custody without an active writer,
unchanged daemon/image, idle GPU 0, and the exact host settings and code delta.
Create only the fresh parent at mode 0700 and require the named child absent;
collision or drift rejects without another name. Save and read back a specific
one-use admission before start. The extra authority sought is exactly one
fresh bounded GPU-0 import-only diagnostic container, including the synthetic
sealed descriptor and real worker/runpy entry. This preparation grants none
of that live authority; all r2/r4/r6/r8/r11 claims and containers remain
consumed and preserved.

### r13 execution of the admitted worker-entry probe (2026-10-20)

Root subsequently accepted the exact r12 command at clean, pushed
`14f44a3ef389c9c04225b058626bb3cfbba61d45` for **one** fresh diagnostic
container. The operator read back the root admission, then checked the remote
clean checkout and pushed ref, helper and trusted verifier SHA-256 above, exact
release/daemon/image/source-lock/manifest/closure/workload/runtime/trainer
commitments, private source/ancestor custody, and absence of a visible active
source writer. The fresh parent was mode 0700 and the named output child was
absent. Windows and WSL reported GPU 0
`GPU-ac4398bb-fc15-d722-b39c-2790b4b5f9cc` with no active compute process.
The one-use current-preflight/command admission was saved and read back in
Nexus as `Windows r13 worker-entry one-use pre-start admission — 2026-10-20`
before Docker effects. These are point-in-time host observations; a read-only
bind does not freeze the retained host tree.

The command above ran **once**, from the remote clean checkout, with no retry.
The retained output is
`/home/profsynapse/.local/share/synaptic-qualification/torch-import-r12-20261020/worker-entry-gpu0-1`.
It returned closed `PASS` at `TORCH_IMPORT`, boundary `BEFORE_UNSLOTH`, with
`python_isolated=true`, `runpy_main_bound=true`, `training_executed=false`,
and no import barrier activation. The separately supplied environment digest
was `1bb0e06d9708dd6fd637f59eaf134dcbe8165f43e120a0b94daad6c79a63699c`;
the observed startup digest was
`a1eeabbf01741b374ca4e410d62119cf8d882b97d33da962257ec7570bdcb08c`,
with the explicitly admitted `LC_CTYPE_C_UTF8` interpreter addition. The
before- and after-Torch environment digests both were
`712d7eee6b813d815b996f2d3db8266c691dcd5d9f02ea5b9b09073409bd2efc`.
These are reconstructed source/plan context digests, not measurements of r2's
historical child environment. `gpu_used` remains unknown: device exposure is
verified, but this diagnostic does not establish CUDA use.

The single retained container is
`d9f8c6dda19ce32a4e3326762cc3ecf929451ef011162ba595b1dd9b12a8636f`.
The durable event sequence is exactly `CREATE_CLAIMED`, `CREATED`,
`START_CLAIMED`, `STARTED`, `TERMINAL_OBSERVED`; the independent daemon
inspection found it `exited` with code 0. It started at
`2026-10-20T16:27:48.793457215Z` and finished at
`2026-10-20T16:27:54.515077872Z`. Inspection confirmed the expected daemon,
image, claim label and name; device request `0`; 2 CPU, 4 GiB, network none,
read-only root, all capabilities dropped, read-only `/source` bind, and
64-MiB `/tmp` and `/artifacts` tmpfs. The container and all prior diagnostic
containers and claims remain retained. The claim digest (over canonical claim
bytes without its final newline) is
`9f22951917cf0eb35ce4e2084c64046087ff4777ce7339bde8039ec04cee6e50`.
Retained file SHA-256 values are:

| File | SHA-256 |
| --- | --- |
| `claim.json` | `86e6202f20064fb12087e532f26debcef6e4047b5d6dcea8bdf3c81ac4648031` |
| `events.jsonl` | `47dd59c98b9da7c15250859f5ee7ccaf3f110070d5a1e73785754e188373953f` |
| `result.json` | `90d84bf65199e6b4aa3d1fd9f88a4bb7d79677902c085d684987ae921bb3f7e5` |

The wrapper, trace/import barrier, `-I -c` entry, synthetic sealed descriptor,
reduced privilege/filesystem/memory settings, and absent historical model and
dataset bytes remain material differences from r2's direct `-I` worker script
entry. The successful r13 import therefore narrows the observed boundary but
does **not** reproduce r2 or establish the cause of its exit 23. It grants no
additional probe, model, or training authority.

### r15 public SFT model-first preparation (2026-10-20)

The retained r14 public-consumer planning root and its r3 configuration are
historical preparation only. No r14 training was submitted. The r2 worker's
exit 23 remains unexplained; the successful reduced import probes do not
establish a runtime fix. The r14 flow exposed an ordering problem: the
short-lived pushed-source proof was minted before a 9.3-GB model staging step.

The Docker example now accepts a separate, fresh `model_output_root` and an
optional private `model_seed_root`. `prepare-model` consumes its own one-use
claim, copies seed members into a fresh private cache through bounded regular
file reads, and uses the unchanged shared model preparer to verify every reused
member against current pinned Hub Git/LFS metadata. Missing members take the
normal cold download path. Corrupt, redirected or changed seed members reject;
old cache bytes and claims remain untouched. The model receipt records the
declared model/configuration/runtime and a complete bounded destination file
inventory. `prepare` checks that receipt and bytes **before** source
authentication, then compiles a fresh public consumer that binds the receipt.
The existing public `start` composition rechecks the receipt and actual bytes
before Docker effects. This adds no image, package, trainer or runtime change;
the Modal/HF closures and runtime locks remain current without refresh.

For review, the proposed next attempt must use a new pushed host commit with
the exact new engine gitlink, a fresh private plan/config/model/source/artifact
root, and a new execution attempt claim. The prior r14 host/gitlink/config and
every earlier model/source/start claim remain consumed or historical; they
cannot be reinterpreted as fresh authority. The ordered commands are the
checked-in example's `plan`, `model-preflight`, `prepare-model`, `prepare`,
public `launch.py plan`, then one separately admitted public `launch.py start`.
The model stage has no hard SDK download deadline; only the subsequent source
proof uses its 600-second validity window. Before any live step, independently
review exact pushed source and genuine host gitlink, wheel/consumer closure,
release and image C pins, daemon/device, source and model custody, prepared
input publication, fresh names, and the original 3-step full-precision SFT
recipe with 900-second/2-CPU/16-GiB/GPU-0/network-none bounds. Do not run a
model request, container or training job on this implementation evidence alone.

The r16 independent review found that cold known-run operations had inherited
the live model-byte check, an absent seeded revision was rejected instead of
treated as a cold miss, and inventory hashing reopened the root pathname after
acquiring a scan descriptor. The r17 host-only repair retains the exact model
receipt inside the public attempt. Start still validates current model bytes
before its sole create; cold known-run operations check the committed receipt,
configuration and run identity without reading weights or the old preparation
root. A missing seed revision now downloads through the unchanged pinned
preparer, while existing redirects and mutated members reject. Inventory hashes
leaves through their held scan-parent descriptors and compares identities before
and after reading. Tiny CPU public-start and transient-ancestor-swap fixtures
exercise these boundaries; they do not qualify the desktop model or diagnose r2.
Public plan authenticates the receipt commitment without reading weights; the
sole live-byte recheck in the public consumer occurs on start before effects.

## r21 public failure and r22 direct-entry comparison — 2026-10-22

The fresh r4 public start was accepted once and the first public observe was
terminal `failed` / `docker_run_failed`, with zero artifacts. Its retained
container `03c1eed8634f7ae9d95c34dfaf2cce38ba6b95085602f341cd1813e02d529f85`
exited 23. The exact 81-byte worker stderr has SHA-256
`95d34d74eae9cb55a223eeda8428ee4387ee79095e5c0f39a0ceb60da9f45513`
and contains only `OFFLINE_SFT_WORKER_REJECTED phase=TRAINER category=IMPORT
milestone=TORCH_IMPORT`. The create-only private terminal evidence record is
`attempt-r4/inputs/public-terminal-evidence.json`, SHA-256
`5eda7a87855631c2109ae4237be533f6019c3762409841966dddaeb15d503e1d`.
No second start, cancel, artifact download, or container cleanup occurred.

Read-only comparison of the retained r4 container and r13 passing probe found:

| Boundary | r13 PASS | r4 public failure |
| --- | --- | --- |
| Image, GPU, network | Same image C; GPU device 0; network none | Same image C; GPU device 0; network none |
| Authenticated source | Worker `2304eeb0…`, runtime `460121aa…`, trainer `75c2ab22…`, bootstrap `4a8d3dd2…`; manifest `38426894…`, 68-member closure `8465ff2b…` | All six hashes identical; r4 source lock `f126c60f…` binds the new host/run, while r13 uses historical lock `d19aecb2…` |
| Workload and child environment | r2 workload `1534308c…`; planned environment digest `a4e07446…`; reconstructed supplied child digest `1bb0e06d…` | r4 workload `0c28609e…`; same planned environment digest and roots; reconstructed child digest `4454b046…`. The **only differing reconstructed child environment key** is `SYNAPTIC_WORKLOAD_FINGERPRINT`. These are source/plan reconstructions, not retained historical child environments. |
| Isolated entry | `-I -c` child imports a diagnostic wrapper, then `runpy.run_path` enters the worker; tracing and import hooks stop before Unsloth | Locked runtime invokes a separate `-I` process on the worker file directly; the worker then enters trainer through `runpy.run_path`. The actual r4 child argv and `sys.modules` were not retained. |
| Host envelope | UID/GID 1000, capabilities dropped, read-only root, 4 GiB, 64-MiB tmpfs artifacts, PID limit 64 | Docker user `root`, default capabilities, writable root, 16 GiB, writable artifact bind with the verified model cache and prepared data; no PID limit. Both request 2 CPUs. |

The r4 Docker mount still points to its retained source read-only and its
artifact root writable; the configured model-snapshot alias points to the
expected, currently unredirected directory. This point-in-time directory check
does not refresh the earlier 14-file byte inventory. Neither container reports
Docker OOM kill or an explicit ulimit. The worker classifies the caught Python
exception by built-in type identity only: `IMPORT` means `ImportError` or a
subclass, without a module name, message, native loader reason, or traceback.
The same pinned source, guard, planned environment and image make a source or
package-version mismatch a poor explanation for the r13/r4 contrast. The
direct interpreter entry, its prior module cache and loader state, plus the
different user/filesystem context remain testable differences; no current
evidence ranks a particular one as the cause. CPython documents that `-I`
excludes the script/current directory and user site from `sys.path`, while
`runpy` leaves cached imports and other side effects in the process; the
upstream Torch 2.11 source loads Python modules and global shared libraries
during `import torch`. These are mechanisms to test, not a diagnosis:
[CPython command line](https://docs.python.org/3.12/using/cmdline.html#cmdoption-I),
[CPython runpy](https://docs.python.org/3.12/library/runpy.html#runpy.run_path),
[upstream Torch 2.11 source](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/__init__.py).

### One proposed discriminating experiment; no live authority

Extend the **existing** bounded diagnostic helper with a direct-worker mode.
Its outer process may use the authenticated r4 runtime source to build the
exact trainer arguments and one sealed descriptor, as r13 did, but must launch
the isolated child as `/opt/unsloth-venv/bin/python3 -I
/source/engine/tuner/runtime/offline_sft_worker.py -- ...` with the r4-derived
environment. No `-c` wrapper, tracing, or import hook may run in that child.
Use r4's prepared dataset bytes for the sealed descriptor, after a bounded
digest check, and a fresh private writable artifact root containing a newly
verified copy of the pinned model snapshot. Match r4's root user, default
capabilities, writable root, 2 CPUs, 16 GiB, GPU 0 and network none. The
artifact and source mounts must be fresh/private and independently checked;
the consumed r4 root remains untouched.

To make fallthrough impossible, stage a fresh diagnostic derivative of the
already authenticated 68-member source closure. Copy every member byte for
byte except `Trainers/sft/train_sft.py`; retain that file **through** the
unique `UNSLOTH_IMPORT` marker at line 229, then replace the remaining suffix
with the fixed `raise SystemExit(42)\n`. The Torch import and every earlier
trainer byte remain identical. Expected derivative trainer SHA-256 is
`a7a8585e2a4607c7c0b49614260fda2814f29887a3b61e4b8b5df6f71f52639a`,
canonical derivative manifest SHA-256 is
`4b5b0ac2b40b9daf85769d133fe16cfabf680b472d0f4b74c2d5cf65e66ccb4a`,
and derivative closure digest is
`eb1b679c294462603f792cf2c66d8c0080443309b25e5114df654fb88b71ccce`.
Bind **both** the original r4 source lock/manifest/closure and these
independently reviewed derivative values before create; recheck source bytes
and custody immediately before start. The derivative never contains the
Unsloth import, model load, trainer main or training code after the stop.

The helper must accept only fixed worker output. `TRAINER/IMPORT/TORCH_IMPORT`
would reproduce the failure boundary under direct child startup and the closer
r4 context; `TRAINER/SYSTEM_EXIT/UNSLOTH_IMPORT` would establish that Torch
returned and the planned stop ran. Any other label or missing terminal proof
is inconclusive. Retain the existing 8-KiB Docker CLI output cap, one claimed
create/start, 120-second execution deadline, exactly one owned stop request
plus independent terminal inspection, and no retry after uncertainty. Use a
fresh output claim such as `torch-import-direct-r22-20261022/direct-worker-gpu0-1`
only after a separate review of the exact implemented command, helper hash,
source and model receipts, fresh roots, idle GPU and one-use admission. The
current helper does not implement this mode, so **there is no runnable live
command or permission to create a container from this proposal**. Meaningful
CPU tests must prove original-prefix equality, derivative closure validation,
direct `-I` child entry with a non-ML Torch stub, the fixed stop before every
Unsloth/model/trainer statement, source drift rejection before create, and
uncertain claims never replaying.

Even a future `UNSLOTH_IMPORT` stop would only narrow the r4 contrast. The
derivative changes the trainer suffix and a fresh artifact root cannot prove
the r4 process's historical module cache or exact filesystem state. Both r2
and r4 failure causes remain unknown.

### r23 direct-worker implementation for separate root review

The existing bounded helper now has `--worker-context --worker-direct`. It
authenticates the original r4 lock, 68-member closure and workload; stages a
private derivative of that same closure; and separately checks the derivative
trainer, manifest and closure against supplied independent hashes. The direct
child is a fresh `/opt/unsloth-venv/bin/python3 -I
/source/engine/tuner/runtime/offline_sft_worker.py -- ...` process. Its parent
uses the authenticated runtime's existing argument builder and a sealed
descriptor containing r4's 2,821-byte prepared input. The child has no
diagnostic `-c` wrapper, trace function or import hook. The derivative retains
the exact original trainer prefix through the `UNSLOTH_IMPORT` marker, followed
only by `raise SystemExit(42)`; the original source remains untouched. Direct
stdout is discarded, and the outer process reads at most 257 stderr bytes in
memory to match fixed closed labels; no raw child stream is retained.

The r22 proposed derivative manifest SHA-256 omitted the terminal newline
required by the worker's canonical manifest parser. The verifier-valid
manifest SHA-256 is
`c6c90c366f3b4043dc0cd651ae9ba130c05d76c370ac1570fab918dea25c31d1`.
The independently calculated trainer SHA-256
`a7a8585e2a4607c7c0b49614260fda2814f29887a3b61e4b8b5df6f71f52639a`
and closure digest
`eb1b679c294462603f792cf2c66d8c0080443309b25e5114df654fb88b71ccce`
are unchanged. The checked-in trusted verifier validates the derived manifest
and all 68 copied members. The helper rechecks original and derivative source
bytes and custody immediately before its sole start.

The implementation creates a fresh private writable artifact root under its
new output claim. It mounts the retained r4 model snapshot at the same
container cache path through a **separate read-only nested bind**. It also
binds the exact prepared dataset file read-only at the original state path.
The helper checks the pinned model receipt and hashes its full file inventory
through held directory descriptors with a 16-GiB/300-second preflight bound,
both before create and before start; it checks the dataset digest and size.
The r25 repair checks that deadline while consuming each directory entry,
before collecting or sorting names. Across the full scan it admits at most
320 entries and 81,600 name bytes, with a 255-byte limit for each name;
duplicate names reject. The 64-file and 256-directory limits still apply.
It never writes to the old r4 artifact, model, source, claim or container.
This avoids the r22 proposal's unnecessary 9.3-GB model copy. The fresh
artifact contents and read-only model bind remain a filesystem difference
from r4; no historical r4 environment or module cache is claimed captured.

This is the **review candidate only**, with no authority yet to launch. A
clean pushed helper checkout and exact helper/verifier hashes must be reviewed
and checked on WSL before any effect. The helper SHA-256 is
`3153b65b6c9216dd4495d6e09dd6c57fece96e5d6bff395cd8b91a1c1df401a4`;
the trusted verifier remains
`2304eeb00667954510c34cc4aacaa818532a301fbeabf04c2fb165d6cd1d70b5`.
The focused local CPU suite passed 44 tests, including forced excessive-entry
and mid-enumeration deadline rejection before file stat, hashing or Docker.
Targeted Ruff and diff checks passed. The command below is unchanged from r23.
The parent
`/home/profsynapse/.local/share/synaptic-qualification/torch-import-direct-r23-20261022`
must be fresh, private mode 0700, and its `direct-worker-gpu0-1` child absent.
Read-only preflight must confirm the current daemon and pinned image C, idle
GPU 0, retained r4 source/model ownership and bytes, and no concurrent writer.
Collision or drift rejects without choosing another name. The command below
uses only the existing helper and has **not been executed**:

```sh
python3 -I scripts/diagnose_docker_torch_import.py \
  --docker /usr/bin/docker \
  --docker-config /home/profsynapse/.local/share/synaptic-qualification/docker-sft-stable-image-c/docker-config \
  --release /home/profsynapse/.local/share/synaptic-qualification/docker-sft-stable-image-c/evidence/runtime-release.json \
  --expected-release-sha256 5f65824ceb5e250a58314da0233b9c07f1d63f1b0a39892d9b36af694938bfe6 \
  --daemon-id 24805003-b125-469b-8080-9da4daacd364 \
  --source-root /home/profsynapse/.local/share/synaptic-qualification/docker-public-plan-4a8b-20261020/attempt-r4/source/engine \
  --source-lock /home/profsynapse/.local/share/synaptic-qualification/docker-public-plan-4a8b-20261020/attempt-r4/material/source-lock.json \
  --expected-source-lock-sha256 f126c60f3ed9f6131c6de6dba7ac032016a1a538db57ccd5499783008e8466b6 \
  --expected-manifest-sha256 3842689403abc5b42e28d58545b88616b81cd8fc738ee5557e97159df65d00ef \
  --expected-closure-digest 8465ff2b7cedb352337e7cedc5bb17cc169892c2eedae0b280569f4a0cde5002 \
  --worker-context --worker-direct --gpu-device 0 \
  --expected-workload-sha256 0c28609e2a73d2c649f5d8f6f963505dbc6e86a7bbee6b9a25b3477e9dfbd2d8 \
  --expected-runtime-sha256 460121aaee3eb83764f67d9307cfb57a3b25c117611d987577b9631b6956a1b3 \
  --expected-trainer-sha256 75c2ab22fbc5dd80b8ddd5e6d0da816d1451cda0cd620c34717336a39a8a7644 \
  --expected-direct-trainer-sha256 a7a8585e2a4607c7c0b49614260fda2814f29887a3b61e4b8b5df6f71f52639a \
  --expected-direct-manifest-sha256 c6c90c366f3b4043dc0cd651ae9ba130c05d76c370ac1570fab918dea25c31d1 \
  --expected-direct-closure-digest eb1b679c294462603f792cf2c66d8c0080443309b25e5114df654fb88b71ccce \
  --artifact-root /home/profsynapse/.local/share/synaptic-qualification/docker-public-plan-4a8b-20261020/attempt-r4/artifacts \
  --model-receipt /home/profsynapse/.local/share/synaptic-qualification/docker-public-plan-4a8b-20261020/attempt-r4/model/model-preparation.json \
  --expected-model-receipt-sha256 e1d2deff0a2a54ff2a5fdd24bdb8b22d55c1ed2d13b2c79307124db4b34b0ea6 \
  --output-dir /home/profsynapse/.local/share/synaptic-qualification/torch-import-direct-r23-20261022/direct-worker-gpu0-1
```

The container envelope requests root, default capabilities, writable root,
2 CPUs, 16 GiB, GPU device 0, network `none`, and the same image C. Existing
helper ownership, 8-KiB Docker CLI output budget, one create/start claim,
120-second execution deadline, one owned stop and independent terminal
inspection remain in force. The only accepted direct child outcomes are its
closed `TRAINER/IMPORT/TORCH_IMPORT` label or the planned
`TRAINER/SYSTEM_EXIT/UNSLOTH_IMPORT` stop; anything else is inconclusive.
Neither result identifies r2 or r4's original import cause by itself. There
has been no r23 container, GPU, ML import or training effect.
