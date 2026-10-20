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

**Proposal for root review; do not execute under the r6 authorization.** A
future reviewed run would require a clean checkout of the new helper commit,
fresh read-only daemon/image/source/custody checks, a new private mode-0700
`/home/profsynapse/.local/share/synaptic-qualification/torch-import-r7-20261020`
parent, an absent `guarded-gpu0-1` child, and a saved/read-back one-use
admission. Its exact command is:

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
