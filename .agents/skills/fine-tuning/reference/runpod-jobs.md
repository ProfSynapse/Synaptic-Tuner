# RunPod One-Shot Jobs Reference

RunPod is the alternative cloud lane for one-shot GPU jobs: clone a git repo at
a pinned commit, run a wrapper script inside a pinned container image on a
single-GPU pod, then terminate the pod. Use it when HF Jobs nodes are flaky
(degraded downloads, zombie containers) or when a job needs a GPU class HF
Jobs does not offer.

This lane is for arbitrary wrapper jobs. Cloud TRAINING stays on the
`tuner.py cloud-run` / `cloud-pipeline` path and its RunPod backend
(`tuner/backends/training/cloud/runpod_backend.py`); see
`reference/cloud-training.md`.

The launcher (`scripts/runpod_run_job.py`) talks to the RunPod REST API
(`https://rest.runpod.io/v1`) for pod create/terminate and to the GraphQL API
for runtime polling. It does NOT use the `runpod` Python SDK. This split was
forced by hard debugging (2026-07-05, 4 dead pods); see the gotchas below.

---

## Requirements

- `RUNPOD_API_KEY` in the environment (required; the script exits 2 without it).
- `HF_TOKEN` in the environment if the wrapper uploads to or downloads from
  HuggingFace (forwarded to the pod when set; never embedded in the startup
  command, so it does not appear in pod logs).
- The target repo must be clonable from the pod (public, or handle auth inside
  the wrapper).
- `--commit` must be a FULL 40-char sha (derive with `git rev-parse`, never
  hand-expand a short sha; the script enforces the length).
- No SDK install needed: the launcher uses only the Python standard library
  (`urllib`), so it runs from any environment with `RUNPOD_API_KEY` set.

---

## Usage

```bash
python3 scripts/runpod_run_job.py \
  --run-tag my-job \
  --repo-url https://github.com/org/repo.git \
  --commit <full-40-char-sha> \
  --wrapper path/inside/repo/job.sh \
  --wrapper-args "--stage extract --surface holdout" \
  --gpu "NVIDIA GeForce RTX 3090" \
  --timeout-min 180
```

Always validate with `--dry-run` first: it prints the resolved pod spec as JSON
(image digest, GPU, disk, env keys redacted, the full `dockerEntrypoint` chain)
without any API call or cost.

Key flags:

- `--image` -- container image, pinned by digest. Default is the same Unsloth
  image digest the RunPod training backend pins, so a job is byte-identical
  across the training and job lanes.
- `--gpu` -- RunPod `gpu_type_id` (default `NVIDIA GeForce RTX 4090`).
- `--cloud-type` -- `COMMUNITY` (default, cheaper) or `SECURE`.
- `--env KEY=VALUE` -- extra pod env vars, repeatable.
- `--timeout-min` -- hard wall-clock cap; the pod is terminated at timeout.
- `--min-download` -- minimum host download speed (Mbps, default 700). Community
  hosts below this stall for tens of minutes pulling a large image; the boot
  timeout then recycles onto a better host (see `--max-hosts`).
- `--allowed-cuda "12.8,12.9,13.0"` -- CUDA versions the host driver must offer;
  a host whose driver is older than the image's CUDA fails container start
  silently.
- `--max-hosts N` -- pods to try before giving up. A community host that never
  starts the container within the boot timeout (stalled image pull) is
  terminated and a fresh pod is created, up to N attempts (~$0.05/boot-fail at
  3090 rates). Anything after boot (job error, timeout) is final, not retried.
- `--probe-only` -- skip clone+wrapper; boot the image, print `nvidia-smi`, and
  exit as soon as real container uptime is observed. Use this to isolate
  boot/image reliability from wrapper logic before spending on a real run.
- `--no-entrypoint-override` -- do not set `dockerEntrypoint`; let the image's
  own ENTRYPOINT/CMD run (diagnostic: isolates whether the override itself
  prevents container start).

---

## Lifecycle and Billing Safety

The script owns the full pod lifecycle: REST `POST /pods` -> wait for REAL
container uptime -> poll every 30s -> exit on pod EXITED -> and ALWAYS
`DELETE /pods/{id}` in a `finally` block, with 3 retries. A wedged or timed-out
run cannot silently keep billing. If termination fails after retries, the
script prints the console URL (https://www.runpod.io/console/pods) for a manual
kill -- verify.

Exit codes: 0 = wrapper finished (pod EXITED), 1 = pod failed / boot timeout /
wall-clock timeout / all `--max-hosts` boot attempts exhausted, 2 = bad
arguments or missing credentials.

---

## Gotchas (all adjudicated 2026-07-05 by local docker repro on the unsloth image)

### 1. Non-root image + `cd /root` crash-loop (the r5/edrt9x saga)

The wrapper command runs as the image's default container user, which is NOT
necessarily root: the unsloth image runs as `unsloth:runtimeusers` with
`HOME=/home/unsloth` and no write access to `/root`. A `cd /root` there fails
"Permission denied", and under `set -e` that aborts in <1s, so RunPod
restart-loops the container with uptime pinned at 1-2s -- indistinguishable from
a boot failure over the API. The launcher uses an image-agnostic writable
workdir instead:

```
{ cd ${WORKDIR:-/workspace} 2>/dev/null || cd "$HOME"; }
```

The braces matter: a bare `cd A || cd B && clone` parses as
`cd A || (cd B && clone)` and SKIPS the clone whenever `cd A` succeeds.

### 2. dockerEntrypoint override (why REST, not the SDK)

Images that ship an ENTRYPOINT (unsloth's `/usr/local/bin/entrypoint.sh`)
receive GraphQL `dockerArgs` as *arguments to the entrypoint*, which can
crash-loop the container forever with uptime pinned at 0 while it bills as
RUNNING. The REST `dockerEntrypoint` field REPLACES the ENTRYPOINT so the job
command actually runs. The SDK also embeds `docker_args` into its GraphQL
mutation with zero escaping, so any double quote corrupts the API call. REST
takes a JSON body: no quoting restrictions.

### 3. REST has no uptime; poll GraphQL

The REST pod object carries no uptime field (only `desiredStatus`,
`lastStartedAt`), so an uptime check against REST reads every pod as
never-booted forever (six healthy-or-not pods were killed at the boot timeout by
exactly that bug). Actual container runtime lives only in the GraphQL
`pod.runtime.uptimeInSeconds`. GraphQL auth is the `?api_key=` query param (a
Bearer header alone 403s), and `api.runpod.io` 403s the default urllib
User-Agent (Cloudflare), so the poller sends an explicit UA.

### 4. Fail-hold: "ran and failed" vs "never booted"

RunPod has no logs API (`runpod-python#400`). On nonzero exit the job command
prints the code and holds the container alive for `FAIL_HOLD_S` (300s) before
exiting nonzero, so the uptime poller can tell "ran and failed" (uptime climbs
to ~300s then EXITED) apart from a boot crash-loop (uptime stuck at 1-2s).
Because there is no log API, a wrapper that wants diagnostics from a failed run
must upload them itself before it exits -- see "Failure telemetry" below.

---

## Outputs and the Artifact Contract

There is no network volume. The wrapper is responsible for uploading its own
outputs before it exits; when the pod reports EXITED the outputs are already
wherever the wrapper put them. Design wrappers so a preempted or re-run pod is
safe (idempotent uploads, resumable work).

Every cloud job (RunPod or Modal) uploads its results + manifest + logs to an
HF staging repo under a provider-tagged path, with the producing repo commit
recorded:

```
runs/runpod/<tag>/     # RunPod wrapper jobs
runs/modal/<tag>/      # Modal training runs
```

- **HF Hub is the durable system of record.** Provider-native storage (Modal
  Volumes, a RunPod pod's local disk) is scratch / checkpoint space only; it is
  ephemeral and must never be the sole copy of a result.
- **The producing repo commit is recorded** in the manifest so any artifact
  traces back to the exact code that made it. For the training scripts this is
  the `repo_commit` field in `manifest.json`; for RunPod wrapper jobs, pass
  `--commit <full-sha>` and stamp the same sha into whatever the wrapper writes.
- **Modal real runs must pass `--publish-final-model`** (and
  `--publish-target-repo`) so the final model lands on the Hub, not just the
  Modal output Volume. The code default is off (a bare `modal run` should not
  silently publish); a real run always opts in.

### Failure telemetry

Because RunPod has no log API, a failed wrapper leaves zero diagnostics unless
it uploads them itself. The recommended pattern (see the Epistemic-Humility AL
wrapper for a worked example): capture the wrapper's own stdout+stderr to a file
from the first line (`exec > >(tee logfile) 2>&1`), and on ANY nonzero exit
(bash `trap ... EXIT`) upload a failure marker plus the tail of that log to
`<run_tag>/_failure/` in the staging repo. Make it best-effort (telemetry
failure must not mask the original exit code) and redact `hf_`/`rpa_` secret
patterns before upload.
