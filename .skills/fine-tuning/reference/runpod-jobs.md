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
- The `runpod` SDK installed where you launch from (`pip install runpod`).
  Gotcha: importing the SDK crashes if the current working directory is on a
  dead/unstattable mount (its state dir uses a cwd-relative path) -- launch
  from a healthy directory.

---

## Usage

```bash
python3 scripts/runpod_run_job.py \
  --run-tag my-job \
  --repo-url https://github.com/org/repo.git \
  --commit <full-40-char-sha> \
  --wrapper path/inside/repo/job.sh \
  --wrapper-args "--stage extract --surface holdout" \
  --gpu "NVIDIA GeForce RTX 4090" \
  --timeout-min 180
```

Always validate with `--dry-run` first: it prints the resolved pod spec
(image digest, GPU, disk, env keys, full `docker_args` chain) without any API
call or cost.

Key flags:

- `--image` -- container image, pinned by digest. Default is the same Unsloth
  image digest the RunPod training backend pins, so a job is byte-identical
  across the training and job lanes.
- `--gpu` -- RunPod `gpu_type_id` (default `NVIDIA GeForce RTX 4090`).
- `--cloud-type` -- `COMMUNITY` (default, cheaper) or `SECURE`.
- `--env KEY=VALUE` -- extra pod env vars, repeatable.
- `--timeout-min` -- hard wall-clock cap; the pod is terminated at timeout.

---

## Lifecycle and Billing Safety

The script owns the full pod lifecycle: `create_pod` -> wait for RUNNING ->
poll every 30s (uptime, GPU/VRAM utilization) -> exit on pod EXITED -> and
ALWAYS `terminate_pod` in a `finally` block, with 3 retries. A wedged or
timed-out run cannot silently keep billing. If termination fails after
retries, the script prints the console URL
(https://www.runpod.io/console/pods) for a manual kill -- verify.

Exit codes: 0 = wrapper finished (pod EXITED), 1 = pod failed / boot timeout
(600s) / wall-clock timeout, 2 = bad arguments or missing credentials.

## Outputs

There is no network volume. The wrapper is responsible for uploading its own
outputs (e.g. to a HuggingFace repo) before it exits; when the pod reports
EXITED the outputs are already wherever the wrapper put them. Design wrappers
so a preempted or re-run pod is safe (idempotent uploads, resumable work).
