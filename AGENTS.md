# Project Agent Notes

This repository has a few cloud-training constraints that are easy to relearn the hard way.

## HF Jobs

- Remote jobs clone and run the exact pushed commit. If the job log shows an older `HEAD`, stop and relaunch from the right SHA instead of debugging stale code.
- Do not upgrade `huggingface_hub` in the main Unsloth training environment just to get Buckets support. `transformers` in the training stack requires `huggingface-hub<1.0`.
- If Buckets support needs a newer Hub client, isolate it in a helper path or subprocess and keep the trainer runtime untouched.
- Pass `HF_TOKEN` into `huggingface_hub.run_job(...)` explicitly with job secrets. Do not assume the cloud job inherits the local shell environment.
- Treat blank `HF_TOKEN` / `HF_API_KEY` values as unset. Empty strings can produce `Authorization: Bearer ` and fail in `httpx` before any request reaches Hugging Face.
- Resolve or create the bucket once up front, then sync against the resolved namespaced bucket ID during the run.
- Avoid repeated bucket creation and `whoami-v2` calls during periodic sync. Cache bucket resolution and keep dashboard polling conservative.
- Use `python tuner.py cloud-eval --run latest --preset full` for remote vLLM evaluation of bucketed HF Jobs runs; it launches a second HF Job and stores outputs under the source run's `evaluations/vllm/` prefix.

## Cloud Artifact UX

- HF Jobs local dashboard parity comes from syncing JSONL training logs to the bucket and replaying them locally.
- Modal may stream usable remote stdout directly; verify that before adding a separate local watcher.
- RunPod currently needs more explicit metric/log plumbing if local dashboard parity is required.
