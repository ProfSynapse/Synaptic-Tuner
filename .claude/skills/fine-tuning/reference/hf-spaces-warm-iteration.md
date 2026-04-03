# HF Spaces Warm Iteration

Use this reference when HF Jobs cold starts are slowing down repeated eval or runtime-debug cycles.

## When to Prefer a GPU Space

Use a Docker GPU Space instead of HF Jobs when you need:
- a warm runtime that keeps downloaded model weights and package caches
- repeated eval/debug iterations against roughly the same package set
- a box you can attach to with Dev Mode or SSH while investigating runtime issues

Do not treat HF Jobs as a warm pool. Jobs are ephemeral and VM reuse is opportunistic at best.

## Recommended Pattern

For repeated eval bring-up:
1. Keep training on HF Jobs.
2. Create a dedicated Docker GPU Space for eval/debug.
3. Add persistent storage.
4. Bake the heavy runtime into the Space image instead of pip-installing it on every run.
5. Cache Hub and pip assets onto persistent storage.
6. Pause or downgrade the Space when you are done.

## Space Settings

- SDK: `Docker`
- Hardware: pick the smallest GPU that can serve the target eval path
- Persistent storage: enable it from the Space Settings tab
- Dev Mode: enable it when you want SSH / VS Code iteration without rebuild cycles

## Cache Layout

Set these environment variables in the Space:

```bash
HF_HOME=/data/.huggingface
PIP_CACHE_DIR=/data/.cache/pip
HF_HUB_ENABLE_HF_TRANSFER=1
```

Why:
- `HF_HOME=/data/.huggingface` keeps model and tokenizer downloads across restarts
- `PIP_CACHE_DIR=/data/.cache/pip` avoids redownloading wheels during image-side debug installs

## Docker Space Shape

The fastest warm iteration loop is:
- base image already contains `vllm`, `transformers`, `peft`, and any parser/runtime deps
- Space start command launches the inference server directly
- repeated tests hit the already-running Space instead of scheduling a new HF Job

For model-family bring-up:
- keep one Space per runtime family, not one giant catch-all image
- examples:
  - `gemma4-vllm-warm`
  - `qwen35-vllm-warm`

## Operational Guidance

- Use HF Jobs for final recorded train/eval artifacts.
- Use GPU Spaces for rapid iteration on runtime compatibility, server flags, and cache-heavy eval loops.
- If the Space is idle, pause it or set an aggressive sleep policy to avoid paying for unused GPU time.
- If you need to resume quickly later, keep persistent storage attached so the caches survive.

## Repo Practice

When a model family needs repeated runtime debugging:
- add the Space workflow to the research/ops notes
- keep the Dockerfile checked into the repo
- keep the image/runtime pinned
- use the Space for quick retries until the runtime is stable enough to move back into HF Jobs
