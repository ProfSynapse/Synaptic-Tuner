# Cloud Training Reference

Cloud training uses the existing SFT and KTO trainers, but persistence and code sync behave differently from local runs.

---

## Exact Source Requirements

Cloud jobs run from the exact git revision you launch:
- tracked worktree must be clean
- current branch must be named
- `HEAD` must already be pushed to `origin/<branch>`

If any of those checks fail, the cloud backend stops before submitting a job.

---

## Provider-Native Storage

Cloud artifacts are durable by default in the provider ecosystem:

| Provider | Default Artifact Backend | Durable Location |
|----------|--------------------------|------------------|
| `hf_jobs` | `hf_bucket` | Hugging Face Bucket |
| `modal` | `modal_volume` | Modal Volume |
| `runpod` | `runpod_network_volume` | RunPod Network Volume |

Remote container filesystems are not treated as durable storage.

---

## Canonical Cloud Run Layout

Every cloud run writes the same logical tree:

```text
runs/{provider}/{method}/{timestamp}-{shortsha}/
├── checkpoints/
├── logs/
├── final_model/
├── training_lineage.json
└── manifest.json
```

`manifest.json` is the quickest way to confirm the artifact backend, commit, and publish settings for a run.

---

## Optional Final-Model Publish

Publishing to Hugging Face Hub is optional and disabled by default.

When enabled:
- only `final_model/` is uploaded
- checkpoints, logs, manifests, and lineage stay in provider-native storage
- the publish target is a Hugging Face model repo

---

## Smoke-Test Workflow

1. Confirm the branch is clean and pushed.
2. Point the trainer config at a remote dataset when possible.
3. Run `python tuner.py cloud`.
4. Choose provider and method.
5. Start with a short smoke test (`max_steps`, small dataset slice, or one epoch).
6. Verify artifacts in provider-native storage before enabling final-model publish.

Recommended first-pass checks:
- `hf_jobs`: inspect the configured bucket prefix under `runs/hf_jobs/...`
- `modal`: inspect the configured Modal Volume path
- `runpod`: inspect the mounted RunPod Network Volume path

---

## HF Jobs Bucket and Auth Gotchas

For `hf_jobs`, a few patterns matter enough to treat as hard rules:

- The job runs from the exact pushed commit. If the remote logs show an older `HEAD`, you launched the wrong SHA and are debugging stale code.
- Keep the main training interpreter compatible with Unsloth and Transformers. If bucket sync needs a newer `huggingface_hub`, install it in an isolated helper path or subprocess, not into the training runtime.
- Pass `HF_TOKEN` into `run_job(...)` explicitly via job secrets. Do not assume the container automatically receives your local token.
- Normalize blank auth values to `None`. An empty `HF_TOKEN` or `HF_API_KEY` can produce `Authorization: Bearer ` and fail before the request is sent.
- Resolve and, if needed, create the bucket once before training starts. During steady-state log sync, use the resolved bucket ID directly.
- Polling and identity checks should be conservative. Frequent bucket creation attempts or repeated `whoami-v2` calls can hit Hugging Face rate limits.

If the training process itself is healthy but uploads fail, inspect bucket auth and sync isolation before touching trainer code.

---

## HF Jobs Cloud Evaluation

You can evaluate a bucketed HF Jobs run on remote GPU without converting to GGUF:

```bash
python tuner.py cloud-eval --run latest --preset full
```

What it does:
- resolves the configured HF bucket and picks the requested run (`latest` works)
- submits a new HF Job on GPU
- downloads the run's `final_model/` LoRA adapter from the bucket
- starts `vLLM` remotely with the adapter loaded on top of the base model
- runs `Evaluator.cli --backend vllm ...`
- syncs evaluation outputs back into the same bucket under:
  `runs/hf_jobs/{method}/{run_slug}/evaluations/vllm/{timestamp}/`

Useful flags:
- `--method sft` or `--method kto` to filter run discovery
- `--scenario behavior_prompts.yaml` to run specific scenarios instead of a preset
- `--tags storageManager,intellectual_humility` to filter cases
- `--upload-to-hf username/model-name --update-model-card` to push evaluation lineage to a model repo

Current constraint:
- the LoRA adapter's `base_model_name_or_path` must point to a hub-accessible model, not a local filesystem path

---

## Recovery and Cleanup

- Resume-from-provider-native-storage is not automatic yet.
- Persistence is guaranteed first; resume flows can be added later.
- Clean up old runs from the provider-native backend explicitly when they are no longer needed.
