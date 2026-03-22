# Cloud Training Reference

Cloud training uses the existing SFT and KTO trainers plus the env-backed GRPO path, but persistence and code sync behave differently from local runs.

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

## HF Jobs Hardware Lookup

Use the checked-in helper when you want the live HF Jobs hardware list and hourly pricing instead of guessing from memory:

```bash
python3 scripts/hf_jobs_hardware.py
python3 scripts/hf_jobs_hardware.py --job-config Trainers/cloud/jobs/nexus_quark_l25_28_env_grpo.yaml
python3 scripts/hf_jobs_hardware.py --sort-by vram --min-vram 40
```

Notes:
- the script queries `GET https://huggingface.co/api/jobs/hardware`
- it uses `HF_TOKEN` or `HF_API_KEY` automatically when present
- `--job-config` highlights the current flavor from a cloud job YAML
- the script now uses a CA bundle automatically, so it should not need ad hoc SSL env vars on normal local setups

## Blind Hardware Planning

Use the planner when you want a back-of-the-envelope stage recommendation without relying on prior run telemetry:

```bash
python tuner.py plan-hardware \
  --experiment-spec Trainers/cloud/experiments/qwen3_4b_full_cycle_full_v2.yaml \
  --optimize-for balanced
```

Current planner inputs:
- model name / parameter scale inferred from the spec
- method (`sft`, `kto`, `grpo`)
- seq length
- 4-bit loading flag
- target batch / effective batch
- live HF Jobs hardware flavors and hourly pricing

Current planner outputs:
- recommended training / evaluation / loss flavor
- recommended training microbatch and gradient accumulation when the spec leaves them unset
- estimated memory footprint and headroom
- relative speed-vs-cost ranking

Use it automatically at launch:

```bash
python tuner.py run-experiment \
  --experiment-spec Trainers/cloud/experiments/qwen3_4b_full_cycle_full_v2.yaml \
  --auto-hardware \
  --optimize-for cost \
  --yes
```

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
- runs `Evaluator.cli --backend unsloth ...` directly in the HF job using the downloaded adapter
- syncs evaluation outputs back into the same bucket under:
  `runs/hf_jobs/{method}/{run_slug}/evaluations/vllm/{timestamp}/`

Saved files to inspect:
- `evaluation_results.json` - canonical machine-readable summary and all records
- `evaluation_results.md` - human-readable report
- `evaluation_lineage.json` - provenance / model-card material
- `logs/eval_progress.jsonl` - incremental progress events used for the local cloud dashboard

Inspection workflow:
1. Find the source training run under `runs/hf_jobs/{method}/{run_slug}/`
2. Open the newest directory under `evaluations/vllm/`
3. Read `evaluation_results.json` first
4. Use `evaluation_results.md` when you want a concise human summary
5. Use `evaluation_lineage.json` if the question is about reproducibility or upload metadata
6. Use `logs/eval_progress.jsonl` only when debugging in-flight or partially failed runs
7. For local inspection from the CLI, use:

```bash
python tuner.py cloud-inspect --run latest --eval-run latest --method sft
```

Interpreting saved failures:
- Do not jump from a failed case count straight to a training conclusion.
- First separate infrastructure or evaluator noise from actual model behavior failures.
- Prefer the structured record fields over raw response text when both are available.
- Classify failures by mechanism:
  wrong action selected relative to the scenario
  response type mismatch
  malformed structured output or parse failure
  missing required fields
  behavior-expectation mismatch
- The useful question is: what did the model do instead of what the evaluation expected?
- Keep this analysis generic. The same method should work across different prompt formats, toolsets, and custom evaluation configs.

Useful flags:
- `--method sft` or `--method kto` to filter run discovery
- `--scenario behavior_prompts.yaml` to run specific scenarios instead of a preset
- `--tags storageManager,intellectual_humility` to filter cases
- `--upload-to-hf username/model-name --update-model-card` to push evaluation lineage to a model repo

Current constraint:
- the LoRA adapter's `base_model_name_or_path` must point to a hub-accessible model, not a local filesystem path

Anti-patterns:
- Do not assume the Unsloth training image is also a stable vLLM-serving runtime. vLLM, Transformers, tokenizers, Triton, and CUDA can drift independently.
- Do not install a newer `huggingface_hub` into the main Unsloth eval interpreter just to satisfy bucket sync. Keep bucket sync in the helper subprocess path.
- Do not trust preset names blindly. The `eval_run.yaml` preset filenames must match the actual files under `Evaluator/config/scenarios/`.

If you want one command for the common path, use:

```bash
python tuner.py cloud-pipeline --method sft --preset full
```

That trains on HF Jobs first, then launches cloud evaluation against the exact finished run. It is the preferred UX for train-followed-by-eval.

---

## Recovery and Cleanup

- Resume-from-provider-native-storage is not automatic yet.
- Persistence is guaranteed first; resume flows can be added later.
- Clean up old runs from the provider-native backend explicitly when they are no longer needed.
