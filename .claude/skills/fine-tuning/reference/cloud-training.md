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

## Protected HF Source Provisioning and Fixed Bootstrap Smoke

This is a narrow source/bootstrap provider-proof lane, not the normal
`cloud-run`, `cloud-pipeline`, or `run-experiment` training path. The local
`hf-source` and `hf-smoke` commands exist. The original JP-S audit returned
REVISE for process-local cancellation, insufficient `JobInfo` identity
agreement, and an incomplete launcher/runtime contract. Bounded remediations are
implemented. JP-S2 passed the frozen 29-file manifest
`55e2c876dd8cc282a43248a3eeaf3f445f6e452ce76ab2d7a0b814b460ef0f41`
with 283 passed/6 skipped, 16 hostile checks, 87 import/generic checks, and no
findings. Final JP-R also PASSed the frozen release/inventory tree. Neither audit
made a live installation or provider claim. Live use still requires clean exact-pushed proof,
the five-pin clean-venv gate, and credential preflight. `cloud.launch` and generic
training remain unavailable.

### Fixed authorization envelope

- At most one paid submission; no retry or replacement without new user approval.
- `cpu-basic`, `python:3.12`, bootstrap verification only.
- No training, publication, ports, SSH, or provider retry.
- Provider timeout 600 seconds; one cancellation attempt after 720 seconds if
  still nonterminal; stop observing after 900 seconds.
- Projected compute no more than USD $0.01; hard total cap USD $2.
- Canonical workload SHA-256:
  `0d1d3454d079ea994a1e3a24b59b772bd4adb40cb441e00cc5801faf5d220841`.

Any wider workload, second submission, higher cost, longer duration, or retry
needs new approval. `SUBMITTING` consumes the authorization even if the provider
response is ambiguous.

### Isolated launcher only

Use a dedicated Python 3.12 environment with exactly these direct pins:
`huggingface_hub==1.27.0`, `jsonschema==4.23.0`, `packaging==24.1`,
`python-dotenv==1.0.1`, and `PyYAML==6.0.2`. Never install this JP dependency set
into the Unsloth/trainer runtime.

```powershell
python scripts/setup_hf_jp_launcher.py `
  --python C:\path\to\python3.12.exe `
  --venv C:\path\to\new\hf-jp-launcher `
  --repo-root <exact-repository-worktree>
```

The setup script refuses a non-3.12 interpreter, missing/extra/duplicate/ranged
or reordered requirements, or an existing target directory. After install it
checks all five distribution versions in isolated Python, imports both protected
handlers without Torch/Transformers/Unsloth, and runs `hf-source --help` and
`hf-smoke --help` from the exact repository worktree with user-site and bytecode
writes disabled. This is a credential-free/provider-free clean-venv gate, not
live provider proof. `python:3.12` is a mutable provider image tag; even a
successful smoke is not digest-pinned image provenance.

### Operator sequence

Only after clean exact-pushed proof, the five-pin clean-venv
gate, and the explicit credential preflight:

```powershell
python tuner.py hf-source `
  --project-root <host-project-root> `
  --experiment-id <experiment-id> `
  --actor <non-secret-operator-id> `
  --authority operator `
  --env-file <explicit-project-env-file> `
  --json
```

`hf-source` requires exact PREPARED tracking state, provisions or verifies only
the descriptor-bound private Profile-C prefix, persists bounded evidence, and
then validates CONSUMABLE. It never submits a job. The explicit env file must be
a regular link-free file inside the project/config boundary and contain exactly
a nonblank `HF_TOKEN`. File/ambient `HF_API_KEY` and ambient `HF_TOKEN` are
rejected. The protected handler parses the file without mutating the environment
or emitting the value.

Create the provider-free exact approval:

```powershell
python tuner.py hf-smoke approve `
  --project-root <host-project-root> `
  --experiment-id <experiment-id> `
  --authorization-reference <recorded-reference> `
  --issued-at <UTC-RFC3339> `
  --expires-at <UTC-RFC3339> `
  --quoted-at <UTC-RFC3339> `
  --hourly-price-usd <official-cpu-basic-hourly-price> `
  --projected-cost-usd <no-more-than-0.01> `
  --json
```

Then, for the one authorized submission only:

```powershell
python tuner.py hf-smoke execute `
  --project-root <host-project-root> `
  --experiment-id <experiment-id> `
  --env-file <explicit-project-env-file> `
  --json

python tuner.py hf-smoke observe `
  --project-root <host-project-root> `
  --experiment-id <experiment-id> `
  --env-file <explicit-project-env-file> `
  --json
```

`execute` revalidates every approval/source/provisioning/workload binding and
atomically writes `SUBMITTING` before credential or provider work. Success records
the normalized provider job identity as `SUBMITTED`; an exception after the call
boundary records terminal `AMBIGUOUS`. Never rerun `execute` after either outcome.
`observe` accepts status/results only when every returned `JobInfo` normalizes to
the exact recorded namespace/job ID. At the cancellation boundary it durably
claims one `synaptic-hf-cancellation-attempt/v1` event. Only the first locked
claimant may call `cancel_job`; resumed/concurrent observers cannot issue another
provider attempt, and an ambiguous cancellation remains consumed.

### Official Hub 1.27 compatibility contract

The isolated adapter probes the complete v1.27.0 method parameter names, order,
kinds, and defaults for `create_bucket`, `bucket_info`, `list_bucket_tree`,
`batch_bucket_files`, and `download_bucket_files` before its first provider read
or mutation. It uses `Volume(type="bucket", source=..., mount_path=...,
path=..., read_only=True)` and the explicit `run_job(..., volumes=...)` surface
for the fixed smoke. These are official-source expectations and hermetic
compatibility checks, not live account/bucket/job proof.

`batch_bucket_files` is officially non-transactional. A failure may leave only
some files uploaded. The operator therefore authenticates immutable byte values
before the one upload attempt, rejects unknown/duplicate/colliding remote entries,
downloads and hashes every exact member, and returns non-retryable
`mutation_ambiguous` for any create/upload/readback uncertainty. Do not retry;
inspect the exact prefix read-only and obtain new user authority for any later
action.

### Remaining gates

- Original JP-S verdict: REVISE for cancellation durability, JobInfo agreement,
  and launcher/runtime completeness.
- JP-S2 security re-audit: PASS on the frozen manifest above; local security
  evidence only, with no live install/provider proof.
- Final JP-R release/inventory re-audit: **PASS**. Evidence: affected **200 passed, 3 skipped**; full JP/HF/import/order **393 passed, 15 skipped**; tracking **249 passed, 1 skipped**; CLI/capability/project/plugin/contract **268 passed, 1 skipped**; broad cloud **673 passed, 15 skipped** with the same five accepted failures and two documented exclusions; **26 Python files compiled**; imports clean; diff check warnings only; public API and `cloud.launch` unchanged; skill sync clean.
- Exact commit push/upstream proof: pending.
- Explicit usable `HF_TOKEN` preflight: pending; no value may enter logs/docs.
- Live provider proof: pending.
- RunPod remains later and requires fresh dated official API/SDK research before
  implementation because its API may have changed.

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

For HF Jobs specifically, bucket-backed artifacts are the primary source of truth once they start appearing:
- training: `logs/training_latest.jsonl`, `logs/stage_summary.json`, `training_lineage.json`
- evaluation: `evaluation_results.json`, `evaluation_results.md`, `evaluation_lineage.json`, then `logs/eval_progress.jsonl`
- loss: `loss_lineage.json`, `loss_summary.json`, `per_example_losses.jsonl`, `high_loss_examples.jsonl`

Use the repo CLI for quick bucket reads/lists:
```bash
python tuner.py bucket analyze --path runs/hf_jobs/sft/<run-prefix>/
python tuner.py bucket read --path runs/hf_jobs/sft/<run-prefix>/logs/training_latest.jsonl --jsonl-latest --pretty
python tuner.py bucket list --path runs/hf_jobs/sft/<run-prefix>/ --limit 20
python tuner.py bucket pull --path runs/hf_jobs/sft/<run-prefix>/analysis/loss/ --dest .
python tuner.py bucket push --path local/notes.json --dest runs/manual_uploads/
```

When a training run has multiple eval reruns or alternate loss benchmarks:
```bash
python tuner.py bucket analyze \
  --path runs/hf_jobs/sft/<run-prefix>/ \
  --eval-path runs/hf_jobs/sft/<run-prefix>/evaluations/vllm/<eval-prefix>/ \
  --loss-path runs/hf_jobs/sft/<run-prefix>/analysis/loss/
```

Keep the checked-in benchmark ledger updated from finished runs:
- [model_hardware_benchmark_ledger.md](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/docs/benchmarks/model_hardware_benchmark_ledger.md)
- [model_hardware_benchmark_ledger.csv](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/docs/benchmarks/model_hardware_benchmark_ledger.csv)

For `run-experiment`, the analysis bundle now appends or updates the ledger automatically using:
- training lineage
- evaluation summary
- loss results when present
- live HF hourly pricing when available

The ledger is derived. The stage lineage artifacts remain the canonical source of truth for train/eval/loss metadata.

Use raw HF job logs mainly for:
- bootstrap failures before the bucket prefix exists
- dependency/runtime crashes before the first artifact sync
- debugging low-level container issues

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

For multi-spec benchmark launches, use the staggered launcher instead of submitting several `run-experiment` commands back-to-back:

```bash
python3 scripts/launch_experiment_batch.py \
  Trainers/cloud/experiments/qwen3_4b_full_cycle_benchmark_l40sx1_pruned.yaml \
  Trainers/cloud/experiments/qwen3_4b_full_cycle_benchmark_a100_large_pruned.yaml \
  --auto-hardware \
  --optimize-for cost \
  --yes
```

Gotcha:
- The launcher defaults to a 5-second stagger. Keep it on unless you have a specific reason to remove it; same-second submissions are harder to monitor and used to collide on timestamp-derived artifact prefixes.
- Do not treat large unused VRAM headroom as a success case on its own. Read `training_lineage.json` and check `capacity_profile`:
  - if peak reserved VRAM is well below half of device memory or headroom is still tens of GB, the run is underpacked
  - for large-memory tiers like `a100-large`, that usually means you left training throughput on the table
  - increase microbatch or otherwise retune before treating the hardware choice as optimized
- On `a100-large` and above, default to aggressive packing:
  - do not reduce batch just because you switched to DoRA, rsLoRA, or another adapter variant
  - start from the highest known-good packed shape for that model family
  - accept that an exploratory OOM is preferable to quietly wasting half the card
  - only back off after a real OOM or reproducible instability signal

---

## HF Jobs Bucket and Auth Gotchas

For `hf_jobs`, a few patterns matter enough to treat as hard rules:

- Never cancel a live HF job, delete bucket data, or relaunch a cloud run unless the user explicitly approves that specific action first.
- Do not infer approval for cancel/delete/relaunch from a request to inspect, check, monitor, compare, or switch focus.
- The job runs from the exact pushed commit. If the remote logs show an older `HEAD`, you launched the wrong SHA and are debugging stale code.
- Keep the main training interpreter compatible with Unsloth and Transformers. If bucket sync needs a newer `huggingface_hub`, install it in an isolated helper path or subprocess, not into the training runtime.
- Pass `HF_TOKEN` into `run_job(...)` explicitly via job secrets. Do not assume the container automatically receives your local token.
- Normalize blank auth values to `None`. An empty `HF_TOKEN` or `HF_API_KEY` can produce `Authorization: Bearer ` and fail before the request is sent.
- Resolve and, if needed, create the bucket once before training starts. During steady-state log sync, use the resolved bucket ID directly.
- Keep HF job labels conservative. Do not put slash-heavy values like raw `bucket_id` or `artifact_prefix` into labels; HF Jobs can reject submission. Recover those values from command args or other metadata instead.
- Polling and identity checks should be conservative. Frequent bucket creation attempts or repeated `whoami-v2` calls can hit Hugging Face rate limits.
- On Windows launch hosts, set `PYTHONIOENCODING=utf-8` for non-JSON cloud
  launches. Rich UI output can contain glyphs such as `★`, and the default
  `cp1252` console path can crash before job submission with
  `UnicodeEncodeError`.
- Keep HF Jobs launcher dependencies isolated from the trainer runtime. A
  launcher-only venv with `huggingface_hub>=1.5.0`, `transformers` 5.x, and CPU
  `torch` can satisfy local CLI imports and Buckets APIs without upgrading the
  Unsloth/KTO training env, which may still require `huggingface_hub<1.0`.
- Do not apply that broad legacy launcher recipe to the protected JP lane. JP
  uses the five exact direct pins listed above under Python 3.12; missing, extra,
  duplicate, reordered, or ranged requirements change the audited runtime
  contract. Transformers, Torch, and Unsloth must remain absent from protected
  launcher imports.
- Do not upgrade generic project dependencies in the active training image
  during HF Jobs bootstrap. Install missing project deps only; curated Unsloth
  images can carry tightly coupled NumPy/SciPy/Transformers/Unsloth stacks, and
  in-place upgrades before `import unsloth` can leave C-extension packages in a
  mixed state. Use explicit stage-local `pip_packages` only when a run is
  intentionally testing a new runtime.
- Quote remote pip requirements that contain shell metacharacters. In HF Jobs
  bash commands, an unquoted token like `huggingface_hub>=1.5.0` can be parsed
  as output redirection rather than a pip requirement. Use shell quoting, e.g.
  `'huggingface_hub>=1.5.0'`.
- Keep bucket-sync overlay installs dependency-isolated. The overlay exists to
  provide a newer Hub/Buckets client to the sync helper, not to replace the
  training image's dependency graph; install it with `--target` and `--no-deps`
  so global image packages remain authoritative.
- Image-profile blockers should be classified before changing training
  hyperparameters. In the Phase 1 HF Jobs smoke, `unsloth/unsloth:latest`
  failed before trainer import with an Unsloth NumPy mid-session mismatch
  (`loaded: 2.2.6, installed: 2.4.1`), while
  `unsloth/unsloth:2026.2.1-pt2.9.0-cu12.8-fixed-numba-numpy-error` failed
  before trainer import at `ModuleNotFoundError: numpy._core.tests` through
  SciPy/Transformers. Treat those as image/runtime-profile failures, not
  dataset or LoRA failures.

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

For experiment orchestration:
- `run-experiment` now defaults to **parallel** post-training execution
- evaluation and exact loss submit as separate sibling jobs after training completes
- analysis waits for both selected post-training stages
- use `post_training.mode: same_job` only when you intentionally want the older embedded eval+loss path for a smoke/fallback run

Same-job exact-loss gotcha:
- `cloud-eval --with-loss` and `post_training.mode: same_job` should rely on the selected eval image's preinstalled ML stack for packages such as `peft`, `torch`, `transformers`, and `numpy`.
- Keep the evaluator runtime overlay separate from the bucket-sync helper overlay. The runtime overlay is the only one exported on evaluator `PYTHONPATH`; the bucket-sync overlay is exposed only through `HF_BUCKET_SYNC_PYTHONPATH`.
- Do not put `huggingface_hub>=1.5.0`, `hf_transfer`, or `hf_xet` on the evaluator `PYTHONPATH`. Those packages are for Buckets only and can violate the base image's `transformers` Hub-version requirements.
- If embedded exact loss fails with `peft is required to load LoRA adapter checkpoints for exact loss scoring`, first verify the eval image actually contains the expected ML stack. If a package must be added, use explicit image-compatible pins or `--no-deps` stage overrides rather than an unconstrained overlay install.

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
- Do not assume multi-GPU HF flavors automatically give you tensor-parallel vLLM. Prequantized BitsAndBytes base models (for example `*-bnb-4bit`) cannot use vLLM tensor parallelism in this path, so multi-GPU eval may need to fall back to single-GPU generation while exact loss still uses all visible GPUs afterward.
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
