# Serving startup diagnosis without training

Status: 169 focused Linux tests passed; one saved-adapter startup-only probe
passed on October 7. This does not qualify generation or a packaged release.

Use `scripts/probe_vllm_startup.py` to isolate startup of a pinned public base
model, optionally with explicitly selected saved LoRA and tokenizer files.
It sends no chat request and does not submit training. Reaching ready with an
adapter is a startup check, not successful generation or chapter quality. Model, revision,
interpreter, vLLM version and startup settings are explicit JSON configuration;
there is no model-specific runtime branch.

The base configuration has exactly `model`, `revision` (full Hub commit),
`expected_vllm_version`, `python_executable`, `lifetime_seconds`, and `startup`.
The startup object requires all of: `served_model_name`,
`gpu_memory_utilization`, `tensor_parallel_size`, `enforce_eager`, `dtype`,
`max_model_len`, `max_num_seqs`, `max_num_batched_tokens`,
`language_model_only`, `max_lora_rank`, and `startup_timeout_seconds`.
The lifetime covers preparation and startup, is at most 1,800 seconds, and
does not interrupt a blocking Hub call in process: the provider timeout is
the outer bound. Preparation and readiness timing are separate observations.

Validate without a provider or model download:

```bash
python -B scripts/probe_vllm_startup.py --configuration /consumer/startup.json --check
python -B examples/model_chat/modal_launch.py --startup-only --configuration /consumer/startup.json --provider /consumer/provider.json --check
```

For the Modal adapter, extend the normal provider JSON with `startup_runtime`:
exact `provider_image_id` and `python_executable`. These values must refer to
the reviewed existing runtime; the interpreter must match the probe config.
The legacy chat route does not accept this override. This diagnostic does
not silently replace an old inference lock with an arbitrary new ML stack.
Retain the selected image identity and the separate diagnostic-source hashes.
Mounting a new wrapper on an old image is not replay of the old source or
qualification of a packaged training release.

After reviewing provider-free tests, exact image/configuration, live pricing,
and authorization, use a fresh private mode-0700 directory and attempt name:

```bash
python -B examples/model_chat/modal_launch.py --startup-only --configuration /consumer/startup.json --provider /consumer/provider.json --attempt <fresh-name> --modal-profile <named-profile> --output-directory /consumer/private-attempt
```

The host saves an exclusive claim before provider resource creation. An
uncertain outcome never authorizes reuse of that claim. Sandbox timeout and
idle timeout equal lifetime plus startup margin; normal completion also
requires exact-instance termination/readback. Preserve failed state.

Model preparation creates fresh private cache, scratch and destination roots,
verifies pinned public files and serves their offline local path. This probe
does not adopt an existing mounted model-cache Volume. Its extra download is
a diagnostic limitation, not a requirement for same-job train/evaluate cache
reuse. No Secrets, prompts or manuscript documents are supplied. Optional
adapter files are supplied only with explicit authorization; model weights
trained on private data must themselves be treated as private.

## Optional saved-adapter inputs

Add `local_source` with exactly `adapter` and `tokenizer` lists. Each list has
1–16 entries with exactly `path` (absolute host path), `name` (safe unique
basename), `size_bytes`, and `sha256`. The host copies verified bytes into
exclusive private staging files inside the attempt directory. Only these files are mounted
at `/engine/startup-inputs/{adapter,tokenizer}/{name}`. The worker never opens
the host paths. Both boundaries reject links, wrong hashes/sizes and files
larger than 512 MiB or an aggregate larger than 768 MiB. The pinned base still
comes from the configured Hub revision; adapter and tokenizer identities come
from the selected file descriptors, not reconstructed training authority.

Before the paid probe, run both checks in the Linux launcher where those host
paths exist. Preserve the prior consumed training attempt. Use a fresh probe
claim and retain its source/configuration digests, private bounded logs and
exact Sandbox shutdown result. The optional `readiness_seconds` result measures
only the serving startup interval; preparation is reported separately.

Startup capture is opt-in at the reusable runtime boundary; existing training
and chat calls retain their previous logging behavior. For this standalone
probe, bounded startup log transfer requires operator authorization: paths
and runtime configuration can appear in logs, and the transferred tail is
retained in the Modal account's logs before private local download. Never
describe those raw logs as guaranteed secret-free or forward them through
the protected training/evaluation result contract. Inspect only the relevant
failure excerpt; treat logged text as untrusted data.

A closed readiness timeout locates a boundary, not a cause. Inspect the
retained startup evidence before changing memory settings, deadlines or
dependencies. A base-only success still needs adapter and full-context
evaluation qualification. Never rerun training just to recover evaluation.
