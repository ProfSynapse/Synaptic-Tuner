# Project Agent Notes

This repository has a few cloud-training constraints that are easy to relearn the hard way.

## Fine-Tuning Workflow Discipline

- For any task in this repo, begin by loading the most relevant canonical skill from `.skills/`. For fine-tuning, cloud training, evaluation, experiment-loop, checkpoint-eval, model-selection, or dataset-publishing work, that starting point is usually the `fine-tuning` skill.
- `.skills/` is the canonical skill source for this repo. `.agents/skills` and `.claude/skills` are synced copies and must match it exactly.
- After changing canonical skills, run `python3 .skills/scripts/sync_skill_trees.py` and verify with `python3 .skills/scripts/sync_skill_trees.py --check`.
- Before building a new script, helper, or one-off workflow to run something, first check whether the needed command, script, CLI surface, or skill guidance already exists in the repo.
- Do not create throwaway scripts just to get a task done if an existing script, CLI, or skill can be used or extended.
- If the capability does not exist, the next step is not an ad hoc workaround. Update the relevant skill and add the proper checked-in script/CLI workflow so the new capability is reusable.
- Prefer repo CLIs and checked-in scripts over manual bucket/API probing whenever those surfaces exist.

## Config-First Generation Discipline

- This repo is format-agnostic. Do not treat the current tool wrapper, CLI shape, or toy dataset format as a runtime truth.
- For generation and evaluation tasks, do not change runtime code to support one user's current tool schema, wrapper, commands, examples, or dataset shape. Use config, scenario YAML, rubric YAML, schema files, or checked-in declarative config instead.
- Only change parser, executor, judge, evaluator, or generation code when the project is intentionally adding a reusable runtime capability that cannot be expressed by existing config surfaces. If that is necessary, stop first and explain why config is insufficient.
- Do not hardcode any current dataset/example format into parser, executor, judge, evaluator, or generation code.
- Tool-call shapes, wrapper names, context fields, command examples, and dataset-specific assumptions must live in config or scenario/rubric YAML, not in code.
- If a generation/eval bug appears to be specific to the current CLI/tool wrapper, the first fix path is config, rubric, or scenario work, not runtime code changes.
- Generic runtime code may validate or transport configured formats, but it must not assume one specific wrapper such as `useTools`, one specific field set, or one specific command structure unless that behavior is itself config-driven.
- When using environment-backed validation during generation, environment/runtime errors must be surfaced into the judge/improver inputs as structured context so the model can correct the response. Do not rely on ad hoc parser/executor repairs as the primary fix path.
- If a process requirement is important enough to affect how generation/eval work is done repeatedly, encode it in `AGENTS.md`, the canonical skill under `.skills/`, and the relevant checked-in config/docs before continuing.
- For new unsteered local batch generation, prefer `batch-generate --engine vllm`. Set `VLLM_BATCH_INVARIANT=1` before process startup and pin the exact vLLM version, documented minimum CUDA compute capability, model and tokenizer revisions, dtype, tensor parallel size, scheduler limits, structured-output backend, model context limit, multimodal limits, decode settings, and JSON Schema when formatting is incidental. Keep generated rows and token artifacts in the consuming project's private output directory.

## HF Jobs

- For the protected paid A10G training smoke, use only `python tuner.py hf-training-smoke {preflight,approve,execute,recover,observe,verify}`: require the exact pushed-source, security/release, isolated CPython 3.12.7 launcher, live quote, and approval gates; allow one submission and one cancel attempt; keep credentials out of the remote job; and verify only the exact 15-file artifact inventory without bulk bucket sync.
- Remote jobs clone and run the exact pushed commit. If the job log shows an older `HEAD`, stop and relaunch from the right SHA instead of debugging stale code.
- HF Jobs may preserve submitted argv in JobInfo while shell-processing it at runtime. Protected launcher payload chunks must use a shell-safe alphabet such as standard Base64; do not use Base85 or any encoding whose alphabet contains shell metacharacters.
- Protected remote smoke diagnostics must use only the closed non-secret stage codes: credential (120), runtime (121), artifact (122), trainer (123), input (124), with failures outside classified phases remaining generic (125). Never expose remote exception text, tracebacks, provider response data, or credential-derived details.
- HF Jobs does not materialize a writable bucket mount for a brand-new empty subprefix. For the protected smoke, `execute` must durably claim authority, prove the exact derived slot empty, upload a CSPRNG-named one-use mount anchor, and re-list it as the slot's only member before mounting that exact slot. The durable submission claim and exact authenticated provider command must bind the generated anchor nonce and canonical payload digest. The credential-free remote job must require that exact anchor, retain its verified file identity through consumption, create `exclusive-sentinel.json` with atomic `O_EXCL`, recheck and remove only its own anchor, and reject every collision; never upload the fixed sentinel through the overwriting Buckets batch API, widen the mount to the artifact parent, or pass credentials into the job.
- For an exact engine-repository worktree in standalone mode, omit `--project-root` and `--manifest` from `hf-source` and `hf-training-smoke`; supplying `--project-root` selects host-project mode and requires `<project-root>/synaptic.yaml`.
- Do not upgrade `huggingface_hub` in the main Unsloth training environment just to get Buckets support. `transformers` in the training stack requires `huggingface-hub<1.0`.
- If Buckets support needs a newer Hub client, isolate it in a helper path or subprocess and keep the trainer runtime untouched.
- Pass `HF_TOKEN` into `huggingface_hub.run_job(...)` explicitly with job secrets. Do not assume the cloud job inherits the local shell environment.
- Treat blank `HF_TOKEN` / `HF_API_KEY` values as unset. Empty strings can produce `Authorization: Bearer ` and fail in `httpx` before any request reaches Hugging Face.
- Resolve or create the bucket once up front, then sync against the resolved namespaced bucket ID during the run.
- Avoid repeated bucket creation and `whoami-v2` calls during periodic sync. Cache bucket resolution and keep dashboard polling conservative.
- Use `python tuner.py cloud-eval --run latest --preset full` for remote HF Jobs evaluation of bucketed runs; the current stable runtime is direct Unsloth inference, not vLLM.
- Use `python tuner.py cloud-pipeline --method sft --preset full` for the common train-then-evaluate path; it hands the exact finished run into cloud eval automatically.
- Use `python3 .skills/fine-tuning/scripts/hf_jobs_hardware.py` before quoting HF hardware availability or pricing; prefer the live HF Jobs hardware endpoint over stale local price assumptions.
- Avoid forcing vLLM into the Unsloth HF Jobs image for this path. If you want vLLM later, treat it as a separate dedicated runtime.
- If a preset resolves but scenario loading fails, inspect `Evaluator/config/eval_run.yaml` for stale filenames before debugging `config_loader.py`.
- HF cloud eval results are saved under the source run's `evaluations/vllm/{timestamp}/` prefix. Inspect `evaluation_results.json` first, then `evaluation_results.md`, then `evaluation_lineage.json`; use `logs/eval_progress.jsonl` only for live/debug state.

## Modal v1

- Modal training is available only behind the provider-neutral public `TrainingAPI`; do not recreate a `modal run` launcher, provider-specific public verb, or engine-owned database.
- The consuming host owns configuration, credentials, grants, lifecycle/preparation persistence, data, and product state. The engine defines `ModalTrainingRepository` as a protocol only.
- Enforce the packaged `modal-runtime-v1.lock.json` at composition and again in the remote source materializer. The CPython 3.11/Linux launcher dependency file must contain the complete transitive closure with exact hashes; install it with `--require-hashes` and never resolve additional packages at runtime.
- Secrets may enter only through explicitly named Modal Secrets. Reject token/secret/password/API-key environment entries and never embed credential values in `Image.env()`.
- Treat mounted Volume paths as hostile shared storage: on the locked Linux runtime, traverse and open through retained parent directory descriptors (`dir_fd`/openat semantics) so ancestor substitution cannot redirect I/O; bounded reads must reject symlink/reparse leaves and compare file identity across the read, while writes remain exclusive and collision-failing.
- A self-consistent provider observation cannot override the packaged image, SDK, Python, dependency, wrapper, worker, SFT runtime, or ML-stack lock. No live preflight or paid smoke may run until the provider-free barrier and independent review are green.

## Cloud Artifact UX

- HF Jobs local dashboard parity comes from syncing JSONL training logs to the bucket and replaying them locally.
- HF Jobs cloud evaluation now uses the same adapter idea: remote JSONL progress, local replay into the existing evaluation dashboard.
- Modal may stream usable remote stdout directly; verify that before adding a separate local watcher.
- RunPod currently needs more explicit metric/log plumbing if local dashboard parity is required.
