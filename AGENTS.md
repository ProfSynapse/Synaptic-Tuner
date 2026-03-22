# Project Agent Notes

This repository has a few cloud-training constraints that are easy to relearn the hard way.

## Fine-Tuning Workflow Discipline

- For any fine-tuning, cloud training, evaluation, experiment-loop, checkpoint-eval, model-selection, or dataset-publishing task, always begin by loading the `fine-tuning` skill first.
- `.skills/` is the canonical skill source for this repo. `.agents/skills` and `.claude/skills` are synced copies and must match it exactly.
- After changing canonical skills, run `python3 .skills/scripts/sync_skill_trees.py` and verify with `python3 .skills/scripts/sync_skill_trees.py --check`.
- Before building a new script, helper, or one-off workflow to run something, first check whether the needed command, script, CLI surface, or skill guidance already exists in the repo.
- Do not create throwaway scripts just to get a task done if an existing script, CLI, or skill can be used or extended.
- If the capability does not exist, the next step is not an ad hoc workaround. Update the relevant skill and add the proper checked-in script/CLI workflow so the new capability is reusable.
- Prefer repo CLIs and checked-in scripts over manual bucket/API probing whenever those surfaces exist.

## HF Jobs

- Remote jobs clone and run the exact pushed commit. If the job log shows an older `HEAD`, stop and relaunch from the right SHA instead of debugging stale code.
- Do not upgrade `huggingface_hub` in the main Unsloth training environment just to get Buckets support. `transformers` in the training stack requires `huggingface-hub<1.0`.
- If Buckets support needs a newer Hub client, isolate it in a helper path or subprocess and keep the trainer runtime untouched.
- Pass `HF_TOKEN` into `huggingface_hub.run_job(...)` explicitly with job secrets. Do not assume the cloud job inherits the local shell environment.
- Treat blank `HF_TOKEN` / `HF_API_KEY` values as unset. Empty strings can produce `Authorization: Bearer ` and fail in `httpx` before any request reaches Hugging Face.
- Resolve or create the bucket once up front, then sync against the resolved namespaced bucket ID during the run.
- Avoid repeated bucket creation and `whoami-v2` calls during periodic sync. Cache bucket resolution and keep dashboard polling conservative.
- Use `python tuner.py cloud-eval --run latest --preset full` for remote HF Jobs evaluation of bucketed runs; the current stable runtime is direct Unsloth inference, not vLLM.
- Use `python tuner.py cloud-pipeline --method sft --preset full` for the common train-then-evaluate path; it hands the exact finished run into cloud eval automatically.
- Use `python3 Trainers/scripts/hf_jobs_hardware.py` before quoting HF hardware availability or pricing; prefer the live HF Jobs hardware endpoint over stale local price assumptions.
- Avoid forcing vLLM into the Unsloth HF Jobs image for this path. If you want vLLM later, treat it as a separate dedicated runtime.
- If a preset resolves but scenario loading fails, inspect `Evaluator/config/eval_run.yaml` for stale filenames before debugging `config_loader.py`.
- HF cloud eval results are saved under the source run's `evaluations/vllm/{timestamp}/` prefix. Inspect `evaluation_results.json` first, then `evaluation_results.md`, then `evaluation_lineage.json`; use `logs/eval_progress.jsonl` only for live/debug state.

## Cloud Artifact UX

- HF Jobs local dashboard parity comes from syncing JSONL training logs to the bucket and replaying them locally.
- HF Jobs cloud evaluation now uses the same adapter idea: remote JSONL progress, local replay into the existing evaluation dashboard.
- Modal may stream usable remote stdout directly; verify that before adding a separate local watcher.
- RunPod currently needs more explicit metric/log plumbing if local dashboard parity is required.
