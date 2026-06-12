# Lessons Learned

Historical context from past development sessions. Non-obvious gotchas and runtime discoveries.

---

## [2026-02-14] SynthChat Parallel Docs Workers (PR #55)

**Feature**: `--workers N` now parallelizes docs-based generation (previously only worked for non-docs scenarios).
**Architecture**: Extracted `_run_parallel_generation()` helper - eliminates ~60 lines of duplication between docs/non-docs paths. Each worker gets instance isolation (fresh generator/engine/LLM clients). Results sorted by `task_id` to preserve document order.
**API Change**: `SynthChatGenerator.generate_single()` is now public (was `_generate_single()`).
**Gotcha**: Input validation clamps `args.workers = max(1, args.workers)` to prevent ValueError from `--workers 0`.
**Files**: `SynthChat/run.py`, `SynthChat/generator.py`

---

## [2026-03-14] HF Jobs Buckets/Auth Runtime Lessons

**Context**: Debugged Hugging Face Jobs cloud training for the fine-tuning pipeline after runs made it through setup and training initialization but failed during bucket sync, local dashboard polling, and auth propagation.

**Key Lessons**:
- HF Jobs runs the exact pushed commit; remote `HEAD` output is the fastest way to confirm whether you are debugging the right code
- Hugging Face Buckets support and Unsloth/Transformers compatibility can require different `huggingface_hub` versions, so bucket sync must stay isolated from the training interpreter
- The isolated bucket-sync overlay must include the Hub client's native xet helper dependency (`hf_xet`). If only `huggingface_hub>=1.5.0` is overlaid, the helper can import an incompatible base-image `hf_xet` and fail final sync with `ImportError: cannot import name 'SKIP_SHA256'`
- Never assume HF Jobs injects the local `HF_TOKEN` into the container; pass it explicitly
- Empty `HF_TOKEN` or `HF_API_KEY` values are worse than missing values because they generate `Authorization: Bearer ` and fail in `httpx` before the request reaches HF
- Repeated bucket creation or `whoami-v2` checks during steady-state sync are enough to hit HF rate limits; resolve once, reuse the canonical bucket ID, and keep polling conservative
- Local cloud TUI parity for HF Jobs depends on syncing training JSONL logs into the bucket during the run and replaying them locally

**Decisions**: Resolve/create the bucket once before launch and normalize bare bucket names to the canonical namespaced bucket ID, remove the `HfFileSystem` fallback for bucket sync, keep the main Unsloth runtime on the image-compatible `huggingface_hub` version, isolate Buckets-only Hub functionality in a helper path installed with `pip --target`, install `hf_xet` in that helper path with the Hub client and transfer helper, pass `HF_TOKEN` into `run_job(...)` explicitly via job secrets, normalize blank auth values to unset, cache bucket resolution to reduce identity calls, and slow HF Jobs dashboard polling to reduce rate-limit pressure.

---

## [2026-03-15] HF Jobs Cloud Evaluation Final Runtime Lessons

**Context**: Continued debugging the new HF Jobs cloud evaluation flow after the initial orchestration worked but the runtime repeatedly failed.

**Key Lessons**:
- Treat HF Jobs cloud evaluation as a separate runtime design problem, not just "training plus a server"
- The Unsloth training image is a good place to run direct Unsloth inference, but it is a poor place to force a fresh vLLM stack unless you fully isolate and pin that environment
- Reuse the same bucket-helper isolation pattern for cloud eval sync that cloud training already needs
- If preset-based eval resolves but file loading fails, inspect `Evaluator/config/eval_run.yaml` before touching the loader; stale scenario filenames are an easy anti-pattern
- For the normal operator flow, `cloud-pipeline` is the right abstraction: train first, then pass the exact artifact prefix into eval instead of rediscovering "latest"
- When asked to inspect cloud eval results, start with `evaluation_results.json`; the markdown and lineage files are secondary views, and `eval_progress.jsonl` is only for runtime/debugging

**Decisions**: Launch the eval helper with `python -m Evaluator.cloud_hf_job`, add a local cloud-eval dashboard adapter/replayer using structured JSONL progress events, add `cloud-pipeline` to train on HF Jobs and then evaluate the exact resulting run automatically, switch the HF Jobs eval runtime from vLLM to direct Unsloth inference, keep bucket sync on the isolated helper subprocess path, fix `eval_run.yaml` presets to point at `tool_prompts.yaml` and `behavior_prompts.yaml`, and standardize the saved eval artifact set as `evaluation_results.json`, `evaluation_results.md`, `evaluation_lineage.json`, and `logs/eval_progress.jsonl`.
