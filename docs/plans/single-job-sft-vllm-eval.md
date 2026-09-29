# One-job SFT and chat evaluation rehearsal

Status: design for the next bounded rehearsal; no full training run is authorized by this plan.

## Outcome

Submit one provider job that trains a LoRA adapter, serves it with vLLM, and
evaluates a small set of chat requests. The job saves evidence and output after
each phase. The next live run remains a short smoke; full training follows only
after this complete path has been qualified. GGUF is a later desktop export
workflow and does not gate cloud training or evaluation.

The workflow contract is provider-neutral. Modal supplies the first execution
adapter, pinned image, Volume mounts, and artifact transport. The host remains
the owner of config, credentials, source and dataset identity, authorization,
and retained run state.

## What the existing smoke proves

The L40S training call `modal-4c9ccfa5f57b6e4b4b0e01c6` completed two steps
and saved a verified Qwen3.5-4B LoRA adapter and tokenizer. Its model preparer
copied the pinned base snapshot into a private training path and committed
verified repository members to the model-cache Volume before the trainer ran.
The training child used that local snapshot offline. The exact proof is in
`docs/review/modal-volume-root-diagnostic-20260927.md`.

Today, the `train-chat` example starts a separate inference Sandbox after
training. The inference image has its own lock and vLLM stack. The exact
successful training image's captured distribution inventory includes vLLM
0.26.0, but packaged training has not qualified that serving path. First test
vLLM and the Qwen3.5 LoRA in that same image. The current training result
accepts exactly five artifact roles; evaluation results need a separate
verified phase output.

## Proposed phases inside one submitted GPU job

1. **Prepare once.** Admit the exact source, dataset, recipe, model revision,
   runtime, and evaluation scenarios. Prepare and verify the
   pinned base model into a retained local snapshot; record its manifest and
   digest. A fresh attempt may download a missing base once at this phase.
2. **Train.** Run the existing isolated, credential-free SFT child for a
   deliberately short smoke. Persist and verify the adapter, tokenizer,
   training metrics, and lineage before moving on. A later phase failure must
   not erase the trained adapter or report the entire workflow as completed.
3. **Serve from local paths.** End the trainer process and free its GPU memory.
   Start a pinned vLLM process in the same job using the already verified local
   base snapshot and newly saved LoRA directory. Bind it to loopback, request
   the adapter by its served name, and fail if it attempts a model download or
   serves a different model/adapter identity. No second model-weight fetch is
   needed between training and inference.
4. **Evaluate and retain.** Send a small, fixed set of chat requests through
   the existing config-driven evaluator. The rehearsal gate checks server
   readiness, adapter selection, successful nonempty responses, and exact
   response/result persistence. Scenario expectations and thresholds live in
   configuration. Preserve responses, pass/fail results, runtime details, and
   lineage. Prose quality after two training steps is informative, not a
   claim that the model is ready for production.
5. **Publish final result.** Stop vLLM and persist the evaluation record to
   provider-native storage alongside the existing verified training artifacts.
   Download and verify the eval record by size and hash. Return one workflow
   result with phase statuses and the verified artifact inventory.

## Implementation order

1. Define the config-first pipeline request and stage/result contract using
   provider-neutral names; keep prompts, thresholds, and model profile in config.
2. Qualify serving in the exact pinned training image, using its captured
   vLLM 0.26.0 installation and the saved Qwen3.5 LoRA. If an actual
   compatibility test fails, review a new image or isolated environment before
   changing pins. Reuse the existing training and serving primitives; add a
   sequential worker boundary instead of chaining submitted provider jobs.
   Do not resolve dependencies at runtime.
3. Add a small, verified evaluation output without widening the existing
   five-role training artifact contract for existing consumers.
4. Prove provider-free stage transitions, local snapshot reuse, failure
   retention, configured gate behavior, exact output digests, and no replay of
   an uncertain submission. Then qualify the composite image and its source
   lock before a fresh bounded L40S run.
5. Run the short end-to-end rehearsal and inspect the saved chat and evaluation
   artifacts. Use that evidence to size and authorize a full run.

## Full training after the rehearsal

Keep the same one-job phase order. Choose the full run's step count and budget
from the measured smoke and the 181/39 train/validation split, rather than
copying the two-step limit or guessing a new number now. Preserve the 39
validation rows as held-out data. Compare base and adapter responses on a
separate, fixed set of context-to-prose prompts, track validation loss and
failure cases, and record the decision before any desktop export.
Only then select whether another epoch or a revised data recipe is warranted.

## Operational tradeoff and open cache decision

One GPU job stays allocated during vLLM evaluation. That avoids a second
provider queue; record measured phase times and actual provider spend.

Reusing the base snapshot **within** the new job is straightforward because
the verified local path already exists. Reusing the previous smoke's cache
Volume in a **new** attempt needs a separate read-only cache-admission design:
fresh attempts currently require unused Volume names and the model preparer
publishes into a create-only cache. Decide whether cross-attempt reuse is
required before changing that resource policy.
