# Environment-Backed Alignment Pipeline

This protocol captures the intended end-to-end training flow for the Nexus
agent models when using SynthChat-generated environment tasks.

## Purpose

Use SynthChat to generate realistic multi-step tool-use tasks inside real
filesystem-like environments, then train models in stages:

1. SFT to learn the tool format and basic behavior
2. Merge and publish the best SFT model under the Nexus naming convention
3. KTO to refine behavior using stored positive/negative rollout examples
4. Env-backed GRPO to optimize directly against multi-step task success in the
   live environment

This is the canonical flow for environment-backed agent training.

## Plain-Language Pipeline

### 1. SynthChat generation

SynthChat generates:
- a workspace / filesystem environment
- a derived task from that environment
- a natural user request for that task
- a multi-step assistant rollout

The assistant then interacts with the environment:
- search
- read
- edit / move / archive / answer
- final text response

Programmatic environment checks determine whether the task was actually solved.
Optional stage reviews determine whether the example is high enough quality to
keep for training.

SynthChat is the task and environment generator.

### 2. SFT

Start from the chosen base model and train on supervised examples so the model
learns:
- the response structure
- the tool wrapper format
- the expected assistant behavior

SFT teaches the model what a correct tool-using assistant response looks like.

### 3. Merge and publish

Take the latest good SFT adapter, merge it into the base model, and publish it
as the named Nexus release artifact.

This published merged model is the source of truth for downstream KTO and GRPO.
Do not continue from an arbitrary local checkpoint when a clean merged/published
SFT model exists.

### 4. KTO

Use stored SynthChat rollouts projected into positive/negative preference data.

KTO teaches:
- which trajectories are preferred
- which failures or lower-quality behaviors should be disfavored

KTO is still an offline stored-data stage.

### 5. Env-backed GRPO

Env-backed GRPO is the online stage.

During training:
- the model receives a prompt for a real environment task
- it acts for multiple turns
- tools execute against the live environment
- it receives tool results and runtime errors back
- reward is computed from whether it solved the task

This is the intended final RL stage for environment-backed agent behavior.

## Core Principles

- SynthChat tasks should be derived from generated environments, not hardcoded
  fixed paths or targets
- SFT should produce a clean merged/published Nexus model before KTO/GRPO
- KTO uses stored rollout data
- true GRPO should use the live environment loop, not only a projected static
  prompt/completion dataset
- cloud env-GRPO should keep the Unsloth image as the base runtime, while newer
  TRL/OpenEnv dependencies live in an isolated runtime layer on top

## Experimental Flywheel

Use a staged loop for env-backed GRPO instead of jumping directly to a full run:

1. Merge the latest SFT model and verify the merged path can load.
2. Generate or select a tiny environment-backed dataset and run the trainer in
   dry-run mode. Confirm raw, filtered, and formatted counts.
3. Run a short GRPO smoke in the background with rollout diagnostics enabled.
   Inspect logs within the first minute and abort if rollout generation is not
   happening or the process is clearly misconfigured.
4. Inspect both aggregate metrics and raw rollouts:
   - reward mean and reward standard deviation
   - KL, loss, and gradient norm
   - environment pass rate
   - final text pass rate
   - stop reasons
   - tool/action names and runtime errors
5. Scale only the dimension that is currently stable:
   - more steps when rewards vary and tool execution is real
   - more examples when generation quality is high
   - more scenario families only after one-per-scenario smoke generation passes
6. If the run has no reward variance, no rollouts, no gradients, or repeated
   malformed actions, stop and fix the prompt/config/runtime before scaling.

Keep each scale step reproducible with checked-in config or target manifests.
Do not bake a specific tool wrapper, command family, or scenario into runtime
code; tool and reward expectations belong in config. Generic runtime additions
are appropriate only when the config surface references reusable primitives that
the runner does not yet support.

## Env-GRPO Smoke Notes

- For local non-vLLM TRL runs, verify that the configured rollout function is
  actually used. Some TRL versions only wire rollout callbacks through vLLM by
  default; a local fallback must be explicitly config-enabled and should write
  rollout JSONL diagnostics.
- Local fallback generation must run the model in inference/eval mode during
  rollout sampling, then restore training mode before optimization. Otherwise
  dropout/training-mode sampling can look much worse than direct inference.
- Use LoRA/PEFT for local env-GRPO smokes on constrained GPUs. Full-model GRPO
  can OOM and can also duplicate large artifacts unnecessarily.
- Ensure the effective generation batch is divisible by `num_generations`.
  A common smoke shape is batch size 4 with 4 generations per prompt.
- Start with shaped rewards when debugging the loop: reward successful
  environment work, apply a penalty for missing final text, and make strict
  final text a later curriculum once the model reliably acts in the environment.
- Prefer partial-credit reward surfaces while debugging multi-step behavior.
  A wrong or incomplete action should usually be penalized according to how far
  it progressed, not treated the same as no action at all. Useful generic
  signals include parsed action validity, expected action presence/order,
  runtime tool status, environment issue severity, and final text completion.
- Keep reward separation large enough that incomplete trajectories remain
  clearly worse than successful ones. If a search/read-only or "ready to
  proceed?" trajectory lands near neutral because partial progress offsets the
  failure penalty, tune the configured penalties/caps before scaling steps or
  examples.
- Before training any behavior that depends on precise fields such as paths,
  identifiers, offsets, or line numbers, inspect the actual model-facing tool
  feedback. If the field is not visible in tool results, either make the
  environment/tool output expose it through config or design the curriculum to
  teach the model how to derive it. Do not silently reshape fixtures to match a
  model's incorrect habit.
- When adding a new environment execution config knob, test both the global
  execution config path and a per-scenario override. It is easy for a runtime
  loader to preserve existing keys while accidentally dropping a newly added
  generic option.
- For edit tasks with line ranges, verify whether read results are raw text or
  line-numbered text. If read output is raw text, then line selection is an
  additional learned behavior and should receive intermediate reward/diagnostic
  coverage instead of being hidden behind a single pass/fail assertion.
- If final text is the main failure after environment success, first test a
  small prompt/config nudge that frames final text as the normal completion
  action: after the needed tool results answer the request or confirm the action
  is complete, stop tools and summarize the result in text. Compare against the
  prior smoke before changing reward code.
- If premature "ready to proceed?" responses are common, prefer a config-level
  prompt augmentation over rewriting generated examples. A generic nudge should
  say to continue with the next useful tool step when the user already requested
  the sequence, while still asking before ambiguous, broad, or destructive work.
  Keep the augmentation in the run YAML so it can be enabled, removed, or
  compared across smokes without changing the source dataset.
- Always keep raw rollout diagnostics. Aggregate rewards can hide whether the
  model solved the task, skipped required exploration, repeated a command, used
  malformed actions, or failed to stop after the environment work was complete.
- One-per-scenario generation smokes should precede mixed GRPO scaling. If most
  scenario rows fail deterministic gates, final review, or environment checks,
  fix the scenario/rubric config before using that family in a larger run.
- If scenario YAML references deterministic gate types the runner does not
  support, either express the checks with supported gates or add generic gate
  implementations. Do not add scenario-specific validation code.
- Run long or uncertain jobs in the background with log files, then poll logs
  and process state. This prevents waiting on a long timeout when a setup error
  is visible in the first minute.
- Clean up failed smoke artifacts and stale containers as part of the loop.
  Keep logs and small diagnostic files; remove duplicate failed checkpoints or
  full-model outputs that are not candidates for evaluation.

## Cloud Runtime Rule

For env-backed GRPO:
- base runtime: Unsloth HF Jobs image
- env-GRPO runtime: isolated venv with modern TRL/OpenEnv

Do not destabilize the legacy trainer runtime by globally upgrading the old
training environment in place.

## Nexus Naming Rule

Merged/published release artifacts should follow:

`Nexus-[SizeClass]-[EngineID].[BuildID]`

Example:
- `Nexus-Quark-L2.5.28`

Where:
- `Quark` is the 1B-4B class
- `L2.5` is the Liquid Mercury base family/version
- `28` is the internal build

## Current Repo Mapping

- SynthChat generation: `python3 -m SynthChat.run ...`
- KTO / cloud orchestration: `python tuner.py ...`
- merge/upload jobs: config-driven HF Jobs as recipes under `Trainers/recipes/*.yaml` (with `target: cloud`)
- env-GRPO entrypoint: `Trainers/grpo/train_env_grpo.py`
- env-GRPO config: `Trainers/grpo/configs/env_config.yaml`

## Operational Checklist

Before launching cloud work:

1. Confirm the worktree is clean and the exact commit is pushed
2. Confirm the source SFT artifact exists in the HF bucket
3. Merge and publish the SFT model under the correct Nexus name
4. Point KTO and env-GRPO at that merged/published SFT model
5. Use cloud for env-GRPO, not the local Mac runtime

