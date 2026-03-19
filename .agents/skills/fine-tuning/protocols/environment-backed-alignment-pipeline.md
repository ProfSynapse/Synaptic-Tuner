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
- merge/upload jobs: config-driven HF Jobs under `Trainers/cloud/jobs/*.yaml`
- env-GRPO entrypoint: `Trainers/grpo/train_env_grpo.py`
- env-GRPO config: `Trainers/grpo/configs/env_config.yaml`

## Operational Checklist

Before launching cloud work:

1. Confirm the worktree is clean and the exact commit is pushed
2. Confirm the source SFT artifact exists in the HF bucket
3. Merge and publish the SFT model under the correct Nexus name
4. Point KTO and env-GRPO at that merged/published SFT model
5. Use cloud for env-GRPO, not the local Mac runtime

