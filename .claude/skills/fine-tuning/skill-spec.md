# Skill spec: Fine-tuning runtime-profile guidance

Status: APPROVED -- 2026-09-21.

## Purpose

Improve the existing project-local fine-tuning skill so agents select a verified named runtime profile for supported model training instead of guessing compatible package combinations.

## Trigger

Use this guidance when an agent plans or runs a supported SFT model locally, chooses its Docker runtime, or prepares the same workload for later provider promotion.

### Should not trigger

- Do not use it to claim compatibility for a model or model size that has not passed its own smoke.
- Do not use it to qualify or rewrite Modal runtime locks; that remains a separate provider-specific promotion step.
- Do not use it for unrelated Python dependency management outside the training runtime.

## Workflow

1. Identify the exact model, revision, and training method.
2. Resolve a checked-in named runtime profile that explicitly admits that model and method.
3. Verify the profile's immutable image and complete captured installed-distribution inventory digest. Treat the inventory as the model runtime's transitive dependency lock; do not reconstruct it from a short list of top-level packages.
4. Compile the training recipe without Docker effects and inspect the resolved profile/image/lineage inputs.
5. Run a dry smoke before training and record the observed model, dataset, GPU, memory, and setup result.
6. Add or broaden a compatibility profile only after equivalent ground-truth smoke evidence exists.

## Inputs

- Exact model name and revision.
- Training method and recipe.
- A checked-in runtime-profile name, immutable image digest, complete installed-distribution inventory digest, and captured Python/CUDA/core-ML runtime facts.
- The verified prepared dataset and its training controls.

## Outputs

- Concise canonical guidance in `.skills/fine-tuning/SKILL.md` and the relevant fine-tuning reference.
- Synced `.agents/skills` and `.claude/skills` mirrors.
- No new standalone or packaged skill.

## Good output

"Pinning the dependency trees for each model ... so there's not guessing required": the Qwen 3.5 4B SFT recipe selects `qwen35-sft-v1`, whose immutable image, full transitive installed-distribution inventory, Python/CUDA facts, and exact model revision were captured from a successful smoke. The skill tells agents to use that profile rather than assemble Torch, CUDA, Transformers, TRL, Unsloth, Torchvision, or transitive pins themselves.

## Bad output

Tell agents to layer an old set of direct `setup.pip` pins over `unsloth/unsloth:latest`, silently resolve transitives at launch, or assume that the verified 4B profile also qualifies untested Qwen 3.5 sizes.

## Constraints

- `.skills/` remains the canonical source; mirrors must be synchronized and checked.
- Keep the skill a slim router and change only stale Qwen/runtime guidance.
- Preserve config-first recipes and the simple one-command local user journey.
- Never invent compatibility claims, dependency versions, or recipes without a successful smoke.
- Do not package or publish the skill; it remains local to this project.

## Out of scope

- Modal image/lock promotion.
- Compatibility claims for Qwen 3.5 0.8B, 2B, or 9B.
- Actual optimization/training, evaluation, GGUF conversion, or publishing.
- A repository-wide redesign of every historical image profile.

## Done

- Stale April-era Qwen overlay guidance is removed or clearly marked historical.
- The skill routes verified Qwen 3.5 4B SFT work through `qwen35-sft-v1` and its immutable inventory.
- The local dry-run sequence and compatibility boundaries are unambiguous.
- Skill-tree synchronization and check both succeed.

## Delivery

Project-local source under `.skills/fine-tuning/`, synchronized to `.agents/skills/fine-tuning/` and `.claude/skills/fine-tuning/`. No packaged `.skill` artifact, per the user's explicit instruction.

## Open decisions

- Start with one grounded profile for Qwen 3.5 4B SFT. Every additional exact model/revision must receive an explicitly admitted, fully pinned dependency inventory after its own successful smoke. A profile may be shared only when each listed model/revision has independently passed against that exact profile.

## Change log

- 2026-09-21: Approved by the user after the grounded Qwen 3.5 4B local Docker smoke and final independent runtime-profile audit.
