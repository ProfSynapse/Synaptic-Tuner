# Vault Workflows v1

This is the first concrete environment-backed rollout spec for training
workflow behavior after tool-schema SFT.

## Purpose

Use one canonical environment-rollout dataset to support both:
- `KTO`: preference shaping for better behavior
- `GRPO`: reward optimization for environment success

Do not train from raw rollout logs directly. Project them into method-specific
views only after filtering.

## Base Scenario Pack

Scenario source:
- [vault_kto_pilot.yaml](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/SynthChat/scenarios/vault_kto_pilot.yaml)

Target counts:
- `3` environment seeds per scenario
- `10` rollouts per seed
- `10` scenarios
- `300` canonical rollout records total for the first local pilot

Target config:
- [targets_vault_workflows_v1.json](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/SynthChat/config/targets_vault_workflows_v1.json)

Target strategy:
- each rollout batch for a scenario shares the same seeded environment
- this keeps environment state comparable across attempts
- metadata should record both `environment_seed_id` and `rollout_index`

## The 10 Scenarios

1. `envfs_promote_inbox_note_with_example`
   Family: `workflow_completion`
   Goal: inspect an example note, promote the inbox note, preserve body, remove source.

2. `envfs_create_daily_note_from_template`
   Family: `format_following`
   Goal: discover a template, read it, create a structured daily note, include the right link.

3. `envfs_triage_correct_note_after_search`
   Family: `retrieval_before_update`
   Goal: find the correct note among distractors, then update only that note.

4. `envfs_update_production_config_preserve_structure`
   Family: `read_before_precise_update`
   Goal: find the right config note, preserve unrelated sections, update only the requested value.

5. `envfs_read_before_replace_settings_yaml`
   Family: `read_before_precise_update`
   Goal: read structured settings first, replace one field, preserve everything else.

6. `envfs_archive_empty_folder_after_verification`
   Family: `destructive_verification`
   Goal: verify emptiness first, archive only the empty folder.

7. `envfs_archive_only_deprecated_notes`
   Family: `destructive_verification`
   Goal: identify which notes are deprecated before archiving only that subset.

8. `envfs_continue_inbox_organization`
   Family: `workflow_completion`
   Goal: continue a prior organization workflow by creating needed folders and moving notes.

9. `envfs_clarify_ambiguous_note_reference`
   Family: `clarify_vs_act`
   Goal: ask a concise clarification question instead of guessing or calling tools.

10. `envfs_confirm_before_bulk_delete`
   Family: `destructive_confirmation`
   Goal: ask for confirmation before broad irreversible cleanup.

## Success Model

For canonical rollouts, keep these signals distinct:

- `schema_passed`
  Meaning: exact structured tool output is valid.

- `environment_passed`
  Meaning: final runtime state matches the assertions.

- `score_tier`
  Meaning: path quality, not just final correctness.
  Expected values: `preferred`, `acceptable`, `partial`, `none`

- `stop_reason`
  Meaning: how the episode ended.
  Useful values include:
  - `environment_passed`
  - `text_response`
  - `max_turns_reached`
  - `max_tool_steps_exceeded`
  - `stuck_repeated_failure`
  - `stuck_no_progress`
  - `schema_validation_failed`
  - `environment_execution_failed`

Hard success rule:
- final environment state is the primary truth for tool scenarios
- text-only clarification/confirmation scenarios succeed when the assistant
  correctly avoids acting and preserves the environment state

Path scoring rule:
- preferred path is best
- acceptable path still counts as success
- non-preferred path is not itself failure if the final state is correct

## Canonical Dataset Record

Each line in the base dataset should conform to:
- [canonical_rollout.schema.json](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/Datasets/environment_rollouts/canonical_rollout.schema.json)

At minimum, each record should capture:
- scenario identity
- system/user prompts
- final assistant response
- full `conversation_trace`
- full `environment_fixture`
- `environment_assertions`
- `episode_trace`
- final result summaries
- labeling metadata for later filtering

## Labeling Rules

Keep rich labels in the canonical dataset. Do not collapse early.

Recommended labels:
- `environment_passed:true|false`
- `schema_passed:true|false`
- `issue:retrieval_failure`
- `issue:structure_failure`
- `issue:workflow_incomplete`
- `issue:destructive_without_verification`
- `issue:clarification_failure`
- `issue:stuck`
- `issue:schema_format_failure`
- `tier:preferred|acceptable|partial|none`
- `family:<scenario_family>`

## KTO Projection

Write KTO output as a separate dataset, not in-place on the base file.

Recommended path:
- `Datasets/kto/vault_workflows_behavior_v1.jsonl`

Projection schema:
- [kto_projection.schema.json](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/Datasets/environment_rollouts/kto_projection.schema.json)

Expected record shape:
- `prompt`
- `completion`
- `label`
- optional metadata for analysis

Positive examples:
- `schema_passed == true`
- `environment_passed == true`
- `score_tier in {preferred, acceptable}`

Negative examples:
- meaningful behavior failures only
- wrong retrieval behavior
- partial workflows
- destructive action without verification
- clarification failures
- stuck/no-progress episodes

Do not mix malformed tool JSON into the main behavior KTO set.
Keep schema-format failures separate unless you are intentionally building an
SFT repair set.

## GRPO Projection

Write GRPO output as a separate dataset, not in-place on the base file.

Recommended path:
- `Datasets/grpo/vault_workflows_rewards_v1.jsonl`

Projection schema:
- [grpo_projection.schema.json](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/Datasets/environment_rollouts/grpo_projection.schema.json)

Expected record shape:
- `prompt`
- reward columns
- scenario metadata
- optional ground-truth columns when a task has a true target action

Recommended first-pass columns:
- `prompt`
- `scenario_id`
- `scenario_family`
- `allowed_tools`
- `environment_passed`
- `score_value`
- `score_tier`
- `stop_reason`
- `schema_passed`
- `tool_call_count`
- `failure_labels`

Recommended first-pass reward intuition:
- strong positive reward for `environment_passed`
- smaller bonus for `preferred` over `acceptable`
- penalty for `stuck_*`
- penalty for malformed tool JSON
- penalty for hard destructive violations

## Separation of Concerns

Keep these datasets separate:
- canonical rollout base dataset
- KTO behavior projection
- GRPO reward projection
- optional schema/SFT repair dataset

Merge later only when you have a clear experimental reason.
