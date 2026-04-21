# CLI Tool Dataset Refresh Plan

> Status: Draft
> Scope: selected `non_thinking` canonical tool datasets, migration pipeline, regeneration queues, merged dataset rebuilds, and cleanup of superseded migration scripts

## Summary

The tool schema for Synaptic Tuner has shifted from older wrapper-era and partially renamed tool formats to a more CLI-oriented catalog defined in [tool-schemas.json](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/tool-schemas.json). The existing canonical tool datasets under [Datasets/tools_datasets](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/Datasets/tools_datasets) are only partially aligned to that new contract.

This plan defines a repo-native refresh workflow that:

1. Inventories the current canonical datasets and classifies examples as auto-migratable, heuristic-migratable, or regenerate-only.
2. Migrates high-confidence examples into the new CLI schema using checked-in scripts under [tools/migrations](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/tools/migrations).
3. Produces explicit regeneration queues for unsupported and low-confidence examples.
4. Generates net-new datasets for tool families that do not have viable legacy source material.
5. Rebuilds downstream merged SFT/KTO artifacts from refreshed canonical sources rather than patching merged files in place.
6. Removes or archives older migration helpers that are now redundant with the new pipeline.

The goal is to make the schema transition reproducible, reviewable, and reusable. This is not a one-off cleanup.

## Goals

- Integrate the CLI-schema refresh into the repo’s existing dataset/tooling structure.
- Preserve as much useful canonical tool data as possible through deterministic migration.
- Separate high-confidence migration from ambiguous transformation and from net-new generation.
- Produce machine-readable reports for inventory, migration outcomes, manual review, and regeneration.
- Rebuild merged dataset artifacts from canonical sources only after canonical refresh is complete.
- Reduce repo ambiguity by retiring superseded migration scripts and stale documentation that still describe older schema transitions.

## Non-Goals

- Directly editing merged dataset artifacts first.
- Reusing older migration helpers without review.
- Hand-migrating every ambiguous content-edit example.
- Designing final scenario prompts for all brand-new tool families in this plan doc.
- Refreshing `thinking/*` datasets in the first implementation pass.
- Updating evaluator compatibility during the first implementation pass.

## Scope Boundaries

### In Scope For Phase 1

Only the selected `non_thinking` canonical datasets:

- `Datasets/tools_datasets/non_thinking/contentManager`
- `Datasets/tools_datasets/non_thinking/memoryManager`
- `Datasets/tools_datasets/non_thinking/promptManager`
- `Datasets/tools_datasets/non_thinking/searchManager`
- `Datasets/tools_datasets/non_thinking/storageManager`

### Explicitly Deferred

The following are deferred until the selected `non_thinking` refresh is complete and stable:

- `Datasets/tools_datasets/thinking/contentManager`
- `Datasets/tools_datasets/thinking/memoryManager`
- `Datasets/tools_datasets/thinking/agentManager`
- `Datasets/tools_datasets/thinking/vaultLibrarian`
- `Datasets/tools_datasets/thinking/vaultManager`

The `thinking` sets have enough semantic drift that they should not complicate the first migration pipeline.

## Current State

### Target Schema

The new catalog in [tool-schemas.json](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/tool-schemas.json) defines 60 tools across 11 agents:

- `canvasManager`
- `composer`
- `contentManager`
- `elevenlabs`
- `ingestManager`
- `memoryManager`
- `promptManager`
- `searchManager`
- `storageManager`
- `taskManager`
- `webTools`

### Canonical Source Pools

The current canonical per-tool dataset sources are concentrated in 10 folders, but only the selected `non_thinking` folders are in the first pass:

- `Datasets/tools_datasets/non_thinking/contentManager`
- `Datasets/tools_datasets/non_thinking/memoryManager`
- `Datasets/tools_datasets/non_thinking/promptManager`
- `Datasets/tools_datasets/non_thinking/searchManager`
- `Datasets/tools_datasets/non_thinking/storageManager`
- `Datasets/tools_datasets/thinking/contentManager`
- `Datasets/tools_datasets/thinking/memoryManager`
- `Datasets/tools_datasets/thinking/agentManager`
- `Datasets/tools_datasets/thinking/vaultLibrarian`
- `Datasets/tools_datasets/thinking/vaultManager`

### Observed Misalignment

- `non_thinking` datasets are already somewhat modernized, but many still contain older parameter names or pre-CLI semantics such as `sourcePath`, `targetPath`, stale `workspaceId` usage, and older prompt-manager execution semantics.
- `thinking` datasets are mixed. Some examples already use partially renamed direct-call tool names, while others still rely on older `agentManager`, session-era memory tools, or content-edit operations that do not map cleanly to the new schema.
- Existing migration helpers in [tools/migrations](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/tools/migrations) and top-level `tools/*.py` reflect the previous transition and should be treated as reference material, not as the new source of truth.

### Current Dataset Inventory

Latest in-scope canonical files:

| Dataset | Latest file | Approx. size |
|---|---|---:|
| non-thinking content | `tools_v2.2.jsonl` | 553 lines |
| non-thinking memory | `tools_v2.3.jsonl` | 584 lines |
| non-thinking prompt | `tools_v2.5.jsonl` | 568 lines |
| non-thinking search | `tools_v2.1.jsonl` | 916 lines |
| non-thinking storage | `tools_v2.1.jsonl` | 1092 lines |
Deferred canonical files:

| Dataset | Latest file | Approx. size |
|---|---|---:|
| thinking agent | `tools_v1.10.jsonl` | 708 lines |
| thinking content | `tools_v1.8.jsonl` | 1131 lines |
| thinking memory | `tools_v1.8.jsonl` | 1219 lines |
| thinking vault librarian | `tools_v1.9.jsonl` | 791 lines |
| thinking vault manager | `tools_v1.10.jsonl` | 1126 lines |

## Placement In Repo

The refresh pipeline should live in the repo’s existing dataset/tooling structure, not as ad hoc top-level scripts.

### Checked-In Code

Primary implementation path:

- [tools/migrations](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/tools/migrations)

Proposed files:

- `tools/migrations/05_inventory_cli_schema_datasets.py`
- `tools/migrations/06_migrate_cli_schema_datasets.py`
- `tools/migrations/07_prepare_cli_schema_regen.py`
- `tools/migrations/cli_schema_rules.py`
- `tools/migrations/cli_schema_utils.py`

Optional thin wrapper:

- `scripts/refresh_cli_tool_datasets.py`

The wrapper should remain thin and simply call the checked-in migration pipeline under `tools/migrations`.

### Reports And Review Artifacts

Generated reports should live alongside the tool datasets:

- `Datasets/tools_datasets/reports/cli_schema/inventory.json`
- `Datasets/tools_datasets/reports/cli_schema/migration_report.md`
- `Datasets/tools_datasets/reports/cli_schema/manual_review.jsonl`
- `Datasets/tools_datasets/reports/cli_schema/regen_queue.jsonl`

### Merge/Rebuild Integration

Merged dataset rebuilds should continue to use the existing merge entrypoints under [Datasets/tools](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/Datasets/tools), especially:

- [Datasets/tools/merge_tools_datasets.py](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/Datasets/tools/merge_tools_datasets.py)
- [Datasets/tools/merge_nonthinking_datasets.py](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/Datasets/tools/merge_nonthinking_datasets.py)

The preferred implementation is to extend current merge flows for the refreshed canonical sets rather than create another disconnected merger.

## Migration Categories

### Mostly Scriptable

These families should migrate via deterministic rules with schema validation:

| Family | Primary actions |
|---|---|
| `storageManager` | Rename `sourcePath -> path`, `targetPath -> newPath`, normalize `open.mode`, preserve `archive`, `copy`, `createFolder`, `list`, `move`, `open` |
| `searchManager` | Normalize direct calls, remove obsolete `workspaceId` from `searchMemory`, clean `fileTypes`, preserve `searchContent`, `searchDirectory`, `searchMemory` |
| `memoryManager` | Migrate `loadWorkspace`, `listWorkspaces`, `listStates`, `loadState`, `archiveWorkspace`, much of `createState`, parts of `createWorkspace` and `updateWorkspace` |
| `promptManager` CRUD/list/image | Migrate `createPrompt`, `getPrompt`, `listPrompts`, `listModels`, `updatePrompt`, `archivePrompt`, `generateImage` |
| `contentManager` basic operations | Migrate `read` and `write` directly |

### Heuristic Migration

These families need transformation logic plus confidence thresholds:

| Area | Why it is ambiguous |
|---|---|
| `contentManager.update` and older content edits | Legacy `append`, `prepend`, `findReplace`, `replaceByLine`, and `replaceContent` need to map into `insert`, `replace`, or `setProperty` |
| `agentManager.toggleAgent` | Maps to `promptManager.updatePrompt` but field naming and intent normalization still need explicit rules |
| `agentManager.executePrompt` and related execution flows | Older agent execution semantics do not always equal current `executePrompts` semantics |
| `memoryManager.createWorkspace` | Older examples may not carry a strong `purpose`; this may need inference from user intent and surrounding conversation |

### Regenerate

These families should be treated as net-new dataset generation work or queued for targeted regeneration:

- `taskManager`
- `canvasManager`
- `composer`
- `ingestManager`
- `elevenlabs`
- `webTools`
- `promptManager.subagent`
- `memoryManager.runWorkflow`
- Low-confidence content-edit examples that cannot be transformed safely
- Old prompt/agent execution examples whose semantics no longer match `executePrompts`

## Dataset Matrix

This is the working migration matrix for the in-scope `non_thinking` canonical files.

| Dataset | Strategy | Expected outcome |
|---|---|---|
| `non_thinking/contentManager/tools_v2.2.jsonl` | Mixed: direct migration for `read` and `write`; heuristic for `update` | Keep roughly 75-85%; regenerate/manual review 15-25% |
| `non_thinking/memoryManager/tools_v2.3.jsonl` | Mostly scripted with some `purpose` inference | Keep roughly 85-92%; regenerate/manual review 8-15% |
| `non_thinking/promptManager/tools_v2.5.jsonl` | Mixed: CRUD/list/image migrated; execution semantics split out | Keep roughly 70-80%; regenerate 20-30% |
| `non_thinking/searchManager/tools_v2.1.jsonl` | Mostly scripted | Keep roughly 90-95%; regenerate/manual review 5-10% |
| `non_thinking/storageManager/tools_v2.1.jsonl` | Mostly scripted | Keep roughly 88-94%; regenerate/manual review 6-12% |
These are planning estimates, not final measured outputs. The inventory script should replace these with observed counts before migration is considered complete.

### Deferred Matrix

The `thinking/*` families are intentionally out of scope for the first pass. They should get a separate plan section or follow-on plan after the `non_thinking` migration pipeline is proven.

## Proposed Pipeline

### Phase 1: Inventory

Create an inventory script that:

- Discovers the latest canonical dataset file in each in-scope source folder.
- Extracts all tool calls from each example.
- Normalizes wrapper/direct-call differences into a common internal representation.
- Classifies every example as `auto`, `heuristic`, or `regen`.
- Writes a machine-readable inventory report and per-dataset summaries.

Primary output:

- `Datasets/tools_datasets/reports/cli_schema/inventory.json`

### Phase 2: Deterministic Migration

Create the migration script that:

- Applies deterministic rename and parameter-normalization rules.
- Replaces tool calls inside examples while preserving the surrounding conversation structure.
- Validates every transformed call against the new schema.
- Writes migrated dataset outputs plus queues for manual review and regeneration.

Primary outputs:

- New versioned per-agent canonical datasets
- `manual_review.jsonl`
- `regen_queue.jsonl`

### Phase 3: Heuristic Migration

Add targeted heuristic transforms only where they are defensible and bounded:

- Content insert/replace/set-property inference
- Toggle-to-updatePrompt mapping
- Workspace-purpose inference

Every heuristic transform should carry a confidence score. Examples below threshold should not be silently migrated.

### Phase 4: Regeneration Preparation

Create a regeneration-prep script that:

- Groups unsupported examples by target tool family
- Emits scenario-ready or prompt-ready inputs for SynthChat generation
- Separates true net-new families from migrated-family backfill

This phase should feed the synthetic dataset generation workflow rather than hide unsupported examples inside the migration script.

### Phase 5: Cleanup Of Superseded Scripts

After the `non_thinking` migration pipeline exists and is validated, remove or archive older migration helpers that no longer represent the supported workflow.

Cleanup candidates:

- [tools/migrate_tool_names.py](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/tools/migrate_tool_names.py)
- [tools/fix_dataset_params.py](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/tools/fix_dataset_params.py)
- [tools/migrations/01_migrate_available_agents_tag.py](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/tools/migrations/01_migrate_available_agents_tag.py)
- [tools/migrations/02_migrate_agent_names_in_calls.py](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/tools/migrations/02_migrate_agent_names_in_calls.py)
- [tools/migrations/03_migrate_tool_names.py](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/tools/migrations/03_migrate_tool_names.py)
- [tools/migrations/04_migrate_content_references.py](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/tools/migrations/04_migrate_content_references.py)

Cleanup rule:

- Do not delete these immediately.
- First replace them with the new CLI-schema pipeline and confirm there are no active references or callers.
- Then either remove them entirely or move them under a clearly marked `legacy/` location if historical retention is still useful.

Related documentation that should be updated in the same cleanup pass because it still describes old tool families and old schema assumptions:

- [tools/README.md](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/tools/README.md)
- [Datasets/tools_datasets/README.md](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/Datasets/tools_datasets/README.md)

### Phase 6: Canonical Refresh Verification

Before merged datasets are rebuilt:

- Validate all refreshed canonical files against the new schema
- Confirm no unsupported legacy tool names remain
- Review a sample of auto-migrated and heuristic-migrated examples from each dataset
- Review regeneration queue sizes against expectations

### Phase 7: Rebuild Merged Outputs

Only after canonical refresh is approved:

- Rebuild merged non-thinking tool datasets
- Rebuild merged tools datasets
- Rebuild any downstream SFT/KTO artifacts that depend on these canonical sources

This includes artifacts such as:

- [Datasets/sft_train_67pct_12.25.jsonl](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/Datasets/sft_train_67pct_12.25.jsonl)
- [Datasets/kto/vault_shared_seed_dynamic_roles_kto_20260316.jsonl](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/Datasets/kto/vault_shared_seed_dynamic_roles_kto_20260316.jsonl)

These should be rebuilt, not hand-migrated.

## Script Responsibilities

### `05_inventory_cli_schema_datasets.py`

Responsibilities:

- Find latest canonical files
- Normalize examples
- Extract tool usage
- Classify examples and calls
- Emit coverage and migration-readiness reports

### `06_migrate_cli_schema_datasets.py`

Responsibilities:

- Load migration rules
- Transform examples
- Validate transformed calls and examples
- Write new versioned canonical outputs
- Write review and regeneration queues

### `07_prepare_cli_schema_regen.py`

Responsibilities:

- Read regeneration queues
- Bucket unsupported items by family and reason
- Produce inputs for targeted synthetic regeneration

### `cli_schema_rules.py`

Responsibilities:

- Define deterministic mappings
- Define heuristic rules and thresholds
- Centralize deprecated-to-current tool names and parameter transforms

### `cli_schema_utils.py`

Responsibilities:

- Shared JSONL I/O
- Schema loading and validation helpers
- Example normalization and replacement helpers
- Dataset version bump helpers

## Validation Requirements

Every migrated output should pass all of the following:

- Tool name exists in the target schema.
- Required parameters match the target schema.
- No stale legacy fields remain unless explicitly allowed.
- Conversation structure remains valid for the dataset format.
- Manual spot checks confirm that the migrated tool call still matches the user request.

Recommended validation artifacts:

- Per-dataset migration counts
- Per-tool success/failure counts
- List of unsupported legacy operations
- Diff-friendly examples of migrated vs original calls for review

## Pseudocode

```python
target_catalog = load_target_catalog("tool-schemas.json")
source_sets = discover_latest_dataset_files("Datasets/tools_datasets")

inventory = []
for dataset in source_sets:
    for example in read_jsonl(dataset):
        normalized = normalize_example(example)
        calls = extract_calls(normalized)
        classification = classify_example(calls, normalized)
        inventory.append((dataset, example.get("id"), classification, calls))

write_inventory_report(inventory)

for dataset in source_sets:
    migrated = []
    manual = []
    regen = []

    for example in read_jsonl(dataset):
        normalized = normalize_example(example)
        new_calls = []

        for call in extract_calls(normalized):
            rule = map_call(call, normalized)

            if rule.kind == "auto":
                new_call = apply_auto_rule(call, rule)
            elif rule.kind == "heuristic":
                new_call, confidence = apply_heuristic_rule(call, normalized, rule)
                if confidence < 0.90:
                    regen.append(tag_for_regen(example, "low_confidence"))
                    break
            else:
                regen.append(tag_for_regen(example, "unsupported"))
                break

            if not validate_call(new_call, target_catalog):
                regen.append(tag_for_regen(example, "schema_fail"))
                break

            new_calls.append(new_call)
        else:
            rewritten = replace_calls(normalized, new_calls)
            if validate_example(rewritten, target_catalog):
                migrated.append(rewritten)
            else:
                manual.append(rewritten)

    write_versioned_outputs(dataset, migrated, manual, regen)

prepare_regen_inputs(regen_queue_path)
```

## Implementation Sequence

1. Write this plan and lock the first pass to selected `non_thinking` managers.
2. Implement `05_inventory_cli_schema_datasets.py`.
3. Run inventory and replace planning estimates with measured counts.
4. Implement shared rules/helpers.
5. Implement `06_migrate_cli_schema_datasets.py`.
6. Migrate the mostly scripted in-scope families first.
7. Add heuristic migration for the bounded ambiguous in-scope cases.
8. Implement `07_prepare_cli_schema_regen.py`.
9. Retire or archive superseded older migration scripts and update stale docs.
10. Review outputs and approve canonical refresh.
11. Rebuild merged dataset artifacts from the refreshed canonical sets.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Legacy content-edit semantics do not map cleanly to CLI operations | High | High | Keep strict confidence thresholds and route low-confidence examples to regeneration |
| Old agent execution examples drift from current `executePrompts` semantics | High | High | Regenerate these flows rather than forcing a lossy migration |
| Canonical datasets contain mixed wrappers/direct-call shapes that break naive tooling | Medium | High | Normalize examples before classification and migration |
| Regeneration queue becomes larger than expected | Medium | Medium | Separate inventory from migration so scope is visible before full execution |
| Merged datasets are rebuilt before canonical refresh stabilizes | Medium | High | Keep merged rebuild as the final phase only |
| Old migration scripts remain in place and confuse future operators | High | Medium | Explicit cleanup pass and README updates after the new pipeline lands |

## Open Questions

- Whether `promptManager.executePrompts` should receive any migrated legacy examples at all, or whether all old agent-execution examples should be regenerated.
- Whether `memoryManager.createWorkspace` should permit heuristic `purpose` inference, or whether missing-purpose examples should be regenerated for clarity.
- Whether net-new families should be generated in one pass or seeded with a small reviewed pilot set first.
- Whether the older migration scripts should be deleted outright or moved under a `legacy/` folder after replacement.

## Success Criteria

This refresh is complete when:

- Every canonical tool dataset is either migrated, explicitly queued for regeneration, or intentionally retired.
- The selected `non_thinking` canonical sets are refreshed without depending on the deferred `thinking` families.
- Refreshed canonical outputs validate against the new CLI schema.
- New and unsupported tool families have clear regeneration inputs and ownership.
- Merged tool datasets have been rebuilt from refreshed canonical sources.
- The migration workflow is documented and checked in so future schema changes use the same structure.
- Superseded migration scripts and stale docs no longer suggest an older supported path.

## Smoke Test Targets

Use checked-in SynthChat target manifests rather than ad hoc CLI JSON blobs when testing the refreshed tool surface.

Current manifests:

- `SynthChat/config/targets_cli_existing_tools_quickcheck.json`
  - Purpose: fastest possible end-to-end check that the highest-risk existing regeneration paths still produce valid examples.
  - Includes: `contentManager_replace`, `promptManager_executePrompts`
  - Count: `2` total examples

- `SynthChat/config/targets_cli_existing_tools_representative.json`
  - Purpose: one representative example for every tool family we currently have source examples for in the existing tool datasets.
  - Excludes future-only families like `taskManager` and `canvasManager`, and excludes current-schema tools with no existing source examples yet such as `contentManager_setProperty`, `memoryManager_runWorkflow`, and `promptManager_subagent`.
  - Count: `29` total examples

Recommended first dry-run commands:

```bash
python -m SynthChat.run generate \
  --targets-file SynthChat/config/targets_cli_existing_tools_quickcheck.json \
  --max-iterations 3 \
  --output Datasets/synthchat/dryrun_cli_existing_tools_quickcheck.jsonl
```

```bash
python -m SynthChat.run generate \
  --targets-file SynthChat/config/targets_cli_existing_tools_representative.json \
  --max-iterations 3 \
  --output Datasets/synthchat/dryrun_cli_existing_tools_representative.jsonl
```

After generation, convert the JSONL to Markdown and review before any broader run.

## Follow-On Work

After the `non_thinking` dataset migration is complete, update SynthChat configs first, then update evaluation and validation.

### Follow-On Phase A: SynthChat Config Alignment

Before evaluator work, align the synthetic data generation layer to the new tool schemas.

Primary areas:

- `SynthChat/scenarios/*`
- `SynthChat/rubrics/*`
- `SynthChat/config/settings.yaml`
- `SynthChat/config/tool_call_formats.yaml`

Objectives:

- Replace stale tool identifiers such as `contentManager_update`, `contentManager_createContent`, and other legacy manager-tool names still embedded in scenarios and rubrics.
- Update scenario `tool` and `expected_tools` fields to the current canonical names.
- Update rubric tables, examples, and validation guidance so they describe the new tool surfaces rather than the legacy ones.
- Decide which wrapper assumptions remain valid in SynthChat and which schema-specific examples need to change.

Current inventory artifacts:

- [Datasets/tools_datasets/reports/cli_schema/synthchat_config_inventory.json](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/Datasets/tools_datasets/reports/cli_schema/synthchat_config_inventory.json)
- [Datasets/tools_datasets/reports/cli_schema/synthchat_config_inventory.md](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/Datasets/tools_datasets/reports/cli_schema/synthchat_config_inventory.md)

Initial inventory highlights:

- `18` SynthChat files with stale tool/schema references
- `14` references to `contentManager_update`
- `3` references to `contentManager_createContent`
- `3` references to `contentManager_replaceContent`
- `3` references to `promptManager_executePrompt`
- `2` `searchMemory` references that still mention `workspaceId`

### Follow-On Phase B: Evaluator Alignment

Only after SynthChat config alignment is stable should evaluator work begin.

Expected follow-on areas:

- `Evaluator/config/scenarios/*` fixtures and expected-tool lists
- Validator and runtime-check flows that assume the legacy wrapper or legacy tool names
