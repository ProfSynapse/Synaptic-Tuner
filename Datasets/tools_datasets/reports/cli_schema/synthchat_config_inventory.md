# SynthChat CLI-Schema Reference Inventory

Generated: 2026-04-21T13:28:38.543594+00:00

- Files with hits: 21

## Special Pattern Counts

- `memory_update_workspace`: 4
- `prompt_execute_prompts`: 6
- `useTools_wrapper`: 101

## Top Stale Tool References


## Impacted Files

- `SynthChat/config/settings.yaml`
  patterns: {'prompt_execute_prompts': 1}
- `SynthChat/config/targets_cli_existing_tools_quickcheck.json`
  patterns: {'prompt_execute_prompts': 1}
- `SynthChat/config/targets_cli_existing_tools_representative.json`
  patterns: {'memory_update_workspace': 1, 'prompt_execute_prompts': 1}
- `SynthChat/config/tool_call_formats.yaml`
  patterns: {'useTools_wrapper': 2}
- `SynthChat/config/workspace_formats.yaml`
  patterns: {'useTools_wrapper': 1}
- `SynthChat/rubrics/commandManager_tools.yaml`
  patterns: {'useTools_wrapper': 11}
- `SynthChat/rubrics/contentManager_tools.yaml`
  patterns: {'useTools_wrapper': 10}
- `SynthChat/rubrics/content_writing_quality.yaml`
  patterns: {'useTools_wrapper': 3}
- `SynthChat/rubrics/createState_transform.yaml`
  patterns: {'useTools_wrapper': 2}
- `SynthChat/rubrics/destructive_safety.yaml`
- `SynthChat/rubrics/memoryManager_tools.yaml`
  patterns: {'useTools_wrapper': 10}
- `SynthChat/rubrics/promptManager_tools.yaml`
  patterns: {'useTools_wrapper': 10}
- `SynthChat/rubrics/searchManager_tools.yaml`
  patterns: {'useTools_wrapper': 11}
- `SynthChat/rubrics/storageManager_tools.yaml`
  patterns: {'useTools_wrapper': 12}
- `SynthChat/rubrics/tool_alignment.yaml`
- `SynthChat/scenarios/content_writing.yaml`
  patterns: {'useTools_wrapper': 6}
- `SynthChat/scenarios/destructive.yaml`
- `SynthChat/scenarios/tool_environments.yaml`
  patterns: {'useTools_wrapper': 5}
- `SynthChat/scenarios/tools.yaml`
  patterns: {'memory_update_workspace': 3, 'prompt_execute_prompts': 3, 'useTools_wrapper': 2}
- `SynthChat/scenarios/vault_kto_pilot.yaml`
  patterns: {'useTools_wrapper': 8}
- `SynthChat/scenarios/vault_shared_seed.yaml`
  patterns: {'useTools_wrapper': 8}
