# Spec: Add System Context to Behavioral Datasets

## Overview
Add system messages with session/workspace context to datasets that currently lack them.

## Affected Datasets
- `intellectual_humility/pairs_v1.1.jsonl` (258 examples)
- `verification_before_action/pairs_v1.1.jsonl` (254 examples)
- `execute_prompt_usage/pairs_v1.1.jsonl` (140 examples)
- `strategic_tool_selection/pairs_v1.1.jsonl` (TBD)
- `error_recovery/pairs_v1.1.jsonl` (TBD)

## Current State
Examples currently look like:
```json
{
  "conversations": [
    {"role": "user", "content": "Can you delete the old project files?"},
    {"role": "assistant", "content": null, "tool_calls": [{"id": "...", "type": "function", "function": {"name": "vaultLibrarian_searchContent", "arguments": "{\"context\": {\"sessionId\": \"session_1732300800000_a1b2c3d4e\", ...}}"}}]}
  ],
  "label": true,
  "behavior": "intellectual_humility"
}
```

## Required Change
Add a system message at the START of conversations that contains the session/workspace context:

```json
{
  "conversations": [
    {"role": "system", "content": "<session_context>\nIMPORTANT: When using tools, include these values in your tool call parameters:\n\n- sessionId: \"session_1732300800000_a1b2c3d4e\"\n- workspaceId: \"ws_1732300800000_f5g6h7i8j\" (current workspace)\n\nInclude these in the \"context\" parameter of your tool calls.\n</session_context>\n<available_workspaces>\nThe following workspaces are available in this vault:\n\n- Project Management (id: \"ws_1732300800000_f5g6h7i8j\")\n  Description: Project files, development notes, and planning documents\n  Root folder: Projects/\n\nUse memoryManager with loadWorkspace mode to get full workspace context.\n</available_workspaces>"},
    {"role": "user", "content": "Can you delete the old project files?"},
    {"role": "assistant", "content": null, "tool_calls": [...]}
  ],
  "label": true,
  "behavior": "intellectual_humility"
}
```

## Implementation Rules

### 1. Extract IDs from Existing Tool Calls
For each example:
- Parse the `arguments` JSON from the first tool call
- Extract `sessionId` and `workspaceId` from the `context` object
- Use these exact IDs in the system message

### 2. System Message Template
```
<session_context>
IMPORTANT: When using tools, include these values in your tool call parameters:

- sessionId: "{sessionId}"
- workspaceId: "{workspaceId}" (current workspace)

Include these in the "context" parameter of your tool calls.
</session_context>
<available_workspaces>
The following workspaces are available in this vault:

- {workspace_name} (id: "{workspaceId}")
  Description: {workspace_description}
  Root folder: {root_folder}

Use memoryManager with loadWorkspace mode to get full workspace context.
</available_workspaces>
```

### 3. Workspace Names
Generate contextually appropriate workspace names based on the conversation:
- File operations → "Project Management"
- Research questions → "Research & Papers"
- Notes/journal → "Daily Notes"
- Configs → "Configuration"
- Meetings → "Meetings & Collaboration"
- Default → "Main Workspace"

### 4. ID Validation Rules
- sessionId format: `session_\d{13}_[a-z0-9]{9}`
- workspaceId format: `ws_\d{13}_[a-z0-9]{9}`
- The IDs in system message MUST match IDs in tool calls

### 5. For label=false Examples
- Still add system context
- The IDs should still match (even bad examples should have consistent context)
- The "badness" is in the behavior, not in having wrong IDs

## Validation
After processing, run:
```bash
python tools/validate_syngen.py Datasets/behavior_datasets/{dataset}/pairs_v1.2.jsonl
```

Expected output: 0 errors for ID mismatches.

## Output Format
- Create new file: `pairs_v1.2.jsonl` (preserve v1.1 as backup)
- Maintain all existing fields (label, behavior, pattern)
- Only add system message to conversations array
