# System Prompt XML Format

When adding system prompts to behavioral datasets, follow this XML structure.

## Format

```xml
<session_context>
IMPORTANT: When using tools, include these values in your tool call parameters:

- sessionId: "session_XXXXXXXXXXXXX_XXXXXXXXX"
- workspaceId: "ws_XXXXXXXXXXXXX_XXXXXXXXX" (current workspace)

Include these in the "context" parameter of your tool calls.
</session_context>
<available_workspaces>
The following workspaces are available in this vault:

- Workspace Name (id: "ws_XXXXXXXXXXXXX_XXXXXXXXX")
  Description: What this workspace is for
  Root folder: FolderName/

Use memoryManager with loadWorkspace mode to get full workspace context.
</available_workspaces>
```

## Optional: Available Agents Section

Only include when the example involves agent operations:

```xml
<available_agents>
The following custom agents are available:

- Agent Name (id: "agent_XXXXXXXXXXXXX_XXXXXXXXX")
  Description of what the agent does

</available_agents>
```

## Key Points

1. **Backfill IDs**: Look at the tool calls in the assistant response. If they contain `sessionId`, `workspaceId`, or `agentId` values, use those EXACT IDs in the system prompt.

2. **Contextual Content**: The workspace names, descriptions, and folder paths should be contextually appropriate for the example.

3. **Output Format**: Add system message as FIRST item in conversations array:
```json
{
  "conversations": [
    {"role": "system", "content": "<session_context>...</session_context>\n<available_workspaces>...</available_workspaces>"},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": null, "tool_calls": [...]}
  ],
  "label": true,
  "behavior": "category_name"
}
```

4. **ID Format**:
   - Session: `session_1732XXXXXXXXX_xxxxxxxxx` (13 digits + 9 alphanumeric)
   - Workspace: `ws_1732XXXXXXXXX_xxxxxxxxx`
   - Agent: `agent_1732XXXXXXXXX_xxxxxxxxx`
