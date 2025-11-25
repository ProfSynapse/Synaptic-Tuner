# Context Injection Specification

## Purpose

This specification defines how to add system prompts to synthetic training data so models learn to use context provided at runtime rather than hallucinating IDs.

## Background

In production, the `SystemPromptBuilder` provides context to the model via system prompts:
- `<session_context>` - Current sessionId and workspaceId
- `<available_workspaces>` - List of workspaces with IDs, names, descriptions
- `<available_agents>` - List of custom agents with IDs, names, descriptions
- `<vault_structure>` - Root folders and files in the vault

The model should USE these provided IDs in tool calls, not hallucinate new ones.

## System Prompt Format

### Session Context Section
```xml
<session_context>
IMPORTANT: When using tools, include these values in your tool call parameters:

- sessionId: "{sessionId}"
- workspaceId: "{workspaceId}" (current workspace)

Include these in the "context" parameter of your tool calls.
</session_context>
```

### Available Workspaces Section
```xml
<available_workspaces>
The following workspaces are available in this vault:

- {workspace_name} (id: "{workspace_id}")
  Description: {description}
  Root folder: {root_folder}

Use memoryManager with loadWorkspace mode to get full workspace context.
</available_workspaces>
```

### Available Agents Section (when agent IDs are referenced)
```xml
<available_agents>
The following custom agents are available:

- {agent_name} (id: "{agent_id}")
  {description}
</available_agents>
```

## Backfill Algorithm

For each existing training example:

### Step 1: Extract IDs from Tool Call Arguments
Parse the `arguments` JSON from the tool call to extract:
- `context.sessionId`
- `context.workspaceId`
- Any agent IDs (e.g., `id` field for agent operations, `agent` field for executePrompt)

### Step 2: Generate Workspace Name from Context
Use clues from the example to create a realistic workspace name:
- `sessionDescription` often hints at the domain (e.g., "Loading budget workspace" → "Budget Tracker")
- `primaryGoal` provides context (e.g., "Plan AI safety podcast episode" → "Podcast Production")
- User message content (e.g., "Update my budget" → "Budget Tracker")

**Name Generation Rules:**
1. Extract domain keywords from sessionDescription, primaryGoal, or user message
2. Create a professional workspace name (Title Case, 2-3 words)
3. Generate a matching description
4. Derive root folder from the name (e.g., "Budget Tracker" → "Finance/")

### Step 3: Handle Special Cases

**"default" workspaceId:**
- Still include in `<session_context>` as `workspaceId: "default"`
- May or may not include `<available_workspaces>` section
- Add note: "Use 'default' when no specific workspace is selected"

**Agent references:**
- If tool call references an agent ID, include `<available_agents>` section
- Extract agent name from the tool call or generate from context

### Step 4: Build System Prompt
Combine sections in order:
1. `<session_context>` (always)
2. `<available_workspaces>` (when workspaceId is not "default", or randomly for variety)
3. `<available_agents>` (when agent IDs are referenced)

### Step 5: Insert System Message
Add as the first message in the conversation:
```json
{
  "role": "system",
  "content": "{generated_system_prompt}"
}
```

## Implementation Instructions for Agents

Each agent processes one tools_dataset file and outputs a new version with system prompts.

### Input
- `Datasets/tools_datasets/{manager}/tools_v1.0.jsonl`

### Output
- `Datasets/tools_datasets/{manager}/tools_v1.1.jsonl`

### Processing Steps

```python
for each line in input_file:
    example = json.loads(line)
    conversations = example["conversations"]

    # Find the assistant message with tool_calls
    tool_call = find_tool_call(conversations)
    if not tool_call:
        # No tool call, skip or copy as-is
        continue

    # Extract IDs from arguments
    args = json.loads(tool_call["function"]["arguments"])
    context = args.get("context", {})
    session_id = context.get("sessionId", "")
    workspace_id = context.get("workspaceId", "default")

    # Extract agent IDs if present
    agent_id = args.get("id") if "agent" in tool_call["function"]["name"].lower() else None
    agent_name = args.get("agent")  # For executePrompt

    # Generate workspace name from context clues
    workspace_name = generate_workspace_name(context, conversations)
    workspace_desc = generate_workspace_description(context)
    root_folder = generate_root_folder(workspace_name)

    # Build system prompt
    system_prompt = build_system_prompt(
        session_id=session_id,
        workspace_id=workspace_id,
        workspace_name=workspace_name,
        workspace_desc=workspace_desc,
        root_folder=root_folder,
        agent_id=agent_id,
        agent_name=agent_name
    )

    # Insert system message at beginning
    system_message = {"role": "system", "content": system_prompt}
    conversations.insert(0, system_message)

    # Write updated example
    output_file.write(json.dumps(example) + "\n")
```

### Workspace Name Generation Heuristics

Map common keywords to workspace names:

| Keywords in Context | Workspace Name | Root Folder |
|---------------------|----------------|-------------|
| budget, expense, finance | Budget Tracker | Finance/ |
| podcast, episode | Podcast Production | Podcast/ |
| research, paper, study | Research Hub | Research/ |
| project, sprint, release | Project Management | Projects/ |
| recipe, cookbook, meal | Recipe Collection | Recipes/ |
| workout, fitness, exercise | Fitness Tracker | Fitness/ |
| meeting, notes, agenda | Meeting Notes | Meetings/ |
| blog, content, post | Content Hub | Content/ |
| code, dev, programming | Development | Dev/ |
| client, presentation | Client Work | Clients/ |
| learning, course, module | Learning Center | Courses/ |
| pet, health, vet | Pet Care | Pets/ |
| car, maintenance, vehicle | Vehicle Tracker | Vehicles/ |
| wellness, meditation | Wellness Journal | Wellness/ |
| agent, automation | Agent Workspace | Agents/ |
| default (no match) | Personal Notes | Notes/ |

### Agent Name Extraction

For agentManager tools:
- `createAgent`: Use `name` from arguments
- `updateAgent`: Use `id` field, derive name from ID (e.g., "agent_code_reviewer" → "Code Reviewer")
- `executePrompt`: Use `agent` field directly as name
- `deleteAgent`: Use `id` field
- `listAgents`: No specific agent (skip agents section)

## Validation Rules

After processing, validate that:

1. **System prompt exists**: First message has `role: "system"`
2. **Session ID matches**: `sessionId` in tool call matches `<session_context>`
3. **Workspace ID matches**: `workspaceId` in tool call matches either:
   - The ID in `<session_context>`, OR
   - An ID listed in `<available_workspaces>`
4. **Agent ID matches** (if applicable): Agent ID in tool call matches `<available_agents>`

## Example Transformations

### Before (current format)
```json
{
  "conversations": [
    {"role": "user", "content": "Show me my current budget status"},
    {"role": "assistant", "content": null, "tool_calls": [{
      "function": {
        "name": "memoryManager_loadWorkspace",
        "arguments": "{\"context\": {\"sessionId\": \"session_1732300800000_a1b2c3d4e\", \"workspaceId\": \"ws_1732300800000_f5g6h7i8j\", \"sessionDescription\": \"Loading budget workspace\"}, \"id\": \"ws_1732300800000_f5g6h7i8j\"}"
      }
    }]}
  ]
}
```

### After (with system prompt)
```json
{
  "conversations": [
    {"role": "system", "content": "<session_context>\nIMPORTANT: When using tools, include these values in your tool call parameters:\n\n- sessionId: \"session_1732300800000_a1b2c3d4e\"\n- workspaceId: \"ws_1732300800000_f5g6h7i8j\" (current workspace)\n\nInclude these in the \"context\" parameter of your tool calls.\n</session_context>\n<available_workspaces>\nThe following workspaces are available in this vault:\n\n- Budget Tracker (id: \"ws_1732300800000_f5g6h7i8j\")\n  Description: Monthly budget and expense tracking\n  Root folder: Finance/\n\nUse memoryManager with loadWorkspace mode to get full workspace context.\n</available_workspaces>"},
    {"role": "user", "content": "Show me my current budget status"},
    {"role": "assistant", "content": null, "tool_calls": [{
      "function": {
        "name": "memoryManager_loadWorkspace",
        "arguments": "{\"context\": {\"sessionId\": \"session_1732300800000_a1b2c3d4e\", \"workspaceId\": \"ws_1732300800000_f5g6h7i8j\", \"sessionDescription\": \"Loading budget workspace\"}, \"id\": \"ws_1732300800000_f5g6h7i8j\"}"
      }
    }]}
  ]
}
```

### Default Workspace Example

```json
{
  "conversations": [
    {"role": "system", "content": "<session_context>\nIMPORTANT: When using tools, include these values in your tool call parameters:\n\n- sessionId: \"session_1731109100000_g9h0i1j2k\"\n- workspaceId: \"default\" (no specific workspace selected)\n\nInclude these in the \"context\" parameter of your tool calls.\nNOTE: Use \"default\" as the workspaceId when no specific workspace context is needed.\n</session_context>"},
    {"role": "user", "content": "Create a new folder called Resources"},
    {"role": "assistant", "content": null, "tool_calls": [{
      "function": {
        "name": "vaultManager_createFolder",
        "arguments": "{\"context\": {\"sessionId\": \"session_1731109100000_g9h0i1j2k\", \"workspaceId\": \"default\", ...}, \"path\": \"Resources\"}"
      }
    }]}
  ]
}
```

## File Outputs

After processing all 5 managers, the following files should exist:
- `Datasets/tools_datasets/agentManager/tools_v1.1.jsonl`
- `Datasets/tools_datasets/contentManager/tools_v1.1.jsonl`
- `Datasets/tools_datasets/memoryManager/tools_v1.1.jsonl`
- `Datasets/tools_datasets/vaultLibrarian/tools_v1.1.jsonl`
- `Datasets/tools_datasets/vaultManager/tools_v1.1.jsonl`

Total expected examples: ~5,289 (same as input, with system prompts added)
