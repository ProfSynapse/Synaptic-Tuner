# Spec: agentManager Dataset Enhancement (v1.3)

## Source
- Input: `Datasets/tools_datasets/agentManager/tools_v1.2.jsonl` (845 examples)
- Output: `Datasets/tools_datasets/agentManager/tools_v1.3.jsonl`

## Task
1. Copy v1.2 to v1.3
2. Hand-write ~200 new behavioral examples and append to v1.3
3. All new examples must have `"label": true` and `"pattern": "text_only"`

## Tools to Focus On

### High Priority
| Tool | Key Params | Clarification Questions |
|------|------------|------------------------|
| `executePrompt` | `agent`, `prompt`, `filepaths`, `provider`, `model` | "Which agent?", "What should it do?", "Any files to include?" |
| `createAgent` | `name`, `description`, `prompt` | "What name?", "What's its purpose?", "What instructions?" |
| `deleteAgent` | `id` | "Which agent?" (if multiple), "Are you sure?" |
| `updateAgent` | `id`, `name`, `description`, `prompt` | "Which agent?", "What to change?" |

### Medium Priority
| Tool | Key Params | Clarification Questions |
|------|------------|------------------------|
| `generateImage` | `prompt`, `provider`, `savePath` | "What image?", "Where to save?" |
| `batchExecutePrompt` | `prompts` | "What tasks?", "Same agent for all?" |

## System Context for Agents
System message MUST include available_agents when agents exist:
```
<available_agents>
The following custom agents are available:

- Research Assistant (id: "agent_1732100000000_research")
  Custom agent for research and summarization tasks

- Code Reviewer (id: "agent_1732100000001_codervw")
  Reviews code and suggests improvements
</available_agents>
```

## Behavioral Examples to Create (~200 total)

### Behavior 1: Uncertainty/Clarification (~80 examples)
Model asks clarifying questions when user request is vague. Assistant response is TEXT ONLY (no tool calls).

**executePrompt scenarios (~35):**
- "Use an agent for this" → ask which agent, what task
- "Have the AI help me" → ask what kind of help, which agent
- "Run this through an agent" → ask which agent, what "this" refers to
- "Get some help analyzing this" → ask what to analyze, which agent
- "Ask the agent about X" → ask which agent if multiple exist
- "Process these files" → ask which agent, what processing

**createAgent scenarios (~25):**
- "Make me a new agent" → ask name, purpose, instructions
- "Create an assistant" → ask what kind, what it should do
- "Set up an agent for research" → ask specific research focus, behavior
- "I need a writing helper" → ask writing style, what it should help with
- "Build me a code reviewer" → ask what languages, what to focus on

**deleteAgent scenarios (~10):**
- "Delete that agent" → ask which one
- "Remove the old agents" → ask which ones count as "old"
- "Get rid of the test agent" → confirm which test agent

**updateAgent scenarios (~10):**
- "Change the agent's instructions" → ask which agent, what changes
- "Update the research agent" → ask what to update
- "Make the agent better at X" → ask how specifically
- "Improve the prompt" → ask which agent, what improvement

### Behavior 2: Pause Before Destructive (~40 examples)
Model confirms before deleting agents or overwriting configurations.

**Delete confirmations (~25):**
```
User: Delete the research agent
Assistant: I'll delete the **Research Assistant** agent (agent_1732100000000_research). This will permanently remove its configuration and saved prompts.

Should I proceed?
```

**Overwrite confirmations (~15):**
```
User: Replace the agent's prompt with this new one
Assistant: This will completely replace the current prompt for **Code Reviewer**. The existing instructions will be lost.

Want me to show you the current prompt first, or proceed with the replacement?
```

### Behavior 3: Summary After Results (~80 examples)
User message contains a Result JSON from a tool call. Assistant provides helpful summary.

**executePrompt summaries (~35):**
```
User: Result: {"success": true, "response": "Based on my analysis of the three files...", "agent": "research_assistant", "tokensUsed": 1547}
Assistant: Research Assistant completed the analysis.

- Processed 3 files
- Used 1,547 tokens

Would you like me to save this to a note, or run it on more files?
```

**createAgent summaries (~25):**
```
User: Result: {"success": true, "agent": {"id": "agent_1732500000000_writhelp", "name": "Writing Helper", "isEnabled": true}}
Assistant: Created **Writing Helper** agent!

- ID: agent_1732500000000_writhelp
- Status: Enabled and ready

You can use it with "run Writing Helper on [files]".
```

**deleteAgent summaries (~10):**
```
User: Result: {"success": true, "deleted": "agent_1732100000000_oldtest"}
Assistant: Deleted the agent permanently.

Your other agents are unaffected. Need to manage any others?
```

**updateAgent summaries (~10):**
```
User: Result: {"success": true, "updated": {"id": "agent_1732100000000_research", "fields": ["prompt", "description"]}}
Assistant: Updated **Research Assistant**:
- Modified: prompt, description
- Changes active immediately

Want to test it on a file?
```

## JSONL Format

Each example must be a single JSON line:
```json
{"conversations": [{"role": "system", "content": "SYSTEM_CONTENT"}, {"role": "user", "content": "USER_MESSAGE"}, {"role": "assistant", "content": "ASSISTANT_RESPONSE"}], "label": true, "behavior": "BEHAVIOR_TYPE", "pattern": "text_only"}
```

**behavior** must be one of: `"uncertainty"`, `"pause_destructive"`, `"summary"`

**System content template:**
```
<session_context>
IMPORTANT: When using tools, include these values in your tool call parameters:

- sessionId: "session_TIMESTAMP_9CHARS"
- workspaceId: "ws_TIMESTAMP_9CHARS" (current workspace)

Include these in the "context" parameter of your tool calls.
</session_context>
<available_workspaces>
The following workspaces are available in this vault:

- WORKSPACE_NAME (id: "ws_TIMESTAMP_9CHARS")
  Description: WORKSPACE_DESCRIPTION
  Root folder: FOLDER/

Use memoryManager with loadWorkspace mode to get full workspace context.
</available_workspaces>
<available_agents>
The following custom agents are available:

- AGENT_NAME (id: "agent_TIMESTAMP_NAME")
  AGENT_DESCRIPTION
</available_agents>
```

Vary timestamps, workspace names, agent names across examples.

## Total New Examples: ~200
