# Spec: agentManager Dataset Enhancement

## Source
- Input: `Datasets/tools_datasets/agentManager/tools_v1.1.jsonl` (769 examples)
- Output: `Datasets/tools_datasets/agentManager/tools_v1.2.jsonl`

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
When agents are available, include them in system message:
```
<available_agents>
The following custom agents are available:

- Research Assistant (id: "agent_1732100000000_research")
  Custom agent for research and summarization tasks

- Code Reviewer (id: "agent_1732100000001_codervw")
  Reviews code and suggests improvements
</available_agents>
```

## Examples to Create

### Clarification Examples (~35)

**executePrompt scenarios (15):**
- "Use an agent for this" → ask which agent, what task
- "Have the AI help me" → ask what kind of help, which agent
- "Run this through an agent" → ask which agent, what "this" refers to
- "Get some help analyzing this" → ask what to analyze, which agent
- "Ask the agent about X" → ask which agent if multiple exist

**createAgent scenarios (10):**
- "Make me a new agent" → ask name, purpose, instructions
- "Create an assistant" → ask what kind, what it should do
- "Set up an agent for research" → ask specific research focus, behavior
- "I need a writing helper" → ask writing style, what it should help with

**deleteAgent scenarios (5):**
- "Delete that agent" → ask which one
- "Remove the old agents" → ask which ones count as "old"
- "Get rid of the test agent" → confirm which test agent

**updateAgent scenarios (5):**
- "Change the agent's instructions" → ask which agent, what changes
- "Update the research agent" → ask what to update
- "Make the agent better at X" → ask how specifically

### Summary Examples (~35)

**executePrompt summaries (15):**
```
User: Result: {"success": true, "response": "Based on my analysis of the three files...", "agent": "research_assistant", "tokensUsed": 1547}
Assistant: # Analysis Complete

**Research Assistant processed your request**
- Analyzed 3 files as requested
- Used 1,547 tokens
- Response ready below

Would you like me to save this analysis to a note?
```

**createAgent summaries (10):**
```
User: Result: {"success": true, "agent": {"id": "agent_1732500000000_writhelp", "name": "Writing Helper", "isEnabled": true}}
Assistant: # Agent Created

**Writing Helper is ready**
- ID: agent_1732500000000_writhelp
- Status: Enabled and available

You can now use this agent with executePrompt.
```

**deleteAgent summaries (5):**
```
User: Result: {"success": true, "deleted": "agent_1732100000000_oldtest"}
Assistant: # Agent Removed

**Deleted agent_1732100000000_oldtest**
- Agent permanently removed
- Any saved prompts for this agent still exist

Need to manage any other agents?
```

**updateAgent summaries (5):**
```
User: Result: {"success": true, "updated": {"id": "agent_1732100000000_research", "fields": ["prompt", "description"]}}
Assistant: # Agent Updated

**Modified Research Assistant**
- Updated: prompt, description
- Changes take effect immediately

Want to test the updated agent?
```

## Total New Examples: ~70
