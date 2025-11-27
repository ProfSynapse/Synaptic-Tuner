# Spec: memoryManager Dataset Enhancement (v1.3)

## Source
- Input: `Datasets/tools_datasets/memoryManager/tools_v1.2.jsonl` (1,421 examples)
- Output: `Datasets/tools_datasets/memoryManager/tools_v1.3.jsonl`

## Task
1. Copy v1.2 to v1.3
2. Hand-write ~300 new behavioral examples and append to v1.3
3. All new examples must have `"label": true` and `"pattern": "text_only"`

## Tools to Focus On

### High Priority
| Tool | Key Params | Clarification Questions |
|------|------------|------------------------|
| `createWorkspace` | `name`, `rootFolder`, `purpose`, `currentGoal` | "What name?", "Which folder?", "What's it for?" |
| `createState` | `name`, `description`, `tags` | "What to call this checkpoint?", "Any tags?" |
| `createSession` | `name`, `description`, `sessionGoal` | "What's this session for?", "Any specific goal?" |

### Medium Priority (Load/Restore)
| Tool | Key Params | Clarification Questions |
|------|------------|------------------------|
| `loadWorkspace` | `id` | "Which workspace?" (if multiple) |
| `loadSession` | `sessionId` | "Which session?" |
| `loadState` | `stateId` | "Which checkpoint?" |

### Lower Priority (Update/Delete)
| Tool | Key Params | Clarification Questions |
|------|------------|------------------------|
| `updateWorkspace` | `workspaceId`, `fieldPath`, `newValue` | "Which workspace?", "What to change?" |
| `deleteWorkspace` | `workspaceId` | "Which workspace?", "Are you sure?" |
| `clearSession` | `sessionId` | "Which session?", "Clear everything?" |

## Behavioral Examples to Create (~300 total)

### Behavior 1: Uncertainty/Clarification (~120 examples)
Model asks clarifying questions when user request is vague. Assistant response is TEXT ONLY (no tool calls).

**createWorkspace scenarios (~40):**
- "Create a workspace" → ask name, root folder, purpose
- "Set up a new workspace for my project" → ask which project, what folder
- "I need a workspace" → ask what for, where to root it
- "Make me a research workspace" → ask specific research area, folder location
- "New workspace please" → ask all the basics
- "Set up a workspace for writing" → ask what kind of writing, folder structure

**createState scenarios (~30):**
- "Save my progress" → ask what to name it, any tags
- "Create a checkpoint" → ask description, what context to capture
- "Bookmark this state" → ask name, reason for saving
- "Save where I am" → ask what to call it
- "Make a backup point" → ask description, tags

**createSession scenarios (~25):**
- "Start a new session" → ask purpose, goal
- "New session for X" → ask more details about X
- "Begin fresh" → ask if they want to link to previous session
- "Start over" → ask if they want to save current session first

**loadWorkspace/Session/State scenarios (~25):**
- "Load my workspace" → ask which one if multiple exist
- "Go back to where I was" → ask which state/session
- "Open the research workspace" → confirm which one if ambiguous
- "Resume my work" → ask which session or state
- "Switch to the other workspace" → ask which one specifically

### Behavior 2: Pause Before Destructive (~60 examples)
Model confirms before clearing sessions, deleting workspaces, or resetting states.

**Session clear confirmations (~25):**
```
User: Clear my session
Assistant: I'll clear the current session data. This will reset our conversation context and any unsaved work.

Should I proceed? You might want to save a state first if there's anything important.
```

**Workspace delete confirmations (~20):**
```
User: Delete the old workspace
Assistant: I'll delete the **Old Projects** workspace (ws_1732100000000_oldproj). This removes:
- The workspace configuration
- All saved states associated with it
- Key file associations

The actual files in the folder are NOT deleted. Should I proceed?
```

**State/reset confirmations (~15):**
```
User: Reset everything
Assistant: This will clear all session data and reset to a fresh state. Any unsaved context will be lost.

Want me to save a checkpoint first, or proceed with the reset?
```

### Behavior 3: Summary After Results (~120 examples)
User message contains a Result JSON from a tool call. Assistant provides helpful summary.

**createWorkspace summaries (~40):**
```
User: Result: {"success": true, "workspace": {"id": "ws_1732500000000_research", "name": "Research Hub", "rootFolder": "Research/"}}
Assistant: Created **Research Hub** workspace!

- Root folder: Research/
- Status: Active

You're now working in this context. What would you like to do first?
```

**createState summaries (~30):**
```
User: Result: {"success": true, "state": {"id": "state_1732500000000_abc", "name": "Pre-refactor checkpoint", "tags": ["backup", "refactor"]}}
Assistant: Saved checkpoint: **Pre-refactor checkpoint**

- Tags: backup, refactor
- Restore anytime with loadState

Safe to proceed with your changes.
```

**createSession summaries (~25):**
```
User: Result: {"success": true, "session": {"id": "session_1732500000000_newproj", "name": "New Project Planning"}}
Assistant: Started **New Project Planning** session.

Fresh context ready. What would you like to work on?
```

**loadWorkspace/Session summaries (~25):**
```
User: Result: {"success": true, "loaded": {"type": "workspace", "id": "ws_1732500000000_research", "name": "Research Hub", "keyFiles": ["Research/index.md"]}}
Assistant: Loaded **Research Hub** workspace.

- Root: Research/
- Key file: Research/index.md

Ready to continue. What's next?
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
```

When showing multiple workspaces for disambiguation, list 2-3 options.

Vary timestamps, workspace names, session purposes, state descriptions across examples.

## Total New Examples: ~300
