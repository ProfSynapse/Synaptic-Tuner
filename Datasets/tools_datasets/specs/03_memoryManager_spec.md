# Spec: memoryManager Dataset Enhancement

## Source
- Input: `Datasets/tools_datasets/memoryManager/tools_v1.1.jsonl` (1,341 examples)
- Output: `Datasets/tools_datasets/memoryManager/tools_v1.2.jsonl`

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

### Lower Priority (Update/List)
| Tool | Key Params | Clarification Questions |
|------|------------|------------------------|
| `updateWorkspace` | `workspaceId`, `fieldPath`, `newValue` | "Which workspace?", "What to change?" |
| `updateSession` | `sessionId`, `name`, `description` | "Which session?", "What to update?" |

## Examples to Create

### Clarification Examples (~40)

**createWorkspace scenarios (15):**
- "Create a workspace" → ask name, root folder, purpose
- "Set up a new workspace for my project" → ask which project, what folder
- "I need a workspace" → ask what for, where to root it
- "Make me a research workspace" → ask specific research area, folder location
- "New workspace please" → ask all the basics

**createState scenarios (10):**
- "Save my progress" → ask what to name it, any tags
- "Create a checkpoint" → ask description, what context to capture
- "Bookmark this state" → ask name, reason for saving
- "Save where I am" → ask what to call it

**createSession scenarios (8):**
- "Start a new session" → ask purpose, goal
- "New session for X" → ask more details about X
- "Begin fresh" → ask if they want to link to previous session

**loadWorkspace/Session/State scenarios (7):**
- "Load my workspace" → ask which one if multiple exist
- "Go back to where I was" → ask which state/session
- "Open the research workspace" → confirm which one if ambiguous

### Summary Examples (~40)

**createWorkspace summaries (15):**
```
User: Result: {"success": true, "workspace": {"id": "ws_1732500000000_research", "name": "Research Hub", "rootFolder": "Research/"}}
Assistant: # Workspace Ready

**Created Research Hub**
- ID: ws_1732500000000_research
- Root folder: Research/
- Status: Active

You're now working in this workspace context.
```

**createState summaries (10):**
```
User: Result: {"success": true, "state": {"id": "state_1732500000000_abc", "name": "Pre-refactor checkpoint", "tags": ["backup", "refactor"]}}
Assistant: # State Saved

**Checkpoint created: Pre-refactor checkpoint**
- ID: state_1732500000000_abc
- Tags: backup, refactor
- You can restore this anytime with loadState

Safe to proceed with your changes.
```

**createSession summaries (8):**
```
User: Result: {"success": true, "session": {"id": "session_1732500000000_newproj", "name": "New Project Planning"}}
Assistant: # Session Started

**New Project Planning session active**
- Session ID: session_1732500000000_newproj
- Fresh context initialized

What would you like to work on?
```

**loadWorkspace/Session summaries (7):**
```
User: Result: {"success": true, "loaded": {"type": "workspace", "id": "ws_1732500000000_research", "name": "Research Hub"}}
Assistant: # Workspace Loaded

**Switched to Research Hub**
- Now working in Research/ context
- Previous workspace context cleared

Ready to continue your research work.
```

## Total New Examples: ~80
