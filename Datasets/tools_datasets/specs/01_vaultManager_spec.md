# Spec: vaultManager Dataset Enhancement (v1.3)

## Source
- Input: `Datasets/tools_datasets/vaultManager/tools_v1.2.jsonl` (1,219 examples)
- Output: `Datasets/tools_datasets/vaultManager/tools_v1.3.jsonl`

## Task
1. Copy v1.2 to v1.3
2. Hand-write ~250 new behavioral examples and append to v1.3
3. All new examples must have `"label": true` and `"pattern": "text_only"`

## Tools to Focus On

### High Priority (Destructive)
| Tool | Key Params | Clarification Questions |
|------|------------|------------------------|
| `deleteFolder` | `path`, `recursive` | "Which folder?", "Delete contents too?" |
| `deleteNote` | `path` | "Which file exactly?", "Are you sure?" |
| `moveFolder` | `path`, `newPath` | "Which folder?", "Where to?" |
| `moveNote` | `path`, `newPath` | "Which file?", "Move where?" |

### Medium Priority
| Tool | Key Params | Clarification Questions |
|------|------------|------------------------|
| `createFolder` | `path` | "What name?", "Where?" |
| `createNote` | `path`, `content` | "What title?", "Which folder?" |
| `listDirectory` | `path`, `filter`, `depth` | "Which folder?", "How deep?" |

## Behavioral Examples to Create (~250 total)

### Behavior 1: Uncertainty/Clarification (~100 examples)
Model asks clarifying questions when user request is vague. Assistant response is TEXT ONLY (no tool calls).

**Delete scenarios (~35):**
- "Delete the old files" → ask which files, where, how old
- "Remove that folder" → ask which folder
- "Clean up the project" → ask what counts as cleanup
- "Get rid of duplicates" → ask how to identify duplicates
- "Clear out the archive" → ask which archive, everything in it?

**Move scenarios (~35):**
- "Move my notes somewhere better" → ask where
- "Relocate the archive" → ask which archive, where to
- "Put these files in the right place" → ask which files, what's "right"
- "Archive the old stuff" → ask what's old, where's archive
- "Organize my project files" → ask how to organize

**Create scenarios (~30):**
- "Make me a new folder" → ask name, location
- "Create a notes folder" → ask where, any subfolders
- "Set up project structure" → ask what structure
- "Add a resources folder" → ask where, what goes in it

### Behavior 2: Pause Before Destructive (~50 examples)
Model confirms before irreversible actions, even when request seems specific. Shows what will happen and asks for confirmation.

**Delete confirmations (~25):**
```
User: Delete the Projects/archive folder
Assistant: I'll delete **Projects/archive** and all its contents. This is permanent and cannot be undone.

Should I proceed?
```

**Overwrite/Move confirmations (~25):**
```
User: Move draft.md to replace final.md
Assistant: This will overwrite **final.md** with the contents of draft.md. The original final.md will be lost.

Want me to proceed, or should I back up final.md first?
```

### Behavior 3: Summary After Results (~100 examples)
User message contains a Result JSON from a tool call. Assistant provides helpful summary of what happened.

**Delete summaries (~35):**
```
User: Result: {"success": true, "deleted": "Projects/old-drafts"}
Assistant: Done! Deleted **Projects/old-drafts** and all its contents.

This was permanent. Let me know if you need anything else cleaned up.
```

**Move summaries (~35):**
```
User: Result: {"success": true, "moved": {"from": "Notes/meeting.md", "to": "Archive/meetings/meeting.md"}}
Assistant: Moved **meeting.md** to Archive/meetings/.

The file is now archived. Want me to move any other meeting notes?
```

**Create summaries (~30):**
```
User: Result: {"success": true, "created": "Projects/NewProject/docs"}
Assistant: Created **Projects/NewProject/docs/** folder.

It's ready for use. Want me to add any subfolders or template files?
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

Vary timestamps (13 digits like 1732100000000), workspace names, folder structures across examples.

## Total New Examples: ~250
