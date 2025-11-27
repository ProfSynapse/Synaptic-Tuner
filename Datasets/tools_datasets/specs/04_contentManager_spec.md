# Spec: contentManager Dataset Enhancement (v1.3)

## Source
- Input: `Datasets/tools_datasets/contentManager/tools_v1.2.jsonl` (1,251 examples)
- Output: `Datasets/tools_datasets/contentManager/tools_v1.3.jsonl`

## Task
1. Copy v1.2 to v1.3
2. Hand-write ~250 new behavioral examples and append to v1.3
3. All new examples must have `"label": true` and `"pattern": "text_only"`

## Tools to Focus On

### High Priority (Destructive/Modifying)
| Tool | Key Params | Clarification Questions |
|------|------------|------------------------|
| `replaceContent` | `filePath`, `oldContent`, `newContent` | "Which file?", "What text to find?", "Replace with what?" |
| `deleteContent` | `filePath`, `content` | "Which file?", "What content to remove?" |
| `appendContent` | `filePath`, `content`, `position` | "Which file?", "What to add?", "Beginning or end?" |

### Medium Priority
| Tool | Key Params | Clarification Questions |
|------|------------|------------------------|
| `createContent` | `filePath`, `content` | "What filename?", "Where?", "What content?" |
| `readContent` | `filePath`, `includeLineNumbers` | "Which file?" |
| `updateContent` | `filePath`, `content` | "Which file?", "Full replacement or partial?" |

## Behavioral Examples to Create (~250 total)

### Behavior 1: Uncertainty/Clarification (~100 examples)
Model asks clarifying questions when user request is vague. Assistant response is TEXT ONLY (no tool calls).

**replaceContent scenarios (~40):**
- "Replace that text" → ask which file, what to find, what to replace with
- "Change the old value to the new one" → ask specifics
- "Fix the typo in my notes" → ask which file, which typo
- "Update the config value" → ask which config, which value
- "Swap X for Y" → ask in which file(s)
- "Change all instances of..." → ask which files to search
- "Fix the spelling mistake" → ask which file, which word

**deleteContent scenarios (~25):**
- "Remove that paragraph" → ask which file, which paragraph
- "Delete the old section" → ask which section, which file
- "Take out the duplicate content" → ask where, what content
- "Clear that part" → ask specifics
- "Remove the header" → ask which file, which header

**appendContent scenarios (~20):**
- "Add this to my notes" → ask which notes file, where to add
- "Put this at the end" → ask which file
- "Insert a section" → ask where, what content
- "Add a paragraph about X" → ask which file, where to insert
- "Append to the log" → ask which log file

**createContent scenarios (~15):**
- "Create a new note" → ask filename, location, initial content
- "Make a file for X" → ask details
- "New document please" → ask what kind, where
- "Start a new meeting notes file" → ask date, attendees, location

### Behavior 2: Pause Before Destructive (~50 examples)
Model confirms before replacing or deleting content in important files.

**Replace confirmations (~25):**
```
User: Replace all instances of "draft" with "final" in the document
Assistant: I'll replace all instances of "draft" with "final" in **report.md**. This will affect every occurrence in the file.

Want me to show you how many matches there are first, or proceed with the replacement?
```

**Delete confirmations (~15):**
```
User: Delete the introduction section
Assistant: I'll remove the **Introduction** section from Notes/paper.md. This content will be permanently deleted.

Should I proceed, or would you like to move it to another file instead?
```

**Overwrite confirmations (~10):**
```
User: Update the entire file with this new content
Assistant: This will completely replace the current content of **config.yaml**. The existing content will be lost.

Want me to back up the current version first?
```

### Behavior 3: Summary After Results (~100 examples)
User message contains a Result JSON from a tool call. Assistant provides helpful summary.

**replaceContent summaries (~40):**
```
User: Result: {"success": true, "file": "config/settings.yaml", "replacements": 3}
Assistant: Updated **config/settings.yaml**:
- Made 3 replacements

Want me to show you the updated file?
```

```
User: Result: {"success": true, "file": "Notes/draft.md", "replacements": 0}
Assistant: No matches found in **Notes/draft.md**.

The text you wanted to replace wasn't found. Want to try different search text?
```

**deleteContent summaries (~25):**
```
User: Result: {"success": true, "file": "Notes/draft.md", "removed": "## Old Section\n\nThis content was outdated..."}
Assistant: Removed content from **Notes/draft.md**:
- Deleted the "Old Section" and its content

Need any other edits?
```

**appendContent summaries (~20):**
```
User: Result: {"success": true, "file": "Projects/log.md", "appended": true, "position": "end"}
Assistant: Added content to **Projects/log.md** at the end.

Anything else to add?
```

**createContent summaries (~15):**
```
User: Result: {"success": true, "created": "Notes/meeting-2024-11-26.md", "size": 245}
Assistant: Created **Notes/meeting-2024-11-26.md** (245 bytes).

The note is ready. Want to add more content?
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

Vary timestamps, workspace names, file paths, content types across examples.

## Total New Examples: ~250
