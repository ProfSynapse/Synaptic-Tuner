# Spec: vaultManager Dataset Enhancement

## Source
- Input: `Datasets/tools_datasets/vaultManager/tools_v1.1.jsonl` (1,139 examples)
- Output: `Datasets/tools_datasets/vaultManager/tools_v1.2.jsonl`

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

## Examples to Create

### Clarification Examples (~40)
Create ~40 handcrafted clarification examples:

**Delete scenarios (15):**
- "Delete the old files" → ask which files, where
- "Remove that folder" → ask which folder
- "Clean up the project" → ask what counts as cleanup
- "Get rid of duplicates" → ask how to identify duplicates
- Vary the vagueness and phrasing

**Move scenarios (15):**
- "Move my notes somewhere better" → ask where
- "Relocate the archive" → ask which archive, where to
- "Put these files in the right place" → ask which files, what's "right"
- "Archive the old stuff" → ask what's old, where's archive

**Create scenarios (10):**
- "Make me a new folder" → ask name, location
- "Create a notes folder" → ask where, any subfolders
- "Set up project structure" → ask what structure

### Summary Examples (~40)
Create ~40 handcrafted summary examples:

**Delete summaries (15):**
```
User: Result: {"success": true, "deleted": "Projects/old-drafts"}
Assistant: # Folder Removed

**Deleted Projects/old-drafts**
- Folder and all contents permanently removed
- This action cannot be undone

Need me to clean up anything else?
```

**Move summaries (15):**
```
User: Result: {"success": true, "moved": {"from": "Notes/meeting.md", "to": "Archive/meetings/meeting.md"}}
Assistant: # Move Complete

**Relocated meeting.md**
- From: Notes/meeting.md
- To: Archive/meetings/meeting.md

The file is now in your archive.
```

**Create summaries (10):**
```
User: Result: {"success": true, "created": "Projects/NewProject/docs"}
Assistant: # Folder Created

**Created Projects/NewProject/docs**
- New folder is ready for use
- Path: Projects/NewProject/docs/

Want me to add any subfolders or files?
```

## Total New Examples: ~80
