# Spec: contentManager Dataset Enhancement

## Source
- Input: `Datasets/tools_datasets/contentManager/tools_v1.1.jsonl` (1,193 examples)
- Output: `Datasets/tools_datasets/contentManager/tools_v1.2.jsonl`

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

## Examples to Create

### Clarification Examples (~30)

**replaceContent scenarios (12):**
- "Replace that text" → ask which file, what to find, what to replace with
- "Change the old value to the new one" → ask specifics
- "Fix the typo in my notes" → ask which file, which typo
- "Update the config value" → ask which config, which value
- "Swap X for Y" → ask in which file(s)

**deleteContent scenarios (8):**
- "Remove that paragraph" → ask which file, which paragraph
- "Delete the old section" → ask which section, which file
- "Take out the duplicate content" → ask where, what content
- "Clear that part" → ask specifics

**appendContent scenarios (5):**
- "Add this to my notes" → ask which notes file, where to add
- "Put this at the end" → ask which file
- "Insert a section" → ask where, what content

**createContent scenarios (5):**
- "Create a new note" → ask filename, location, initial content
- "Make a file for X" → ask details
- "New document please" → ask what kind, where

### Summary Examples (~30)

**replaceContent summaries (12):**
```
User: Result: {"success": true, "file": "config/settings.yaml", "replacements": 3}
Assistant: # Content Updated

**Modified config/settings.yaml**
- Made 3 replacements
- Old text replaced with new text throughout

Want me to show you the updated file?
```

**deleteContent summaries (8):**
```
User: Result: {"success": true, "file": "Notes/draft.md", "removed": "## Old Section\n\nThis content was outdated..."}
Assistant: # Content Removed

**Edited Notes/draft.md**
- Removed the "Old Section" heading and content
- File saved with changes

Need any other edits?
```

**appendContent summaries (5):**
```
User: Result: {"success": true, "file": "Projects/log.md", "appended": true, "position": "end"}
Assistant: # Content Added

**Appended to Projects/log.md**
- New content added at end of file
- File saved

Anything else to add?
```

**createContent summaries (5):**
```
User: Result: {"success": true, "created": "Notes/meeting-2024-11-26.md", "size": 245}
Assistant: # Note Created

**Created Notes/meeting-2024-11-26.md**
- New file with 245 characters
- Ready for editing

Want to add more content?
```

## Total New Examples: ~60
