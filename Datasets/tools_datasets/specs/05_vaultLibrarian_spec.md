# Spec: vaultLibrarian Dataset Enhancement (v1.3)

## Source
- Input: `Datasets/tools_datasets/vaultLibrarian/tools_v1.2.jsonl` (915 examples)
- Output: `Datasets/tools_datasets/vaultLibrarian/tools_v1.3.jsonl`

## Task
1. Copy v1.2 to v1.3
2. Hand-write ~200 new behavioral examples and append to v1.3
3. All new examples must have `"label": true` and `"pattern": "text_only"`

## Tools to Focus On

### Primary Focus: Search Clarification
| Tool | Key Params | Clarification Questions |
|------|------------|------------------------|
| `searchContent` | `query`, `paths`, `limit`, `includeContent` | "What keywords?", "Which folders?", "Need full content or just names?" |
| `searchDirectory` | `query`, `paths`, `searchType`, `fileTypes`, `dateRange` | "What pattern?", "Where to look?", "File types?", "Date range?" |
| `searchMemory` | `query`, `workspaceId`, `memoryTypes` | "What to search for?", "Sessions, states, or both?" |

### Secondary: Batch Operations
| Tool | Key Params | Clarification Questions |
|------|------------|------------------------|
| `batch` | `searches`, `mergeResults` | "What searches?", "Combine results?" |
| `batchUpdate` | `paths`, `operation` | "Which files?", "What operation?" |

## Key Insight
vaultLibrarian is about FINDING things. Clarification should focus on:
- What to search for (query terms)
- Where to search (paths, scope)
- What kind of results (content vs names, file types)
- NOT about destructive operations (that's vaultManager)

## Behavioral Examples to Create (~200 total)

### Behavior 1: Uncertainty/Clarification (~90 examples)
Model asks clarifying questions when search request is vague. Assistant response is TEXT ONLY (no tool calls).

**searchContent scenarios (~40):**
- "Find my notes about X" → ask which folders, how recent
- "Search for that thing" → ask what keywords
- "Look for the meeting notes" → ask which meeting, date range
- "Find files mentioning the project" → ask which project name
- "Search my vault" → ask for keywords, scope
- "Where did I write about Y" → ask if want content or just paths
- "Find everything about Z" → ask if that's the exact keyword
- "Look for recent changes" → ask what topic, how recent
- "Find that idea I had" → ask any keywords you remember

**searchDirectory scenarios (~30):**
- "Find all markdown files" → ask in which folders
- "List the PDFs" → ask where to look
- "What files are in Projects" → ask how deep, any filters
- "Find files from last week" → ask which folders, file types
- "Show me the images" → ask which directories
- "Find large files" → ask what size threshold, where
- "List everything in Archive" → ask file types, depth

**searchMemory scenarios (~20):**
- "What did we do last session" → ask which session if multiple
- "Find my saved states" → ask any filters, tags
- "Search my history" → ask what to search for
- "What was I working on" → ask time frame, workspace
- "Find that checkpoint" → ask name or tags

### Behavior 2: Pause Before Batch Operations (~30 examples)
Model confirms before batch operations that affect many files.

**Batch update confirmations (~20):**
```
User: Tag all files in Projects with #active
Assistant: I found 47 files in Projects/. I'll add the #active tag to all of them.

Should I proceed with all 47 files, or would you like to filter this list first?
```

```
User: Update all the links in my vault
Assistant: This will scan and update links across your entire vault. This could affect hundreds of files.

Want me to start with a specific folder, or show you a preview first?
```

**Batch search confirmations (~10):**
```
User: Search for everything everywhere
Assistant: A vault-wide search with no filters could return thousands of results and take a while.

Can you give me some keywords or narrow down the folders to search?
```

### Behavior 3: Summary After Results (~80 examples)
User message contains a Result JSON from a tool call. Assistant provides helpful summary.

**searchContent summaries - results found (~25):**
```
User: Result: {"success": true, "results": [{"path": "Projects/ai-research/notes.md", "score": 0.95}, {"path": "Archive/old-research.md", "score": 0.72}], "count": 2, "query": "neural networks"}
Assistant: Found 2 files matching "neural networks":

1. **Projects/ai-research/notes.md** (95% match)
2. **Archive/old-research.md** (72% match)

Want me to open either of these?
```

**searchContent summaries - no results (~10):**
```
User: Result: {"success": true, "results": [], "count": 0, "query": "quantum computing"}
Assistant: No results for "quantum computing".

Try different keywords? I could search for related terms or check specific folders.
```

**searchContent summaries - many results (~10):**
```
User: Result: {"success": true, "results": [...], "count": 47, "query": "project"}
Assistant: Found 47 files matching "project" - that's a lot!

Want me to:
- Filter by folder?
- Sort by date?
- Narrow down with more keywords?
```

**searchDirectory summaries (~20):**
```
User: Result: {"success": true, "files": [{"path": "Images/diagram1.png", "size": 245000}, {"path": "Images/screenshot.png", "size": 189000}], "count": 2, "type": "png"}
Assistant: Found 2 PNG files:

- Images/diagram1.png (245 KB)
- Images/screenshot.png (189 KB)

Need to do anything with these?
```

**searchMemory summaries (~15):**
```
User: Result: {"success": true, "sessions": [{"id": "session_1732400000000_abc", "name": "Research Session", "created": "2024-11-24"}], "count": 1}
Assistant: Found 1 session: **Research Session** (Nov 24, 2024)

Want to load this session?
```

```
User: Result: {"success": true, "states": [{"id": "state_1732300000000_xyz", "name": "Before refactor", "tags": ["backup"]}, {"id": "state_1732350000000_abc", "name": "After cleanup", "tags": ["milestone"]}], "count": 2}
Assistant: Found 2 saved states:

1. **Before refactor** (tagged: backup)
2. **After cleanup** (tagged: milestone)

Which would you like to restore?
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

Vary timestamps, workspace names, search queries, result counts across examples.

## Total New Examples: ~200
