# Spec: vaultLibrarian Dataset Enhancement

## Source
- Input: `Datasets/tools_datasets/vaultLibrarian/tools_v1.1.jsonl` (844 examples)
- Output: `Datasets/tools_datasets/vaultLibrarian/tools_v1.2.jsonl`

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

## Key Insight
vaultLibrarian is about FINDING things. Clarification should focus on:
- What to search for (query terms)
- Where to search (paths, scope)
- What kind of results (content vs names, file types)
- NOT about destructive operations (that's vaultManager)

## Examples to Create

### Clarification Examples (~45)

**searchContent scenarios (20):**
- "Find my notes about X" → ask which folders, how recent
- "Search for that thing" → ask what keywords
- "Look for the meeting notes" → ask which meeting, date range
- "Find files mentioning the project" → ask which project name
- "Search my vault" → ask for keywords, scope
- "Where did I write about Y" → ask if want content or just paths
- "Find everything about Z" → ask if that's the exact keyword
- "Look for recent changes" → ask what topic, how recent

**searchDirectory scenarios (15):**
- "Find all markdown files" → ask in which folders
- "List the PDFs" → ask where to look
- "What files are in Projects" → ask how deep, any filters
- "Find files from last week" → ask which folders, file types
- "Show me the images" → ask which directories
- "Find large files" → ask what size threshold, where

**searchMemory scenarios (10):**
- "What did we do last session" → ask which session if multiple
- "Find my saved states" → ask any filters, tags
- "Search my history" → ask what to search for
- "What was I working on" → ask time frame, workspace

### Summary Examples (~35)

**searchContent summaries (15):**
```
User: Result: {"success": true, "results": [{"path": "Projects/ai-research/notes.md", "score": 0.95}, {"path": "Archive/old-research.md", "score": 0.72}], "count": 2, "query": "neural networks"}
Assistant: # Search Complete

**Searched for "neural networks"**
- Found 2 matching files

**Results:**
- Projects/ai-research/notes.md (95% match)
- Archive/old-research.md (72% match)

Want me to open either of these?
```

```
User: Result: {"success": true, "results": [], "count": 0, "query": "quantum computing"}
Assistant: # No Results

**Searched for "quantum computing"**
- No matching files found in the searched paths

Try different keywords or expand the search scope?
```

**searchDirectory summaries (12):**
```
User: Result: {"success": true, "files": [{"path": "Images/diagram1.png", "size": 245000}, {"path": "Images/screenshot.png", "size": 189000}], "count": 2, "type": "png"}
Assistant: # Files Found

**Found 2 PNG files**
- Images/diagram1.png (245 KB)
- Images/screenshot.png (189 KB)

Need to do anything with these?
```

```
User: Result: {"success": true, "files": [...], "count": 47, "path": "Projects/"}
Assistant: # Directory Listed

**Found 47 items in Projects/**
- Mix of folders and files
- Sorted by name

Want me to filter these or search within them?
```

**searchMemory summaries (8):**
```
User: Result: {"success": true, "sessions": [{"id": "session_1732400000000_abc", "name": "Research Session", "created": "2024-11-24"}], "count": 1}
Assistant: # Memory Search Results

**Found 1 matching session**
- Research Session (Nov 24, 2024)
- ID: session_1732400000000_abc

Want to load this session?
```

```
User: Result: {"success": true, "states": [{"id": "state_1732300000000_xyz", "name": "Before refactor", "tags": ["backup"]}, {"id": "state_1732350000000_abc", "name": "After cleanup", "tags": ["milestone"]}], "count": 2}
Assistant: # States Found

**Found 2 saved states**
- "Before refactor" (tagged: backup)
- "After cleanup" (tagged: milestone)

Want to restore either of these?
```

## Total New Examples: ~80
