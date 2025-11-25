# Context Efficiency Training Dataset

## Purpose

Teach the model to be **context-aware** and operate efficiently within a sliding context window by:
- Reading files strategically (by line, by section, not entire large files)
- Using `includeContent: false` when content isn't needed
- Setting appropriate search limits
- Leveraging sessionMemory instead of re-reading
- Breaking operations into chunks
- Being strategic about what context to load

## Core Concept: Sliding Context Window

The model has **limited context capacity** and should minimize unnecessary context consumption. The context object (especially `sessionMemory`) exists to track what's been done WITHOUT re-reading everything.

## Positive Patterns (Context-Efficient)

### 1. Read by Line, Not Entire File
```json
// GOOD: Read specific lines
{
  "tool_call": "contentManager_readContent",
  "arguments": {
    "filePath": "docs/large-spec.md",
    "offset": 100,
    "limit": 50,
    "includeLineNumbers": true
  }
}

// BAD: Read entire 10,000-line file
{
  "tool_call": "contentManager_readContent",
  "arguments": {
    "filePath": "docs/large-spec.md"
  }
}
```

### 2. Search Without Content When Just Locating
```json
// GOOD: Just find files
{
  "tool_call": "vaultLibrarian_searchContent",
  "arguments": {
    "query": "API documentation",
    "includeContent": false,
    "limit": 20
  }
}

// BAD: Include all content
{
  "tool_call": "vaultLibrarian_searchContent",
  "arguments": {
    "query": "API documentation",
    "includeContent": true,
    "limit": 100
  }
}
```

### 3. Use sessionMemory Instead of Re-reading
```json
// GOOD: Reference prior read in sessionMemory
{
  "sessionMemory": "Previously read lines 1-50 of config.yaml. Found database section at line 23. API section starts at line 45. Now reading API section specifically."
}

// BAD: Re-read entire file
{
  "sessionMemory": "Need to check API configuration.",
  "tool": "readContent on entire config.yaml again"
}
```

### 4. Appropriate Search Limits
```json
// GOOD: Reasonable limit
{
  "tool_call": "vaultLibrarian_searchContent",
  "arguments": {
    "query": "meeting notes",
    "limit": 20
  }
}

// BAD: Excessive limit
{
  "tool_call": "vaultLibrarian_searchContent",
  "arguments": {
    "query": "meeting notes",
    "limit": 500
  }
}
```

### 5. Chunked Operations for Large Tasks
```json
// GOOD: Process in chunks
{
  "sessionMemory": "Processing files 1-10 of 100. Completed analysis of first batch. Next: files 11-20."
}

// BAD: Try to load everything at once
{
  "sessionMemory": "Reading all 100 files simultaneously"
}
```

## Negative Patterns (Context-Wasteful)

### 1. Reading Entire Large Files
- Read 5000-line log file when only need last 50 lines
- Read entire documentation when only need one section
- Load full config when only checking one value

### 2. Unnecessary Content Inclusion
- Search with `includeContent: true` when just need file paths
- List with full content when just need file names
- Include file contents in search when only verifying existence

### 3. Re-reading Instead of Using Context
- Read same file multiple times instead of referencing sessionMemory
- Re-search for information already found
- Don't track progress in sessionMemory

### 4. Excessive Limits
- Search with limit: 1000 when 20 would suffice
- List all sessions when only need recent few
- Batch operations without pagination

### 5. No Chunking for Large Operations
- Try to process 100 files in one operation
- Load entire large dataset into context
- No incremental progress tracking

## Dataset Structure

Generate 140 examples (70 pairs):

### 1. Line-Based Reading (40 examples, 20 pairs)
- Positive: Read specific line ranges for large files
- Negative: Read entire large files when only need section

### 2. Content Inclusion Strategy (30 examples, 15 pairs)
- Positive: Use `includeContent: false` when appropriate
- Negative: Always include content unnecessarily

### 3. SessionMemory Leverage (30 examples, 15 pairs)
- Positive: Track findings in sessionMemory, avoid re-reading
- Negative: Re-read files unnecessarily, poor memory usage

### 4. Search Limits (20 examples, 10 pairs)
- Positive: Appropriate limits (10-50 for most cases)
- Negative: Excessive limits (500+) or no limits

### 5. Chunked Operations (20 examples, 10 pairs)
- Positive: Process large tasks in chunks with progress tracking
- Negative: Try to load everything at once

## Example Scenarios

**Context-Efficient Operations:**
- Reading last 100 lines of 10,000-line log file
- Searching for files without loading content
- Reading specific section of documentation (lines 500-600)
- Checking file existence without reading content
- Processing 100 files in batches of 10
- Tracking progress in sessionMemory instead of re-reading
- Finding line number of error, then reading ±10 lines around it
- Reading table of contents, then specific chapter

**Context-Wasteful Operations:**
- Reading entire changelog when only need latest entry
- Including all file contents when just listing files
- Re-reading config file 3 times instead of using sessionMemory
- Searching with limit: 1000 when only need first few results
- Loading all 50 meeting notes simultaneously
- Reading full specification when only need API section
- No progress tracking in sessionMemory for multi-step tasks

## Key Teaching Points

1. **Assume limited context** - Model should act like it has a sliding window
2. **sessionMemory is your cache** - Use it to track what you've seen
3. **Read strategically** - By line, by section, not everything
4. **includeContent: false** - When you just need to locate files
5. **Appropriate limits** - 20-50 for most searches, not 500+
6. **Chunk large operations** - 10-20 items at a time, track progress
7. **Don't re-read** - If it's in sessionMemory, reference it
8. **Read minimally** - Only load what you need for current step

## Validation

```bash
python tools/validate_syngen.py Datasets/context_efficiency/pairs_v1.0.jsonl
```

## File Format

Same as other datasets:
```json
{
  "conversations": [
    {"role": "user", "content": "Find the error in the 50,000 line log file"},
    {"role": "assistant", "content": "tool_call: contentManager_readContent\narguments: {\"context\": {...}, \"filePath\": \"logs/app.log\", \"offset\": -100, \"limit\": 100, \"includeLineNumbers\": true}"}
  ],
  "label": true,
  "behavior": "context_efficiency",
  "pattern": "line_based_reading"
}
```

## Context Object Requirements

**sessionMemory should show context-awareness:**
- Positive: "Previously read lines 1-50, found config section at line 23. Now reading API section (lines 45-80) instead of re-reading entire file."
- Negative: "Reading config file." (no awareness of what was already loaded)

**toolContext should explain efficiency:**
- Positive: "Using includeContent: false to locate files without loading content, saving context for actual file reading."
- Negative: "Searching for files." (no efficiency consideration)
