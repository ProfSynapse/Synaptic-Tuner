# Claudesidian-MCP Tool Schema Verification Reference

**Purpose**: Single source of truth for actual tool schemas extracted from the claudesidian-mcp codebase. Used to validate synthetic examples before generation.

**Last Updated**: 2025-11-07

---

## ContentManager Agent

### contentManager_appendContent

**File**: `.obsidian/plugins/claudesidian-mcp/src/agents/contentManager/modes/appendContentMode.ts`

**Parameters**:
- `filePath` (string, REQUIRED) - Path to the file to append to
- `content` (string, REQUIRED) - Content to append to the file
- `context` (object, REQUIRED) - Standard context object

**Result**:
```json
{
  "success": boolean,
  "error": "string (if success=false)",
  "data": {
    "filePath": "string",
    "appendedLength": "number",
    "totalLength": "number"
  },
  "workspaceContext": {
    "workspaceId": "string",
    "workspacePath": ["string"],
    "activeWorkspace": boolean
  }
}
```

### contentManager_createContent

**File**: `.obsidian/plugins/claudesidian-mcp/src/agents/contentManager/modes/createContentMode.ts`

**Parameters**:
- `filePath` (string, REQUIRED) - Path to the file to create
- `content` (string, REQUIRED) - Content to write to the file
- `context` (object, REQUIRED) - Standard context object

**Result**:
```json
{
  "success": boolean,
  "error": "string (if success=false)",
  "data": {
    "filePath": "string",
    "created": "timestamp (number)"
  },
  "workspaceContext": { /* as above */ }
}
```

**Notes**: Returns creation timestamp in `created` field, NOT `createdAt`

---

## VaultManager Agent

### vaultManager_createFolder

**File**: `.obsidian/plugins/claudesidian-mcp/src/agents/vaultManager/modes/createFolderMode.ts`

**Parameters**:
- `path` (string, REQUIRED) - Path of the folder to create
- `context` (object, REQUIRED) - Standard context object

**Result**:
```json
{
  "success": boolean,
  "error": "string (if success=false)",
  "data": {
    "path": "string",
    "existed": "boolean (whether folder already existed)"
  }
}
```

### vaultManager_moveNote

**File**: `.obsidian/plugins/claudesidian-mcp/src/agents/vaultManager/modes/moveNoteMode.ts`

**Parameters**:
- `path` (string, REQUIRED) - Path to the note
- `newPath` (string, REQUIRED) - New path for the note
- `overwrite` (boolean, OPTIONAL) - Whether to overwrite if note exists at new path
- `context` (object, REQUIRED) - Standard context object

**Result**:
```json
{
  "success": boolean,
  "error": "string (if success=false)",
  "path": "string",
  "newPath": "string",
  "recommendations": [{
    "type": "string",
    "message": "string"
  }]
}
```

**Notes**: Returns `recommendations` array (formerly called nudges), not in `data` wrapper

---

## VaultLibrarian Agent

### vaultLibrarian_searchContent

**File**: `.obsidian/plugins/claudesidian-mcp/src/agents/vaultLibrarian/modes/searchContentMode.ts`

**Parameters**:
- `query` (string, REQUIRED) - Search query (cannot be empty)
- `limit` (number, OPTIONAL, default 10) - Max results to return
- `includeContent` (boolean, OPTIONAL, default true) - Whether to include file content in results
- `snippetLength` (number, OPTIONAL, default 200) - Length of content snippet
- `paths` (string[], OPTIONAL) - Filter by specific paths
- `context` (object, REQUIRED) - Standard context object

**Result**:
```json
{
  "success": boolean,
  "query": "string",
  "results": [{
    "filePath": "string",
    "title": "string",
    "content": "string",
    "score": "number (0-1)",
    "searchMethod": "fuzzy|keyword|combined",
    "frontmatter": "object (optional)",
    "metadata": {
      "fileExtension": "string",
      "parentFolder": "string",
      "modifiedTime": "number (timestamp)"
    }
  }],
  "totalResults": "number",
  "executionTime": "number (milliseconds)",
  "error": "string (optional, only if query empty)"
}
```

**Notes**: 
- Does NOT wrap data in `data` object
- Returns `results` array at top level
- Includes `executionTime` in milliseconds
- If query is empty, returns error message

---

## MemoryManager Agent

### memoryManager_createSession

**File**: `.obsidian/plugins/claudesidian-mcp/src/agents/memoryManager/modes/sessions/CreateSessionMode.ts`

**Parameters**:
- `name` (string, REQUIRED) - Session name
- `description` (string, OPTIONAL) - Session description
- `sessionGoal` (string, OPTIONAL) - Goal for the session
- `generateContextTrace` (boolean, OPTIONAL, default true) - Whether to generate memory traces
- `workspaceContext` (object, OPTIONAL) - Workspace to bind session to
- `context` (object, REQUIRED) - Standard context object

**Result**:
```json
{
  "success": boolean,
  "error": "string (if success=false)",
  "data": {
    "sessionId": "string (format: session_${timestamp}_${randomString})",
    "name": "string",
    "workspaceId": "string",
    "createdAt": "number (timestamp)"
  },
  "workspaceContext": {
    "workspaceId": "string",
    "workspacePath": ["string"],
    "activeWorkspace": boolean
  }
}
```

---

## Standard Context Object (All Tools)

**Required fields**:
```json
{
  "sessionId": "string (format: session_${timestamp}_${randomString})",
  "workspaceId": "string (format: ws_${timestamp}_${randomString})",
  "sessionDescription": "string",
  "sessionMemory": "string (accumulated context from prior calls)",
  "toolContext": "string (why this tool is being used)",
  "primaryGoal": "string (user's main objective)",
  "subgoal": "string (specific subgoal for this step)"
}
```

**Format Examples**:
- SessionId: `session_1731004200001_h5k2n7m3p`
- WorkspaceId: `ws_1731004150000_creative_prime_xyz`

---

## Common Error Messages

### Missing Parameters
- "Path is required"
- "File path is required"
- "Content is required"
- "Query parameter is required and cannot be empty"
- "Name is required"

### File Not Found
- "Note not found at path: {path}"
- "File not found: {path}"

### Invalid Paths
- "Invalid path: {path}"
- "Path cannot contain '..'"
- "Cannot access system folder"

### File Operations
- "File already exists: {path}"
- "Cannot overwrite existing file"

### Search Errors
- "No results found for query: {query}"
- "Invalid search limit: must be between 1 and 100"

---

## Key Discrepancies Found (2025-11-07 Audit)

1. **CreateContent result field**: Uses `created` (timestamp), NOT `createdAt`
2. **SearchContent result structure**: Does NOT wrap in `data` object - results are at top level
3. **MoveNote recommendations**: Returns `recommendations` array, not wrapped in `data`
4. **SearchContent timing**: Returns `executionTime` (milliseconds), NOT `duration`
5. **MemoryManager createSession**: Returns `createdAt` in data, sessionId format is specific

---

## Next Steps

- [ ] Complete audit of all 8 existing examples in copilot.jsonl
- [ ] Fix discrepancies
- [ ] Resume new example generation using verified schemas
