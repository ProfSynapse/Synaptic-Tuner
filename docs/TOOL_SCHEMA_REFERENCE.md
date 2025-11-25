# Tool Schema Reference - Verified Parameters

**Purpose**: Quick reference for generating training examples with correct parameters
**Source**: `/Users/jrosenbaum/Documents/Code/tools/tool_schemas.json`
**Last Updated**: 2025-11-07

## ContentManager Tools

### contentManager_replaceByLine
**Required Parameters**:
- `filePath` (string)
- `startLine` (number)
- `endLine` (number)
- `newContent` (string)
- `context` (object)

**Optional Parameters**: None

**Notes**: Replace content in specific line range. Both startLine and endLine are required.

---

### contentManager_deleteContent
**Required Parameters**:
- `filePath` (string)
- `content` (string)
- `context` (object)

**Optional Parameters**:
- `similarityThreshold` (number)

**Notes**: Deletes FIRST matching content by default. No `deleteAll` parameter exists.

---

### contentManager_batchContent
**Required Parameters**:
- `operations` (ContentOperation[])
- `context` (object)

**Optional Parameters**: None

**Notes**: Array of operations. Each operation has type and specific parameters.

---

## VaultManager Tools

### vaultManager_duplicateNote
**Required Parameters**:
- `sourcePath` (string)
- `targetPath` (string)
- `context` (object)

**Optional Parameters**:
- `overwrite` (boolean)
- `autoIncrement` (boolean)

**Notes**: NOT `path`/`newPath` - must use `sourcePath`/`targetPath`

---

### vaultManager_editFolder
**Required Parameters**:
- `path` (string)
- `newPath` (string)
- `context` (object)

**Optional Parameters**: None

**Notes**: Requires FULL new path, not just `newName`

---

## VaultLibrarian Tools

### vaultLibrarian_searchDirectory
**Required Parameters**:
- `query` (string)
- `paths` (string[]) ← ARRAY required
- `context` (object)

**Optional Parameters**:
- `searchType` ('files' | 'folders' | 'both')
- `fileTypes` (string[])
- `depth` (number)
- `includeContent` (boolean)

**Notes**: 
- `paths` must be an ARRAY even for single path
- NO `filterType` or `dateRange` parameters exist

---

### vaultLibrarian_searchMemory
**Required Parameters**:
- `query` (string)
- `workspaceId` (string) ← Required!
- `context` (object)

**Optional Parameters**:
- `memoryTypes` (MemoryType[])
- `searchMethod` ('semantic' | 'exact' | 'mixed')

**Notes**: `workspaceId` is REQUIRED, not optional

---

### vaultLibrarian_batch
**Required Parameters**:
- `searches` (UniversalSearchParams[])
- `context` (object)

**Optional Parameters**:
- `mergeResults` (boolean)
- `maxConcurrency` (number)

**Notes**: Array of search operations

---

## MemoryManager Tools

### memoryManager_loadState
**Required Parameters**:
- `stateId` (string) ← Must use ID, not name
- `context` (object)

**Optional Parameters**: None

**Notes**: NO `name` parameter - must use `stateId` from listStates

---

### memoryManager_loadSession
**Required Parameters**:
- `sessionId` (string)
- `context` (object)

**Optional Parameters**: None

---

## AgentManager Tools

### agentManager_executePrompt
**Required Parameters**:
- `prompt` (string)
- `context` (object)

**Optional Parameters**:
- `agent` (string)
- `filepaths` (string[])
- `provider` (string)
- `model` (string)
- `temperature` (number)

**Notes**: 
- NO `content` parameter exists
- `prompt` is required
- `filepaths` (plural) for file context

---

### agentManager_generateImage
**Required Parameters**:
- `prompt` (string)
- `provider` ('google')
- `savePath` (string)
- `context` (object)

**Optional Parameters**:
- `model` ('imagen-4' | 'imagen-4-ultra' | 'imagen-4-fast')
- `aspectRatio` (AspectRatio)
- `numberOfImages` (number)
- `sampleImageSize` ('1K' | '2K')

**Notes**: All three required params must be present

---

### agentManager_createAgent
**Required Parameters**:
- `name` (string)
- `description` (string)
- `prompt` (string)
- `context` (object)

**Optional Parameters**:
- `isEnabled` (boolean)

---

## Common Patterns

### Path Parameters
- `filePath` - Single file reference
- `path` - Generic path reference
- `sourcePath` / `targetPath` - Copy/duplicate operations
- `newPath` - Rename/move target (full path required)
- `savePath` - Save location for generated content

### Array Parameters (MUST be arrays)
- `paths` - Array of path strings
- `filepaths` - Array of file paths
- `managers` - Array of manager names
- `operations` - Array of operation objects
- `searches` - Array of search parameters

### ID Parameters (Always strings)
- `sessionId` - Session identifier
- `workspaceId` - Workspace identifier
- `stateId` - State snapshot identifier

### Context Object (Required for ALL tools)
```json
{
  "sessionId": "session_timestamp_random",
  "workspaceId": "ws_timestamp_random",
  "sessionDescription": "Brief task description",
  "sessionMemory": "Previous context or empty string",
  "toolContext": "Specific context for this tool call",
  "primaryGoal": "User's main objective",
  "subgoal": "Current step toward goal"
}
```

## Common Mistakes to Avoid

1. ❌ Using `path`/`newPath` for duplicateNote → ✅ Use `sourcePath`/`targetPath`
2. ❌ Using `newName` for editFolder → ✅ Use `newPath` (full path)
3. ❌ Using `path: "string"` for searchDirectory → ✅ Use `paths: ["string"]`
4. ❌ Using `name` for loadState → ✅ Use `stateId` (get from listStates)
5. ❌ Adding `content` to executePrompt → ✅ Use `filepaths` array instead
6. ❌ Adding `deleteAll`/`replaceAll` flags → ✅ These don't exist in schemas
7. ❌ Forgetting `workspaceId` in searchMemory → ✅ It's required
8. ❌ Using single values for array params → ✅ Always use arrays: `["value"]`
