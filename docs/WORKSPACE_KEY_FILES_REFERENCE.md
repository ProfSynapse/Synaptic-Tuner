# Workspace System - Key Files Reference

## Core Type Definitions

### Workspace Types
**File**: `/Users/jrosenbaum/Documents/Code/.obsidian/plugins/claudesidian-mcp/src/database/types/workspace/WorkspaceTypes.ts`

Contains:
- `Workspace` - Simple base interface
- `ProjectWorkspace` - Extended with legacy fields
- `WorkspaceContext` - Intelligence layer with purpose, goals, workflows
- `ItemStatus` - Status enum (not_started, in_progress, completed)

### Storage Types
**File**: `/Users/jrosenbaum/Documents/Code/.obsidian/plugins/claudesidian-mcp/src/types/storage/StorageTypes.ts`

Contains:
- `IndividualWorkspace` - Actual stored format in .json files
- `SessionData` - Session structure nested in workspace
- `MemoryTrace` - Individual trace data
- `StateData` - Saved state/checkpoint
- `WorkspaceMetadata` - Index metadata (lightweight, no sessions)
- `WorkspaceIndex` - Full index structure with search fields

### Parameter Types
**File**: `/Users/jrosenbaum/Documents/Code/.obsidian/plugins/claudesidian-mcp/src/database/types/workspace/ParameterTypes.ts`

Contains:
- `CreateWorkspaceParameters` - What to provide when creating workspace
- `CreateWorkspaceResult` - What you get back
- `LoadWorkspaceParameters` - What to provide when loading
- `LoadWorkspaceResult` - Full response with context, workflows, sessions, states
- `ListWorkspacesParameters` - Sorting and filtering options
- `CreateStateParameters` - State snapshot parameters
- Plus legacy parameter types for backward compatibility

### Session Types
**File**: `/Users/jrosenbaum/Documents/Code/.obsidian/plugins/claudesidian-mcp/src/database/types/session/SessionTypes.ts`

Contains:
- `WorkspaceSession` - Session tracking
- `StateSnapshot` - Complete state for resumption
- `State` - Simple state interface
- `WorkspaceStateSnapshot` - Extended with legacy fields

---

## Implementation Files

### WorkspaceService
**File**: `/Users/jrosenbaum/Documents/Code/.obsidian/plugins/claudesidian-mcp/src/services/WorkspaceService.ts`

Methods:
- `listWorkspaces(limit)` - Get workspaces from index
- `getWorkspaces(options)` - Get with sorting/filtering
- `getWorkspace(id)` - Load full workspace from disk
- `getAllWorkspaces()` - Load all (expensive)
- `createWorkspace(data)` - Create and store new workspace
- `updateWorkspace(id, updates)` - Modify workspace
- `updateLastAccessed(id)` - Update timestamp
- `deleteWorkspace(id)` - Remove workspace
- `addSession(workspaceId, sessionData)` - Add session
- `updateSession()` - Modify session
- `deleteSession()` - Remove session
- `getSession()` - Retrieve session
- `addMemoryTrace()` - Add trace to session
- `getMemoryTraces()` - Get all traces
- `addState()` - Save state checkpoint
- `getState()` - Retrieve state
- `searchWorkspaces(query, limit)` - Full-text search
- `getWorkspaceByFolder(folder)` - Folder lookup
- `getActiveWorkspace()` - Get active workspace

### LoadWorkspaceMode
**File**: `/Users/jrosenbaum/Documents/Code/.obsidian/plugins/claudesidian-mcp/src/agents/memoryManager/modes/workspaces/LoadWorkspaceMode.ts`

What it does:
1. Takes workspace ID as input
2. Loads full workspace from disk
3. Collects context, workflows, preferences
4. Fetches sessions and states (limited by 'limit' param)
5. Collects recent files and directory structure
6. Resolves dedicated agent if configured
7. Returns comprehensive briefing for LLM

Key services used:
- `WorkspaceDataFetcher` - Fetches sessions/states
- `WorkspaceAgentResolver` - Resolves dedicated agent
- `WorkspaceContextBuilder` - Builds context briefing
- `WorkspaceFileCollector` - Collects workspace files

### CreateWorkspaceMode
**File**: `/Users/jrosenbaum/Documents/Code/.obsidian/plugins/claudesidian-mcp/src/agents/memoryManager/modes/workspaces/CreateWorkspaceMode.ts`

What it does:
1. Takes workspace creation parameters
2. Validates all required fields
3. Creates root folder if needed
4. Builds WorkspaceContext from parameters
5. Auto-detects key files
6. Handles dedicated agent setup
7. Creates workspace and stores to disk
8. Updates index

### ListWorkspacesMode
**File**: `/Users/jrosenbaum/Documents/Code/.obsidian/plugins/claudesidian-mcp/src/agents/memoryManager/modes/workspaces/ListWorkspacesMode.ts`

What it does:
1. Retrieves workspace metadata from index
2. Applies sorting (name, created, lastAccessed)
3. Applies ordering (asc/desc)
4. Applies limit
5. Returns list of workspace summaries

---

## Storage Location Files

### Index Manager
**File**: `/Users/jrosenbaum/Documents/Code/.obsidian/plugins/claudesidian-mcp/src/services/storage/IndexManager.ts`

Manages:
- Loading/saving `.workspaces/index.json`
- Updating index with new workspaces
- Removing workspaces from index
- Fast lookup operations

### File System Service
**File**: `/Users/jrosenbaum/Documents/Code/.obsidian/plugins/claudesidian-mcp/src/services/storage/FileSystemService.ts`

Handles:
- Reading individual workspace files
- Writing workspace files
- Listing workspace IDs
- Deleting workspace files

---

## Memory System Files

### MemoryService
**File**: `/Users/jrosenbaum/Documents/Code/.obsidian/plugins/claudesidian-mcp/src/agents/memoryManager/services/MemoryService.ts`

Provides:
- Session management
- State management
- Memory trace storage
- Workspace context persistence

### Workspace Data Fetcher
**File**: `/Users/jrosenbaum/Documents/Code/.obsidian/plugins/claudesidian-mcp/src/agents/memoryManager/services/WorkspaceDataFetcher.ts`

Fetches:
- Sessions for workspace
- States for workspace
- Memory traces
- Recent activity summaries

### Workspace Context Builder
**File**: `/Users/jrosenbaum/Documents/Code/.obsidian/plugins/claudesidian-mcp/src/agents/memoryManager/services/WorkspaceContextBuilder.ts`

Builds:
- Context briefings from workspace data
- Workflow descriptions
- Preference summaries
- Key file listings

### Workspace File Collector
**File**: `/Users/jrosenbaum/Documents/Code/.obsidian/plugins/claudesidian-mcp/src/agents/memoryManager/services/WorkspaceFileCollector.ts`

Collects:
- Recent files in workspace
- Directory structure/tree
- File paths for navigation
- Workspace file listing

---

## Tool Schema Definition

**File**: `/Users/jrosenbaum/Documents/Code/tools/tool_schemas.json`

Defines all workspace-related tools with their parameters and return types:
- `memoryManager_createWorkspace`
- `memoryManager_loadWorkspace`
- `memoryManager_listWorkspaces`
- `memoryManager_updateWorkspace`
- `memoryManager_createSession`
- `memoryManager_loadSession`
- `memoryManager_createState`
- `memoryManager_loadState`

---

## Utility Files

### Context Utils
**File**: `/Users/jrosenbaum/Documents/Code/.obsidian/plugins/claudesidian-mcp/src/utils/contextUtils.ts`

Utilities:
- `parseWorkspaceContext(context)` - Parse context string/object
- Context validation
- Context serialization

### Workspace Utils
**File**: `/Users/jrosenbaum/Documents/Code/.obsidian/plugins/claudesidian-mcp/src/utils/workspaceUtils.ts`

Utilities:
- `fileIsInWorkspace()` - Check if file belongs to workspace
- `getWorkspacesForFile()` - Find all workspaces containing file
- `getBestWorkspaceForFile()` - Find most specific workspace for file
- `updateWorkspaceActivityForFile()` - Track file activity (deprecated)

---

## Main Files to Understand

### Priority 1 (Start Here)
1. **WorkspaceTypes.ts** - Understand the data model
2. **StorageTypes.ts** - Understand storage format
3. **ParameterTypes.ts** - Understand request/response contracts
4. **LoadWorkspaceMode.ts** - Understand load flow

### Priority 2 (Understand Operations)
5. **WorkspaceService.ts** - All workspace operations
6. **CreateWorkspaceMode.ts** - Creation flow
7. **ListWorkspacesMode.ts** - Listing/filtering
8. **SessionTypes.ts** - Session/state structure

### Priority 3 (Infrastructure)
9. **FileSystemService.ts** - File I/O
10. **IndexManager.ts** - Index management
11. **WorkspaceDataFetcher.ts** - Data collection
12. **WorkspaceContextBuilder.ts** - Context building

---

## Quick Navigation by Task

### Want to create a workspace?
1. Read: `ParameterTypes.ts` - See what parameters are needed
2. Read: `CreateWorkspaceMode.ts` - See creation flow
3. Check: `WorkspaceService.createWorkspace()` - See storage logic

### Want to load a workspace?
1. Read: `LoadWorkspaceMode.ts` - See full load process
2. Check: `WorkspaceDataFetcher.ts` - See data collection
3. Check: `WorkspaceContextBuilder.ts` - See context building

### Want to understand sessions/states?
1. Read: `SessionTypes.ts` - See structure
2. Read: `StorageTypes.ts` - See how nested
3. Check: `WorkspaceService.addSession()` - See operations

### Want to query workspaces?
1. Read: `ParameterTypes.ts` - ListWorkspacesParameters
2. Read: `ListWorkspacesMode.ts` - See query implementation
3. Check: `WorkspaceService.getWorkspaces()` - See sorting logic

### Want to understand storage?
1. Read: `StorageTypes.ts` - Understand format
2. Check: `FileSystemService.ts` - File operations
3. Check: `IndexManager.ts` - Index structure

---

## Data Flow Summary

### Creation Flow
```
CreateWorkspaceParameters
    ↓
CreateWorkspaceMode.execute()
    ↓
Build WorkspaceContext
    ↓
Create IndividualWorkspace
    ↓
FileSystemService.writeWorkspace()
    ↓
IndexManager.updateWorkspaceInIndex()
    ↓
Return CreateWorkspaceResult
```

### Load Flow
```
LoadWorkspaceParameters (id)
    ↓
LoadWorkspaceMode.execute()
    ↓
WorkspaceService.getWorkspace(id)
    ↓
Parallel Collection:
  ├─ ContextBuilder.buildContextBriefing()
  ├─ DataFetcher.fetchWorkspaceSessions()
  ├─ FileCollector.getRecentFilesInWorkspace()
  └─ AgentResolver.fetchWorkspaceAgent()
    ↓
Assemble LoadWorkspaceResult
    ↓
Return to Client
```

### List Flow
```
ListWorkspacesParameters
    ↓
ListWorkspacesMode.execute()
    ↓
WorkspaceService.getWorkspaces(options)
    ↓
Read from WorkspaceIndex (fast!)
    ↓
Apply sorting & filtering
    ↓
Return ListWorkspacesResult
```

---

## Testing Resources

### Example Workspace JSON
See real examples in `WORKSPACE_ANALYSIS_REPORT.md` section 2 and 7

### Example Parameters
See real examples in `WORKSPACE_ANALYSIS_REPORT.md` section 8

### Example Responses
See real examples in `WORKSPACE_ANALYSIS_REPORT.md` section 2

---

## Configuration Files

**Plugin Main Config**: `manifest.json`
**Tool Schema Config**: `tools/tool_schemas.json`
**Agent Config**: `src/config/agentConfigs.ts`

---

## Important Notes

1. **Split-File Architecture**: Each workspace is a separate .json file in `.workspaces/` directory
2. **Index for Speed**: Metadata index enables fast searches without loading all workspace files
3. **Nested Sessions**: Sessions and states are stored inside workspace files, not separately
4. **Context-Driven**: WorkspaceContext is the intelligence layer that guides AI behavior
5. **File References**: Workspaces reference files in vault, don't duplicate them
6. **Backward Compatibility**: Multiple interfaces maintained for legacy support

