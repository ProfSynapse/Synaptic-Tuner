# Claudesidian MCP Workspace Documentation

Complete documentation of the workspace data structure and loadWorkspace response.

## Start Here

Begin with one of these documents:

1. **WORKSPACE_DOCUMENTATION_INDEX.md** - Quick orientation and navigation guide
2. **WORKSPACE_ANALYSIS_REPORT.md** - Complete technical specification (START WITH SECTION 1-2)

## The 4 Documents

```
README.md (this file)
│
├─ WORKSPACE_DOCUMENTATION_INDEX.md (9.4KB)
│  └─ Quick start, navigation guide, key concepts
│
├─ WORKSPACE_ANALYSIS_REPORT.md (21KB) ← MOST COMPREHENSIVE
│  ├─ Section 1: Workspace Structure (types, fields)
│  ├─ Section 2: LoadWorkspace Response (full structure + example)
│  ├─ Section 3: Workspace Contents (file organization)
│  ├─ Section 4: Metadata & Indexing (fast lookups)
│  ├─ Section 5: Sessions & States (nesting, resumption)
│  ├─ Section 6: Tool Schemas (definitions)
│  ├─ Section 7: Example Workspaces (3 real examples)
│  ├─ Section 8: Creation Parameters (what you send)
│  ├─ Section 9: Search Capabilities (query operations)
│  ├─ Section 10: Design Principles (why it works)
│  └─ Section 11: Workflow Execution (step-by-step)
│
├─ WORKSPACE_ARCHITECTURE_DIAGRAM.md (18KB)
│  ├─ 1. Workspace Data Hierarchy (tree view)
│  ├─ 2. Storage Architecture (file layout)
│  ├─ 3. LoadWorkspace Data Flow (process diagram)
│  ├─ 4. Session & State Hierarchy (nesting)
│  ├─ 5. Index Structure (search indexes)
│  ├─ 6. Request/Response Types (message structures)
│  ├─ 7. Context-Driven AI Workflow (how AI uses it)
│  └─ 8. Workspace Creation Parameters Flow (creation)
│
└─ WORKSPACE_KEY_FILES_REFERENCE.md (11KB)
   ├─ Core Type Definitions (with file paths)
   ├─ Implementation Files (services, modes)
   ├─ Storage & Infrastructure (file I/O)
   ├─ Navigation by Task (how to find things)
   ├─ Data Flow Summary (creation, load, list)
   └─ Quick code location reference
```

## Quick Facts

- **What is this?** A workspace is a bounded context for related work with purpose, goals, and workflows
- **How big?** Each workspace has metadata (light) and sessions/states (nested inside)
- **How stored?** Split-file architecture: `.workspaces/{id}.json` per workspace + index for fast lookups
- **What does loadWorkspace return?** Everything an AI needs: context, workflows, files, sessions, states, agent
- **Why sessions/states?** Sessions track activity, states save checkpoints for work resumption

## Workspace Structure at a Glance

```typescript
Workspace {
  id: "ws_..."
  name: "Job Search 2025"
  rootFolder: "Job Search/2025"
  created: 1729990000000
  lastAccessed: 1730005000000
  
  context: {
    purpose: "Apply for marketing manager positions"
    currentGoal: "Submit 10 applications this week"
    workflows: [{name, when, steps}]
    keyFiles: ["Resume.md", "tracker.md"]
    preferences: "Professional tone. Tech companies."
    dedicatedAgent?: {agentId, agentName}
  }
  
  sessions: {
    session_123: {
      id: "session_123"
      name: "Google Application Week"
      startTime: 1730000000000
      
      memoryTraces: {...}
      states: {
        state_456: {
          name: "Google Cover Letter Draft"
          snapshot: {
            conversationContext: "..."
            activeTask: "..."
            activeFiles: ["..."]
            nextSteps: ["..."]
            reasoning: "..."
          }
        }
      }
    }
  }
}
```

## LoadWorkspace Response at a Glance

```typescript
{
  success: true,
  data: {
    context: {name, purpose, rootFolder, recentActivity}
    workflows: ["Workflow 1: steps...", "Workflow 2: steps..."]
    workspaceStructure: ["file1", "dir/file2", ...]
    recentFiles: [{path, modified}, ...]
    keyFiles: {path: "purpose", ...}
    preferences: "user guidelines"
    sessions: [{id, name, created}, ...]
    states: [{id, name, sessionId, created}, ...]
    agent?: {agentId, agentName, systemPrompt}
  },
  workspaceContext: {
    workspaceId: "ws_..."
    workspacePath: [...]
  }
}
```

## 5-Minute Overview

1. **What's a Workspace?** Container for related work with purpose, goals, workflows
2. **What's WorkspaceContext?** Intelligence layer that guides AI behavior
3. **What does loadWorkspace return?** Everything AI needs to understand and work with workspace
4. **How are sessions nested?** Sessions inside workspaces, states inside sessions
5. **How does resumption work?** StateSnapshot captures everything: what was happening, what files, next steps

## For Different Audiences

### I'm learning the system
Start with WORKSPACE_ANALYSIS_REPORT.md sections 1-2

### I want visual diagrams
Go to WORKSPACE_ARCHITECTURE_DIAGRAM.md

### I need to find source code
Use WORKSPACE_KEY_FILES_REFERENCE.md

### I want real examples
See WORKSPACE_ANALYSIS_REPORT.md section 7

### I want quick navigation
Use WORKSPACE_DOCUMENTATION_INDEX.md

## Key Concepts

- **Workspace**: Bounded context for related work
- **WorkspaceContext**: Intelligence layer (purpose, goals, workflows, preferences)
- **Sessions**: Track activity with memory traces and states
- **States**: Saved checkpoints for work resumption
- **Split-File Architecture**: Each workspace = one .json file, sessions nested inside
- **Index**: Fast metadata lookups without loading full workspace files

## Tool Schema Entry

```json
{
  "memoryManager_loadWorkspace": {
    "tool_name": "memoryManager_loadWorkspace",
    "agent": "memoryManager",
    "mode": "loadWorkspace",
    "class_name": "LoadWorkspaceMode",
    "parameters": [
      {"name": "id", "optional": false, "type": "string"},
      {"name": "limit", "optional": true, "type": "number"}
    ],
    "required_params": ["id"],
    "file_path": "src/agents/memoryManager/modes/workspaces/LoadWorkspaceMode.ts"
  }
}
```

## File Locations

All source code under:
```
/Users/jrosenbaum/Documents/Code/.obsidian/plugins/claudesidian-mcp/
```

Key files:
- `src/database/types/workspace/WorkspaceTypes.ts` - Workspace types
- `src/types/storage/StorageTypes.ts` - Storage format
- `src/services/WorkspaceService.ts` - All operations
- `src/agents/memoryManager/modes/workspaces/LoadWorkspaceMode.ts` - Load implementation

## Statistics

- 4 comprehensive documents
- 1,820 lines of documentation
- 30+ type definitions
- 8 ASCII diagrams
- 20+ code examples
- 20+ source files mapped
- 3 real-world examples

## What's Included

- Complete type definitions with explanations
- Full LoadWorkspace response structure
- Real-world JSON examples
- 8 ASCII architecture diagrams
- Visual data flow diagrams
- Session and state structures
- Workspace indexing system
- Tool schemas and definitions
- Example workspace structures
- Creation parameter specifications
- Search and query capabilities
- Design principles explained
- Workflow execution patterns
- Source code file mapping
- Quick reference guides

## Generated

November 9, 2025
Claudesidian MCP Plugin v3.0.7
Total analysis time: Comprehensive codebase investigation

## Next Steps

1. Open WORKSPACE_DOCUMENTATION_INDEX.md for quick orientation
2. Read WORKSPACE_ANALYSIS_REPORT.md sections 1-4 for core understanding
3. Check WORKSPACE_ARCHITECTURE_DIAGRAM.md for visual reference
4. Use WORKSPACE_KEY_FILES_REFERENCE.md when you need source code

---

All documents are in Markdown format. Open with any text editor or Markdown viewer.
