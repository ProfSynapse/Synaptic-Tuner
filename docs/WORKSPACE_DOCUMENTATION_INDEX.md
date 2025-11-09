# Workspace System Documentation Index

This directory contains comprehensive documentation of the Claudesidian MCP workspace data structure and load response system.

## Documents Generated (3 files, 50KB)

### 1. WORKSPACE_ANALYSIS_REPORT.md (21KB)
**Most Comprehensive - START HERE**

Complete technical specification of the workspace system including:

- **Workspace Structure** (11 sections)
  - Core Workspace Types (Workspace, ProjectWorkspace)
  - WorkspaceContext (intelligence layer)
  - Storage Format (IndividualWorkspace)
  - LoadWorkspace Response (full structure with types)
  - Real-world response examples with actual JSON

- **Workspace Metadata & Indexing**
  - WorkspaceIndex structure
  - Fast metadata lookups
  - Search capabilities

- **Sessions & States**
  - SessionData structure
  - MemoryTrace types
  - StateSnapshot for work resumption

- **Tool Schemas**
  - Tool definitions in tool_schemas.json
  - All workspace-related tools

- **Example Workspaces**
  - Job Application workspace
  - Research Project workspace
  - Software Development workspace

- **Design Principles**
  - Split-file architecture
  - Context-driven design
  - Lightweight operations
  - State resumption
  - Backward compatibility

- **Workflow Execution Cycle**
  - Step-by-step workflow process
  - State save/restore patterns

### 2. WORKSPACE_ARCHITECTURE_DIAGRAM.md (18KB)
**Visual Reference - DIAGRAMS & FLOWS**

Visual diagrams and ASCII art showing:

1. **Workspace Data Hierarchy** - Tree structure of all fields
2. **Storage Architecture** - File system layout with .workspaces/ structure
3. **LoadWorkspace Data Flow** - Step-by-step process diagram
4. **Session & State Hierarchy** - Nesting and relationships
5. **Index Structure** - Fast lookups and search indexes
6. **Request/Response Message Types** - All parameter and result types
7. **Context-Driven AI Workflow** - How AI uses workspace context
8. **Workspace Creation Parameters Flow** - Creation process

### 3. WORKSPACE_KEY_FILES_REFERENCE.md (11KB)
**Quick Reference - SOURCE CODE LOCATIONS**

Maps all workspace functionality to source files:

- **Core Type Definitions** (with file paths)
  - WorkspaceTypes.ts
  - StorageTypes.ts
  - ParameterTypes.ts
  - SessionTypes.ts

- **Implementation Files**
  - WorkspaceService.ts (methods list)
  - LoadWorkspaceMode.ts
  - CreateWorkspaceMode.ts
  - ListWorkspacesMode.ts

- **Storage & Infrastructure**
  - IndexManager.ts
  - FileSystemService.ts
  - WorkspaceDataFetcher.ts
  - WorkspaceContextBuilder.ts
  - WorkspaceFileCollector.ts

- **Navigation by Task**
  - How to create a workspace
  - How to load a workspace
  - How to understand sessions/states
  - How to query workspaces
  - How to understand storage

- **Data Flow Summary**
  - Creation flow
  - Load flow
  - List flow

---

## Quick Start Guide

### I want to understand the workspace data structure
-> Read **WORKSPACE_ANALYSIS_REPORT.md** sections 1-4

### I want to see what LoadWorkspace returns
-> Read **WORKSPACE_ANALYSIS_REPORT.md** section 2, especially the real-world example

### I want visual diagrams
-> See **WORKSPACE_ARCHITECTURE_DIAGRAM.md** all sections

### I want to find source code
-> Use **WORKSPACE_KEY_FILES_REFERENCE.md** to locate files
-> Then check the absolute paths listed (e.g., `/Users/jrosenbaum/Documents/Code/.obsidian/plugins/...`)

### I want to understand workflows
-> Read **WORKSPACE_ANALYSIS_REPORT.md** section 11
-> Then see **WORKSPACE_ARCHITECTURE_DIAGRAM.md** section 7

### I want real examples
-> **WORKSPACE_ANALYSIS_REPORT.md** section 7 (Job Search, Research, Development examples)
-> **WORKSPACE_ANALYSIS_REPORT.md** section 2 (JSON response example)

---

## Document Organization

Each document is organized for different purposes:

| Document | Purpose | Best For |
|----------|---------|----------|
| ANALYSIS_REPORT | Complete specification | Understanding the full system, contracts, types |
| ARCHITECTURE_DIAGRAM | Visual reference | Understanding data flow, hierarchy, relationships |
| KEY_FILES_REFERENCE | Source code mapping | Finding implementation, navigating codebase |

---

## Key Concepts Explained

### Workspace
A bounded context around a set of related tasks. Contains:
- **Purpose**: What is this workspace for?
- **Current Goal**: What are we doing right now?
- **Workflows**: Defined processes for different situations
- **Key Files**: Important files for quick reference
- **Preferences**: User guidelines for AI behavior

### WorkspaceContext
The "intelligence layer" that guides AI behavior. Includes everything above plus:
- Dedicated agent (optional)
- Preferences as actionable guidelines

### LoadWorkspace Response
Returns comprehensive briefing including:
- Context (purpose, goal, activity)
- Workflows (human-readable steps)
- File structure (all files/folders)
- Recent files (what was modified)
- Key files (important references)
- Sessions (ongoing work)
- States (saved checkpoints)
- Optional dedicated agent

### Sessions
Tracks activity within a workspace:
- Memory traces (what happened)
- States (checkpoints saved)
- Active status
- Name and description

### States
Saved snapshots for resumption:
- Conversation context (what was happening)
- Active task (current work)
- Active files (what files were open)
- Next steps (what to do next)
- Workspace context snapshot (state at save time)

### Split-File Architecture
- Each workspace = separate .json file in `.workspaces/`
- Index for fast metadata lookups
- Sessions nested inside workspaces
- No monolithic data structures
- Files referenced, not duplicated

---

## Structure of Workspace Object

```
Workspace
├── id: string
├── name: string
├── rootFolder: string
├── created: number
├── lastAccessed: number
├── context: WorkspaceContext
│   ├── purpose: string
│   ├── currentGoal: string
│   ├── workflows: [{name, when, steps}]
│   ├── keyFiles: string[]
│   ├── preferences: string
│   └── dedicatedAgent?: {agentId, agentName}
└── sessions: {[sessionId]: SessionData}
    └── SessionData
        ├── id: string
        ├── name: string
        ├── startTime: number
        ├── memoryTraces: {[traceId]: MemoryTrace}
        └── states: {[stateId]: StateData}
            └── StateData
                ├── id: string
                ├── name: string
                └── snapshot: StateSnapshot
                    ├── conversationContext: string
                    ├── activeTask: string
                    ├── activeFiles: string[]
                    ├── nextSteps: string[]
                    └── reasoning: string
```

---

## LoadWorkspace Tool Summary

**What you send:**
```typescript
{
  id: "ws_1729990000000_abc123",  // Required
  limit: 3                         // Optional, default 3
}
```

**What you get back:**
```typescript
{
  success: true,
  data: {
    context: { /* briefing */ },
    workflows: [ /* human-readable */ ],
    workspaceStructure: [ /* all files */ ],
    recentFiles: [ { path, modified } ],
    keyFiles: { /* path: purpose */ },
    preferences: "...",
    sessions: [ { id, name, created } ],
    states: [ { id, name, sessionId, created } ],
    agent?: { agentId, agentName, systemPrompt }
  },
  workspaceContext: {
    workspaceId: string,
    workspacePath: string[]
  }
}
```

---

## Related Tools

| Tool | Purpose |
|------|---------|
| `memoryManager_createWorkspace` | Create new workspace with context |
| `memoryManager_loadWorkspace` | Load workspace + restore context (THIS ONE) |
| `memoryManager_listWorkspaces` | List available workspaces |
| `memoryManager_updateWorkspace` | Modify workspace properties |
| `memoryManager_createSession` | Start new session |
| `memoryManager_loadSession` | Restore previous session |
| `memoryManager_createState` | Save checkpoint |
| `memoryManager_loadState` | Restore from checkpoint |

---

## File Locations

All source files referenced are located under:
```
/Users/jrosenbaum/Documents/Code/.obsidian/plugins/claudesidian-mcp/
```

Key subdirectories:
- `src/database/types/workspace/` - Type definitions
- `src/types/storage/` - Storage types
- `src/services/` - Service implementations
- `src/agents/memoryManager/modes/workspaces/` - Mode implementations

---

## How to Use This Documentation

1. **First Time?** Read WORKSPACE_ANALYSIS_REPORT.md sections 1-2
2. **Need Visual Help?** Check WORKSPACE_ARCHITECTURE_DIAGRAM.md
3. **Want to Code?** Use WORKSPACE_KEY_FILES_REFERENCE.md to find files
4. **Building Something?** Look for relevant example in ANALYSIS_REPORT section 7

---

## Document Statistics

- **Total Pages**: ~14-15 pages when printed
- **Total Words**: ~12,000 words
- **Code Examples**: 20+
- **Diagrams**: 8
- **TypeScript Interfaces**: 30+
- **Source Files Referenced**: 20+

---

## Key Takeaways

1. Workspaces are purpose-driven containers with goals and workflows
2. LoadWorkspace returns everything an AI needs to understand and work with a workspace
3. Sessions and states allow work resumption with full context
4. WorkspaceContext is the intelligence layer that guides AI behavior
5. Split-file architecture enables fast indexing and scalability
6. Sessions nest within workspaces, states nest within sessions
7. Multiple tools available for complete workspace lifecycle management

---

Generated: November 9, 2025
Source: Claudesidian MCP Plugin v3.0.7
Documentation Type: System Architecture & Data Structure Reference
