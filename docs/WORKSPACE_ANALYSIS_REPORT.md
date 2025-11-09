# Workspace Data Structure & Load Response Analysis

## Executive Summary

The Claudesidian MCP workspace system uses a **split-file JSON storage architecture** for workspaces, sessions, conversations, and states. When a workspace is loaded via the `loadWorkspace` tool, it returns a comprehensive briefing with context, workflows, file structure, sessions, states, and agent information.

---

## 1. WORKSPACE STRUCTURE

### Core Workspace Types

#### **Simple Workspace (Base Interface)**
Located: `src/database/types/workspace/WorkspaceTypes.ts`

```typescript
interface Workspace {
  id: string;                              // Unique workspace identifier
  name: string;                            // Display name
  context?: WorkspaceContext;              // Optional context briefing
  rootFolder: string;                      // Root directory path
  created: number;                         // Unix timestamp
  lastAccessed: number;                    // Unix timestamp
}
```

#### **Extended ProjectWorkspace (Backward Compatible)**
```typescript
interface ProjectWorkspace extends Workspace {
  // Core functionality
  isActive?: boolean;

  // Legacy optional fields
  description?: string;
  relatedFolders?: string[];
  relatedFiles?: string[];
  associatedNotes?: string[];
  keyFileInstructions?: string;
  activityHistory?: Array<{
    timestamp: number;
    action: 'view' | 'edit' | 'create' | 'tool';
    toolName?: string;
    duration?: number;
    context?: string;
  }>;
  preferences?: Record<string, any>;
  projectPlan?: string;
  checkpoints?: Array<{
    id: string;
    date: number;
    description: string;
    completed: boolean;
  }>;
  completionStatus?: Record<string, {
    status: 'not_started' | 'in_progress' | 'completed';
    completedDate?: number;
    completionNotes?: string;
  }>;
}
```

### Workspace Context (The Intelligence Layer)

Located: `src/database/types/workspace/WorkspaceTypes.ts`

```typescript
interface WorkspaceContext {
  purpose: string;                         // What is this workspace for?
                                          // Example: "Apply for marketing manager positions"

  currentGoal: string;                    // What are you trying to accomplish right now?
                                          // Example: "Submit 10 applications this week"

  workflows: Array<{
    name: string;                         // Workflow name
    when: string;                         // When to use this workflow
    steps: string;                        // Steps as newline-separated string
  }>;

  keyFiles: string[];                     // Important files for quick reference
                                          // Example: ["path/to/resume.md", "path/to/portfolio.md"]

  preferences: string;                    // User preferences and guidelines
                                          // Example: "Use professional tone. Focus on tech companies."

  dedicatedAgent?: {
    agentId: string;                      // Unique agent identifier
    agentName: string;                    // Display name
  };
}
```

### Storage Format (IndividualWorkspace)

Located: `src/types/storage/StorageTypes.ts`

The actual stored format in `.workspaces/{id}.json`:

```typescript
interface IndividualWorkspace {
  id: string;
  name: string;
  description?: string;
  rootFolder: string;
  created: number;
  lastAccessed: number;
  isActive?: boolean;
  context?: WorkspaceContext;
  sessions: Record<string, SessionData>;  // Sessions nested in workspace
}
```

---

## 2. LOADWORKSPACE RESPONSE (FULL STRUCTURE)

### Tool Definition
- **Agent**: `memoryManager`
- **Mode**: `loadWorkspace`
- **File**: `src/agents/memoryManager/modes/workspaces/LoadWorkspaceMode.ts`
- **Tool Schema Entry**: `memoryManager_loadWorkspace` in `tools/tool_schemas.json`

### Request Parameters

```typescript
interface LoadWorkspaceParameters extends CommonParameters {
  id: string;                    // Required: Workspace ID to load
  limit?: number;                // Optional: Default 3
                                // Limits sessions, states, and recentActivity results
}
```

### Full Response Structure

```typescript
interface LoadWorkspaceResult extends CommonResult {
  success: boolean;
  error?: string;                           // If success=false
  
  data: {
    // Context Briefing
    context: {
      name: string;                        // Workspace name
      description?: string;
      purpose?: string;                    // From WorkspaceContext
      rootFolder: string;
      recentActivity: string[];            // Limited to 'limit' items
    };

    // Workflow Operations
    workflows: string[];                   // Human-readable workflow steps
                                          // Format: "Workflow Name:\nStep 1\nStep 2\nStep 3"

    // File Structure
    workspaceStructure: string[];          // Tree of all files/folders in workspace
                                          // Nested path format for navigation

    // Recent Files
    recentFiles: Array<{
      path: string;                        // File path
      modified: number;                    // Unix timestamp
    }>;

    // Key Files Mapping
    keyFiles: Record<string, string>;      // { filePath: "purpose/notes" }

    // Preferences
    preferences: string;                   // User guidelines from WorkspaceContext

    // Sessions
    sessions: Array<{
      id: string;
      name: string;
      description?: string;
      created: number;                     // Session start time
    }>;

    // Saved States
    states: Array<{
      id: string;
      name: string;
      description?: string;
      sessionId: string;                   // Associated session
      created: number;                     // Save timestamp
      tags?: string[];
    }>;

    // Dedicated Agent (Optional)
    agent?: {
      agentId: string;
      agentName: string;
      systemPrompt?: string;               // If dedicated agent is configured
    };
  };

  // Context for workspace awareness
  workspaceContext: {
    workspaceId: string;
    workspacePath: string[];               // File structure for reference
  };
}
```

### Real-World Response Example

```json
{
  "success": true,
  "data": {
    "context": {
      "name": "Job Search 2025",
      "description": "Marketing manager position applications",
      "purpose": "Apply for marketing manager positions at tech companies",
      "rootFolder": "Job Search/2025",
      "recentActivity": [
        "Customized cover letter for Google position",
        "Updated resume with Q4 achievements",
        "Tracked 12 application submissions"
      ]
    },
    "workflows": [
      "New Application:\n1. Research company deeply\n2. Customize cover letter\n3. Tailor resume\n4. Review application\n5. Submit\n6. Track in tracker",
      "Follow-up:\n1. Wait 2 weeks\n2. Check application status\n3. Send follow-up email\n4. Log interaction",
      "Interview Prep:\n1. Review job description\n2. Prepare stories using STAR\n3. Practice common questions\n4. Research company culture\n5. Prepare questions for interviewer"
    ],
    "workspaceStructure": [
      "Job Search/",
      "Job Search/2025/",
      "Job Search/2025/Applications/",
      "Job Search/2025/Applications/google-marketing-manager.md",
      "Job Search/2025/Applications/amazon-brand-manager.md",
      "Job Search/2025/Cover Letters/",
      "Job Search/2025/Cover Letters/template.md",
      "Job Search/2025/Cover Letters/google-customized.md",
      "Job Search/2025/Resume-2025.md",
      "Job Search/2025/tracker.md"
    ],
    "recentFiles": [
      {
        "path": "Job Search/2025/Cover Letters/google-customized.md",
        "modified": 1730000000000
      },
      {
        "path": "Job Search/2025/tracker.md",
        "modified": 1729990000000
      }
    ],
    "keyFiles": {
      "Job Search/2025/Resume-2025.md": "Primary resume for all applications",
      "Job Search/2025/Applications/tracker.md": "Central tracking of all applications and responses",
      "Job Search/2025/Cover Letters/template.md": "Base cover letter template"
    },
    "preferences": "Use professional tone. Focus on tech companies with strong brand. Keep cover letters under 300 words. Highlight data-driven achievements.",
    "sessions": [
      {
        "id": "session_1730000000000_abc123",
        "name": "Google Application Week",
        "description": "Preparing and submitting applications for Google marketing positions",
        "created": 1730000000000
      }
    ],
    "states": [
      {
        "id": "state_1730000000000_xyz789",
        "name": "Google Cover Letter Draft",
        "description": "Cover letter customized for Google position - ready for review",
        "sessionId": "session_1730000000000_abc123",
        "created": 1729999000000,
        "tags": ["google", "cover-letter", "draft"]
      }
    ],
    "agent": {
      "agentId": "agent_marketing_specialist",
      "agentName": "Marketing Specialist",
      "systemPrompt": "You are a marketing expert specializing in tech company positioning..."
    }
  },
  "workspaceContext": {
    "workspaceId": "ws_1729990000000_abc123",
    "workspacePath": [
      "Job Search/2025/",
      "Job Search/2025/Applications/",
      "Job Search/2025/Cover Letters/"
    ]
  }
}
```

---

## 3. WORKSPACE CONTENTS (TYPICAL STRUCTURE)

### File Organization

```
Vault Root/
├── .workspaces/                              # All workspace metadata
│   ├── index.json                            # Workspace index
│   ├── ws_1729990000000_abc123.json          # Workspace file
│   ├── ws_1730000000000_def456.json
│   └── ...
│
├── .conversations/                           # All conversations
│   ├── index.json                            # Conversation index
│   ├── conv_1729990000000_abc123.json
│   └── ...
│
└── Job Search/                               # Workspace root folder
    ├── 2025/
    │   ├── Applications/
    │   │   ├── google-marketing-manager.md
    │   │   ├── amazon-brand-manager.md
    │   │   └── tracking-sheet.csv
    │   ├── Cover Letters/
    │   │   ├── template.md
    │   │   ├── google-customized.md
    │   │   └── amazon-customized.md
    │   ├── Resume-2025.md
    │   ├── tracker.md
    │   └── research/
    │       ├── google-company-notes.md
    │       └── industry-salary-data.md
    └── 2024/
        └── ...
```

### Key Files Nested in Workspace

The workspace stores metadata about important files without duplicating them. Files remain in their original locations, but the workspace tracks them for quick access.

---

## 4. WORKSPACE METADATA (INDEXING SYSTEM)

### Workspace Index Structure

Located: `.workspaces/index.json`

```typescript
interface WorkspaceIndex {
  workspaces: Record<string, WorkspaceMetadata>;
  byName: Record<string, string[]>;          // Quick name lookup
  byDescription: Record<string, string[]>;   // Description search
  byFolder: Record<string, string>;          // Folder-to-workspace mapping
  sessionsByWorkspace: Record<string, string[]>;  // Session IDs per workspace
  lastUpdated: number;
}

interface WorkspaceMetadata {
  id: string;
  name: string;
  description?: string;
  rootFolder: string;
  created: number;
  lastAccessed: number;
  isActive?: boolean;
  sessionCount: number;
  traceCount: number;                        // Memory traces
}
```

### Example Index

```json
{
  "workspaces": {
    "ws_1729990000000_abc123": {
      "id": "ws_1729990000000_abc123",
      "name": "Job Search 2025",
      "description": "Marketing manager position applications",
      "rootFolder": "Job Search/2025",
      "created": 1729990000000,
      "lastAccessed": 1730005000000,
      "isActive": true,
      "sessionCount": 3,
      "traceCount": 47
    }
  },
  "byName": {
    "job": ["ws_1729990000000_abc123"],
    "search": ["ws_1729990000000_abc123"],
    "2025": ["ws_1729990000000_abc123"]
  },
  "byFolder": {
    "Job Search/2025": "ws_1729990000000_abc123"
  },
  "sessionsByWorkspace": {
    "ws_1729990000000_abc123": [
      "session_1730000000000_abc123",
      "session_1729995000000_def456"
    ]
  },
  "lastUpdated": 1730005000000
}
```

---

## 5. SESSIONS (NESTED IN WORKSPACE)

### Session Data Structure

Located: Nested in `IndividualWorkspace.sessions`

```typescript
interface SessionData {
  id: string;
  name?: string;
  description?: string;
  startTime: number;                         // Unix timestamp
  endTime?: number;
  isActive: boolean;
  memoryTraces: Record<string, MemoryTrace>;
  states: Record<string, StateData>;         // Saved checkpoints
}

interface MemoryTrace {
  id: string;
  timestamp: number;
  type: string;                              // e.g., "tool_call", "note_created"
  content: string;
  metadata?: {
    tool?: string;
    params?: any;
    result?: any;
    relatedFiles?: string[];
  };
}

interface StateData {
  id: string;
  name: string;
  created: number;
  snapshot: WorkspaceStateSnapshot;          // Full state at save time
}
```

### State Snapshot (Work Resumption)

```typescript
interface StateSnapshot {
  workspaceContext: WorkspaceContext;        // Workspace state at save
  conversationContext: string;               // What was happening
  activeTask: string;                        // Current task
  activeFiles: string[];                     // Files being edited
  nextSteps: string[];                       // Immediate actions
  reasoning: string;                         // Why saved now
}
```

---

## 6. TOOL SCHEMAS & DEFINITIONS

### Tool Registry Entry

From `tools/tool_schemas.json`:

```json
{
  "memoryManager_loadWorkspace": {
    "tool_name": "memoryManager_loadWorkspace",
    "agent": "memoryManager",
    "mode": "loadWorkspace",
    "class_name": "LoadWorkspaceMode",
    "params_interface": "LoadWorkspaceParameters",
    "parameters": [
      {
        "name": "id",
        "optional": false,
        "type": "string",
        "description": "Workspace ID to load"
      },
      {
        "name": "limit",
        "optional": true,
        "type": "number",
        "description": "Max results (default 3)"
      }
    ],
    "required_params": ["id"],
    "file_path": "src/agents/memoryManager/modes/workspaces/LoadWorkspaceMode.ts"
  }
}
```

### Related Workspace Tools

| Tool | Purpose |
|------|---------|
| `memoryManager_createWorkspace` | Create new workspace with context |
| `memoryManager_loadWorkspace` | Load workspace and restore context |
| `memoryManager_listWorkspaces` | List available workspaces |
| `memoryManager_updateWorkspace` | Modify workspace properties |
| `memoryManager_createSession` | Start new session in workspace |
| `memoryManager_loadSession` | Restore previous session |
| `memoryManager_createState` | Save workspace checkpoint |
| `memoryManager_loadState` | Restore from saved state |

---

## 7. EXAMPLE WORKSPACE STRUCTURES

### Structure 1: Job Application Workspace

```
Purpose: "Apply for marketing manager positions"
Current Goal: "Submit 10 applications this week"

Workflows:
1. "New Application"
   - Research company
   - Customize cover letter
   - Tailor resume
   - Submit
   - Track in tracker

2. "Interview Prep"
   - Review job description
   - Prepare STAR stories
   - Practice common questions
   - Research company culture

Key Files:
- "Job Search/2025/Resume-2025.md"
- "Job Search/2025/tracker.md"
- "Job Search/2025/Cover Letters/template.md"

Preferences: "Use professional tone. Focus on tech companies."
```

### Structure 2: Research Project Workspace

```
Purpose: "Market research analysis for Q4 planning"
Current Goal: "Deliver analysis report by Friday"

Workflows:
1. "Data Collection"
   - Gather sources
   - Extract key data
   - Compile findings

2. "Analysis"
   - Identify trends
   - Create visualizations
   - Write interpretations

Key Files:
- "Research/2025/Q4/raw-data.csv"
- "Research/2025/Q4/data-analysis.ipynb"
- "Research/2025/Q4/report-template.md"
```

### Structure 3: Software Development Workspace

```
Purpose: "Backend refactoring for performance"
Current Goal: "Complete database query optimization"

Workflows:
1. "Code Review"
   - Run benchmarks
   - Analyze bottlenecks
   - Implement improvements

2. "Testing"
   - Write unit tests
   - Integration testing
   - Performance validation

Key Files:
- "src/database/queries.ts"
- "tests/performance.test.ts"
- "docs/REFACTORING.md"
```

---

## 8. CREATION PARAMETERS (What Gets Set)

### CreateWorkspace Request

```typescript
interface CreateWorkspaceParameters {
  name: string;                              // Required: "Job Search 2025"
  rootFolder: string;                        // Required: "Job Search/2025"
  purpose: string;                           // Required: "Apply for marketing positions"
  currentGoal: string;                       // Required: "Submit 10 applications"
  workflows: Array<{                         // Required: At least one
    name: string;                            // "New Application"
    when: string;                            // "When applying to new position"
    steps: string;                           // "Step 1\nStep 2\nStep 3"
  }>;
  keyFiles?: string[];                       // Optional: ["path/to/resume.md"]
  preferences?: string;                      // Optional: "Professional tone"
  dedicatedAgentId?: string;                 // Optional: Custom agent for workspace
  description?: string;                      // Optional: Longer description
  relatedFolders?: string[];                 // Optional: Additional folders
  relatedFiles?: string[];                   // Optional: Additional files
}
```

---

## 9. SEARCH & QUERY CAPABILITIES

### ListWorkspaces with Sorting

```typescript
interface ListWorkspacesParameters {
  sortBy?: 'name' | 'created' | 'lastAccessed';  // Default: lastAccessed
  order?: 'asc' | 'desc';                         // Default: desc
  limit?: number;
}
```

### Search Operations

- **Name search**: Case-insensitive, word-based
- **Description search**: Full text across description field
- **Folder matching**: Find workspace by root folder
- **Active workspace**: Get currently active workspace
- **Custom queries**: Flexible filtering in WorkspaceService

---

## 10. KEY DESIGN PRINCIPLES

### 1. **Split-File Architecture**
- Each workspace lives in its own `.json` file
- Sessions and traces nested within workspace
- Index for fast metadata lookups
- No monolithic data structures

### 2. **Context-Driven Design**
- WorkspaceContext contains all LLM-relevant info
- Purpose, goals, and workflows guide AI reasoning
- Key files provide quick reference points
- Preferences encode user guidelines

### 3. **Lightweight Operations**
- Index provides O(1) lookups
- Metadata includes only essential fields
- Limit parameter prevents data bloat
- Recent activity tracking for quick context

### 4. **State Resumption**
- StateSnapshot captures complete work state
- activeFiles, nextSteps enable instant resumption
- Session nesting allows context-scoped memory
- Tags and descriptions for organization

### 5. **Backward Compatibility**
- ProjectWorkspace extends Workspace
- Optional legacy fields preserved
- IndividualWorkspace for storage, Workspace for API
- Multiple parameter types supported

---

## 11. WORKFLOW EXECUTION CYCLE

### Typical Workflow

1. **Load Workspace**
   ```
   memoryManager_loadWorkspace(id: "ws_xxx")
   ↓
   Returns full context, workflows, sessions, states
   ```

2. **Examine Context**
   ```
   AI reads:
   - Purpose and current goal
   - Available workflows
   - Key files and preferences
   - Recent activity
   ```

3. **Execute Workflow Step**
   ```
   AI selects workflow "New Application"
   Executes steps:
   1. Research company → Create note
   2. Customize cover letter → Edit file
   3. Submit → Track in tracker
   ```

4. **Save Progress**
   ```
   memoryManager_createState(
     name: "Google Cover Letter Draft",
     conversationContext: "Working on Google position",
     activeFiles: ["Cover Letters/google-customized.md"],
     nextSteps: ["Review cover letter", "Submit application"]
   )
   ```

5. **Resume Later**
   ```
   memoryManager_loadState(stateId: "state_xxx")
   ↓
   AI resumes with full context from when state was saved
   ```

---

## Summary Table

| Aspect | Details |
|--------|---------|
| **Storage Location** | `.workspaces/{id}.json` for data, `.workspaces/index.json` for metadata |
| **Key Fields** | id, name, context, rootFolder, created, lastAccessed, sessions |
| **Context Fields** | purpose, currentGoal, workflows, keyFiles, preferences, dedicatedAgent |
| **Load Response** | context briefing, workflows, structure, recent files, key files, preferences, sessions, states |
| **Sessions** | Stored nested in workspace, contain memory traces and states |
| **States** | Saved checkpoints with full snapshot for resumption |
| **Indexing** | Fast metadata lookups by name, description, folder |
| **Required for Creation** | name, rootFolder, purpose, currentGoal, workflows |
| **Optional for Creation** | keyFiles, preferences, dedicatedAgentId, description, related items |

