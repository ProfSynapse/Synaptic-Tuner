# Workspace Architecture & Data Flow Diagrams

## 1. Workspace Data Hierarchy

```
Workspace (Root)
│
├── id: "ws_1729990000000_abc123"
├── name: "Job Search 2025"
├── rootFolder: "Job Search/2025"
├── created: 1729990000000
├── lastAccessed: 1730005000000
│
└── context: WorkspaceContext
    ├── purpose: "Apply for marketing manager positions"
    ├── currentGoal: "Submit 10 applications this week"
    │
    ├── workflows: [
    │   ├── {
    │   │   name: "New Application",
    │   │   when: "When applying to new position",
    │   │   steps: "Research company\nCustomize cover letter\nApply\nTrack"
    │   ├── },
    │   └── {...}
    │]
    │
    ├── keyFiles: ["Resume-2025.md", "tracker.md", "template.md"]
    │
    ├── preferences: "Use professional tone. Focus on tech companies."
    │
    └── dedicatedAgent?: {
        agentId: "agent_marketing_specialist",
        agentName: "Marketing Specialist"
    }

└── sessions: Record<sessionId, SessionData>
    │
    └── session_1730000000000_abc123
        ├── id: "session_1730000000000_abc123"
        ├── name: "Google Application Week"
        ├── startTime: 1730000000000
        ├── isActive: true
        │
        ├── memoryTraces: Record<traceId, MemoryTrace>
        │   ├── trace_1: { timestamp, type, content, metadata }
        │   └── trace_2: { ... }
        │
        └── states: Record<stateId, StateData>
            │
            └── state_1730000000000_xyz789
                ├── name: "Google Cover Letter Draft"
                ├── created: 1729999000000
                │
                └── snapshot: StateSnapshot
                    ├── conversationContext: "Working on Google position"
                    ├── activeTask: "Finish cover letter"
                    ├── activeFiles: ["Cover Letters/google-customized.md"]
                    ├── nextSteps: ["Review", "Submit"]
                    ├── reasoning: "Need to save before context limit"
                    │
                    └── workspaceContext: (copy of parent context)
```

## 2. Storage Architecture (File System Layout)

```
Vault Root/
│
├── .workspaces/                     # Workspace storage
│   ├── index.json                   # Fast metadata lookups
│   │   ├── workspaces: {}           # All workspace metadata
│   │   ├── byName: {}               # Name-based search
│   │   ├── byDescription: {}        # Description search
│   │   ├── byFolder: {}             # Folder mapping
│   │   └── sessionsByWorkspace: {}  # Session tracking
│   │
│   ├── ws_1729990000000_abc123.json # Individual workspace file
│   │   ├── id, name, context
│   │   └── sessions: {}             # All sessions for this workspace
│   │
│   └── ws_1730000000000_def456.json # Another workspace
│
├── .conversations/                  # Conversation storage
│   ├── index.json
│   ├── conv_1729990000000_abc123.json
│   └── ...
│
└── Job Search/                      # Actual vault content (referenced, not duplicated)
    ├── 2025/
    │   ├── Applications/
    │   │   ├── google-marketing-manager.md
    │   │   └── amazon-brand-manager.md
    │   ├── Cover Letters/
    │   │   ├── template.md
    │   │   └── google-customized.md
    │   ├── Resume-2025.md
    │   ├── tracker.md
    │   └── research/
    │       └── ...
    └── 2024/
        └── ...
```

## 3. LoadWorkspace Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ Client Request: memoryManager_loadWorkspace({ id: "ws_xxx" })   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           v
┌──────────────────────────────────────────────────────────────────┐
│ LoadWorkspaceMode.execute()                                      │
│                                                                  │
│  1. Get WorkspaceService from agent                             │
│  2. Load workspace by ID from disk                              │
│  3. Update lastAccessed timestamp                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           v
┌──────────────────────────────────────────────────────────────────┐
│ Parallel Data Collection                                         │
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ ContextBuilder  │  │ DataFetcher     │  │ FileCollector   │ │
│  │                 │  │                 │  │                 │ │
│  │ • Purpose       │  │ • Sessions      │  │ • Recent files  │ │
│  │ • Current goal  │  │ • States        │  │ • Structure     │ │
│  │ • Workflows     │  │ • Memory traces │  │ • Paths         │ │
│  │ • Preferences   │  │                 │  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           v
┌──────────────────────────────────────────────────────────────────┐
│ AgentResolver (if dedicatedAgent configured)                    │
│  • Fetch agent system prompt                                     │
│  • Resolve agent configuration                                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           v
┌──────────────────────────────────────────────────────────────────┐
│ Assemble LoadWorkspaceResult                                    │
│                                                                  │
│  data: {                                                         │
│    context: { ... briefing ... },                               │
│    workflows: [ "Workflow 1: steps", "Workflow 2: steps" ],     │
│    workspaceStructure: [ "file/paths/to/all/files" ],           │
│    recentFiles: [ { path, modified }, ... ],                    │
│    keyFiles: { path: "purpose", ... },                          │
│    preferences: "user guidelines",                               │
│    sessions: [ { id, name, created }, ... ],                    │
│    states: [ { id, name, sessionId, created }, ... ],           │
│    agent?: { agentId, agentName, systemPrompt }                 │
│  }                                                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           v
┌─────────────────────────────────────────────────────────────────┐
│ Return to Client                                                 │
│ {                                                               │
│   success: true,                                                │
│   data: { ... },                                               │
│   workspaceContext: { workspaceId, workspacePath }             │
│ }                                                               │
└─────────────────────────────────────────────────────────────────┘
```

## 4. Session & State Hierarchy

```
Workspace
│
└── sessions: Record<sessionId, SessionData>
    │
    ├── Session 1: "Google Application Week"
    │   │
    │   ├── memoryTraces
    │   │   ├── trace_1: { timestamp, type: "tool_call", content, metadata }
    │   │   ├── trace_2: { timestamp, type: "note_created", ... }
    │   │   └── trace_3: { ... }
    │   │
    │   └── states: Record<stateId, StateData>
    │       │
    │       └── State 1: "Google Cover Letter Draft"
    │           │
    │           └── snapshot: StateSnapshot
    │               ├── conversationContext
    │               ├── activeTask
    │               ├── activeFiles
    │               ├── nextSteps
    │               ├── reasoning
    │               └── workspaceContext: (preserved context at save time)
    │
    └── Session 2: "Amazon Application Week"
        ├── memoryTraces: [...]
        └── states: [...]
```

## 5. Index Structure for Fast Lookups

```
.workspaces/index.json
│
├── workspaces
│   ├── ws_xxx: WorkspaceMetadata
│   │   ├── id
│   │   ├── name
│   │   ├── rootFolder
│   │   ├── created
│   │   ├── lastAccessed (for sorting)
│   │   ├── sessionCount (cached)
│   │   └── traceCount (cached)
│   └── ws_yyy: { ... }
│
├── byName              # Fast name search
│   ├── "job" → [ws_xxx, ws_aaa]
│   ├── "search" → [ws_xxx]
│   └── "2025" → [ws_xxx, ws_yyy]
│
├── byDescription       # Full-text search ready
│   ├── "marketing" → [ws_xxx]
│   └── "positions" → [ws_xxx]
│
├── byFolder            # Folder-to-workspace mapping
│   ├── "Job Search/2025" → ws_xxx
│   └── "Research/Q4" → ws_yyy
│
└── sessionsByWorkspace # Quick session lookup
    ├── ws_xxx → [session_1, session_2, session_3]
    └── ws_yyy → [session_4]
```

## 6. Request/Response Message Types

```
CREATE WORKSPACE Request
└─ CreateWorkspaceParameters
   ├── name (required)
   ├── rootFolder (required)
   ├── purpose (required)
   ├── currentGoal (required)
   ├── workflows (required) [{name, when, steps}]
   ├── keyFiles (optional)
   ├── preferences (optional)
   ├── dedicatedAgentId (optional)
   └── ... legacy fields

CREATE WORKSPACE Response
└─ CreateWorkspaceResult
   ├── success: boolean
   ├── error?: string
   └── data
      ├── workspaceId: string
      └── workspace: ProjectWorkspace

─────────────────────────

LOAD WORKSPACE Request
└─ LoadWorkspaceParameters
   ├── id (required)
   └── limit (optional, default: 3)

LOAD WORKSPACE Response
└─ LoadWorkspaceResult
   ├── success: boolean
   ├── error?: string
   ├── data
   │   ├── context: { name, description, purpose, rootFolder, recentActivity }
   │   ├── workflows: string[]
   │   ├── workspaceStructure: string[]
   │   ├── recentFiles: [{ path, modified }]
   │   ├── keyFiles: Record<path, purpose>
   │   ├── preferences: string
   │   ├── sessions: [{ id, name, description, created }]
   │   ├── states: [{ id, name, description, sessionId, created, tags }]
   │   └── agent?: { agentId, agentName, systemPrompt }
   │
   └── workspaceContext
       ├── workspaceId: string
       └── workspacePath: string[]

─────────────────────────

LIST WORKSPACES Request
└─ ListWorkspacesParameters
   ├── sortBy: 'name' | 'created' | 'lastAccessed'
   ├── order: 'asc' | 'desc'
   └── limit: number

LIST WORKSPACES Response
└─ ListWorkspacesResult
   ├── success: boolean
   └── data
       └── workspaces: [{
           ├── id
           ├── name
           ├── description
           ├── rootFolder
           ├── lastAccessed
           └── childCount
       }]
```

## 7. Context-Driven AI Workflow

```
LOAD WORKSPACE
   │
   v
AI reads:
   ├── "Purpose: Apply for marketing manager positions"
   ├── "Current Goal: Submit 10 applications this week"
   │
   ├── Available Workflows:
   │   ├── "New Application" → (research, customize, apply, track)
   │   ├── "Interview Prep" → (review, prepare, practice, research)
   │   └── "Follow-up" → (wait, check, send, log)
   │
   ├── Key Files:
   │   ├── Resume-2025.md → "Primary resume for all applications"
   │   ├── tracker.md → "Central tracking of applications"
   │   └── template.md → "Base cover letter template"
   │
   ├── Preferences: "Professional tone. Focus on tech companies."
   │
   └── Recent Sessions:
       ├── "Google Application Week" → session_xxx
       ├── "Amazon Interview Prep" → session_yyy
       └── ...
   │
   v
AI selects workflow based on context:
   "I should follow the 'New Application' workflow"
   │
   ├─ Step 1: Research company → vaultLibrarian_searchContent
   ├─ Step 2: Customize letter → contentManager_readContent + appendContent
   ├─ Step 3: Track → vaultManager_openNote + contentManager_appendContent
   └─ Step 4: Save progress → memoryManager_createState
   │
   v
SAVE STATE
   └─ createState({
      name: "Amazon Marketing Manager Application",
      conversationContext: "Working on Amazon application",
      activeTask: "Finishing cover letter customization",
      activeFiles: ["Cover Letters/amazon-customized.md"],
      nextSteps: ["Review cover letter", "Submit application"],
      reasoning: "Checkpoint before submission"
   })
   │
   v
Later: LOAD STATE
   └─ Restore full context:
      ├── What was I doing? (conversationContext)
      ├── What's next? (nextSteps)
      ├── Which files? (activeFiles)
      └── What's the workspace context? (workspaceContext snapshot)
```

## 8. Workspace Creation Parameters Flow

```
User Input
   │
   ├─ Name: "Job Search 2025"
   ├─ Root Folder: "Job Search/2025"
   ├─ Purpose: "Apply for marketing manager positions"
   ├─ Current Goal: "Submit 10 applications this week"
   │
   ├─ Workflows:
   │   ├─ New Application
   │   │  └─ Steps: "Research\nCustomize cover letter\nApply\nTrack"
   │   └─ Interview Prep
   │      └─ Steps: "Review job description\nPrepare stories\nPractice\nResearch"
   │
   ├─ Key Files: ["Resume-2025.md", "tracker.md"]
   ├─ Preferences: "Professional tone. Tech companies."
   └─ Dedicated Agent: "agent_marketing_specialist" (optional)
   │
   v
createWorkspace() validation
   ├─ Check all required fields present
   ├─ Validate rootFolder path exists or create
   ├─ Load dedicated agent details if specified
   └─ Auto-detect additional key files
   │
   v
Build WorkspaceContext
   ├─ purpose: "Apply for marketing manager positions"
   ├─ currentGoal: "Submit 10 applications this week"
   ├─ workflows: [{ name, when, steps }]
   ├─ keyFiles: [auto-detected + provided]
   ├─ preferences: "Professional tone. Tech companies."
   └─ dedicatedAgent: { agentId, agentName }
   │
   v
Create IndividualWorkspace
   ├─ id: "ws_" + timestamp + random
   ├─ name, rootFolder
   ├─ created: Date.now()
   ├─ lastAccessed: Date.now()
   ├─ context: (from above)
   └─ sessions: {} (empty)
   │
   v
Store & Index
   ├─ Write to .workspaces/{id}.json
   ├─ Update .workspaces/index.json
   │   ├─ Add workspaceMetadata
   │   ├─ Update byName, byDescription, byFolder
   │   └─ Update lastUpdated
   │
   v
Return CreateWorkspaceResult
   ├─ success: true
   ├─ data:
   │   ├─ workspaceId: "ws_..."
   │   └─ workspace: ProjectWorkspace
```

---

These diagrams show the complete architecture and data flow of the workspace system!
