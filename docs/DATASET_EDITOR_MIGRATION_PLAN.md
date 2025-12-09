# Dataset Editor Migration Plan: Streamlit to Next.js

## 1. Overview
This document outlines the plan to migrate the existing Streamlit-based Dataset Editor to a modern web application using **Next.js** and **shadcn/ui**. The goal is to improve performance, interactivity, and maintainability by moving to a client-side state model with server-side file operations.

## 2. Tech Stack

| Component | Choice | Reasoning |
| :--- | :--- | :--- |
| **Framework** | **Next.js 14 (App Router)** | Built-in routing, Server Actions for filesystem access, easy deployment. |
| **Language** | **TypeScript** | Type safety for complex JSON structures (Thinking Blocks, Tool Calls). |
| **Styling** | **Tailwind CSS** | Rapid UI development, required by shadcn/ui. |
| **UI Library** | **shadcn/ui** | High-quality, accessible, copy-paste components (Radix UI based). |
| **Tables** | **@tanstack/react-table** | Headless, highly performant table logic (sorting, filtering, pagination). |
| **Forms** | **React Hook Form + Zod** | Robust form validation and state management. |
| **Icons** | **Lucide React** | Consistent, clean iconography. |
| **LLM Client** | **Standard Fetch / AI SDK** | Direct API calls to OpenRouter/Ollama from Server Actions. |

## 3. Architecture

### 3.1. Directory Structure
```
app/
├── page.tsx                    # Dataset Library (Card View)
├── layout.tsx                  # Global layout (Sidebar, Theme provider)
├── dataset/
│   ├── [filename]/
│   │   ├── page.tsx            # Dataset Explorer (Table View)
│   │   └── [index]/
│   │       └── page.tsx        # Entry Editor (Form View)
lib/
├── actions.ts                  # Server Actions (FS operations, LLM calls)
├── utils.ts                    # Helper functions (Thinking block parser)
├── types.ts                    # TypeScript interfaces
components/
├── ui/                         # shadcn components (Button, Input, etc.)
├── dataset-card.tsx            # Card component for library
├── data-table.tsx              # Reusable TanStack table
├── thinking-editor.tsx         # Specialized editor for thinking blocks
└── tool-call-editor.tsx        # Recursive JSON editor for tools
```

### 3.2. Data Flow
1.  **Read**: Server Components fetch data using **Server Actions** (reading local JSONL files).
2.  **Render**: Data is passed to Client Components.
3.  **Edit**: Client Components manage local form state (instant feedback).
4.  **Save**: User clicks "Save", triggering a Server Action to write back to the JSONL file.
5.  **Regenerate**: Server Action calls LLM API and returns new JSON to update Client state.

## 4. Data Models (`lib/types.ts`)

```typescript
export interface ThinkingBlock {
  goal: string;
  memory: string;
  requirements: string[];
  plan: string[];
  confidence: number;
  assessment: {
    complex: boolean;
    risky: boolean;
  };
}

export interface ToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string; // JSON string
  };
}

export interface Message {
  role: "system" | "user" | "assistant" | "tool";
  content: string | null;
  tool_calls?: ToolCall[];
  // Computed fields for UI
  parsedThinking?: ThinkingBlock;
}

export interface DatasetEntry {
  conversations: Message[];
  source?: string;
}
```

## 5. Step-by-Step Implementation Plan

### Phase 1: Setup & Scaffolding
1.  Initialize Next.js project: `npx create-next-app@latest dataset-editor --typescript --tailwind --eslint`.
2.  Initialize shadcn/ui: `npx shadcn-ui@latest init`.
3.  Install core components: `button`, `card`, `input`, `textarea`, `table`, `dialog`, `form`, `select`, `switch`.
4.  Install dependencies: `lucide-react`, `@tanstack/react-table`, `zod`, `react-hook-form`.

### Phase 2: Core Data Access (Server Actions)
Implement `lib/actions.ts` to handle filesystem operations.

**Pseudocode (`lib/actions.ts`):**
```typescript
'use server'
import fs from 'fs/promises'

export async function getDatasets() {
  // Scan directories (Datasets/behavior_datasets, etc.)
  // Return list of { name, path, size, count }
}

export async function getDatasetContent(filePath: string) {
  // Read file line by line
  // Parse JSON
  // Return Array<DatasetEntry>
}

export async function saveDatasetEntry(filePath: string, index: number, entry: DatasetEntry) {
  // Read all lines
  // Replace line at index
  // Write back to file
}
```

### Phase 3: Dataset Library (Home Page)
Create the landing page that lists available datasets.
*   **UI**: Grid of `Card` components.
*   **Logic**: Call `getDatasets()` on load.
*   **Interactivity**: Clicking a card navigates to `/dataset/[filename]`.

### Phase 4: Dataset Explorer (Table View)
Create the table view to list examples within a dataset.
*   **UI**: `DataTable` component using `@tanstack/react-table`.
*   **Columns**: "Index", "First User Query", "Has Tools", "Turn Count".
*   **Interactivity**:
    *   Click row -> Navigate to Editor.
    *   Search bar -> Client-side filtering of the table data.

### Phase 5: Entry Editor & Thinking Logic
This is the most complex part. We need to port the Python regex logic for `<thinking>` tags to TypeScript.

**Pseudocode (`lib/utils.ts`):**
```typescript
export function parseThinking(content: string): ThinkingBlock | null {
  // Regex to extract content between <thinking> tags
  // JSON.parse the content
  // Return typed object
}

export function constructThinking(block: ThinkingBlock): string {
  // JSON.stringify block
  // Wrap in <thinking> tags
}
```

**UI Components:**
*   `ThinkingEditor`: A form with specific inputs for Goal, Memory, Plan (array input), etc.
*   `ToolCallEditor`: A recursive JSON tree editor (or simple text area with validation) for tool arguments.

### Phase 6: LLM Integration
Connect the "Regenerate" button.
*   **Config**: Store API keys in `.env.local`.
*   **Server Action**: `regenerateThinking(currentBlock: ThinkingBlock)`
    *   Reads system prompt from `config/system_prompts.yaml`.
    *   Calls OpenRouter/Ollama API.
    *   Returns new `ThinkingBlock`.

## 6. Migration Checklist

- [ ] **Project Init**: Create Next.js app.
- [ ] **Server Actions**: Implement file reading/writing.
- [ ] **Parsers**: Port `parse_thinking` and `construct_thinking` to TS.
- [ ] **Home Page**: List datasets.
- [ ] **Table View**: View dataset rows.
- [ ] **Editor View**:
    - [ ] Message list rendering.
    - [ ] Thinking block form.
    - [ ] Tool call form.
- [ ] **LLM**: Implement regeneration action.
- [ ] **Testing**: Verify read/write integrity on sample files.

## 7. Future Improvements (Post-Migration)
*   **Virtualization**: Use `react-window` for datasets with 10k+ rows to avoid rendering DOM for all rows.
*   **Diff View**: Show side-by-side diffs when regenerating thinking blocks.
*   **Bulk Actions**: Select multiple rows in the table to delete or export.
