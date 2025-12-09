# Synaptic Tuner Web UI Architecture

## 1. Overview
This document outlines the architecture for the **Synaptic Tuner Web UI**, a comprehensive frontend for the Synaptic Tuner CLI. It unifies dataset management, training (SFT/KTO), model evaluation, and deployment into a single modern web application.

## 2. Tech Stack

| Component | Choice | Reasoning |
| :--- | :--- | :--- |
| **Framework** | **Next.js 14 (App Router)** | Server Actions for filesystem/subprocess control, easy deployment. |
| **Language** | **TypeScript** | Type safety for complex configs and data structures. |
| **UI Library** | **shadcn/ui** | Professional, accessible components (Radix UI + Tailwind). |
| **State** | **React Query / SWR** | Efficient data fetching and caching. |
| **Streaming** | **Server-Sent Events (SSE)** | Real-time streaming of training logs/terminal output. |
| **Charts** | **Recharts** | Visualizing training metrics (loss curves). |
| **Editor** | **Monaco Editor** | For editing YAML configs and JSONL files. |

## 3. Core Modules

### 3.1. Dashboard (Home)
*   **System Status**: GPU usage (via `nvidia-smi`), Disk space.
*   **Recent Activity**: Last training runs, recent dataset edits.
*   **Quick Actions**: "Start SFT", "New Dataset", "Run Eval".

### 3.2. Dataset Manager (`/datasets`)
*   **Library**: Card view of available datasets (JSONL).
*   **Explorer**: Table view of dataset entries (TanStack Table).
*   **Editor**: Form-based editor for `<thinking>` blocks and tool calls.
*   **Improvement Engine**: Integration with LLM backends to regenerate thinking blocks.

### 3.3. Training Manager (`/training`)
*   **Job Launcher**:
    *   **Auto-detected Platform**: Automatically identifies GPU (RTX) or Apple Silicon (MLX) backend.
    *   Select Method: SFT vs KTO.
    *   Config Editor: Visual form + YAML fallback for `config.yaml`.
*   **Job Monitor**:
    *   Real-time terminal output (streaming).
    *   Live Loss Curves (parsing logs or W&B API).
    *   Stop/Kill job button.
*   **History**: List of past runs with status (Success/Fail) and metrics.

### 3.4. Model Manager (`/models`)
*   **Local Models**: List checkpoints in `output/` directories.
*   **Upload**: Interface for `shared.upload.cli.upload_cli`.
    *   Select Checkpoint.
    *   Set HF Repo ID.
    *   Toggle GGUF Conversion (Quantization options).
*   **GGUF**: Standalone GGUF conversion tool.

### 3.5. Evaluation Manager (`/evaluation`)
*   **Runner**: Interface for `Evaluator/cli.py`.
    *   Select Model (Local/Ollama/LM Studio).
    *   Select Test Set.
    *   Configure Prompts.
    *   **Live Progress**: Stream evaluation logs and progress bars to the UI.
*   **Results & Comparison**:
    *   Scorecards (Pass/Fail rates).
    *   Detailed breakdown of tool usage.
    *   **Historical Trends**: Unified charts comparing the last 3-5 runs to track regression/improvement.
    *   **Side-by-Side**: Compare specific metrics between selected runs.

### 3.6. Settings (`/settings`)
*   **Environment**: Editor for `.env` file (HF_TOKEN, WANDB_API_KEY, LLM Keys).
*   **Paths**: Configure paths to Python environments or external tools.

## 4. Backend Architecture (Server Actions)

The Next.js Server Actions will act as the bridge to the existing Python scripts.

### 4.1. Subprocess Manager
A utility to spawn and manage long-running Python processes.

```typescript
// lib/process-manager.ts
export async function spawnJob(command: string[], args: string[], onLog: (line: string) => void) {
  // Spawns python process
  // Streams stdout/stderr to callback
  // Returns process ID
}
```

### 4.2. Config Manager
Utilities to read/write the YAML configuration files used by the trainers.

```typescript
// lib/config-manager.ts
export async function getTrainingConfig(method: 'sft' | 'kto') {
  // Reads Trainers/rtx3090_{method}/configs/config.yaml
  // Returns parsed JSON
}

export async function saveTrainingConfig(method: 'sft' | 'kto', config: any) {
  // Converts JSON to YAML
  // Writes to file
}
```

### 4.3. Log Parser
A utility to parse the raw stdout from training scripts to extract metrics for charting.

```typescript
// lib/log-parser.ts
export function parseTrainingLine(line: string) {
  // Regex to extract: {'loss': 0.123, 'epoch': 1.5}
  // Returns metric object
}
```

## 5. Directory Structure

```
app/
├── page.tsx                    # Dashboard
├── datasets/                   # Dataset Manager
├── training/
│   ├── page.tsx                # Job List
│   ├── new/                    # Config Editor
│   └── [id]/                   # Live Monitor
├── models/                     # Model Manager
├── evaluation/                 # Evaluation Manager
└── settings/                   # Settings
lib/
├── actions/                    # Server Actions
│   ├── datasets.ts
│   ├── training.ts
│   ├── models.ts
│   └── settings.ts
├── hooks/                      # React Hooks
└── utils/                      # Helpers
components/
├── ui/                         # shadcn components
├── terminal.tsx                # XTerm.js or simple log viewer
├── config-form.tsx             # Dynamic form for YAML configs
└── metrics-chart.tsx           # Recharts wrapper
```

## 6. Implementation Task List

### Phase 1: Foundation & Setup
- [x] **Project Initialization**
    - [x] Create `web-ui/` directory
    - [x] `web-ui/package.json` (Next.js, React, Tailwind, shadcn-ui, lucide-react, recharts, xterm)
    - [x] `web-ui/tsconfig.json`
    - [x] `web-ui/tailwind.config.ts`
    - [x] `web-ui/.env.local` (Mapped from root `.env`)
    - [x] Update root `.gitignore`
- [x] **Layout & Navigation**
    - [x] `web-ui/app/layout.tsx` (Sidebar provider, Theme provider)
    - [ ] `web-ui/components/app-sidebar.tsx` (Navigation menu)
    - [x] `web-ui/app/page.tsx` (Dashboard skeleton)

### Phase 2: Dataset Manager
- [x] **Server Actions**
    - [x] `web-ui/lib/actions/datasets.ts` (List files, Read JSONL, Save Entry)
    - [x] `web-ui/lib/utils/thinking-parser.ts` (Port regex logic from `dataset_editor.py`)
- [x] **UI Components**
    - [x] `web-ui/app/datasets/page.tsx` (Library Card View)
    - [x] `web-ui/app/datasets/[filename]/page.tsx` (TanStack Table View)
    - [x] `web-ui/components/thinking-editor.tsx` (Form for Goal, Plan, etc.)
    - [x] `web-ui/components/tool-editor.tsx` (JSON editor for tool calls)
- [ ] **LLM Integration**
    - [ ] `web-ui/lib/actions/llm.ts` (OpenRouter/Ollama clients)
    - [ ] Connect "Regenerate" button in Editor

### Phase 3: Training Manager
- [ ] **Server Actions**
    - [ ] `web-ui/lib/actions/training.ts` (Job spawning, Config reading)
- [ ] **UI Components**
    - [ ] `web-ui/app/training/page.tsx` (Job Launcher & History)
    - [ ] `web-ui/components/config-form.tsx` (YAML config editor)
    - [ ] `web-ui/components/terminal-viewer.tsx` (XTerm.js streaming)
    - [ ] `web-ui/components/metrics-chart.tsx` (Recharts loss curve)

### Phase 4: Evaluation Manager
- [ ] **Server Actions**
    - [ ] `web-ui/lib/actions/evaluation.ts` (Runner spawning)
- [ ] **UI Components**
    - [ ] `web-ui/app/evaluation/page.tsx` (Runner interface)
    - [ ] `web-ui/components/eval-progress.tsx` (Live progress bars)
    - [ ] `web-ui/components/eval-results.tsx` (Scorecards & Comparison charts)

### Phase 5: Model Manager & Settings
- [ ] **Model Management**
    - [ ] `web-ui/app/models/page.tsx` (List local models)
    - [ ] `web-ui/lib/actions/models.ts` (Upload/GGUF logic)
- [ ] **Settings**
    - [ ] `web-ui/app/settings/page.tsx` (Env var editor)
    - [ ] `web-ui/lib/actions/settings.ts` (Read/Write .env)
