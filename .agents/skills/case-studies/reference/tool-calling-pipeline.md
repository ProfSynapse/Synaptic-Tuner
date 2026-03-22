# Case Study: Tool-Calling Training Pipeline

A complete walkthrough of how the project teaches a language model to call tools correctly — from raw API schemas to a fine-tuned model that selects the right tool, constructs valid arguments, and maintains session context.

---

## Stage 1: Define the Capability

### What We're Teaching

The model must learn to:
1. **Call `useTools`** with the correct agent/tool/params structure via a single unified function
2. **Maintain context** — pass session IDs, workspace IDs, memory, and goal in every call
3. **Select the right agent and tool** from 5 managers with 29 total tools
4. **Reason before acting** — use `<thinking>` blocks for complex/risky operations
5. **Show judgment** — ask for clarification on vague or destructive requests

### The Source of Truth: Tool Schemas

Everything starts with `tool-schemas.json` (v2.0.0) — the canonical definition of every tool, its parameters, and its validation rules. It includes:

- A **migrations** map showing old → new agent/tool names (for backward compatibility)
- A **context** schema defining the required fields for every `useTools` call
- A **tools** object with per-tool JSON schemas
- An **agents** summary listing each agent's tool count and tool names

### The `useTools` Pattern

All tool calls go through a single function called `useTools`. It wraps context + one or more tool calls:

```json
{
  "name": "useTools",
  "arguments": {
    "context": {
      "workspaceId": "ws_1731020200000_q5r4s3t2u",
      "sessionId": "session_1731020200000_l9m8n7o6p",
      "memory": "User is organizing AI research materials and wants to add a Resources folder.",
      "goal": "Create a Resources folder inside AI Research."
    },
    "calls": [
      {
        "agent": "storageManager",
        "tool": "createFolder",
        "params": {
          "path": "AI Research/Resources"
        }
      }
    ]
  }
}
```

**Key features of `useTools`:**
- **Single function** — the model always calls `useTools`, never individual tool functions
- **`calls` array** — supports multiple tool calls in one invocation (batching)
- **Each call** specifies `agent`, `tool`, and `params` separately

### The Context Object Contract

Every `useTools` call must include a context object with 4 required fields:

```json
{
  "context": {
    "workspaceId": "ws_1731020200000_q5r4s3t2u",
    "sessionId": "session_1731020200000_l9m8n7o6p",
    "memory": "Conversation essence — 1-3 sentences of what's happened so far",
    "goal": "Current objective — 1-3 sentences of what this call achieves"
  }
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `workspaceId` | ✅ | Scope identifier — use `"default"` for global |
| `sessionId` | ✅ | Session name (system assigns ID) |
| `memory` | ✅ | Conversation essence (1-3 sentences) — NEVER empty |
| `goal` | ✅ | Current objective (1-3 sentences) |
| `constraints` | ❌ | Optional rules/limits (1-3 sentences) |

**Critical rule:** `memory` must never be empty. This was a recurring failure in early training.

### Agent Categories (Current v2.0)

The tools are organized into 5 managers:

| Agent | Purpose | Tools |
|-------|---------|-------|
| `contentManager` | Read, update, write note content | `read`, `update`, `write` |
| `memoryManager` | Session states, workspaces | `createState`, `listStates`, `loadState`, `createWorkspace`, `listWorkspaces`, `loadWorkspace`, `updateWorkspace`, `archiveWorkspace` |
| `promptManager` | Prompt/agent lifecycle, LLM execution, subagents | `createPrompt`, `updatePrompt`, `archivePrompt`, `getPrompt`, `listPrompts`, `executePrompts`, `generateImage`, `listModels`, `subagent` |
| `searchManager` | Content search, directory search, memory search | `searchContent`, `searchDirectory`, `searchMemory` |
| `storageManager` | File/folder operations | `archive`, `copy`, `createFolder`, `list`, `move`, `open` |

**Migration note:** Older datasets may reference legacy names. The schema includes a migrations map:
- `agentManager` → `promptManager`
- `vaultManager` → `storageManager`
- `vaultLibrarian` → `searchManager`

### System Prompt Format

System prompts provide runtime context the model must USE. The current format includes these XML sections:

```xml
<session_context>
IMPORTANT: When using tools, include these values in your tool call parameters:
- sessionId: "session_1731071640123_d7m2v9c4r"
- workspaceId: "default" (no specific workspace selected)
Include these in the "context" parameter of your tool calls.
</session_context>

<vault_structure>
Folders:
 - Studios/
 - Projects/
 - Archive/
Files:
 - README.md
 - Home.md
</vault_structure>

<available_workspaces>
- Default Workspace (id: "default")
  Description: General catch-all vault
  Root folder: /

- Studio Hub (id: "ws_studio_hub")
  Description: Dedicated workspace for creative studios
  Root folder: Studios/
</available_workspaces>

<available_prompts>
- name: FileManagerAgent
  description: Executes file system operations.
- name: ContentEditor
  description: Assists with markdown editing.
</available_prompts>

<selected_workspace name="Default Workspace" id="default">
{
  "context": { ... },
  "workspaceStructure": [ ... ],
  "recentFiles": [ ... ],
  "keyFiles": [ ... ],
  "workflows": [ ... ],
  "preferences": "...",
  "sessions": []
}
</selected_workspace>
```

**Key sections:**
- `<session_context>` — IDs the model must inject into `useTools` context
- `<vault_structure>` — Top-level folders and files
- `<available_workspaces>` — Workspaces with IDs, descriptions, root folders
- `<available_prompts>` — Named prompts (formerly "agents") available for use
- `<selected_workspace>` — Full details of the active workspace including structure, workflows, preferences

### Behavioral Rubrics

Beyond structural correctness, we define 6 behavioral patterns the model should exhibit:

| Behavior | What It Means | Priority |
|----------|--------------|----------|
| Intellectual Humility | Ask before acting on ambiguity | HIGH |
| Verification Before Action | Read before deleting/overwriting | HIGH |
| Context Continuity | Rich memory, workflow tracking | HIGH |
| Strategic Tool Selection | Batch when efficient, right tool for the job | MEDIUM |
| Error Recovery | Handle failures, adapt strategy | MEDIUM |
| Workspace Awareness | Use workspace metadata in decisions | MEDIUM |

Each behavior has a YAML rubric in `Datasets/behavior_rubrics/` defining positive indicators, negative indicators, and trigger scenarios.

---

## Stage 2: Create Training Data

### Strategy: Three Data Sources

```
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│  1. HANDCRAFTED     │  │  2. SYNTHCHAT        │  │  3. SELF-PLAY       │
│     SEED EXAMPLES   │  │     GENERATION       │  │     GENERATION      │
│                     │  │                      │  │                     │
│  ~50 gold-standard  │  │  Scale to thousands  │  │  Model generates    │
│  examples, one per  │  │  using LLM-guided    │  │  its own training   │
│  tool, handcrafted  │  │  scenario templates  │  │  data via temp      │
│  for accuracy       │  │  + rubric validation │  │  variation + eval   │
└────────┬────────────┘  └────────┬─────────────┘  └────────┬────────────┘
         │                       │                          │
         └───────────────────────┼──────────────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │   MERGED DATASET        │
                    │   Validated & balanced   │
                    └─────────────────────────┘
```

### 2a. Handcrafted Seed Examples

These are the first ~50 examples, written by hand with extreme care. One per tool minimum.

**Location:** `Datasets/tools_datasets/{thinking|non_thinking}/{agent_category}/tools_v{version}.jsonl`

**Process:**
1. Read the tool schema for the target tool in `tool-schemas.json`
2. Read the spec: `Datasets/tools_datasets/specs/{agent}_spec.md`
3. Follow the shared guidelines: `Datasets/tools_datasets/specs/00_shared_guidelines.md`
4. Write the example with unique IDs, realistic scenarios, and correct parameters
5. Validate against the tool schema

**Example — Complete training example (label: true):**

```json
{
  "conversations": [
    {
      "role": "system",
      "content": "<session_context>...</session_context>\n<vault_structure>...</vault_structure>\n<available_workspaces>...</available_workspaces>\n<available_prompts>...</available_prompts>\n<selected_workspace>...</selected_workspace>"
    },
    {
      "role": "user",
      "content": "Create a Resources folder within my AI Research project folder."
    },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_x7k9m2qz1",
          "type": "function",
          "function": {
            "name": "useTools",
            "arguments": "{\"context\": {\"workspaceId\": \"ws_1731020200000_q5r4s3t2u\", \"sessionId\": \"session_1731020200000_l9m8n7o6p\", \"memory\": \"User is organizing AI research materials and wants to add a Resources folder.\", \"goal\": \"Create a Resources folder inside AI Research.\"}, \"calls\": [{\"agent\": \"storageManager\", \"tool\": \"createFolder\", \"params\": {\"path\": \"AI Research/Resources\"}}]}"
          }
        }
      ]
    }
  ],
  "label": true
}
```

**Multi-tool batching example:**

```json
{
  "function": {
    "name": "useTools",
    "arguments": "{\"context\": {\"workspaceId\": \"default\", \"sessionId\": \"session_1731071640123_d7m2v9c4r\", \"memory\": \"User is reorganizing the vault, moving the LightRay board into the archive.\", \"goal\": \"Move LightRay to archive and open Hero.md from archived location.\"}, \"calls\": [{\"agent\": \"storageManager\", \"tool\": \"move\", \"params\": {\"path\": \"Studios/Boards/LightRay\", \"newPath\": \"Studios/Archive/LightRay\", \"overwrite\": false}}, {\"agent\": \"storageManager\", \"tool\": \"open\", \"params\": {\"path\": \"Studios/Archive/LightRay/Hero.md\"}}]}"
  }
}
```

**Two additional example types beyond tool calls:**

1. **Clarification examples** — model asks questions instead of acting on vague/destructive requests:
   ```json
   {
     "role": "assistant",
     "content": "I'd like to clarify a few things:\n\n- Which file specifically?\n- Should I include subfolders?\n\nThis will help me target the right content."
   }
   ```

2. **Summary examples** — model summarizes tool results for the user:
   ```json
   {
     "role": "assistant",
     "content": "# Files Relocated\n\n**Moved 3 notes to Projects/Q4:**\n- meeting-notes.md\n- budget-draft.md\n- timeline.md\n\nAnything else to organize?"
   }
   ```

### 2b. SynthChat Generation (Scaling Up)

Once you have seed examples, use SynthChat to generate thousands more.

**Scenario file:** `SynthChat/scenarios/tools.yaml`

Each tool gets a scenario definition with three prompt templates:
- `system` — generates a realistic vault/workspace context with IDs, vault structure, available prompts, and selected workspace
- `user` — generates a natural user request (often intentionally vague)
- `assistant` — generates the `useTools` call with correct context and agent/tool/params

**Running generation:**
```bash
python -m SynthChat.run generate \
  --scenarios tools \
  --workers 4 \
  --output Datasets/synthchat/tools_generated.jsonl
```

### 2c. Self-Play Generation (Model Trains Itself)

After the first training round, use the trained model to generate its own data:

```bash
python Tools/selfplay_generator.py \
  --model claudesidian-mcp \
  --prompt-set Evaluator/prompts/tool_prompts.json \
  --output Datasets/syngen_selfplay.jsonl \
  --num-examples 1000 \
  --num-variations 3
```

**How it works:**
1. Load prompts (one per tool), each with system prompt + user question
2. For each prompt, generate 3 responses at different temperatures (0.3, 0.7, 0.9)
3. Low temperature → mostly correct → `label: true`
4. High temperature → more errors → `label: false`
5. Validate each response against tool schemas
6. Interleave true/false in alternating pattern
7. Output ready for KTO training

---

## Stage 3: Validate & Improve

### Structural Validation

Run the schema validator on every dataset before training:

```bash
python3 .skills/synethetic-data-generation/scripts/validate_syngen.py Datasets/tools_generated.jsonl
```

**What it checks:**
- ✅ Valid JSON/JSONL format
- ✅ `conversations` array with correct role sequence
- ✅ `useTools` call structure (context + calls array)
- ✅ Context object present with all 4 required fields (`workspaceId`, `sessionId`, `memory`, `goal`)
- ✅ `memory` is not empty
- ✅ Each call has `agent`, `tool`, `params`
- ✅ Agent name is valid (`contentManager`, `memoryManager`, `promptManager`, `searchManager`, `storageManager`)
- ✅ Tool name exists for that agent in `tool-schemas.json`
- ✅ Required parameters present per tool schema

### Rubric-Based Quality Validation

Use SynthChat's improvement loop to score and fix examples:

```bash
python -m SynthChat.run validate \
  -i Datasets/tools_generated.jsonl \
  --rubrics system_prompt_format,tool_alignment
```

If quality is low, run the improvement loop:

```bash
python -m SynthChat.run improve \
  -i Datasets/tools_generated.jsonl \
  --rubrics system_prompt_format,tool_alignment \
  --max-iterations 10
```

### Manual Review

Sample 10-20% of examples and check:
- Are user requests natural and varied?
- Do positive examples demonstrate ALL positive behavioral indicators?
- Are negative examples realistic mistakes (not nonsense)?
- For paired examples, do they use the identical user request?
- Does the `useTools` context use IDs from the system prompt (not hallucinated)?

---

## Stage 4: Train

### Training Sequence

```
SFT (learn format)  →  KTO (learn preferences)  →  GRPO (optimize rewards)
      ↓                       ↓                          ↓
  "What to do"          "Which is better"          "Maximize score"
```

### SFT: Teaching the `useTools` Format

**Dataset:** Positive examples only (all `label: true` or no labels).

```bash
cd Trainers/rtx3090_sft

python train_sft.py \
  --model-size 7b \
  --local-file ../../Datasets/tools_sft_merged.jsonl \
  --num-epochs 3 \
  --learning-rate 2e-4
```

**What SFT learns:**
- The `useTools` wrapper format (context + calls array)
- Context object structure (4 required fields, using IDs from system prompt)
- Agent/tool/params routing (which agent owns which tool)
- Multi-call batching (multiple calls in one `useTools` invocation)
- `<thinking>` block format for complex tasks
- Response patterns (clarification, summary, direct action)

**Key SFT settings:**
- Learning rate: `2e-4` (aggressive — learning new patterns)
- Epochs: `3` (enough to memorize format)
- Packing: `true` (2.5-5x faster)
- Completion-only loss: `true` (only learn from assistant responses)

### KTO: Teaching Judgment

**Dataset:** Interleaved true/false pairs. Same user request, different quality responses.

```bash
cd Trainers/rtx3090_kto

python train_kto.py \
  --model-size 7b \
  --local-file ../../Datasets/behavior_merged_kto_balanced.jsonl \
  --num-epochs 1 \
  --learning-rate 1e-6
```

**What KTO learns:**
- Prefer clarification over blind action on vague requests
- Prefer verification before destructive operations (archive/overwrite)
- Prefer rich context (`memory` and `goal`) over lazy/empty fields
- Prefer batch operations over repetitive single calls
- Prefer using workspace metadata (workflows, preferences) in reasoning

**Key KTO settings:**
- Learning rate: `1e-6` (very low — refining, not relearning)
- Epochs: `1` (preference signals are strong)
- Dataset must be interleaved true/false (data loader handles this)

### GRPO: Optimizing for Rewards (Optional)

```bash
cd Trainers/rtx3090_grpo
# Edit configs/config.yaml to set model.lora_path to KTO checkpoint
python train_grpo.py
```

---

## Stage 5: Evaluate

### Run Evaluation

```bash
python -m Evaluator.cli \
  --backend unsloth \
  --model ./Trainers/rtx3090_sft/sft_output_rtx3090/TIMESTAMP/final_model \
  --scenario behavior_prompts.yaml tool_prompts.yaml \
  --output Evaluator/results/eval_v1.json \
  --markdown Evaluator/results/eval_v1.md
```

### Key Metrics

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| `pass_rate` | Overall correctness | > 80% |
| `schema_pass_rate` | Tool selection accuracy | > 90% |
| `behavior_pass_rate` | Behavioral quality | > 70% |
| `by_tag` breakdown | Per-capability performance | Varies |

### Status Meanings

| Status | Meaning |
|--------|---------|
| **PASS** | Correct tool + good behavior |
| **WARN** | Right tool, but suboptimal behavior (KTO signal) |
| **FAIL** | Wrong tool, missing tool, or error |

**WARN results are gold for KTO** — they're "correct but could be better" examples.

---

## Stage 6: Iterate

### Common Failure Patterns and Fixes

| Failure Pattern | Root Cause | Fix |
|----------------|-----------|-----|
| Wrong agent/tool selected | Insufficient tool diversity in SFT | Add more examples for confused agent/tool pairs |
| Missing context fields | Too few examples showing all 4 fields | Add examples emphasizing complete context |
| Empty `memory` | Bad examples in training data | Validate & clean data, add rich-memory examples |
| No clarification on vague requests | Missing clarification examples | Add behavior KTO pairs for intellectual humility |
| Tool call after destructive request | Missing verification behavior | Add verification-before-action KTO pairs |
| Single calls instead of batching | Not enough multi-call examples | Add examples with 2-3 calls in one `useTools` invocation |
| Hallucinated IDs | Not using system prompt context | More examples where context IDs match system prompt exactly |

### The Improvement Loop

```
Evaluate → Identify failures → Generate targeted data → Retrain → Re-evaluate
    ↑                                                                    │
    └────────────────────────────────────────────────────────────────────┘
```

---

## Real-World Timeline

| Phase | Duration (7B model) | Output |
|-------|-------------------|--------|
| Schema & rubric design | 2-4 hours | `tool-schemas.json`, behavior rubrics |
| Handcrafted seeds | 4-8 hours | ~50 gold examples |
| SynthChat generation | 1-2 hours | ~2000+ examples |
| Validation & cleanup | 1-2 hours | Clean dataset |
| SFT training | ~45 min | LoRA adapters |
| KTO training | ~30 min | Refined adapters |
| Evaluation | ~15 min | Results JSON |
| Iteration (per round) | 2-4 hours | Improved dataset + model |

---

## File Map

```
tool-schemas.json                                  ← Source of truth (v2.0) for all tool definitions
Datasets/tools_datasets/non_thinking/              ← Non-thinking tool examples by agent
Datasets/tools_datasets/thinking/                  ← Thinking (with <thinking> blocks) by agent
Datasets/tools_datasets/specs/                     ← Authoring guidelines per agent
Datasets/behavior_rubrics/                         ← YAML rubrics for behavioral patterns
Datasets/behavior_datasets/                        ← KTO pairs by behavior
SynthChat/scenarios/tools.yaml                     ← Generation templates per tool
SynthChat/rubrics/                                 ← Quality rubrics for validation
Evaluator/config/scenarios/                        ← Test scenario YAMLs
Trainers/rtx3090_sft/                              ← SFT trainer
Trainers/rtx3090_kto/                              ← KTO trainer
Trainers/rtx3090_grpo/                             ← GRPO trainer
```
