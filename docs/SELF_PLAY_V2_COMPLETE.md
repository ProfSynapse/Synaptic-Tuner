# True Self-Play Generation v2 - Complete System

## What We Built

A sophisticated self-play system where your fine-tuned model generates **both sides** of the conversation with:

✅ **Manager-specific prompts** - Differentiated by tool category
✅ **Behavioral agents** - Elicit specific behaviors without teaching to the test
✅ **Sampling variation** - Vary temp, top_p for maximum diversity
✅ **No system prompts** in responses - Model should already know how to respond

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  TRUE SELF-PLAY GENERATION                                │
├──────────────────────────────────────────────────────────┤
│                                                            │
│  1. Prompt Generation (70% tool-based, 30% behavioral)   │
│     ↓                                                      │
│                                                            │
│  Tool-Based (70%):                                         │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ Manager-specific system prompt tells model:          │  │
│  │ "You're in Obsidian vault, using vaultManager"      │  │
│  │                                                       │  │
│  │ Modes: createFolder, deleteFolder, renameFolder...  │  │
│  │                                                       │  │
│  │ Examples: "Create folder called 'Project Ideas'..."  │  │
│  │                                                       │  │
│  │ Sampling variation:                                  │  │
│  │  - Temperature: 0.6-1.2 (varied creativity)         │  │
│  │  - Top-p: 0.85-0.98 (nucleus sampling)              │  │
│  │  - Max tokens: 50-150 (varied length)               │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                            │
│  Behavioral (30%):                                         │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ Intellectual Humility:                               │  │
│  │   "Search for notes about 'Project Nebula' -         │  │
│  │    I think that was the codename but not certain"    │  │
│  │                                                       │  │
│  │ Error Recovery:                                      │  │
│  │   "Delete folder 'Temp' but keep important files"   │  │
│  │                                                       │  │
│  │ Verification Before Action:                          │  │
│  │   "Delete all notes older than 6 months"             │  │
│  │                                                       │  │
│  │ Strategic Tool Selection:                            │  │
│  │   "Set up new workspace with folder structure..."    │  │
│  │                                                       │  │
│  │ Workspace Awareness:                                 │  │
│  │   "Switch to Phoenix workspace, show blockers"      │  │
│  └─────────────────────────────────────────────────────┘  │
│     ↓                                                      │
│                                                            │
│  Generated Prompt:                                         │
│  "Create a new folder called 'Meeting Notes'"             │
│     ↓                                                      │
│                                                            │
│  2. Response Generation (NO SYSTEM PROMPT!)               │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ User: "Create a new folder called 'Meeting Notes'"   │  │
│  │                                                       │  │
│  │ (No system prompt - model is fine-tuned!)            │  │
│  │                                                       │  │
│  │ Sampling variation:                                  │  │
│  │  - Temperature: 0.3-0.9 (some conservative, some    │  │
│  │    creative)                                         │  │
│  │  - Top-p: 0.90-0.98 (varied nucleus sampling)       │  │
│  └─────────────────────────────────────────────────────┘  │
│     ↓                                                      │
│                                                            │
│  Model Response:                                           │
│  "tool_call: vaultManager_createFolder                    │
│   arguments: {...}"                                        │
│     ↓                                                      │
│                                                            │
│  3. Validation                                            │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ ✓ Context object present?                            │  │
│  │ ✓ All 7 fields present?                              │  │
│  │ ✓ Tool parameters correct?                           │  │
│  │ ✓ Tool call syntax valid?                            │  │
│  │                                                       │  │
│  │ → Label: true/false                                  │  │
│  └─────────────────────────────────────────────────────┘  │
│     ↓                                                      │
│                                                            │
│  4. Collect & Save                                        │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ {                                                     │  │
│  │   "conversations": [                                  │  │
│  │     {"role": "user", "content": "..."},               │  │
│  │     {"role": "assistant", "content": "tool_call..."} │  │
│  │   ],                                                  │  │
│  │   "label": true/false                                 │  │
│  │ }                                                     │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                            │
│  5. Interleave True/False → Save to JSONL                │
│                                                            │
└──────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Manager-Specific Prompts

**5 categories**, each with specific modes and examples:

| Manager | Description | Example Modes |
|---------|-------------|---------------|
| `vaultManager` | Folder/structure management | createFolder, deleteFolder, moveFolder |
| `contentManager` | Note CRUD operations | createNote, updateNote, deleteNote |
| `memoryManager` | Workspaces, sessions, context | createWorkspace, loadWorkspace, saveSession |
| `vaultLibrarian` | Search, batch operations | searchNotes, findByTag, batchUpdate |
| `agentManager` | Agent lifecycle | createAgent, executePrompt, deleteAgent |

### 2. Behavioral Agents

**5 behaviors**, designed to elicit specific responses:

| Behavior | Purpose | Example Prompt |
|----------|---------|----------------|
| `intellectual_humility` | Admit uncertainty | "Search for 'Project Nebula' - not certain of name" |
| `error_recovery` | Handle errors gracefully | "Delete 'Temp' but keep important files" |
| `verification_before_action` | Confirm destructive actions | "Delete all notes older than 6 months" |
| `workspace_awareness` | Manage context properly | "Switch to Phoenix workspace, show blockers" |

### 3. Sampling Variation

**Prompt Generation:**
- Temperature: 0.6-1.2 (higher = more creative prompts)
- Top-p: 0.85-0.98
- Max tokens: 50-150 (varied length)

**Response Generation:**
- Temperature: 0.3-0.9 (some conservative, some creative)
- Top-p: 0.90-0.98
- **NO system prompt** - Model is fine-tuned!

## Usage

### Basic: Model generates its own prompts
```bash
python Tools/selfplay_generator_v2.py \
  --model claudesidian-mcp \
  --output Datasets/syngen_selfplay.jsonl \
  --num-examples 1000
```

**What happens:**
- 70% tool-based prompts (manager-specific)
- 30% behavioral prompts
- Model generates prompts → Model responds → Validate → Collect

### With MCP execution (future)
```bash
python Tools/selfplay_generator_v2.py \
  --model claudesidian-mcp \
  --output Datasets/syngen_selfplay_mcp.jsonl \
  --num-examples 1000 \
  --execute-mcp \
  --vault-path ~/obsidian-test-vault
```

## Example Generation

### Tool-Based Prompt

**System prompt to model:**
```
You are working in an Obsidian vault with a powerful assistant.

Your current focus is using the **vaultManager** tool category:
Manages vault structure - creates/deletes folders, organizes files

Available modes in vaultManager:
createFolder, deleteFolder, renameFolder, moveFolder

Generate a realistic user request that would use one of these vaultManager tools.

Example requests for vaultManager:
- "Create a new folder called 'Project Ideas' in the root directory"
- "Delete the 'Archive' folder and all its contents"
```

**Model generates:**
```
"Create a folder called 'Q4 Planning' in the Projects directory"
```

**Then model responds (NO system prompt):**
```
tool_call: vaultManager_createFolder
arguments: {
  "context": {
    "sessionId": "session_...",
    "workspaceId": "ws_default",
    ...
  },
  "path": "Projects/Q4 Planning"
}
```

### Behavioral Prompt

**Generated prompt:**
```
"Search for notes about 'Project Nebula' - I think that was the codename but not certain"
```

**Expected behavior:**
- Model should acknowledge uncertainty
- Use appropriate search tool
- Express caution in response

**Model response:**
```
tool_call: vaultLibrarian_searchNotes
arguments: {
  "context": {...},
  "query": "Project Nebula"
}

I'll search for "Project Nebula" though I should note that since you're uncertain
about the exact name, the results might not match what you're looking for.
I'll show you what I find and you can confirm if it's the right project.
```

## Why This Works

✅ **No teaching to test** - Doesn't use evaluation prompts
✅ **True self-play** - Model generates both sides
✅ **Infinite diversity** - New prompts every time
✅ **Manager coverage** - All 5 categories represented
✅ **Behavioral coverage** - Tests key behaviors
✅ **Sampling variation** - Maximum diversity
✅ **No system prompts in responses** - Model internalized patterns

## Recommended Workflow

### Phase 1: Initial Training (SFT)
```bash
cd Trainers/rtx3090_sft
python train_sft.py --model-size 7b
```

### Phase 2: True Self-Play Generation
```bash
python Tools/selfplay_generator_v2.py \
  --model your-sft-model \
  --output Datasets/syngen_selfplay_v1.jsonl \
  --num-examples 2000
```

### Phase 3: KTO Refinement
```bash
cd Trainers/rtx3090_kto
python train_kto.py \
  --model-size 7b \
  --local-file ../../Datasets/syngen_selfplay_v1.jsonl
```

### Phase 4: Iterate
```bash
# Use refined model for next generation
python Tools/selfplay_generator_v2.py \
  --model your-kto-model \
  --output Datasets/syngen_selfplay_v2.jsonl \
  --num-examples 2000
```

## Configuration Options

### Behavioral probability
```bash
# More behavioral prompts (40%)
# Edit selfplay_generator_v2.py:
behavioral_probability=0.4
```

### Manager distribution
```python
# In PromptGenerator._generate_from_model():
manager = random.choice(self.managers)

# Weighted selection:
manager = random.choices(
    self.managers,
    weights=[0.2, 0.3, 0.2, 0.2, 0.1],  # vaultManager, contentManager, etc.
    k=1
)[0]
```

### Sampling ranges
```python
# In generate_response():
temperature = random.uniform(0.5, 1.0)  # Adjust range
top_p = random.uniform(0.92, 0.98)      # Tighter nucleus
```

## Comparison: v1 vs v2

| Feature | v1 (Guided) | v2 (True Self-Play) |
|---------|-------------|---------------------|
| Prompt source | Eval prompts | Model-generated |
| Response system prompt | Yes | No (fine-tuned!) |
| Manager-specific | No | Yes (5 categories) |
| Behavioral agents | No | Yes (5 behaviors) |
| Sampling variation | Some | Extensive |
| Teaching to test | Yes | No |
| Diversity | Limited | Infinite |
| **Recommended** | Initial coverage | **Main training** |

## Next Steps

1. Try it: `python Tools/selfplay_generator_v2.py --model your-model --output test.jsonl --num-examples 100`
2. Inspect: `head -5 test.jsonl | jq`
3. Validate: `python tools/validate_syngen.py test.jsonl`
4. Train: `cd Trainers/rtx3090_kto && python train_kto.py --local-file ../../test.jsonl`
5. Evaluate: Check if model improves on behavioral tests
6. Iterate: Generate more data with refined model

## See Also

- [Self-Play True vs Guided](SELF_PLAY_TRUE_VS_GUIDED.md)
- [How Self-Play Works](SELF_PLAY_HOW_IT_WORKS.md)
- [KTO Training Reference](../KTO_TRAINING_REFERENCE.md)
