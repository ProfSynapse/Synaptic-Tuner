# How Self-Play Generation Works

This document explains the mechanics of self-play generation with concrete examples.

## Overview

The self-play system uses **existing prompt sets** (from `Evaluator/prompts/`) which contain:
1. ✅ **System prompts** - Context about the session, workspace, agents
2. ✅ **User questions** - Specific tasks to perform
3. ✅ **Expected tools** - What tools should be called
4. ✅ **Validation context** - IDs to verify against

## Concrete Example

Here's what happens when you run self-play with a single prompt:

### Step 1: Load Prompt from File

From `Evaluator/prompts/tool_prompts.json`:

```json
{
  "id": "agentManager_createAgent",
  "question": "Create a new agent called \"Vault Curator\" with the description \"Culls outdated workspace docs for the Atlas program\". Use the model \"gpt-4o-mini\" and set the temperature to 0.3.",
  "tags": ["agentManager", "single-tool"],
  "expected_tools": ["agentManager_createAgent"],
  "system": "<session_context>\nIMPORTANT: When using tools, include these values in your tool call parameters:\n\n- sessionId: \"session_1732300800000_eval01234\"\n- workspaceId: \"ws_1732300800000_atlasroll\" (current workspace)\n\nInclude these in the \"context\" parameter of your tool calls.\n</session_context>\n<available_workspaces>\nThe following workspaces are available in this vault:\n\n- Atlas Rollout (id: \"ws_1732300800000_atlasroll\")\n  Description: Project tracking for Atlas customer rollout\n  Root folder: Projects/\n\nUse memoryManager with loadWorkspace mode to get full workspace context.\n</available_workspaces>\n<available_agents>\nThe following custom agents are available:\n\n- Workspace Auditor (id: \"agent_1732300800000_workspace_auditor\")\n  Audits workspace contents and reports issues\n\n- Release Briefing (id: \"agent_1732300800001_release_briefing\")\n  Generates release briefing summaries\n\n</available_agents>",
  "expected_context": {
    "session_id": "session_1732300800000_eval01234",
    "workspace_id": "ws_1732300800000_atlasroll"
  }
}
```

### Step 2: Build Chat Messages

The system constructs a chat in **ChatML format**:

```python
messages = [
    {
        "role": "system",
        "content": "<session_context>\nIMPORTANT: When using tools, include these values...\n</session_context>\n<available_workspaces>...\n</available_workspaces>\n<available_agents>...\n</available_agents>"
    },
    {
        "role": "user",
        "content": "Create a new agent called \"Vault Curator\" with the description \"Culls outdated workspace docs for the Atlas program\". Use the model \"gpt-4o-mini\" and set the temperature to 0.3."
    }
]
```

**Key points:**
- ✅ **System prompt** provides ALL the context (session IDs, workspaces, agents)
- ✅ **User prompt** is the specific task/question
- ✅ Model must use IDs from system prompt in tool calls

### Step 3: Send to LM Studio with Temperature Variation

The self-play generator sends this to your model with **different temperatures**:

```python
# Variation 1: Temperature 0.3 (conservative)
response_1 = client.chat(messages, temperature=0.3)

# Variation 2: Temperature 0.7 (balanced)
response_2 = client.chat(messages, temperature=0.7)

# Variation 3: Temperature 0.9 (creative/diverse)
response_3 = client.chat(messages, temperature=0.9)
```

**Why multiple temperatures?**
- Low temp (0.3): Mostly correct responses → labeled `true`
- High temp (0.9): More diverse, some incorrect → mix of `true`/`false`
- Creates natural distribution of good and bad examples

### Step 4: Model Response Example

**Good response** (temperature 0.3):

```
tool_call: agentManager_createAgent
arguments: {
  "context": {
    "sessionId": "session_1732300800000_eval01234",
    "workspaceId": "ws_1732300800000_atlasroll",
    "sessionDescription": "Creating Vault Curator agent",
    "sessionMemory": "User requested agent creation for workspace document management",
    "toolContext": "Creating new agent to automate workspace curation",
    "primaryGoal": "Create Vault Curator agent",
    "subgoal": "Initialize agent with specified configuration"
  },
  "name": "Vault Curator",
  "description": "Culls outdated workspace docs for the Atlas program",
  "model": "gpt-4o-mini",
  "temperature": 0.3
}

Result: {
  "success": true,
  "agentId": "agent_1732300900000_vault_curator",
  "message": "Agent created successfully"
}

I've created the Vault Curator agent with the specified settings. The agent is now available with ID agent_1732300900000_vault_curator and will help manage outdated documents in the Atlas program workspace.
```

**Bad response** (temperature 1.2 - very high):

```
tool_call: agentManager_createAgent
arguments: {
  "context": {
    "sessionId": "session_WRONG_ID",
    "workspaceId": "ws_1732300800000_atlasroll"
    // MISSING: sessionDescription, sessionMemory, toolContext, primaryGoal, subgoal
  },
  "name": "Vault Curator",
  "description": "Culls outdated workspace docs for the Atlas program"
  // MISSING: model, temperature parameters
}
```

### Step 5: Validation

The validator checks the response:

**Good response validation:**
```python
✓ Context object present and first
✓ All 7 context fields present
✓ sessionId matches system prompt: "session_1732300800000_eval01234"
✓ workspaceId matches system prompt: "ws_1732300800000_atlasroll"
✓ All required parameters present
✓ Tool schema matches

→ Label: true (desirable)
```

**Bad response validation:**
```python
✗ sessionId does not match system prompt
✗ Missing context fields: sessionDescription, sessionMemory, toolContext, primaryGoal, subgoal
✗ Missing required parameters: model, temperature

→ Label: false (undesirable)
```

### Step 6: Collect in ChatML Format

**Good example** (saved to dataset):

```json
{
  "conversations": [
    {
      "role": "user",
      "content": "Create a new agent called \"Vault Curator\" with the description \"Culls outdated workspace docs for the Atlas program\". Use the model \"gpt-4o-mini\" and set the temperature to 0.3."
    },
    {
      "role": "assistant",
      "content": "tool_call: agentManager_createAgent\narguments: {...}\n\nResult: {...}\n\nI've created the Vault Curator agent..."
    }
  ],
  "label": true
}
```

**Note:** System prompt is **NOT included** in the saved dataset! Why?
- Models trained with SFT should **internalize** the context pattern
- System prompts are for **evaluation** only
- Training without system prompts makes models more robust

**Bad example** (saved to dataset):

```json
{
  "conversations": [
    {
      "role": "user",
      "content": "Create a new agent called \"Vault Curator\"..."
    },
    {
      "role": "assistant",
      "content": "tool_call: agentManager_createAgent\narguments: {...with errors...}\n"
    }
  ],
  "label": false
}
```

### Step 7: Interleaving

The system interleaves examples in **True/False/True/False** pattern:

```jsonl
{"conversations": [...], "label": true}   # Good example
{"conversations": [...], "label": false}  # Bad example
{"conversations": [...], "label": true}   # Good example
{"conversations": [...], "label": false}  # Bad example
...
```

**Why interleave?**
- KTO training requires mixed batches (TRL bug workaround)
- Prevents model collapse from seeing only good or bad examples
- Better gradient flow during training

## Full Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Load Prompt Set (tool_prompts.json)                       │
│    - 47 prompts covering all tools                           │
│    - Each has system prompt + user question                  │
└──────────────────┬────────────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. For Each Prompt × Num Variations (e.g., 3)                │
│                                                               │
│    messages = [                                               │
│        {"role": "system", "content": "<session_context>..."}  │
│        {"role": "user", "content": "Create a new agent..."}   │
│    ]                                                          │
└──────────────────┬────────────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Send to LM Studio with Temperature Variation              │
│                                                               │
│    Variation 1: T=0.3  → Mostly correct                      │
│    Variation 2: T=0.7  → Balanced                            │
│    Variation 3: T=0.9  → More errors                         │
└──────────────────┬────────────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Receive Model Response                                    │
│                                                               │
│    "tool_call: agentManager_createAgent                       │
│     arguments: {...}                                          │
│     Result: {...}                                             │
│     Response text..."                                         │
└──────────────────┬────────────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Validate Response (using validate_syngen.py)              │
│                                                               │
│    Check:                                                     │
│    ✓ Context object present and first                        │
│    ✓ All 7 context fields present                            │
│    ✓ sessionId matches system prompt                         │
│    ✓ workspaceId matches system prompt                       │
│    ✓ Tool parameters match schema                            │
│    ✓ Tool call syntax correct                                │
│                                                               │
│    → Valid: label = true                                      │
│    → Invalid: label = false                                   │
└──────────────────┬────────────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. Collect in ChatML Format (NO SYSTEM PROMPT)               │
│                                                               │
│    {                                                          │
│      "conversations": [                                       │
│        {"role": "user", "content": "..."},                    │
│        {"role": "assistant", "content": "tool_call: ..."}     │
│      ],                                                       │
│      "label": true/false                                      │
│    }                                                          │
└──────────────────┬────────────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. Interleave True/False/True/False                          │
│                                                               │
│    Separate valid and invalid examples                       │
│    Balance counts                                             │
│    Interleave: T, F, T, F, T, F, ...                          │
└──────────────────┬────────────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ 8. Write to JSONL File                                        │
│                                                               │
│    syngen_selfplay_20251204_143000.jsonl                     │
│    - Ready for KTO training                                   │
│    - Perfectly interleaved                                    │
│    - Balanced dataset                                         │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Why use existing prompt sets?

✅ **Pre-designed prompts** covering all tools
✅ **Consistent format** with proper context
✅ **Validation metadata** for cross-checking
✅ **Known expected tools** for verification

Alternative (not implemented yet):
- Generate random variations of prompts
- Mutate existing prompts
- Create new scenarios programmatically

### 2. Why include system prompts during generation but not in saved data?

**During generation:**
- System prompt provides **context** (session IDs, workspaces, agents)
- Model needs this to generate **correct** tool calls
- Validator uses this to **cross-check** IDs

**In saved data:**
- System prompts are **evaluation-only** scaffolding
- Training data should be **clean** user/assistant pairs
- Models should **internalize** the pattern, not rely on system prompts
- Makes models more **robust** and **generalizable**

This matches your existing SFT training approach (no system prompts).

### 3. Why multiple temperature variations?

**Creates natural diversity:**
- Low temp (0.3-0.5): ~80-90% correct → labeled `true`
- Medium temp (0.7): ~60-70% correct → mix
- High temp (0.9-1.2): ~40-50% correct → more `false`

**Result:** Balanced dataset without manual curation!

### 4. Why remove system prompts from saved examples?

Looking at your existing training datasets:

```jsonl
# Your current SFT dataset format (syngen_tools_sft_11.18.25.jsonl)
{
  "conversations": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "tool_call: ..."}
  ],
  "label": true
}
# NO system prompt! Model internalizes the pattern.
```

The self-play generator **matches this format** - system prompts are used for generation/validation only, not saved to training data.

## Prompt Set Statistics

From `Evaluator/prompts/tool_prompts.json`:
- **47 prompts** (one per tool)
- **5 manager categories** (vaultManager, contentManager, memoryManager, vaultLibrarian, agentManager)
- **Each prompt includes:**
  - Unique ID
  - User question (task to perform)
  - System prompt (context with IDs)
  - Expected tools
  - Tags for filtering

## Example: Full Generation Run

```bash
./Tools/run_selfplay.sh --quick
```

**What happens:**
1. Load 47 prompts from `tool_prompts.json`
2. For each prompt, generate 3 variations (different temperatures)
3. Total attempts: 47 × 3 = 141 (but stops at 100 for `--quick`)
4. Each generation:
   - Builds messages with system + user prompt
   - Sends to LM Studio
   - Validates response
   - Labels as `true` or `false`
   - Collects example
5. After 100 examples collected:
   - Separate valid (true) and invalid (false)
   - Balance counts (use min of both)
   - Interleave True/False/True/False
   - Write to `Datasets/syngen_selfplay_YYYYMMDD_HHMMSS.jsonl`

**Expected output:**
```
Generation Statistics:
  Total attempts: 100
  Valid examples: 65
  Invalid examples: 35
  Success rate: 65%

Interleaving examples for KTO training...
  Valid examples: 65
  Invalid examples: 35
  Interleaved dataset size: 70 (35 pairs)

✓ Dataset written to Datasets/syngen_selfplay_20251204_143000.jsonl
```

## Next Steps

Now that you understand how it works, you can:

1. **Run it:** `./Tools/run_selfplay.sh --quick`
2. **Inspect output:** `head -5 Datasets/syngen_selfplay_*.jsonl | jq`
3. **Check validation:** `python tools/validate_syngen.py Datasets/syngen_selfplay_*.jsonl`
4. **Train with it:** `cd Trainers/rtx3090_kto && python train_kto.py --local-file ../../Datasets/syngen_selfplay_*.jsonl`

## Customization Ideas

### Create your own prompt set

```json
[
  {
    "id": "custom_task_1",
    "question": "Your custom task here",
    "system": "<session_context>...</session_context>",
    "expected_tools": ["toolName"],
    "tags": ["custom"]
  }
]
```

Then use it:
```bash
python Tools/selfplay_generator.py \
  --model your-model \
  --prompt-set path/to/custom_prompts.json \
  --output Datasets/custom_selfplay.jsonl
```

### Adjust temperature range

Edit `Tools/selfplay_generator.py`:

```python
class PromptVariator:
    def __init__(self):
        # More conservative (fewer errors)
        self.temperature_variations = [0.2, 0.4, 0.6, 0.8]

        # More diverse (more errors)
        self.temperature_variations = [0.5, 0.8, 1.0, 1.3]
```

### Generate more variations per prompt

```bash
python Tools/selfplay_generator.py \
  --model your-model \
  --prompt-set Evaluator/prompts/tool_prompts.json \
  --num-variations 5 \  # Instead of 3
  --output Datasets/diverse_selfplay.jsonl
```
