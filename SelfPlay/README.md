# SelfPlay - Synthetic Dataset Generation

A sophisticated 3-prompt pipeline for generating synthetic training data where a fine-tuned model creates both the user request and assistant response through self-play.

## Overview

This system generates training data through **three sequential prompts**:

1. **Prompt 1**: Generate realistic workspace environment
2. **Prompt 2**: Generate vague user request (based on environment)
3. **Prompt 3**: Generate assistant response with tool calls

The pipeline is intentionally single-tool and single-response focused—no tool chaining or multi-step workflows.

### Why 3 Prompts?

- **Prompt 1** creates context that both Prompt 2 and Prompt 3 can use
- **Prompt 2** generates realistic, vague user requests (like real users)
- **Prompt 3** generates responses without system prompts (model is fine-tuned)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Prompt 1: Generate Environment (LLM)                       │
├─────────────────────────────────────────────────────────────┤
│  Input:  "Generate a realistic Obsidian workspace..."       │
│  Output: Workspace Name: Project Atlas                      │
│          Description: Customer rollout tracking...          │
│          Root Folder: Projects/Atlas/                       │
│          Folder Structure: [detailed tree]                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Auto-Generate IDs (Programmatic)                           │
├─────────────────────────────────────────────────────────────┤
│  session_1732300800000_a1b2c3d4e                            │
│  ws_1732300800000_f5g6h7i8j                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Build Session Context (Inject IDs)                         │
├─────────────────────────────────────────────────────────────┤
│  <session_context>                                          │
│    sessionId: "session_1732300800000_a1b2c3d4e"             │
│    workspaceId: "ws_1732300800000_f5g6h7i8j"                │
│  </session_context>                                         │
│  <current_workspace>                                        │
│    [workspace details from Prompt 1]                        │
│  </current_workspace>                                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Prompt 2: Generate User Request (LLM)                      │
├─────────────────────────────────────────────────────────────┤
│  Input:  [Session context] + Tool/behavior instructions     │
│  Output: "Can you create a folder for Q4 planing stuff?     │
│           I think in the project area"                      │
│                                                              │
│  Note: VAGUE like real users (misspellings, unclear paths)  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Prompt 3: Generate Assistant Response (LLM)                │
├─────────────────────────────────────────────────────────────┤
│  Input:  User: [user request from Prompt 2]                 │
│          (NO system prompt - model is fine-tuned)           │
│  Output: <tool_call>                                        │
│          {                                                   │
│            "name": "vaultManager_createFolder",             │
│            "arguments": {                                   │
│              "context": {...},                              │
│              "path": "Projects/Atlas/Planning/Q4-Planning"  │
│            }                                                 │
│          }                                                   │
│          </tool_call>                                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Validation & JSONL Export                                  │
├─────────────────────────────────────────────────────────────┤
│  {                                                           │
│    "conversations": [                                        │
│      {"role": "user", "content": "Can you create..."},      │
│      {"role": "assistant", "content": "<tool_call>..."}     │
│    ],                                                        │
│    "label": true                                             │
│  }                                                           │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
SelfPlay/
├── __init__.py              # Package initialization
├── id_utils.py             # Auto-generate session/workspace IDs
├── generator.py            # Main 3-prompt pipeline
├── README.md               # This file
└── configs/
    ├── agents.yaml         # 49 tools organized by agent
    ├── behaviors.yaml      # 4 behavioral scenarios
    └── prompts/
        ├── environment.yaml          # Prompt 1 templates
        ├── user_tool.yaml           # Prompt 2 (tool-based)
        └── user_behavioral.yaml     # Prompt 2 (behavioral)
```

## Configuration Files

### `configs/prompts/environment.yaml`

Defines how to generate workspace environments with 4 variations:

- **simple**: 3-5 files, minimal structure (20%)
- **complex**: 10-15 files, realistic mess (50%) ⭐ Most common
- **project_focused**: Structured project workspace (20%)
- **knowledge_base**: Topic-organized research notes (10%)

### `configs/prompts/user_tool.yaml`

Instructions for generating **vague, realistic** user requests for tool-based scenarios.

Key features:
- Misspellings: "gols" instead of "goals"
- Partial recall: "I think it's in..."
- Content description: "my notes about tacos" (file is actually "mexican-restaurants.md")
- Location vagueness: "somewhere in projects"

### `configs/prompts/user_behavioral.yaml`

Instructions for generating user requests that elicit specific behaviors:

- **intellectual_humility**: Ambiguous requests requiring clarification
- **error_recovery**: Conflicting requirements needing alternatives
- **verification_before_action**: Bulk operations needing confirmation
- **workspace_awareness**: Workspace-specific context handling

### `configs/agents.yaml`

All 49 tools organized by agent:

- **vaultManager** (9 tools): Folder/file management
- **contentManager** (10 tools): Note content operations
- **memoryManager** (13 tools): Sessions, states, workspaces
- **vaultLibrarian** (4 tools): Search and discovery
- **agentManager** (10 tools): Custom agents
- **commandManager** (2 tools): Obsidian commands
- **meta** (1 tool): Tool discovery

### `configs/behaviors.yaml`

4 core behaviors with validation criteria:

- **intellectual_humility** (30%): Asks clarifying questions
- **error_recovery** (25%): Handles conflicts gracefully
- **verification_before_action** (25%): Confirms before destructive ops
- **workspace_awareness** (20%): Maintains context

## Usage

### Quick Start (Recommended)

**Using the run script:**

```bash
# Quick test (10 examples)
./SelfPlay/run.sh --quick

# Standard generation (100 examples)
./SelfPlay/run.sh --standard

# Large batch (1000 examples)
./SelfPlay/run.sh --large

# Custom number
./SelfPlay/run.sh --num 500

# Custom output file
./SelfPlay/run.sh --num 200 --output my_data.jsonl
```

**Using Python directly:**

```bash
# Generate 100 examples with validation
python3 SelfPlay/run_generation.py --num-examples 100

# Generate without validation (faster)
python3 SelfPlay/run_generation.py --num-examples 1000 --no-validate

# Custom LM Studio host (for WSL + Windows LM Studio)
python3 SelfPlay/run_generation.py \
    --host 192.168.1.104 \
    --port 1234 \
    --num-examples 500
```

**What it does:**
1. Connects to LM Studio
2. Generates examples using 3-prompt pipeline
3. Validates each example automatically
4. Separates valid (label=true) and invalid (label=false) examples
5. Saves to JSONL files:
   - `selfplay_output.jsonl` - Valid examples
   - `selfplay_output_invalid.jsonl` - Invalid examples (for contrastive learning)

### Programmatic Usage

```python
from SelfPlay.generator import SelfPlayGenerator
from Evaluator.lmstudio_client import LMStudioClient
from Evaluator.config import LMStudioSettings

# Initialize LM Studio client
settings = LMStudioSettings(model="local-model")
client = LMStudioClient(settings=settings)

# Initialize generator with client
generator = SelfPlayGenerator(model_client=client)

# Generate batch with validation
results = generator.generate_batch(
    num_examples=100,
    output_file="output.jsonl",
    validate=True,
    save_invalid=True
)

print(f"Valid: {len(results['valid'])}")
print(f"Invalid: {len(results['invalid'])}")
```

### Generate Specific Type

```python
# Generate only tool-based examples
tool_example = generator.generate_single_example(generation_type="tool")

# Generate only behavioral examples
behavioral_example = generator.generate_single_example(generation_type="behavioral")

# Validate single example
is_valid, report = generator.validate_example(tool_example)
if not is_valid:
    for issue in report.issues:
        print(f"{issue.level}: {issue.message}")
```

## ID Generation

Session and workspace IDs are auto-generated following the required format:

```python
from SelfPlay.id_utils import generate_ids, validate_session_id

# Generate IDs
session_id, workspace_id = generate_ids()
# Returns: ('session_1732300800000_a1b2c3d4e', 'ws_1732300800000_f5g6h7i8j')

# Validate
is_valid = validate_session_id(session_id)  # True
```

Format:
- **Session**: `session_{13-digit-timestamp}_{9-char-random}`
- **Workspace**: `ws_{13-digit-timestamp}_{9-char-random}`

## Tool Call Formats

The system should handle 3 different tool call formats:

### Qwen Format (Default)
```xml
<tool_call>
{
  "name": "vaultManager_createFolder",
  "arguments": {...}
}
</tool_call>
```

### Mistral Format
```
[TOOL_CALLS] [{"name": "vaultManager_createFolder", "arguments": {...}}]
```

### OpenAI Format
```
tool_call: vaultManager_createFolder
arguments: {...}
```

## Output Format

All examples are saved in ChatML format:

**Tool-based:**
```json
{
  "conversations": [
    {"role": "user", "content": "Can you create a folder for Q4 planning?"},
    {"role": "assistant", "content": "<tool_call>...</tool_call>"}
  ],
  "metadata": {
    "session_id": "session_1732300800000_a1b2c3d4e",
    "workspace_id": "ws_1732300800000_f5g6h7i8j",
    "environment_variation": "complex",
    "type": "tool",
    "agent": "vaultManager",
    "tools": ["vaultManager_createFolder"]
  }
}
```

**Behavioral:**
```json
{
  "conversations": [
    {"role": "system", "content": "<session_context>...</session_context>"},
    {"role": "user", "content": "Can you delete the old project files?"},
    {"role": "assistant", "content": "I'd like to clarify..."}
  ],
  "behavior": "intellectual_humility",
  "metadata": {
    "session_id": "session_1732300800000_a1b2c3d4e",
    "workspace_id": "ws_1732300800000_f5g6h7i8j",
    "environment_variation": "project_focused",
    "type": "behavioral",
    "behavior": "intellectual_humility"
  }
}
```

## Generation Distribution

**Type Distribution:**
- 70% Tool-based
- 30% Behavioral

**Environment Variations:**
- 50% Complex (realistic mess)
- 20% Simple (minimal)
- 20% Project-focused
- 10% Knowledge base

**Behavioral Distribution:**
- 25% Intellectual humility
- 20% Error recovery
- 20% Verification before action
- 20% Strategic tool selection
- 15% Workspace awareness

## Getting Started

### Prerequisites

1. **LM Studio running** with a model loaded
2. **Server started** in LM Studio (click "Start Server")
3. **Python environment** with dependencies installed

### Quick Test

```bash
# Test the pipeline with a single example
python3 SelfPlay/test_pipeline.py

# Generate a small batch
./SelfPlay/run.sh --quick
```

### Production Generation

```bash
# Generate 1000 examples for training
./SelfPlay/run.sh --large

# Or custom amount
./SelfPlay/run.sh --num 500 --output my_training_data.jsonl
```

### Using Generated Data

The output files are ready for KTO training:
- **Valid examples** (`label=true`): Model generated correct tool calls
- **Invalid examples** (`label=false`): Model made errors (useful for contrastive learning)

**For KTO training**, combine both files and interleave them (True/False/True/False pattern):

```bash
# Combine and interleave for KTO
python tools/interleave_dataset.py \
    SelfPlay/selfplay_output.jsonl \
    SelfPlay/selfplay_output_invalid.jsonl \
    -o Datasets/selfplay_kto_ready.jsonl
```

## Next Steps

1. ✅ **Integrate LM Studio Client**: Complete - generator.py fully integrated
2. ✅ **Implement Validation**: Complete - auto-validates all examples
3. **Add Tool Call Parser**: Support Qwen/Mistral/OpenAI formats (TODO)
4. **Run Generation**: Create 1000+ examples using `./SelfPlay/run.sh --large`
5. **Validate Dataset**: Check distribution and quality with `tools/analyze_tool_coverage.py`
6. **Train with KTO**: Use interleaved dataset for model refinement

## Key Design Decisions

1. **IDs are programmatic, NOT LLM-generated**: Ensures correct format
2. **Environment is LLM-generated**: Ensures diversity and realism
3. **No system prompt for Prompt 3**: Fine-tuned model already knows tool calling
4. **System prompt for behavioral Prompt 3**: Session context needed for behavior
5. **Vagueness is intentional**: Real users are vague, models should handle it

## See Also

- [docs/SELF_PLAY_META_GENERATION.md](../docs/SELF_PLAY_META_GENERATION.md) - Detailed architecture
- [configs/prompts/environment.yaml](configs/prompts/environment.yaml) - Environment generation
- [configs/agents.yaml](configs/agents.yaml) - All 49 tools
- [configs/behaviors.yaml](configs/behaviors.yaml) - 4 behavioral scenarios
