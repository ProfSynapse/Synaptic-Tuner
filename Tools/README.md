# Tools Directory

Utilities for dataset validation, analysis, and synthetic data generation.

## Dataset Validation & Analysis

### validate_syngen.py
Comprehensive validator for synthetic datasets in ChatML format.

**Features:**
- Validates ChatML conversation structure
- Checks context objects (all 7 required fields)
- Validates against tool schemas (`tool_schemas.json`)
- Cross-validates IDs with system prompt context
- Supports both desirable (`label=true`) and undesirable (`label=false`) examples

**Usage:**
```bash
python tools/validate_syngen.py Datasets/syngen_tools_11.18.25.jsonl
```

**Output:**
```
Example line 42:
  [ERROR] Tool call #1 (vaultManager_createFolder): Missing required context field 'sessionMemory'
  [WARN] Tool call #1: Unexpected parameter 'extra_param' not in schema

Validated 4649 example(s): 2 failed (ignoring 2324 label=false examples).
```

### analyze_tool_coverage.py
Analyzes tool usage distribution across a dataset.

**Usage:**
```bash
python tools/analyze_tool_coverage.py Datasets/syngen_tools_11.18.25.jsonl
```

**Output:**
- Tool call frequency distribution
- Coverage by manager (vaultManager, contentManager, etc.)
- Missing tools
- Statistics

### convert_to_mcp_format.py
Converts ChatML format to Anthropic Messages API / MCP format.

**Usage:**
```bash
python tools/convert_to_mcp_format.py \
  input.jsonl \
  output.jsonl \
  --validate
```

**Format conversion:**
```
# From ChatML:
tool_call: toolName
arguments: {...}

# To MCP:
{
  "type": "tool_use",
  "id": "toolu_XYZ...",
  "name": "toolName",
  "input": {...}
}
```

## Self-Play Data Generation

### selfplay_generator.py
**NEW!** Generate synthetic training data using your fine-tuned model.

**Features:**
- Uses LM Studio to query your fine-tuned model
- Automatically validates responses
- Collects both correct and incorrect examples
- Creates interleaved KTO datasets (True/False/True/False)
- Optional MCP execution for functional validation (coming soon)

**Quick start:**
```bash
# Interactive mode (recommended)
./Tools/run_selfplay.sh

# Or directly:
python Tools/selfplay_generator.py \
  --model claudesidian-mcp \
  --prompt-set Evaluator/prompts/tool_prompts.json \
  --output Datasets/syngen_selfplay_$(date +%Y%m%d).jsonl \
  --num-examples 1000 \
  --temperature 0.7
```

**See:** [docs/SELF_PLAY_GENERATION.md](../docs/SELF_PLAY_GENERATION.md) for full documentation.

### run_selfplay.sh / run_selfplay.ps1
Interactive wrappers for self-play generation.

**Features:**
- Auto-detects LM Studio connection
- Lists available models
- Interactive configuration
- Progress tracking
- Helpful next steps after generation

**Usage:**
```bash
# Linux/WSL
./Tools/run_selfplay.sh              # Interactive
./Tools/run_selfplay.sh --quick      # 100 examples
./Tools/run_selfplay.sh --standard   # 1000 examples
./Tools/run_selfplay.sh --large      # 5000 examples

# PowerShell
.\Tools\run_selfplay.ps1             # Interactive
.\Tools\run_selfplay.ps1 -Quick      # 100 examples
.\Tools\run_selfplay.ps1 -Standard   # 1000 examples
.\Tools\run_selfplay.ps1 -Large      # 5000 examples
```

### mcp_executor.py
MCP tool executor for functional validation (placeholder).

**Coming soon:**
- Execute tool calls against real Obsidian vault
- Verify functional correctness
- Integration with selfplay_generator.py

**Future usage:**
```bash
python Tools/selfplay_generator.py \
  --model claudesidian-mcp \
  --prompt-set Evaluator/prompts/tool_prompts.json \
  --output Datasets/syngen_selfplay_mcp.jsonl \
  --num-examples 1000 \
  --execute-mcp \
  --vault-path ~/obsidian-test-vault
```

## Tool Schemas

### tool_schemas.json
Central schema definitions for all 47+ Claudesidian tools.

**Used by:**
- `validate_syngen.py` for parameter validation
- Training data generation
- Model evaluation

**Format:**
```json
{
  "toolName": {
    "required_params": ["context", "param1"],
    "parameters": [
      {
        "name": "param1",
        "type": "string",
        "description": "..."
      }
    ],
    "context_schema": {
      "fields": [...]
    }
  }
}
```

## Workflow Examples

### Validate a dataset before training
```bash
python tools/validate_syngen.py Datasets/syngen_tools_11.18.25.jsonl
```

### Analyze tool coverage
```bash
python tools/analyze_tool_coverage.py Datasets/syngen_tools_11.18.25.jsonl
```

### Generate synthetic data with self-play
```bash
# Quick test (100 examples)
./Tools/run_selfplay.sh --quick

# Full generation (1000 examples)
./Tools/run_selfplay.sh --standard

# Validate generated data
python tools/validate_syngen.py Datasets/syngen_selfplay_*.jsonl
```

### Convert to MCP format
```bash
python tools/convert_to_mcp_format.py \
  Datasets/syngen_tools_11.18.25.jsonl \
  Datasets/syngen_tools_mcp_format.jsonl \
  --validate
```

## File Organization

```
Tools/
├── README.md                      # This file
├── validate_syngen.py             # Dataset validator
├── analyze_tool_coverage.py       # Coverage analysis
├── convert_to_mcp_format.py       # Format converter
├── selfplay_generator.py          # Self-play data generation
├── mcp_executor.py                # MCP tool executor (placeholder)
├── run_selfplay.sh                # Self-play wrapper (Bash)
├── run_selfplay.ps1               # Self-play wrapper (PowerShell)
└── tool_schemas.json              # Tool schema definitions
```

## See Also

- [Self-Play Generation Guide](../docs/SELF_PLAY_GENERATION.md)
- [Dataset Validation Reference](../docs/SCHEMA_VERIFICATION_REFERENCE.md)
- [KTO Training Reference](../KTO_TRAINING_REFERENCE.md)
- [Main Documentation](../CLAUDE.md)
