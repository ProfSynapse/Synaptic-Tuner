# Dataset Improvement Engine

Automated quality improvement for synthetic training datasets using LLM-based enhancement.

## Overview

The Dataset Improvement Engine uses a shared LLM client system to automatically improve the quality of synthetic training data by enhancing thinking blocks in tool-calling examples. Supports multiple LLM providers: OpenRouter, LM Studio, and Ollama.

## Features

- **Multi-Provider LLM Support**: OpenRouter (cloud), LM Studio (local), Ollama (local)
- **LLM-Powered Improvements**: Uses configured model to enhance thinking blocks
- **Quality Validation**: Validates improvements against schema rules
- **Batch Processing**: Processes examples in configurable batches
- **Sequential Processing**: Waits for each batch before proceeding (no parallel requests)
- **Dry Run Mode**: Preview changes before applying
- **Automatic Backups**: Creates backups before modifying files
- **Interactive CLI**: User-friendly dataset/version selection
- **Progress Logging**: Real-time progress and success rates

## Setup

1. **Install dependencies**:
   ```bash
   pip install pyyaml requests python-dotenv
   ```

2. **Create `.env` file in repo root** (if not already exists):
   ```bash
   cd /path/to/Toolset-Training
   # Add improvement engine settings to root .env (see improvement_engine/.env.example)
   ```

3. **Configure LLM backend in root `.env`**:
   ```bash
   # Edit root .env file
   IMPROVEMENT_BACKEND=lmstudio  # or openrouter, ollama
   IMPROVEMENT_MODEL=local-model

   # If using OpenRouter:
   OPENROUTER_API_KEY=your_api_key_here

   # If using LM Studio:
   LMSTUDIO_HOST=192.168.1.104
   LMSTUDIO_PORT=1234

   # If using Ollama:
   OLLAMA_HOST=localhost
   OLLAMA_PORT=11434
   ```

## Usage

### Via Unified CLI (Recommended)

```bash
# From repo root
./run.sh improve

# Or directly
python tuner.py improve
```

### Interactive Workflow

The CLI will guide you through:

1. **Select dataset category**:
   - Behavior datasets
   - Tools (thinking)
   - Tools (non-thinking)

2. **Select agent** (e.g., contentManager, vaultManager)

3. **Select version to improve** (e.g., v1.5 → v1.6)

4. **Configure settings**:
   - Batch size (default: 10)
   - Start line (default: 1)
   - End line (default: all)
   - Dry run? (yes/no)

5. **Review and proceed**

### Example Session

```
Dataset Improvement Engine

Select dataset category:
  1. behavior (5 agents)
  2. tools_thinking (5 agents)
  3. tools_non_thinking (5 agents)

Choice: 2

Select agent from tools_thinking:
  1. agentManager (5 versions)
  2. contentManager (6 versions)
  3. memoryManager (6 versions)
  4. vaultLibrarian (4 versions)
  5. vaultManager (4 versions)

Choice: 2

Available contentManager versions:
  1. v1.0 (1194 examples)
  2. v1.1 (1193 examples)
  3. v1.2 (1251 examples)
  4. v1.3 (1495 examples)
  5. v1.4 (1491 examples)
  6. v1.5 (1491 examples)

Select version to improve: 6

Batch size [default: 10]: 20
Start from line [default: 1]: 1
End at line [default: 1491 (all)]: 100
Dry run first? [y/n]: y

Configuration Summary
Input:  v1.5 (1491 examples)
Output: v1.6
Model:  openai/gpt-5-mini
Batch:  20 examples (sequential)
Range:  Lines 1-100 (5 batches)

Proceed? [y/n]: y

Batch 1/5 (lines 1-20)
  → Sending to gpt-4o-mini...
  → Received responses
  → Validating JSON... ✓ 20/20 valid
  ✓ Completed in 3.2s

...

Summary
Total processed: 100
Successful: 97 (97%)
Failed: 3 (3%)
Total time: 16.4s

DRY RUN - No files modified
```

## LLM-Based Dataset Labeling

**NEW!** The improvement engine now includes automated dataset labeling using LLM-as-a-judge to evaluate and label training examples.

### Features

- **LLM-as-a-Judge**: Automated quality evaluation using LLM
- **Multiple Backends**: LM Studio (local), OpenRouter (cloud), Ollama (local)
- **Boolean Labels**: Clean, simple true/false labels for quality metrics
- **Batch Processing**: Efficient processing with configurable batch sizes
- **KTO Compatible**: `label` field serves dual purpose (quality + KTO training)
- **Configurable Criteria**: Define evaluation criteria in YAML

### Quick Start

```bash
# Label with LM Studio (default, local)
python -m improvement_engine.label_dataset \
  --input Datasets/tools_datasets/thinking/vaultManager/tools_v1.3.jsonl \
  --output Datasets/tools_datasets/thinking/vaultManager/tools_v1.3_labeled.jsonl

# Label with OpenRouter (cloud)
python -m improvement_engine.label_dataset \
  --input dataset.jsonl \
  --output dataset_labeled.jsonl \
  --backend openrouter \
  --model openai/gpt-4o-mini

# Label in-place (modifies original file)
python -m improvement_engine.label_dataset \
  --input dataset.jsonl \
  --in-place \
  --backend lmstudio
```

### Label Schema

The LLM applies **5 boolean labels** to each example:

```json
{
  "conversations": [...],
  "label": true,              // Good (true) or Bad (false) quality - ALSO used for KTO
  "excellent": false,         // Exemplary/reference quality
  "hallucinated": false,      // Contains hallucinations
  "poor_reasoning": false,    // Weak/illogical reasoning
  "context_mismatch": false   // Doesn't align with context
}
```

**Key Points:**
- `label` serves **dual purpose**: quality indicator AND KTO training label
- All fields are optional booleans (only set if LLM judges them)
- Original `label` field preserved if already present

### Configuration

Evaluation criteria are defined in `improvement_engine/rubrics/quality_labels.yaml`:

```yaml
# System prompt for LLM judge
system_prompt: |
  You are an expert AI quality evaluator analyzing training examples.

# Criteria for each label
criteria:
  label:
    true_description: |
      GOOD QUALITY - Tool selection appropriate, parameters correct,
      clear reasoning, addresses request completely...

    false_description: |
      BAD QUALITY - Wrong tool, incorrect parameters, flawed reasoning,
      doesn't address request...

  excellent:
    description: |
      EXEMPLARY QUALITY - Perfect tool selection, comprehensive thinking,
      accurate response. Use sparingly (top 5-10%)...

  hallucinated:
    description: |
      Makes claims not supported by context. Invents file paths,
      functions, or details not mentioned...

  poor_reasoning:
    description: |
      Goal unclear, memory lacks context, requirements/plan duplicates,
      confidence doesn't match risk...

  context_mismatch:
    description: |
      Misunderstands intent, ignores conversation history,
      responds to different request...
```

**Customize the criteria** to match your quality standards!

### Labeling Workflow

1. **LLM receives criteria**: System prompt includes all evaluation criteria from config
2. **For each example**: LLM analyzes conversation, thinking block, tool calls
3. **LLM returns JSON**: Boolean labels + reasoning for decisions
4. **Validation**: Checks JSON schema and required fields
5. **Save**: Applies labels to example (in-place or new file)
6. **Batch Processing**: Processes N examples at a time (default: 10)

**The LLM judges all 5 aspects for every example** based on your configured criteria.

### Example Output

**Console Output:**
```
Starting LLM-based labeling for: tools_v1.3.jsonl
Backend: lmstudio, Model: qwen2.5-coder:7b
Processing lines 1-100 (100 examples)

✓ Line 1: label=true, excellent=false, hallucinated=false
✓ Line 2: label=true, excellent=true, hallucinated=false
✓ Line 3: label=false, excellent=false, hallucinated=true
...
Progress: 100/100 (100%) | Success: 97 | Failed: 3

════════════════════════════════════════════════════════════════
LABELING SUMMARY
════════════════════════════════════════════════════════════════
Total processed: 100/100
Successfully labeled: 97 (97%)
Failed: 3

Label Distribution:
  good: 78
  hallucinated: 12
  poor_reasoning: 8
  context_mismatch: 5
  excellent: 15
════════════════════════════════════════════════════════════════
```

**JSON Output:**

Examples are saved with boolean labels applied:

```json
{
  "conversations": [...],
  "label": true,
  "excellent": false,
  "hallucinated": false,
  "poor_reasoning": false,
  "context_mismatch": false
}
```

### CLI Options

```bash
# Backend selection
--backend lmstudio              # Use LM Studio (default, local)
--backend openrouter            # Use OpenRouter (cloud)
--backend ollama                # Use Ollama (local)

# Model selection
--model qwen2.5-coder:7b       # Specify model (backend-specific)
--model openai/gpt-4o-mini     # For OpenRouter

# Line range
--start-line 10 --end-line 50  # Label specific range

# In-place editing
--in-place                     # Modify original file

# LLM settings
--temperature 0.3              # Lower = more consistent (default: 0.3)
--max-tokens 500               # Response size (default: 500)

# Preview mode
--dry-run                      # Preview without saving

# Custom config
--config path/to/config.yaml   # Use custom evaluation criteria
```

### Use Cases

- **Dataset Quality Analysis**: Understand distribution of good vs bad examples
- **KTO Training Prep**: `label` field automatically set for chosen/rejected pairs
- **Hallucination Detection**: Identify examples where model makes things up
- **Reasoning Quality**: Find examples with poor thinking blocks
- **Context Alignment**: Detect mismatches between context and response
- **Exemplar Identification**: Find top-quality examples for few-shot prompting

### Integration with KTO Training

The `label` field serves dual purpose:

1. **Quality Indicator**: Good (true) vs bad (false) quality
2. **KTO Training**: Used directly as chosen/rejected pairs

**Training workflow:**
```bash
# 1. Label dataset
python -m improvement_engine.label_dataset \
  --input raw_dataset.jsonl \
  --output labeled_dataset.jsonl

# 2. Filter by quality if desired
# (e.g., remove hallucinated=true examples)

# 3. Use for KTO training
python train_kto.py --dataset labeled_dataset.jsonl
```

**Benefits:**
- No manual labeling needed
- Consistent evaluation criteria
- Scales to large datasets
- Preserves KTO compatibility

## Architecture

```
improvement_engine/
├── config/                      # YAML configuration files
│   ├── quality_guidelines.yaml
│   ├── system_prompts.yaml
│   ├── validation_rules.yaml
│   └── labeling_categories.yaml  # NEW: Labeling categories and display config
├── core/                        # Data models and interfaces
│   ├── models.py                 # Includes labeling models
│   └── exceptions.py
├── services/                    # Business logic
│   ├── llm_service.py           # LLM API client
│   ├── validator.py             # Schema validation
│   ├── file_handler.py          # JSONL file operations
│   ├── improvement_service.py   # Improvement orchestration
│   └── labeling_service.py      # NEW: Labeling orchestration
├── ui/                          # NEW: User interface components
│   ├── __init__.py
│   └── interactive_display.py   # Console UI for labeling
├── utils/                       # Utilities
│   ├── logger.py
│   ├── backup.py
│   ├── yaml_loader.py
│   ├── dataset_scanner.py
│   └── progress_tracker.py      # NEW: Progress tracking for labeling
└── label_dataset.py             # NEW: Labeling CLI entry point
```

## Quality Improvements

The engine improves thinking blocks by:

1. **Goal Specificity**: Adding file paths and concrete actions
2. **Memory Enrichment**: Adding WHY/WHEN context and quantitative details (min 50 chars)
3. **Requirements vs Plan**: Ensuring distinct verification checks vs execution steps
4. **Confidence Calibration**: Risk-based scoring (0.3-0.5 risky, 0.85-0.95 safe)
5. **Assessment Accuracy**: Setting complexity and risk flags correctly

### Before

```json
{
  "goal": "Update documentation",
  "memory": "User reviewing files",
  "requirements": ["Update docs", "Documentation update"],
  "confidence": 0.96,
  "plan": ["Update docs", "Documentation update"]
}
```

### After

```json
{
  "goal": "Update README.md installation section with new dependency requirements",
  "memory": "User reviewed 3 config files modified yesterday during authentication refactor. Need to verify API keys are properly externalized before deployment to production environment.",
  "requirements": [
    "Verify README.md exists and is writable",
    "Check current installation section format",
    "Confirm new dependencies are documented"
  ],
  "confidence": 0.72,
  "plan": [
    "Read current README.md content",
    "Update installation section with dependency list",
    "Verify changes are properly formatted",
    "Write updated content back to file"
  ]
}
```

## Configuration

### System Prompts (`config/system_prompts.yaml`)

Contains the LLM prompt that defines improvement rules. Edit to customize improvement behavior.

### Quality Guidelines (`config/quality_guidelines.yaml`)

Defines what makes a good vs bad example. Used by validation and as reference for LLM improvements.

### Validation Rules (`config/validation_rules.yaml`)

JSON schema validation rules. Ensures improved examples are structurally valid.

## Batch Processing

- Processes examples **sequentially** in batches
- Waits for batch N to complete before starting batch N+1
- Default batch size: 10 examples
- Configurable via CLI prompt

## Safety Features

1. **Automatic Backups**: Creates timestamped backup before any changes
2. **Dry Run Mode**: Preview improvements without writing
3. **Validation Gates**: JSON must parse and pass schema validation
4. **Progress Logging**: Track successes/failures in real-time
5. **Rollback Capability**: Restore from backups if needed

## Cost Estimation

Approximate costs using gpt-4o-mini:
- ~1491 examples @ $0.00015 per 1K tokens (input) + $0.0006 per 1K tokens (output)
- Average: ~500 tokens input + ~600 tokens output per example
- Estimated: **~$1.50 per 1000 examples**

## Troubleshooting

### API Key Issues
```
Error: OPENROUTER_API_KEY not found in .env file
```
**Solution**: Add your API key to the root `.env` file in the repository root with the settings from `improvement_engine/.env.example`.

### Validation Failures
```
✗ Line 42: Validation failed
```
**Solution**: Check logs for specific validation errors. LLM may have returned invalid JSON.

### Rate Limiting
```
API request failed: 429 Too Many Requests
```
**Solution**: Reduce batch size or add delay between batches.

## Rubric YAML Configuration

Rubrics define how to judge and improve specific scopes of training data. Each rubric is a YAML file in `improvement_engine/rubrics/`.

### Rubric Structure

```yaml
name: My Rubric Name
description: What this rubric evaluates
scope: system_prompt | thinking | response  # Which part to evaluate
pass_threshold: 0.8  # Minimum score to pass (0.0-1.0)

judge_prompt: |
  # Instructions for LLM judge to score the content
  Return JSON: {"myrubric_score": 0.0-1.0}

improver_prompt: |
  # Instructions for LLM to improve the content
  Output ONLY the improved content.

output_schema:
  # JSON schema for judge response validation
  type: object
  properties:
    myrubric_score:
      type: number
      minimum: 0.0
      maximum: 1.0
  required: [myrubric_score]

content_schema:
  # JSON schema for improved content validation (thinking blocks)
  type: object
  properties:
    goal:
      type: string
    # ... other fields
  required: [goal]

schema_validation:
  # Composable validation rules (see below)
```

### Two Types of Content Validation

The engine has two distinct validation mechanisms:

1. **`content_schema`** - Validates the **structure** of improved content (thinking blocks)
   - Checks required fields exist (goal, memory, requirements, plan, etc.)
   - Validates field types (string, array, object, number)
   - Enforces constraints (min/max values, required nested properties)
   - Used for `thinking` scope validation

2. **`schema_validation`** - Validates **string content** against composable rules
   - XML tag presence
   - JSON structure in sections
   - YAML structure
   - Regex patterns with cross-reference support
   - Used for `system_prompt` scope validation

#### content_schema Example (Thinking Blocks)

```yaml
content_schema:
  type: object
  properties:
    goal:
      type: string
      description: Specific, actionable goal statement
    memory:
      type: string
      description: User intent without fabricated dates/history
    requirements:
      type: array
      description: Dynamic state checks
    plan:
      type: array
      description: Direct action steps
    assessment:
      type: object
      properties:
        risky:
          type: boolean
        complex:
          type: boolean
      required: [risky, complex]
    confidence:
      type: number
      minimum: 0.0
      maximum: 1.0
  required: [goal, memory, requirements, plan, assessment, confidence]
```

This ensures every improved thinking block has the correct structure.

### Schema Validation Types

The `schema_validation` section supports composable validation types that run in sequence:

```yaml
schema_validation:
  types: [xml, json, yaml, regex]  # Order = execution order
```

#### XML Validation

Check for required XML tags:

```yaml
xml:
  required_tags:
    - '<session_context>'
    - '<vault_structure>'
    - '<selected_workspace'
```

#### JSON Validation

Validate JSON structure in XML sections:

```yaml
json:
  sections:
    - tag: 'selected_workspace'
      extract_pattern: '\{[\s\S]*\}'
      required_fields:
        - 'context'
        - 'workspaceStructure'
        - name: 'workflows'
          type: 'array'
          min_items: 1
        - name: 'preferences'
          type: 'string'
          min_length: 1
```

#### YAML Validation

Validate YAML structure in XML sections:

```yaml
yaml:
  sections:
    - tag: 'available_agents'
      min_items: 2
      item_fields: ['name', 'description']
```

#### Regex Validation

Pattern matching with optional tag scoping:

```yaml
regex:
  patterns:
    - pattern: '\btool_call:\s*\w+'
      description: 'Must contain tool_call format'
    - pattern: '{workspace_root}'  # Interpolated variable
      in_tag: 'vault_structure'
      description: 'Workspace rootFolder must exist in vault_structure'
```

### Cross-Reference Validation

The engine supports extracting values from one section and checking they exist in another. This works by composing `json` extraction with `regex` validation.

#### Step 1: Extract Values

In the `json` section, use `extract` to store values:

```yaml
json:
  sections:
    - tag: 'selected_workspace'
      extract_pattern: '\{[\s\S]*\}'
      required_fields: [...]
      extract:
        - path: 'context.rootFolder'    # Dot notation path
          as: 'workspace_root'          # Variable name
          skip_if: '/'                  # Skip if value equals this
```

#### Step 2: Reference Values

In the `regex` section, use `{var_name}` to reference extracted values:

```yaml
regex:
  patterns:
    - pattern: '{workspace_root}'       # Interpolated from extract
      in_tag: 'vault_structure'         # Scope to this tag
      description: 'Workspace rootFolder must exist in vault_structure'
```

#### How It Works

1. **Types run in order**: `[xml, json, yaml, regex]`
2. **json extracts**: `context.rootFolder = "Vehicles/"` → stored as `workspace_root`
3. **skip_if applies**: If value is `"/"`, extraction is skipped
4. **regex interpolates**: `{workspace_root}` → `"Vehicles/"`
5. **in_tag scopes**: Search only within `<vault_structure>` content
6. **Error if not found**: `"Vehicles/' not found in <vault_structure>"`

#### Full Example

```yaml
schema_validation:
  types: [xml, json, yaml, regex]

  xml:
    required_tags:
      - '<vault_structure>'
      - '<selected_workspace'

  json:
    sections:
      - tag: 'selected_workspace'
        extract_pattern: '\{[\s\S]*\}'
        required_fields:
          - 'context'
          - name: 'workflows'
            type: 'array'
            min_items: 1
        extract:
          - path: 'context.rootFolder'
            as: 'workspace_root'
            skip_if: '/'

  yaml:
    sections:
      - tag: 'available_agents'
        min_items: 2
        item_fields: ['name', 'description']

  regex:
    patterns:
      - pattern: '{workspace_root}'
        in_tag: 'vault_structure'
        description: 'Workspace rootFolder must exist in vault_structure'
```

### Available Rubrics

List available rubrics:

```bash
python -m improvement_engine.services.rubric_runner --list
```

Current rubrics:
- `system_prompt_format` - Validates system prompt structure and hierarchy
- `thinking_quality` - Evaluates goal, memory, requirements/plan, confidence
- `requirements_plan` - Ensures requirements vs plan distinction
- `factuality` - Checks for fabricated dates/history in memory field

### Running Rubrics

```bash
python -m improvement_engine.services.rubric_runner \
  --file path/to/dataset.jsonl \
  --output path/to/output.jsonl \
  --rubrics system_prompt_format,thinking_quality \
  --backend lmstudio \
  --start-line 1 \
  --end-line 10 \
  --max-iterations 3
```

## Development

### Adding New Validation Rules

Edit `config/validation_rules.yaml`:
```yaml
thinking_block_schema:
  constraints:
    my_new_field:
      min_length: 10
      max_length: 100
```

### Customizing System Prompt

Edit `config/system_prompts.yaml`:
```yaml
improvement_prompt: |
  Your custom instructions here...
```

### Creating a New Rubric

1. Create `rubrics/my_rubric.yaml` with the structure above
2. Define `judge_prompt` for scoring
3. Define `improver_prompt` for improvements
4. Add `output_schema` for judge response validation
5. Optionally add `schema_validation` for structural checks
6. Test with: `python -m improvement_engine.services.rubric_runner --rubrics my_rubric ...`

### Testing

```bash
# Dry run on small sample
python tuner.py improve
# Select dataset, use batch_size=5, end_line=50, dry_run=yes
```

## Integration with Training Pipeline

1. **Improve dataset**: `python tuner.py improve`
2. **Train with improved data**: `python tuner.py train`
3. **Upload model**: `python tuner.py upload`
4. **Evaluate**: `python tuner.py eval`

Or use the pipeline:
```bash
# Full pipeline with custom dataset
python tuner.py pipeline
```

## License

Part of Toolset-Training project.
