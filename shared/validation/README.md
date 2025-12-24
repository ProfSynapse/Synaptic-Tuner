# Shared Validation Module

Unified validation infrastructure for Synaptic-Tuner, providing format-agnostic response parsing and config-driven validation across all systems.

## Quick Start

```python
from shared.validation import (
    # Parsing
    parse_response,
    ParsedResponse,
    ParsedToolCall,

    # Validators
    StructureValidator,
    CrossScopeValidator,

    # Rubric management
    RubricLoader,
    RubricCache,
    RubricRepository,
)
```

## Architecture

```
shared/validation/
├── __init__.py           # Public API exports
│
├── parsing/              # Layer 1: Format Detection
│   ├── response_parser.py   # Main parser (auto-detects format)
│   ├── tool_call_parser.py  # Qwen/Mistral format handlers
│   ├── enums.py             # ResponseType, ToolCallFormat
│   └── utilities.py         # Shared helpers
│
├── validators/           # Layer 2: Schema Validation
│   ├── structure_validator.py    # Field/pattern/tool validation
│   ├── cross_scope_validator.py  # Cross-field validation
│   ├── base.py                   # Protocol and base class
│   └── content/                  # Content-type validators
│       ├── xml_validator.py
│       ├── json_validator.py
│       ├── yaml_validator.py
│       ├── regex_validator.py
│       ├── code_validator.py
│       └── registry.py
│
└── rubric/               # Rubric Management
    ├── rubric_loader.py      # File I/O
    ├── rubric_cache.py       # In-memory caching
    └── rubric_repository.py  # High-level API
```

## Two-Layer Design

### Layer 1: Format Parsing (Format-Agnostic)

Automatically detects and parses different tool call formats:

| Model | Format | Detection |
|-------|--------|-----------|
| Qwen | `<tool_call>...</tool_call>` | XML-like tags |
| Mistral | `[TOOL_CALLS] [...]` | Bracket prefix |
| ChatML | `tool_call: ...\narguments: ...` | Text markers |
| OpenAI | `{"tool_calls": [...]}` | Structured dict |

```python
from shared.validation import parse_response

# Works with any format!
parsed = parse_response(model_output)

# Access normalized data
if parsed.has_tool_calls:
    tool_name = parsed.first_tool_call.name
    arguments = parsed.first_tool_call.arguments

# Check format detected
print(parsed.format_detected)  # "qwen", "mistral", "chatml", "openai"
```

### Layer 2: Schema Validation (Config-Driven)

Validates the normalized structure using YAML configuration:

```python
from shared.validation import StructureValidator

validator = StructureValidator()

# Validate with config
validations = [
    {"field": "name", "type": "string"},
    {"field": "count", "type": "number", "min": 1},
]
is_valid, errors = validator.validate(data, validations)
```

## Validation Config Format

Same format as rubrics - works in SynthChat, Evaluator, and Trainer:

```yaml
name: "tool_calling"
description: "Validate tool call structure"
scope: response

validations:
  # Field validation
  - field: "context.sessionId"
    type: string
    error: "Missing sessionId"

  # Pattern matching
  - match: "<vault_structure>"
    type: xml
    error: "Missing vault_structure tag"

  # Tool manifest validation
  - tools:
      useTools:
        _required: ["context", "calls"]
        context:
          sessionId: string
          workspaceId: string

  # Cross-scope validation
  - cross_scope:
      from: "tool_calls[0].arguments.context.sessionId"
      to: "system_prompt"
      match_pattern: "sessionId:\\s*([\\w-]+)"
```

## Content Validators

Specialized validators for different content types:

| Validator | Purpose | Config Key |
|-----------|---------|------------|
| XmlContentValidator | XML tag presence | `xml` |
| JsonContentValidator | JSON structure | `json` |
| YamlContentValidator | YAML structure | `yaml` |
| RegexContentValidator | Pattern matching | `regex` |
| CodeContentValidator | Code syntax | `code` |

## Usage by System

### SynthChat

```python
from shared.validation import StructureValidator, parse_response

def improve_example(example):
    parsed = parse_response(example["assistant_content"])
    is_valid, errors = validator.validate(parsed, rubric["validations"])
    if not is_valid:
        # Trigger improvement...
```

### Evaluator

```python
from shared.validation import parse_response, StructureValidator

def evaluate_response(response):
    parsed = parse_response(response)
    # Validate against rubric...
```

### Trainer (Future - Phase 2)

```python
from shared.validation import parse_response, StructureValidator

def compute_fitness(model_output):
    parsed = parse_response(model_output)
    is_valid, errors = validator.validate(parsed.to_dict(), config)
    return 1.0 if is_valid else max(0.0, 1.0 - len(errors) / 5)
```

## Adding New Format Handlers

When a new model uses a different tool call format:

1. Add format detection to `parsing/tool_call_parser.py`
2. Add parsing logic
3. All existing validation configs work unchanged

Example for a hypothetical new format:

```python
def is_newmodel_tool_call(content: str) -> bool:
    return "[TOOLS]" in content

def parse_newmodel_tool_calls(content: str) -> Optional[List[Dict]]:
    # Extract and parse...
```

## Rubric Management

```python
from shared.validation import RubricLoader, RubricRepository
from pathlib import Path

# Direct loading
loader = RubricLoader(Path("rubrics/"))
rubric = loader.load_from_file("tool_calling")

# With caching
repo = RubricRepository(rubrics_dir=Path("rubrics/"))
rubric = repo.get("tool_calling")  # Cached after first load
```

## Design Principles

1. **Format-Agnostic**: Parsing layer handles model differences
2. **Config-Driven**: Validation rules are YAML, not code
3. **No Duplication**: Single source of truth for all systems
4. **Backward Compatible**: Old imports still work via re-exports

## Related Documentation

- [Evolutionary Fine-Tuning Design](../docs/EVOLUTIONARY_FINETUNING.md) - Full architecture doc
- [Behavior Rubric Guide](../Datasets/BEHAVIOR_RUBRIC_GUIDE.md) - Rubric writing guide
