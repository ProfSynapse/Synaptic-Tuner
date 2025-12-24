# Unified Validation & Evolutionary Fine-Tuning

Design document for unifying config-driven validation across the codebase, then building evolutionary training on top.

> **Status:**
> - ✅ **Phase 1 COMPLETE** - Unified validation architecture implemented in `shared/validation/`
> - ⏳ **Phase 2 PENDING** - Evolutionary fine-tuning (ready to start)

## Table of Contents

1. [Philosophy](#philosophy)
2. [Phase 1: Unified Validation Architecture](#phase-1-unified-validation-architecture) ✅
3. [Phase 2: Evolutionary Fine-Tuning](#phase-2-evolutionary-fine-tuning)
4. [Implementation Roadmap](#implementation-roadmap)

---

## Philosophy

### Core Principles

1. **Config-Driven Everything**
   - Developers configure behavior through YAML, not code changes
   - New use cases = new config files, not new Python modules
   - Same config format across all systems (SynthChat, Evaluator, Trainer)

2. **No Code Duplication**
   - Shared infrastructure lives in `shared/`
   - All systems import from the same source
   - Single source of truth for validation logic

3. **Format-Agnostic Validation**
   - Parsing layer handles model-specific formats (Qwen, Mistral, ChatML, OpenAI)
   - Validation layer operates on normalized structures
   - Adding a new model format = one code change in parsing, zero changes elsewhere

4. **Maximum Flexibility**
   - No hardcoded field names or schemas
   - Validation rules are data, not code
   - Same rubric YAML works in SynthChat, Evaluator, and Trainer

---

## Phase 1: Unified Validation Architecture

> ✅ **COMPLETED** - All validation code now lives in `shared/validation/`

### Quick Usage

```python
# Import from shared validation
from shared.validation import (
    parse_response,           # Format-agnostic parsing
    ParsedResponse,           # Normalized response
    StructureValidator,       # Config-driven validation
    RubricLoader,            # Rubric management
)

# Parse any format (Qwen, Mistral, ChatML, OpenAI)
parsed = parse_response(model_output)
if parsed.has_tool_calls:
    tool_name = parsed.first_tool_call.name
    args = parsed.first_tool_call.arguments

# Validate with config
validator = StructureValidator()
is_valid, errors = validator.validate(data, validations)
```

### The Problem (Solved)

Three systems need the same validation logic:

| System | Purpose | Status |
|--------|---------|--------|
| **SynthChat** | Validate/improve training examples | ✅ Uses `shared.validation` |
| **Evaluator** | Validate model responses during eval | ✅ Uses `shared.validation` |
| **Trainer** | Compute fitness during training | ⏳ Ready for Phase 2 |

#### Previous Architecture (Fragmented) - NOW FIXED

The old fragmented structure has been consolidated into `shared/validation/`:

```
# OLD (fragmented):
SynthChat/services/validators/     ← Had canonical validators
Evaluator/                         ← Had hacky importlib imports
Trainers/                          ← Had no access

# NEW (unified):
shared/validation/                 ← Single source of truth
├── parsing/                       ← Format-agnostic parsing
├── validators/                    ← Config-driven validation
└── rubric/                        ← Rubric management
```

**Problems SOLVED:**
1. ~~`Evaluator/rubric_validator.py` lines 15-28 use hacky `importlib`~~ → Now uses clean imports
2. ~~Format parsing lives in Evaluator, schema validation in SynthChat~~ → Both in shared/
3. ~~No clean way for Trainers to access either~~ → Direct imports from shared/
4. ~~Duplication of coordination logic between systems~~ → Single implementation

### The Solution (Implemented)

All validation now lives in `shared/validation/` with clean imports everywhere.

#### Current Architecture (Unified)

```
shared/validation/                      ← SINGLE SOURCE OF TRUTH
├── __init__.py                         ← Public API exports
│
├── parsing/                            ← Layer 1: Format Detection
│   ├── __init__.py
│   ├── tool_call_parser.py             (FROM Evaluator)
│   ├── response_parser.py              (FROM Evaluator)
│   └── formats/                        ← Extensible format handlers
│       ├── __init__.py
│       ├── base.py                     ← Abstract format handler
│       ├── qwen.py                     ← <tool_call>...</tool_call>
│       ├── mistral.py                  ← [TOOL_CALLS] [...]
│       ├── chatml.py                   ← tool_call: ...\narguments: ...
│       ├── openai.py                   ← {"tool_calls": [...]}
│       └── registry.py                 ← Format auto-detection
│
├── validators/                         ← Layer 2: Schema Validation
│   ├── __init__.py
│   ├── structure_validator.py          (FROM SynthChat)
│   ├── cross_scope_validator.py        (FROM SynthChat)
│   └── content/                        ← Content-type validators
│       ├── __init__.py
│       ├── base.py                     (FROM SynthChat)
│       ├── json_validator.py           (FROM SynthChat)
│       ├── xml_validator.py            (FROM SynthChat)
│       ├── regex_validator.py          (FROM SynthChat)
│       ├── yaml_validator.py           (FROM SynthChat)
│       ├── code_validator.py           (FROM SynthChat)
│       └── registry.py                 (FROM SynthChat)
│
├── rubric/                             ← Rubric Management
│   ├── __init__.py
│   ├── loader.py                       (FROM SynthChat services/data/)
│   ├── cache.py                        (FROM SynthChat services/data/)
│   └── repository.py                   (FROM SynthChat services/data/)
│
├── results.py                          ← Unified result types
│
└── fitness.py                          ← Layer 3: Scoring (for Trainer)

───────────────────────────────────────────────────────────────────────

SynthChat/services/validators/          ← Thin re-exports
└── __init__.py
    # Backwards compatibility
    from shared.validation.validators import (
        StructureValidator,
        CrossScopeValidator,
    )
    from shared.validation.validators.content import (
        ContentValidatorRegistry,
        # ... etc
    )

Evaluator/                              ← Clean imports
├── rubric_validator.py
│   from shared.validation import RubricValidator
├── schema_validator.py
│   from shared.validation.parsing import parse_response
└── tool_call_parser.py
    # DELETED - now in shared/validation/parsing/

Trainers/                               ← Direct access
└── rtx3090_sft/
    from shared.validation import FitnessEvaluator
```

### Two-Layer Architecture

The key insight: **Format parsing and schema validation are separate concerns.**

```
┌─────────────────────────────────────────────────────────────────────┐
│                       MODEL OUTPUT (any format)                     │
│                                                                     │
│  Qwen:    "<tool_call>{"name": "useTools", ...}</tool_call>"       │
│  Mistral: "[TOOL_CALLS] [{"name": "useTools", ...}]"               │
│  ChatML:  "tool_call: useTools\narguments: {...}"                  │
│  OpenAI:  {"tool_calls": [{"function": {"name": "useTools"}}]}     │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 LAYER 1: FORMAT PARSING                             │
│                 shared/validation/parsing/                          │
│                                                                     │
│  • Auto-detects format (Qwen, Mistral, ChatML, OpenAI)             │
│  • Handles model-specific markers and syntax                        │
│  • Normalizes to common structure                                   │
│  • Extensible: add new formats without changing validation          │
│                                                                     │
│  INPUT:  Raw string or dict (any format)                           │
│  OUTPUT: ParsedResponse(tool_calls=[ToolCall(name, arguments)])    │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 LAYER 2: SCHEMA VALIDATION                          │
│                 shared/validation/validators/                       │
│                                                                     │
│  • Config-driven rules (from YAML - same format as rubrics)        │
│  • Validates normalized structure (format-agnostic)                │
│  • Tool manifest validation                                         │
│  • Field existence, types, constraints                              │
│  • Cross-scope validation                                           │
│                                                                     │
│  INPUT:  ParsedResponse (normalized)                               │
│  OUTPUT: (is_valid: bool, errors: List[str])                       │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                     ┌─────────────┼─────────────┐
                     │             │             │
                     ▼             ▼             ▼
              ┌──────────┐  ┌──────────┐  ┌──────────┐
              │SynthChat │  │Evaluator │  │ Trainer  │
              │          │  │          │  │          │
              │ Improve  │  │ Report   │  │ Fitness  │
              │ examples │  │ results  │  │ score    │
              └──────────┘  └──────────┘  └──────────┘
```

### Why Format-Agnostic Matters

Different models represent tool calls differently:

| Model | Format | Marker |
|-------|--------|--------|
| Qwen | XML-like | `<tool_call>...</tool_call>` |
| Mistral | Bracket prefix | `[TOOL_CALLS] [...]` |
| ChatML | Text markers | `tool_call: ...\narguments: ...` |
| OpenAI | Structured dict | `{"tool_calls": [...]}` |

**The validation config should NOT know about formats:**

```yaml
# BAD - Hardcodes format, breaks when switching models
validations:
  - match: "<tool_call>"           # Only works for Qwen
    type: contains
  - match: "[TOOL_CALLS]"          # Only works for Mistral
    type: contains
```

```yaml
# GOOD - Validates parsed structure, works with any model
validations:
  - tools:
      useTools:
        _required: ["context", "calls"]
        context:
          sessionId: string
          workspaceId: string
```

The `tools:` validation type operates on **parsed tool calls**, not raw text. The parsing layer handles format detection automatically.

### Adding a New Model Format

When a new model comes out with a different tool call format:

1. **Add format handler** to `shared/validation/parsing/formats/`:
```python
# shared/validation/parsing/formats/newmodel.py

class NewModelFormatHandler(BaseFormatHandler):
    """Handle NewModel's [TOOLS]...[/TOOLS] format."""

    def can_parse(self, content: str) -> bool:
        return "[TOOLS]" in content

    def parse(self, content: str) -> List[ToolCall]:
        # Extract and parse tool calls
        ...
```

2. **Register in registry**:
```python
# shared/validation/parsing/formats/registry.py

FORMAT_HANDLERS = [
    NewModelFormatHandler(),  # Add new format
    QwenFormatHandler(),
    MistralFormatHandler(),
    ChatMLFormatHandler(),
    OpenAIFormatHandler(),
]
```

3. **All existing validation configs work unchanged** - they validate the parsed structure, not the raw format.

### Config Format: Same as Rubrics

The validation config uses the **exact same format** as existing rubrics. No new syntax to learn.

```yaml
# This works as both:
# - A SynthChat rubric (for improvement engine)
# - An Evaluator rubric (for model evaluation)
# - A Trainer fitness config (for evolutionary training)

name: "tool_calling"
description: "Validate tool call structure and context"
scope: response

validations:
  # Tool manifest validation
  - tools:
      useTools:
        _required: ["context", "calls"]
        context:
          _required: ["sessionId", "workspaceId"]
          sessionId: string
          workspaceId: string
        calls:
          _item_schema:
            _required: ["agent", "tool", "params"]
            agent: string
            tool: string
            params: object
          _subtools:
            vaultManager:
              moveNote:
                _required: ["path", "destination"]
                path: string
                destination: string
            contentManager:
              readNote:
                _required: ["path"]
                path: string

  # Cross-scope validation (IDs match system prompt)
  - cross_scope:
      from: "tool_calls[0].arguments.context.sessionId"
      to: "system_prompt"
      match_pattern: "sessionId:\\s*([\\w-]+)"
    error: "sessionId doesn't match system prompt"

# Optional: scoring config (used by Trainer)
scoring:
  method: "error_count"
  params:
    max_errors_before_zero: 5
```

### Unified Result Types

All systems use the same result types:

```python
# shared/validation/results.py

@dataclass
class ToolCall:
    """Normalized tool call (format-agnostic)."""
    name: str
    arguments: Dict[str, Any]

@dataclass
class ParsedResponse:
    """Result of format parsing."""
    tool_calls: List[ToolCall]
    text_content: Optional[str]
    raw_content: str
    format_detected: str  # "qwen", "mistral", "chatml", "openai"

@dataclass
class ValidationResult:
    """Result of schema validation."""
    passed: bool
    errors: List[str]
    validated_tools: List[ToolCall]

@dataclass
class FitnessResult:
    """Result of fitness evaluation (for Trainer)."""
    score: float  # 0.0 - 1.0
    validation_result: ValidationResult
    level_scores: Dict[str, float]  # Per-validation-level breakdown
```

### Public API

Clean, simple imports for all systems:

```python
# shared/validation/__init__.py

# Layer 1: Parsing
from .parsing import (
    parse_response,
    parse_tool_calls,
    ParsedResponse,
    ToolCall,
)

# Layer 2: Validation
from .validators import (
    StructureValidator,
    CrossScopeValidator,
    validate_against_rubric,
    ValidationResult,
)

# Rubric Management
from .rubric import (
    RubricLoader,
    RubricCache,
    load_rubric,
)

# Layer 3: Fitness (for Trainer)
from .fitness import (
    FitnessEvaluator,
    FitnessResult,
)
```

**Usage in each system:**

```python
# SynthChat - improvement engine
from shared.validation import StructureValidator, parse_response

def improve_example(example):
    parsed = parse_response(example["assistant_content"])
    is_valid, errors = validator.validate(parsed, rubric["validations"])
    if not is_valid:
        # Trigger improvement...

# Evaluator - model evaluation
from shared.validation import parse_response, validate_against_rubric

def evaluate_response(response, rubric_key):
    parsed = parse_response(response)
    result = validate_against_rubric(parsed, rubric_key)
    return result.to_dict()

# Trainer - evolutionary fitness
from shared.validation import FitnessEvaluator

evaluator = FitnessEvaluator(config_path="configs/fitness/tool_calling.yaml")
score = evaluator.evaluate(model_output, example)  # Returns 0.0-1.0
```

---

## Phase 2: Evolutionary Fine-Tuning

> **Prerequisite:** Phase 1 must be complete. Evolutionary training depends on unified validation for fitness evaluation.

### The Problem

Training a model to use tools correctly involves:
- **Discrete decisions**: Which tool to call (gradient signal is noisy)
- **Structured output**: Exact JSON format required
- **Pattern matching**: Intent → tool mapping

Standard gradient descent struggles because:
```
Gradient says: "Adjust weights in this direction"
But we don't know if that direction improves tool selection
We just apply it and hope
```

### The Solution: Gradient + ES Hybrid

Instead of blindly applying gradients, test candidates first:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SINGLE TRAINING STEP                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. FORWARD + BACKWARD (standard, done once)                        │
│     Input → Model → Loss → Gradient G                              │
│                                                                     │
│  2. CANDIDATE GENERATION                                            │
│     G + noise₁ → Candidate 1                                       │
│     G + noise₂ → Candidate 2                                       │
│     G × scale  → Candidate 3                                       │
│     G (pure)   → Candidate 4                                       │
│                                                                     │
│  3. FITNESS EVALUATION (uses unified validation!)                   │
│     For each candidate:                                             │
│       - Temporarily apply to weights                                │
│       - Forward pass on eval batch                                  │
│       - Parse output (Layer 1 - format agnostic)                   │
│       - Validate (Layer 2 - config-driven)                         │
│       - Compute fitness score                                       │
│       - Revert weights                                              │
│                                                                     │
│  4. SELECTION                                                       │
│     Apply only the highest-fitness candidate                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Fitness Evaluation

Uses the unified validation from Phase 1:

```python
# shared/validation/fitness.py

class FitnessEvaluator:
    """
    Config-driven fitness evaluation.
    Wraps unified parsing + validation layers.
    """

    def __init__(self, config_path: Path):
        # Load validation config (same format as rubrics!)
        self.config = load_yaml(config_path)
        self.validations = self.config.get("validations", [])
        self.scoring = self.config.get("scoring", {})

        # Use shared validators
        self.validator = StructureValidator()

    def evaluate(self, model_output: str, example: Dict) -> float:
        # Layer 1: Parse (format-agnostic)
        parsed = parse_response(model_output)

        if not parsed.tool_calls:
            return 0.0  # No valid tool calls

        # Layer 2: Validate (config-driven)
        data = {
            "tool_calls": [
                {"name": tc.name, "arguments": tc.arguments}
                for tc in parsed.tool_calls
            ]
        }

        is_valid, errors = self.validator.validate(
            data=data,
            validations=self.validations,
        )

        # Layer 3: Score
        return self._compute_score(is_valid, errors)

    def _compute_score(self, is_valid: bool, errors: List[str]) -> float:
        method = self.scoring.get("method", "error_count")

        if method == "binary":
            return 1.0 if is_valid else 0.0
        elif method == "error_count":
            max_errors = self.scoring.get("params", {}).get("max_errors_before_zero", 5)
            return max(0.0, 1.0 - len(errors) / max_errors)

        return 0.5
```

### Training Config

Minimal additions to existing config:

```yaml
# Trainers/rtx3090_sft/configs/config.yaml

model:
  model_name: "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"

training:
  learning_rate: 2e-4
  # ... existing fields ...

# NEW: Evolutionary section (optional, disabled by default)
evolutionary:
  enabled: false

  # Points to validation config (same format as rubrics!)
  validation_config: "configs/fitness/tool_calling.yaml"

  candidates: 4
  eval_batch_size: 4

  strategy:
    type: "gradient_noise"
    params:
      noise_scale: 0.1
```

### Compute Analysis

For 2B model on 24GB VRAM:

| Operation | Time | Count | Total |
|-----------|------|-------|-------|
| Backward pass | 300ms | 1 | 300ms |
| Forward pass (generate) | 100ms | 4 × 4 | 1600ms |
| Parse + validate | ~1ms | 16 | ~16ms |
| **Total per step** | | | **~1900ms** |

~4-5x slower per step, but potentially fewer steps to converge.

Memory fits easily: ~8.5GB used of 24GB available.

---

## Implementation Roadmap

### Phase 1: Unified Validation (Do First)

#### Step 1.1: Create shared/validation/ structure

```bash
mkdir -p shared/validation/{parsing/formats,validators/content,rubric}
```

#### Step 1.2: Move parsing from Evaluator

```
Evaluator/tool_call_parser.py → shared/validation/parsing/tool_call_parser.py
Evaluator/response_parser.py  → shared/validation/parsing/response_parser.py
```

Refactor into format handlers:
```
shared/validation/parsing/formats/
├── base.py       # Abstract handler
├── qwen.py       # Extract from tool_call_parser.py
├── mistral.py    # Extract from tool_call_parser.py
├── chatml.py     # Extract from response_parser.py
├── openai.py     # Handle dict format
└── registry.py   # Auto-detection logic
```

#### Step 1.3: Move validators from SynthChat

```
SynthChat/services/validators/structure_validator.py    → shared/validation/validators/
SynthChat/services/validators/cross_scope_validator.py  → shared/validation/validators/
SynthChat/services/validators/base.py                   → shared/validation/validators/
SynthChat/services/validators/content/                  → shared/validation/validators/content/
```

#### Step 1.4: Move rubric loading from SynthChat

```
SynthChat/services/data/rubric_loader.py     → shared/validation/rubric/loader.py
SynthChat/services/data/rubric_cache.py      → shared/validation/rubric/cache.py
SynthChat/services/data/rubric_repository.py → shared/validation/rubric/repository.py
```

#### Step 1.5: Create unified result types

```python
# shared/validation/results.py
# Define ToolCall, ParsedResponse, ValidationResult, FitnessResult
```

#### Step 1.6: Create public API

```python
# shared/validation/__init__.py
# Export clean public interface
```

#### Step 1.7: Update SynthChat imports

```python
# SynthChat/services/validators/__init__.py
from shared.validation.validators import *
from shared.validation.validators.content import *
```

#### Step 1.8: Update Evaluator imports

```python
# Evaluator/rubric_validator.py
from shared.validation import RubricValidator, StructureValidator
# Remove hacky importlib code!

# Evaluator/schema_validator.py
from shared.validation.parsing import parse_response
```

#### Step 1.9: Delete duplicated files

```bash
rm Evaluator/tool_call_parser.py  # Now in shared/
# Keep thin re-export wrappers for backwards compatibility
```

#### Step 1.10: Test all systems still work

```bash
# Test SynthChat
python -m SynthChat.services.rubric_runner --list

# Test Evaluator
python -m Evaluator.cli --help

# Run any existing tests
pytest
```

### Phase 2: Evolutionary Training (After Phase 1)

#### Step 2.1: Create fitness evaluator

```python
# shared/validation/fitness.py
# Thin wrapper: parse → validate → score
```

#### Step 2.2: Create evolutionary components

```
shared/evolutionary/
├── candidate_generator.py
├── strategies/
│   ├── gradient_noise.py
│   ├── scale_variation.py
│   └── component_dropout.py
└── trainer_wrapper.py
```

#### Step 2.3: Integrate into Trainers

```python
# Trainers/rtx3090_sft/train_sft.py
if config.evolutionary.enabled:
    from shared.evolutionary import EvolutionaryTrainerWrapper
    trainer = EvolutionaryTrainerWrapper(trainer, config.evolutionary)
```

#### Step 2.4: Add fitness config

```yaml
# Trainers/rtx3090_sft/configs/fitness/tool_calling.yaml
# Same format as rubrics!
```

---

## Summary

**Phase 1** unifies validation across all systems:
- Single source of truth in `shared/validation/`
- Format-agnostic parsing layer
- Config-driven validation layer
- Same rubric format everywhere
- Clean imports, no hacks

**Phase 2** builds evolutionary training on top:
- Uses unified validation for fitness evaluation
- Config-driven fitness scoring
- Gradient + ES hybrid approach
- Minimal changes to existing training code

**The key insight:** Tool calls are structured data. Format parsing normalizes any model output. Config-driven validation checks the structure. The same validation pipeline works for data generation (SynthChat), model evaluation (Evaluator), and training (evolutionary fitness).
