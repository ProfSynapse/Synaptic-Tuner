# GRPO/GSPO Reward Rubrics

YAML-driven reward definitions for GRPO/GSPO training.

## Architecture

```
rewards/
├── _schema.yaml          # Shared schema reference (tool_schema.yaml)
├── context_completeness.yaml
├── tool_selection.yaml
├── json_structure.yaml
└── format.yaml
```

## How It Works

1. **Schema Reference**: Rule-based rubrics reference `tool_schema.yaml` for structure definitions
2. **StructureValidator**: Uses `shared/validation` for extraction and validation
3. **Two scoring paths**:
   - **Deterministic**: rule-based scoring against ground truth (default — see `args_match.yaml`, `format.yaml`, etc.)
   - **LLM-judge**: see `judge_rubric.yaml` — opt-in path that scores completions
     via `shared/judge` for non-verifiable criteria (instruction following,
     formatting, safety). Used by setting `type: judge_rubric` in the YAML.

## Adding a New Reward

1. Create `new_reward.yaml` in this directory
2. Define `validation` rules using the schema
3. Define `scoring` rules
4. Add to `config.yaml` rewards.items list

## Example Rubric

```yaml
name: my_reward
description: What this reward measures

# Reference shared schema
schema_ref: tool_format.wrapper_structure.context

# Validation rules (uses StructureValidator)
validation:
  - field: memory
    type: string
    min: 10

# Scoring
scoring:
  strategy: proportional  # or: binary, weighted
  max_score: 1.0
```
