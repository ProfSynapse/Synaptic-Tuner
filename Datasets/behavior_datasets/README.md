# Behavior-Based KTO Training Datasets

This directory contains separate datasets for each behavioral pattern we're teaching through preference learning (KTO).

## Directory Structure

```
behavior_datasets/
├── intellectual_humility/          # Verification, asking, escalation
├── context_continuity/             # Rich context, workflow tracking
├── verification_before_action/     # Safety before destructive ops
├── strategic_tool_selection/       # Efficiency, batch operations
├── error_recovery/                 # Handling failures, adaptation
└── workspace_awareness/            # Using workspace metadata
```

## Philosophy

Each behavior dataset is **independent** and **version-controlled**, allowing:
- ✅ Separate evolution and refinement
- ✅ Individual quality control
- ✅ Modular training (train on specific behaviors)
- ✅ Easy combination later (merge when ready)
- ✅ Behavior-specific evaluation

## Dataset Format

All datasets use the standard ChatML format with KTO labels:

```jsonl
{
  "conversations": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "tool_call: toolName\narguments: {...}"}
  ],
  "label": true,
  "behavior": "behavior_name"
}
```

**Critical Requirements:**
- ✅ Single-turn format (no Result objects in assistant content)
- ✅ Paired examples (same user request, different responses)
- ✅ Interleaved labels (True/False/True/False pattern for KTO)
- ✅ Behavior field identifies which pattern this teaches
- ✅ All context fields present (7 required fields)
- ✅ sessionMemory never empty

## Workflow

### 1. Generate Pairs

For each behavior:
1. Read the rubric: `../behavior_rubrics/{behavior_name}.yaml`
2. Follow generation guide: `../BEHAVIOR_RUBRIC_GUIDE.md`
3. Create paired examples (good + bad for same user request)
4. Save to behavior folder: `{behavior_name}/pairs_v{version}.jsonl`

### 2. Validate

Run **both validators** on each dataset:

```bash
# Validator 1: Schema and structure
python tools/validate_syngen.py Datasets/behavior_datasets/{behavior_name}/pairs_v1.0.jsonl

# Validator 2: Behavior-specific (when implemented)
python tools/validate_behavior_pairs.py \
  --dataset Datasets/behavior_datasets/{behavior_name}/pairs_v1.0.jsonl \
  --rubric Datasets/behavior_rubrics/{behavior_name}.yaml
```

**Must pass:**
- Zero schema validation errors
- All tool calls use valid tools from schema
- All required parameters present
- Context objects complete (7 fields)
- sessionMemory never empty
- toolContext is STRING (not object)
- Behavior-specific quality criteria met

### 3. Review

Manual review of 10-20% sample:
- User requests are natural and varied
- Positive examples demonstrate ALL positive indicators
- Negative examples show realistic mistakes (not nonsense)
- Pairs use identical user requests
- Labels correctly assigned (true=good, false=bad)

### 4. Interleave

Prepare for KTO training:

```bash
python tools/interleave_dataset.py \
  --input Datasets/behavior_datasets/{behavior_name}/pairs_v1.0.jsonl \
  --output Datasets/behavior_datasets/{behavior_name}/interleaved_v1.0.jsonl
```

Verify pattern: `True, False, True, False, True, False...`

### 5. Merge (Optional)

Combine multiple behavior datasets:

```bash
python tools/merge_behavior_datasets.py \
  --behaviors intellectual_humility context_continuity verification_before_action \
  --output Datasets/behavior_training_combined_v1.0.jsonl \
  --interleave
```

## Priority Levels

Behaviors are prioritized for training focus:

**HIGH Priority** (Train first):
- ✅ Intellectual Humility - Prevents overconfident mistakes
- ✅ Context Continuity - Enables multi-step operations
- ✅ Verification Before Action - Safety-critical

**MEDIUM Priority** (Train second):
- Strategic Tool Selection - Efficiency improvements
- Error Recovery - Robustness enhancements

**LOW-MEDIUM Priority** (Train third):
- Workspace Awareness - UX improvements for structured users

## Target Volumes

Per behavior dataset:
- **Pairs:** 100-200 (same user request, different responses)
- **Total examples:** 200-400 (after separating pairs)
- **Quality threshold:** 4-5/5 average

Combined dataset:
- **Total pairs:** 600-1200
- **Total examples:** 1200-2400
- **After merge with SFT:** ~6700-7900 examples

## Validation Checklist

Before considering a behavior dataset "complete":

**Schema Validation:**
- [ ] `validate_syngen.py` passes with 0 errors
- [ ] All tools exist in tool_schemas.json
- [ ] All required parameters present
- [ ] All context objects complete

**Quality Validation:**
- [ ] 10-20% manual review completed
- [ ] Positive examples: sessionMemory 80-150 chars with specifics
- [ ] Positive examples: toolContext 60-120 chars explaining WHY
- [ ] Negative examples: sessionMemory <50 chars or generic
- [ ] Negative examples: toolContext <40 chars just restating action
- [ ] User requests are natural (not "Result:" format)
- [ ] Pairs use identical user requests

**Behavior Validation:**
- [ ] Positive examples demonstrate ALL positive indicators from rubric
- [ ] Negative examples show realistic anti-patterns
- [ ] Trigger scenarios appropriately represented
- [ ] Tool coverage matches rubric expectations

**Structure Validation:**
- [ ] Perfect 1:1 label balance (50% true, 50% false)
- [ ] Interleaved pattern: True/False/True/False...
- [ ] behavior field present and correct
- [ ] Single-turn format (no Result objects)

## Notes

### Why Separate Datasets?

1. **Independent Evolution:** Each behavior can be refined without affecting others
2. **Targeted Training:** Train specific behaviors that need improvement
3. **Quality Control:** Easier to review and validate smaller focused datasets
4. **Versioning:** Track improvements to each behavior separately
5. **Debugging:** Identify which behaviors model learns well vs poorly

### When to Merge?

Merge when:
- Each behavior dataset is validated and high-quality
- Ready for full training run
- Want to combine multiple behaviors in single KTO session
- Creating a "release" version of behavior training data

Keep separate when:
- Still generating/refining examples
- Testing individual behaviors
- Debugging quality issues
- Experimenting with different approaches

### Versioning Scheme

Use semantic versioning for datasets:
- `pairs_v1.0.jsonl` - Initial generation
- `pairs_v1.1.jsonl` - Minor refinements (fixed some examples)
- `pairs_v2.0.jsonl` - Major revision (regenerated with new rubric)
- `interleaved_v1.0.jsonl` - Processed for training

## Quick Start

### Generate First Behavior Dataset

```bash
# 1. Read the rubric
cat Datasets/behavior_rubrics/intellectual_humility.yaml

# 2. Read generation guide
less Datasets/BEHAVIOR_RUBRIC_GUIDE.md

# 3. Generate 10 seed pairs manually (template in guide)
# Save to: Datasets/behavior_datasets/intellectual_humility/seed_pairs_v1.0.jsonl

# 4. Use seed pairs to prompt teacher model for more
# (Use generation template from guide)

# 5. Validate
python tools/validate_syngen.py \
  Datasets/behavior_datasets/intellectual_humility/pairs_v1.0.jsonl

# 6. Review sample
head -20 Datasets/behavior_datasets/intellectual_humility/pairs_v1.0.jsonl | python3 -m json.tool

# 7. Interleave
python tools/interleave_dataset.py \
  --input Datasets/behavior_datasets/intellectual_humility/pairs_v1.0.jsonl \
  --output Datasets/behavior_datasets/intellectual_humility/interleaved_v1.0.jsonl

# 8. Ready for training!
```

## References

- **Master Rubric Guide:** `../BEHAVIOR_RUBRIC_GUIDE.md`
- **YAML Rubrics:** `../behavior_rubrics/`
- **Tool Schemas:** `../../tools/tool_schemas.json`
- **Validator:** `../../tools/validate_syngen.py`
- **Training Guide:** `../../Trainers/rtx3090_kto/README.md`
