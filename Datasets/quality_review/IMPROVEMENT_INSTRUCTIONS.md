# Dataset Quality Improvement - Instructions

## Context

A comprehensive quality review of the entire training dataset (5,505 examples) has been completed using 184 parallel agents across 7 rounds. The dataset is now fully scored and ready for improvement.

## Current State

**Scored Dataset:** `Datasets/quality_review/scored_complete_relabeled.jsonl`
- **Total examples:** 5,518 (100% coverage)
- **True (desirable):** 1,837 examples (33.3%)
- **False (undesirable):** 3,681 examples (66.7%)

**Quality Breakdown:**
- Overall quality: 2.83/5.0
- Need improvement: 3,415 examples (61.9%)

**Relabeling Criteria Applied:**
- `overall_quality < 3.0` → labeled `false`
- `sessionMemory_quality == 1` → labeled `false`

## Your Mission

**Improve the poor-quality examples and create an interleaved KTO training dataset.**

### Objectives:

1. **Improve False-Labeled Examples:**
   - Load `Datasets/quality_review/scored_complete_relabeled.jsonl`
   - Filter for examples where `label == false`
   - Read the `quality_scores.notes` field to understand WHY each failed
   - Enhance examples following the improvement guidelines below

2. **Create Interleaved Dataset:**
   - Interleave improved (true) with original poor (false) examples
   - Pattern: `True, False, True, False, True, False...`
   - This creates contrastive pairs for KTO preference learning

3. **Validate All Changes:**
   - Use `python tools/validate_syngen.py <file>` after improvements
   - Ensure tool schemas match `tools/tool_schemas.json`
   - Verify context object pattern compliance

## Key Files & Resources

**Input Files:**
- `Datasets/quality_review/scored_complete_relabeled.jsonl` - Scored dataset with labels
- `Datasets/quality_review/quality_triage_report.md` - Comprehensive analysis
- `tools/INTERACTION_QUALITY_RUBRIC.md` - Scoring rubric (reference for quality standards)

**Validation Tools:**
- `tools/validate_syngen.py` - Schema validator (USE THIS!)
- `tools/tool_schemas.json` - Canonical tool schemas
- `tools/analyze_tool_coverage.py` - Tool coverage analysis

**Source Dataset:**
- `Datasets/syngen_tools_sft_merged_complete_11.21.25.jsonl` - Original 5,505 examples

## Common Quality Issues to Fix

Based on the quality review, here are the most common issues (from `quality_scores.notes`):

### 1. Empty sessionMemory (most critical)
**Problem:** `sessionMemory: ""`
**Fix:** Add meaningful prior context
```json
// BAD
"sessionMemory": ""

// GOOD
"sessionMemory": "Listed Projects/ (23 files), searched 'async' (12 matches), now examining core/utils.ts"
```

### 2. Generic toolContext
**Problem:** Just restates action without explaining WHY
**Fix:** Add workflow reasoning
```json
// BAD
"toolContext": "Searching for files"

// GOOD
"toolContext": "Need to locate all config files before batch update to ensure consistent environment variables across services"
```

### 3. Missing Result Objects
**Problem:** Response shows only tool call, no Result
**Fix:** Add complete tool execution with Result
```json
// Add Result section showing:
{
  "success": true,
  "data": [...],
  "executionTime": "145ms",
  "totalResults": 12
}
```

### 4. Weak Goal Hierarchies
**Problem:** primaryGoal and subgoal are too similar
**Fix:** Make primaryGoal strategic, subgoal tactical
```json
// BAD
"primaryGoal": "List agents",
"subgoal": "Show all agents"

// GOOD
"primaryGoal": "Audit agent configuration for optimization",
"subgoal": "Inventory all active agents to identify redundancies"
```

### 5. Unnatural Prompts
**Problem:** Result objects used as user messages
**Fix:** Convert to natural user language
```json
// BAD (user message)
"Result: {\"error\": \"Agent not found\"}"

// GOOD (user message)
"The agent lookup failed - can you check if the agent still exists?"
```

## Improvement Workflow

### Step 1: Extract Poor Examples
```python
import json

# Load scored dataset
with open('Datasets/quality_review/scored_complete_relabeled.jsonl') as f:
    examples = [json.loads(line) for line in f if line.strip()]

# Filter for false (poor quality) examples
poor_examples = [ex for ex in examples if ex['label'] == False]
print(f"Found {len(poor_examples)} poor examples to improve")
```

### Step 2: Improve Examples
For each poor example:
1. Read `quality_scores.notes` to understand issues
2. Check dimension scores (which are < 3?)
3. Apply fixes based on the notes
4. Preserve `_line_number` for traceability

### Step 3: Create Interleaved Dataset
```python
# Interleave: improved (true) with original poor (false)
interleaved = []
for i, improved in enumerate(improved_examples):
    interleaved.append(improved)  # True
    if i < len(poor_examples):
        interleaved.append(poor_examples[i])  # False
```

### Step 4: Validate
```bash
# Validate improved dataset
python tools/validate_syngen.py Datasets/quality_review/improved_interleaved.jsonl
```

## Context Object Pattern (CRITICAL)

Every tool call MUST include this complete context as the FIRST parameter:

```json
{
  "context": {
    "sessionId": "session_1731015400000_a1b2c3d4e",
    "workspaceId": "ws_1731015400000_f5g6h7i8j",
    "sessionDescription": "Brief summary of session",
    "sessionMemory": "Never empty - prior context with specifics",
    "toolContext": "Why calling this tool in this workflow",
    "primaryGoal": "User's strategic objective",
    "subgoal": "Tactical step toward primaryGoal"
  },
  "otherParams": "..."
}
```

**All 7 fields required. sessionMemory must NEVER be empty.**

## Success Criteria

Your improved dataset should:
1. ✅ Pass `validate_syngen.py` with zero errors
2. ✅ Have True/False interleaved pattern
3. ✅ Improved examples score ≥3.5 on quality dimensions
4. ✅ All context fields populated with meaningful content
5. ✅ Clear workflow reasoning in toolContext
6. ✅ Complete tool execution (call → Result → response)

## Output Files

Create these files:
- `Datasets/quality_review/improved_interleaved.jsonl` - Final KTO training dataset
- `Datasets/quality_review/improvement_report.md` - Summary of changes made
- `Datasets/quality_review/validation_results.txt` - Output from validate_syngen.py

## Notes

- Use parallel agents if improving many examples (batches of 100-200)
- Refer to top-scoring examples (overall_quality ≥ 4.0) as templates
- Keep original poor examples for contrastive learning (don't delete)
- The interleaved pattern is MANDATORY for KTO training (prevents homogeneous batch crashes)

## Ready to Start?

The scored dataset is ready at `Datasets/quality_review/scored_complete_relabeled.jsonl`. Start by analyzing the most common issues in the quality report, then begin improving examples systematically.

Good luck! 🚀
