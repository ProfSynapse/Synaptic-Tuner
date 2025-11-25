# Response Pattern Dataset

**Version:** 1.0
**Created:** 2025-11-25
**Status:** In Progress

---

## Purpose

Teaches models **when to stop using tools** and respond with text vs when to continue tool execution.

### Critical Gap Addressed

Current datasets (behavior_merged_kto_v1.1.jsonl, syngen_tools_sft_11.24.25_cleaned.jsonl) contain:
- **100% tool-only responses** (content = null, tool_calls present)
- **0% text-only responses** (no summarization, clarification, or explanations)
- **0% tool+text hybrid responses**

Models trained on these datasets don't learn:
1. When task is complete and should present results
2. When to ask for clarification instead of assuming
3. When to explain actions vs execute silently
4. How to summarize tool results for users

---

## Three Response Patterns

### Pattern 1: Text-Only Response
**Files:** `text_only_pairs_v1.0.jsonl`
**Target:** 150-200 examples (75-100 pairs)

**Teaches:**
- When task is complete → present results
- When unclear → ask clarification
- When error occurs → explain constructively
- When results ready → summarize for user

### Pattern 2: Tool-Only Response
**Files:** `tool_only_pairs_v1.0.jsonl`
**Target:** 150-200 examples (75-100 pairs)

**Teaches:**
- When in workflow middle → continue silently
- When next step obvious → execute without explaining
- When automation expected → maintain flow
- When single target clear → act directly

### Pattern 3: Tool+Text Response
**Files:** `tool_text_pairs_v1.0.jsonl`
**Target:** 150-200 examples (75-100 pairs)

**Teaches:**
- When complex action → explain then execute
- When multiple options → preview choice
- When teaching moment → narrate process
- When surprising action → justify before doing

---

## Dataset Structure

Each pattern file contains KTO-ready pairs:
- **Positive examples (label: true):** Correct pattern choice
- **Negative examples (label: false):** Wrong pattern choice (realistic mistake)
- **Interleaved:** Perfect True/False/True/False pattern

### Example Structure

**Text-Only (Positive):**
```json
{
  "conversations": [
    {
      "role": "user",
      "content": "Result: {\"success\": true, \"results\": [...], \"count\": 3}"
    },
    {
      "role": "assistant",
      "content": "I found 3 files... [summary]. Would you like me to...?"
    }
  ],
  "label": true,
  "pattern": "text_only"
}
```

**Text-Only (Negative - should have responded with text):**
```json
{
  "conversations": [
    {
      "role": "user",
      "content": "Result: {\"success\": true, \"results\": [...], \"count\": 3}"
    },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [...]  // Proceeds without asking - WRONG
    }
  ],
  "label": false,
  "pattern": "text_only"
}
```

---

## Generation Strategy

### Parallel Agent Generation

Three agents working in parallel:
1. **Agent 1:** Pattern 1 (text_only) → 75-100 pairs
2. **Agent 2:** Pattern 2 (tool_only) → 75-100 pairs
3. **Agent 3:** Pattern 3 (tool_text) → 75-100 pairs

Each agent:
- Hand-crafts examples following GENERATION_SPEC.md
- Ensures diverse tool coverage (15+ tools)
- Validates every 20 pairs
- Produces interleaved output

### Merge Strategy

After individual pattern files complete:
- Validate each file independently
- Optionally merge into `response_patterns_merged_kto_v1.0.jsonl`
- Or keep separate for targeted training

---

## Quality Requirements

### Must Pass
- ✅ 100% syngen validation pass rate
- ✅ Perfect interleaving (True/False/True/False)
- ✅ 50% True, 50% False label balance
- ✅ All 7 context fields complete
- ✅ sessionMemory never empty
- ✅ toolContext is STRING (not object)

### Quality Targets
- ✅ 15+ different tools per pattern
- ✅ All 5 agent families represented
- ✅ 8+ scenario domains covered
- ✅ Realistic user messages and tool results
- ✅ Clear behavioral contrast in pairs

---

## Usage

### Training
Use for KTO preference learning after SFT:
```bash
# Train with response patterns
python train_kto.py \
  --local-file ../../Datasets/behavior_datasets/response_patterns/text_only_pairs_v1.0.jsonl

# Or use merged dataset
python train_kto.py \
  --local-file ../../Datasets/behavior_datasets/response_patterns/response_patterns_merged_kto_v1.0.jsonl
```

### Validation
```bash
# Validate individual pattern files
python tools/validate_syngen.py \
  Datasets/behavior_datasets/response_patterns/text_only_pairs_v1.0.jsonl

# Check interleaving
python tools/check_interleaving.py \
  Datasets/behavior_datasets/response_patterns/text_only_pairs_v1.0.jsonl
```

---

## Expected Impact

After training on this dataset, models should demonstrate:

1. **Completion Recognition** (>85% accuracy)
   - Stops and summarizes when results ready
   - Doesn't continue tool chains unnecessarily
   - Presents findings clearly to user

2. **Clarification Behavior** (>80% accuracy)
   - Asks questions when ambiguous
   - Confirms before risky operations
   - Presents options when multiple choices exist

3. **Workflow Efficiency** (>90% accuracy)
   - Continues obvious workflows without interruption
   - Doesn't over-explain simple operations
   - Maintains automation when appropriate

4. **Context Provision** (>85% accuracy)
   - Explains complex operations before executing
   - Narrates multi-step processes appropriately
   - Balances transparency with efficiency

---

## Files

- `GENERATION_SPEC.md` - Complete specification for agents
- `README.md` - This file
- `text_only_pairs_v1.0.jsonl` - Pattern 1 dataset (in progress)
- `tool_only_pairs_v1.0.jsonl` - Pattern 2 dataset (in progress)
- `tool_text_pairs_v1.0.jsonl` - Pattern 3 dataset (in progress)
- `response_patterns_merged_kto_v1.0.jsonl` - Merged dataset (future)
- `coverage_report.md` - Tool/scenario coverage (future)
- `validation_report.md` - Quality metrics (future)

---

## Status

**Phase:** Generation
**Progress:** 0/450-600 examples complete

### Pattern 1: Text-Only
- [ ] 150-200 examples
- [ ] Validation passed
- [ ] Coverage verified

### Pattern 2: Tool-Only
- [ ] 150-200 examples
- [ ] Validation passed
- [ ] Coverage verified

### Pattern 3: Tool+Text
- [ ] 150-200 examples
- [ ] Validation passed
- [ ] Coverage verified

---

**Next Steps:**
1. Spawn 3 agents in parallel
2. Generate examples following GENERATION_SPEC.md
3. Validate each pattern file
4. Review quality samples
5. Merge if desired
6. Train with KTO

---

**Contact:** Generated via Claude Code parallel agent workflow
**Last Updated:** 2025-11-25
