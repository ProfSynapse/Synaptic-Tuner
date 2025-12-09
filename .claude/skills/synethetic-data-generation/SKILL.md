---
name: dataset-improvement
description: Extract, review, and replace lines in JSONL dataset files for hand-crafted quality improvements. Use when improving synthetic training data, fixing thinking blocks, or manually editing dataset examples.
allowed-tools: Read, Bash, Write
---

# Dataset Improvement Skill

Manually improve quality of thinking blocks in synthetic training dataset JSONL files through systematic extraction and replacement.

## Instructions

This Skill provides bash scripts to:
1. Extract specific lines from JSONL files for manual review
2. Replace lines with hand-crafted improvements
3. Manage backups automatically

### Workflow

1. **Extract lines for review:**
   ```bash
   cd .claude/skills/synethetic-data-generation
   ./scripts/improve_dataset.sh <file> <start_line> [count]
   ```

2. **Review extracted examples** - Identify quality issues:
   - Requirements duplicating plan?
   - Weak or generic memory?
   - Incorrect confidence calibration?
   - Missing contextual details?

3. **Hand-craft improvements** following quality guidelines (see below)

4. **Replace line with improved version:**
   ```bash
   ./scripts/replace_lines.sh <file> <line_num> '<improved_json_content>'
   ```

5. **Repeat** for additional lines in batch

### Quality Guidelines

**Goal:** Make specific and actionable
- ❌ "Update content"
- ✅ "Append 'New entry' to Content/log.md to maintain chronological record"

**Memory:** Add rich context (WHY, WHAT BEFORE, broader situation)
- ❌ "User updating content"
- ✅ "User maintaining activity log for Content Hub. Previously added 8 entries this week tracking blog post drafts and milestones. Log serves as project timeline for content team."

**Requirements:** Verification checks (DISTINCT from plan)
- Must focus on what to CHECK before acting
- Example: `["Verify file exists", "Confirm write permissions", "Check file isn't locked"]`

**Plan:** Execution steps (DISTINCT from requirements)
- Must focus on what to DO
- Example: `["Verify file exists", "Execute append", "Confirm success", "Report to user"]`

**Confidence:** Risk-calibrated scoring
- 0.3-0.5: Risky (delete, replace, overwrite)
- 0.6-0.8: Moderate (update, batch operations)
- 0.85-0.95: Safe (read, list, search, append)

## Examples

### Example 1: Extract 5 lines from contentManager dataset

```bash
./scripts/improve_dataset.sh ../../Datasets/tools_datasets/thinking/contentManager/tools_v1.5.jsonl 100 5
```

Output shows lines 100-104 for review.

### Example 2: Replace line 100 with improved version

```bash
./scripts/replace_lines.sh ../../Datasets/tools_datasets/thinking/contentManager/tools_v1.5.jsonl 100 '{"conversations":[...]}'
```

Creates backup at `tools_v1.5.jsonl.backup` and replaces line 100.

### Example 3: Process dataset in batches

```bash
# Extract batch of 10
./scripts/improve_dataset.sh dataset.jsonl 1 10

# Review and improve each line manually

# Replace each improved line
./scripts/replace_lines.sh dataset.jsonl 1 '<improved_line_1>'
./scripts/replace_lines.sh dataset.jsonl 2 '<improved_line_2>'
# ... continue for all 10

# Move to next batch
./scripts/improve_dataset.sh dataset.jsonl 11 10
```

## Best Practices

1. **Work in small batches** - Process 10-20 examples at a time
2. **Track progress** - Keep notes on which lines completed
3. **Validate JSON** - Use `jq` to verify syntax: `echo '<json>' | jq`
4. **Review backups** - Check `.backup` files before continuing
5. **Use line numbers** - Files use 1-indexed line numbers

## Safety Features

- Automatic `.backup` files created before replacement
- Line-by-line processing prevents bulk errors
- Manual review required at each step
- JSON validation recommended before replacement

## Related Tools

- `jq` - Parse and validate JSON
- `diff` - Compare original vs improved
- `wc -l` - Count total lines in dataset
- `sed -n 'Np'` - Preview single line N
