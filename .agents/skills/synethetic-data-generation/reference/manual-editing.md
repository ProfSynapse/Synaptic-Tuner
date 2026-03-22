# Manual Dataset Editing Reference

For hand-crafting improvements to individual JSONL lines when automated improvement isn't sufficient.

---

## Scripts

Located at `.claude/skills/synethetic-data-generation/scripts/`

### Extract Lines for Review

```bash
./scripts/improve_dataset.sh <file> <start_line> [count]
```

**Example:** Extract 5 lines starting at line 100:
```bash
./scripts/improve_dataset.sh ../../Datasets/tools_v1.5.jsonl 100 5
```

### Replace a Line

```bash
./scripts/replace_lines.sh <file> <line_num> '<improved_json>'
```

Creates automatic `.backup` file before replacing.

**Example:**
```bash
./scripts/replace_lines.sh ../../Datasets/tools_v1.5.jsonl 100 '{"conversations":[...]}'
```

### Batch Workflow

```bash
# Extract batch
./scripts/improve_dataset.sh dataset.jsonl 1 10

# Review, hand-craft improvements, replace each
./scripts/replace_lines.sh dataset.jsonl 1 '<improved_line>'
./scripts/replace_lines.sh dataset.jsonl 2 '<improved_line>'
# ...

# Next batch
./scripts/improve_dataset.sh dataset.jsonl 11 10
```

---

## Quality Guidelines

### Goal — Make Specific and Actionable

- Bad: "Update content"
- Good: "Append 'New entry' to Content/log.md to maintain chronological record"

### Memory — Rich Context

- Bad: "User updating content"
- Good: "User maintaining activity log for Content Hub. Previously added entries tracking blog drafts."

### Requirements — Verification Checks (DISTINCT from plan)

Must check things NOT visible in system prompt:
- "Verify file isn't locked"
- "Check no active references"
- "Confirm write permissions"

### Plan — Execution Steps (DISTINCT from requirements)

Direct actions, not re-verification:
- "Execute append operation"
- "Confirm success"
- "Report result to user"

### Confidence — Risk-Calibrated

| Risk Level | Confidence Range | Examples |
|-----------|-----------------|----------|
| Safe | 0.85-0.95 | read, list, search, append |
| Moderate | 0.6-0.8 | update, batch operations |
| Risky | 0.3-0.5 | delete, replace, overwrite |

---

## Safety

- Automatic `.backup` files before replacement
- Line-by-line processing prevents bulk errors
- Always validate JSON with `jq` before replacing: `echo '<json>' | jq`
- Use `wc -l dataset.jsonl` to check total lines
- Work in small batches (10-20 lines)
