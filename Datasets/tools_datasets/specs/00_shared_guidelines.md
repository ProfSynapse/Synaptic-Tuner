# Shared Guidelines: SFT Dataset Enhancement

## Overview
We are adding two types of text-only examples to each manager dataset:
1. **Clarification examples** - Model asks questions instead of acting on vague requests
2. **Summary examples** - Model summarizes what happened after tool results

## IMPORTANT: Handcrafted Examples Only
- DO NOT use scripts to generate examples
- Manually read existing examples to understand patterns
- Craft each new example by hand
- Ensure variety in phrasing and scenarios

## Format Requirements

### System Message (required)
Every example MUST have a system message with session context:
```json
{
  "role": "system",
  "content": "<session_context>\nIMPORTANT: When using tools, include these values in your tool call parameters:\n\n- sessionId: \"session_1732500000000_abc123def\"\n- workspaceId: \"ws_1732500000000_ghi456jkl\" (current workspace)\n\nInclude these in the \"context\" parameter of your tool calls.\n</session_context>\n<available_workspaces>\nThe following workspaces are available in this vault:\n\n- Project Hub (id: \"ws_1732500000000_ghi456jkl\")\n  Description: Main project workspace\n  Root folder: Projects/\n\nUse memoryManager with loadWorkspace mode to get full workspace context.\n</available_workspaces>"
}
```

### ID Format
- sessionId: `session_\d{13}_[a-z0-9]{9}` (13-digit timestamp + 9 lowercase alphanumeric)
- workspaceId: `ws_\d{13}_[a-z0-9]{9}`
- Each example needs UNIQUE IDs

### Labels
- All new examples should have `"label": true`
- We may create `label: false` versions later for KTO

---

## Type 1: Clarification Examples

### When to Ask (not act)
- Request is vague about WHAT to target
- Request is destructive (delete, overwrite, replace)
- Request involves creating something without details
- Multiple valid interpretations exist

### Clarification Response Format
```json
{
  "role": "assistant",
  "content": "I'd like to clarify a few things:\n\n- [Question about specific param from schema]\n- [Question about scope/target]\n- [Question about confirmation if destructive]\n\nThis will help me [reason]."
}
```

### Question Sources (from tool schemas)
Use the tool's parameters to guide questions:
- `path` → "Which file/folder specifically?"
- `query` → "What keywords should I search for?"
- `paths` → "Which folders should I look in?"
- `recursive` → "Should I include all subfolders?"
- `name` → "What should I call it?"
- `description` → "What's its purpose?"

---

## Type 2: Summary Examples

### Conversation Structure
```json
{
  "conversations": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Result: {\"success\": true, \"deleted\": [...], ...}"},
    {"role": "assistant", "content": "# Heading\n\n**Action taken**\n- Detail 1\n- Detail 2\n\nAnything else?"}
  ]
}
```

### Summary Response Format
```
# [Contextual Heading]

**[Bold action description]**
- Bullet detail with file paths
- Another detail
- Counts/stats if relevant

[Optional follow-up question or next step offer]
```

### Contextual Headings (vary these)
| Action | Heading Options |
|--------|-----------------|
| Search | "Search Complete", "Found It", "Results Ready" |
| Delete | "Files Removed", "Cleanup Done", "Deleted" |
| Move | "Move Complete", "Files Relocated", "Moved" |
| Create | "Created", "Ready", "Set Up" |
| Update | "Updated", "Changes Saved", "Done" |

### Tone
- Professional, not overly excited
- No exclamation marks in headings
- Concise but informative
- Include file paths when relevant

---

## Validation
After creating examples, validate with:
```bash
python tools/validate_syngen.py Datasets/tools_datasets/{manager}/tools_v1.2.jsonl
```

## Output
- Read from: `tools_v1.1.jsonl`
- Write to: `tools_v1.2.jsonl` (original + new examples appended)
