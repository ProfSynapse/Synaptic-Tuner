# Destructive Action Confirmation Generator

## Overview

This system generates synthetic training examples teaching the model to:
1. **Execute high-risk operations AFTER user confirmation**
2. **Show HIGH RISK + HIGH CONFIDENCE (correct pattern)**
3. **Use correct tool schemas**

### Key Training Signal

```
Risk ↑ + User Permission ✓ → Confidence ↑
```

**HIGH RISK** operations CAN and SHOULD have **HIGH CONFIDENCE** when user explicitly authorizes them.

## The Problem We're Solving

Without these examples, models might learn incorrect patterns:
- ❌ High risk always means low confidence
- ❌ Never execute destructive operations
- ❌ Ask for confirmation even when already given

With these examples, models learn:
- ✅ User permission resolves uncertainty
- ✅ High risk + authorization = High confidence
- ✅ Execute correctly with proper tool schemas

## Generated Example Structure

### Single-Turn Format

```
SYSTEM: <workspace description with fake files/folders>
USER: "I give you permission to delete [specific item]"
ASSISTANT: <thinking: HIGH RISK + HIGH CONFIDENCE> + <tool_call> + <completion>
```

### Key Components

1. **System Prompt** - Fake workspace with realistic file structure
2. **User Request** - Explicit permission to delete specific item
3. **Thinking Block** - Shows:
   - HIGH RISK assessment
   - User permission verification
   - HIGH CONFIDENCE (8-10/10)
   - Requirements checklist
   - Execution plan
4. **Tool Call** - Correct schema with full context object
5. **Completion** - Confirms successful execution

## Configuration Files

### `configs/destructive_single_turn.yaml`

Defines:
- **Destructive tools** for each agent (vaultManager, contentManager, agentManager)
- **Permission patterns** (variations of how users give permission)
- **Thinking template** (HIGH RISK + HIGH CONFIDENCE structure)
- **Fake workspace structures** (realistic file/folder examples)
- **Risk level descriptions**
- **Confidence levels** based on permission type

## Files Created

1. **`configs/destructive_single_turn.yaml`** - Configuration
2. **`single_turn_destructive_gen.py`** - Generator implementation
3. **`test_destructive_gen.py`** - Original multi-turn test (deprecated)
4. **`destructive_confirmation_generator.py`** - Original multi-turn (deprecated)
5. **`run_destructive_gen.py`** - Runner for multi-turn (deprecated)

## Usage

### Quick Test (3 Examples)

```bash
cd synth_chat
python single_turn_destructive_gen.py --test
```

### Generate Examples

```bash
# Generate 50 examples
python single_turn_destructive_gen.py --count 50 --output my_examples.jsonl

# Generate 100 examples
python single_turn_destructive_gen.py -n 100 -o destructive_training.jsonl
```

### Example Output

```json
{
  "conversations": [
    {
      "role": "system",
      "content": "<session_context>...<workspace_structure>Projects/Archive/Q1-2023/..."
    },
    {
      "role": "user",
      "content": "I give you permission to delete Projects/Archive/Q1-2023/"
    },
    {
      "role": "assistant",
      "content": "<thinking>\nRisk Level: HIGH\nConfidence: 9/10 (HIGH)\nUser explicitly authorized: \"I give you permission to delete...\"\n</thinking>\n\n<tool_call>{\"name\": \"vaultManager_deleteFolder\", ...}</tool_call>"
    }
  ]
}
```

## Destructive Tools Covered

### vaultManager
- **deleteFolder** (HIGH risk) - Delete folders with contents
- **deleteNote** (HIGH risk) - Delete specific files

### contentManager
- **deleteContent** (MEDIUM risk) - Delete content from within files

### agentManager
- **deleteAgent** (MEDIUM risk) - Delete custom agents

## Thinking Block Pattern

The thinking block follows this structure:

```
<thinking>
Understanding the Request:
- User requests deletion of: [target]
- Target: [specific path/id]
- Operation: [tool_name]

Risk Assessment:
- Risk Level: HIGH/MEDIUM
- Impact: [description of consequences]
- Irreversibility: This action cannot be undone
- [Additional risk factors]

Requirements Check:
✓ Specific target identified
✓ User explicitly authorized deletion: "[permission phrase]"
✓ Target exists in workspace
✓ Tool available
✓ User understands irreversibility

Confidence Assessment:
- Confidence Level: [8-10]/10 (HIGH)
- User provided explicit permission
- Target is clearly specified
- Operation is straightforward
- All requirements satisfied

Plan:
1. Verify target path/id
2. Execute [tool_name] with proper context
3. Confirm deletion completion

Proceeding with deletion as authorized by user.
</thinking>
```

## Tool Schema Validation

All tool calls use correct schemas from `Tools/tool_schemas.json`:

### vaultManager_deleteFolder

```json
{
  "name": "vaultManager_deleteFolder",
  "arguments": {
    "context": {
      "sessionId": "session_...",
      "workspaceId": "ws_...",
      "sessionDescription": "...",
      "sessionMemory": "...",
      "toolContext": "...",
      "primaryGoal": "...",
      "subgoal": "..."
    },
    "path": "Projects/Archive/Q1-2023/",
    "recursive": true
  }
}
```

## Permission Patterns

The generator uses varied permission phrases:

### Explicit Permission (40% weight)
- "I give you permission to delete X"
- "You have my permission to delete X"
- "I authorize you to delete X"

### Confirmation First (30% weight)
- "Yes, delete X"
- "Confirmed, delete X"
- "Go ahead and delete X"

### Specific Instruction (30% weight)
- "Delete X for me"
- "Remove X please"
- "Get rid of X"

## Fake Workspace Examples

The generator creates realistic workspace structures:

### Project Archive Structure
```
Projects/
  Active/
    Q4-Goals.md
  Archive/
    Q1-2023/ (contains 15 files)
    Q2-2023/ (contains 23 files)
```

### General Cleanup Structure
```
Notes/
  Daily/
    2024-12-08.md
  Temp/
    Drafts/ (contains 5 files)
    Test-Notes/ (contains 3 files)
```

### Agent Management Structure
```
Settings/Agents/
  Active:
    - Daily Summarizer (agt_daily_summary)
  Experimental:
    - Test Analytics (agt_test_analytics)
```

## Adding to Existing Datasets

To add generated examples to agent datasets:

```bash
# Generate and append to datasets
python single_turn_destructive_gen.py --count 50 --add-to-datasets

# This will:
# 1. Generate 50 examples
# 2. Group by agent (vaultManager, contentManager, agentManager)
# 3. Append to latest dataset files in Datasets/tools_datasets/thinking/
```

## Integration with Existing System

This generator is **standalone** but can be integrated with the main `synth_chat` system:

```python
from single_turn_destructive_gen import SingleTurnDestructiveGenerator

# Initialize
generator = SingleTurnDestructiveGenerator()

# Generate examples
examples = generator.generate_batch(count=100)

# Each example has:
# - System prompt with fake workspace
# - User request with explicit permission
# - Assistant response with thinking + tool call
```

## Training Impact

After training on these examples, the model learns:

### ✅ Correct Behaviors
1. **Execute after authorization** - Don't hesitate when user gives permission
2. **High confidence from permission** - User authorization increases confidence
3. **Proper risk assessment** - Acknowledge high risk while proceeding
4. **Correct tool schemas** - Use proper context objects and parameters

### ❌ Prevents Wrong Behaviors
1. Never executing destructive operations
2. Always having low confidence on high-risk operations
3. Re-asking for confirmation when already given
4. Using incorrect tool schemas

## Distribution by Tool

The generator creates examples with these weights:
- **vaultManager_deleteFolder**: 35% (most common)
- **vaultManager_deleteNote**: 30%
- **contentManager_deleteContent**: 20%
- **agentManager_deleteAgent**: 15%

This distribution reflects real-world usage patterns.

## Validation

Each generated example validates:

### Thinking Block
- ✓ Contains "Risk Level: HIGH" or "MEDIUM"
- ✓ Contains "Confidence Level: [8-10]/10" or "HIGH"
- ✓ References user permission/authorization
- ✓ Has "Requirements Check" section
- ✓ Has "Plan:" section

### Tool Call
- ✓ Correct tool name matching schema
- ✓ Complete context object (all 7 fields)
- ✓ Specific target path/id
- ✓ Proper parameter types

### Overall Structure
- ✓ System prompt with fake workspace
- ✓ User gives explicit permission
- ✓ Response has thinking + tool call + completion

## Future Enhancements

Potential improvements:
1. **LLM Integration** - Use actual LLM to generate more varied responses
2. **Batch Operations** - Add examples with multiple deletions
3. **Conditional Deletions** - Add filters and conditions
4. **Error Scenarios** - Add examples where deletion fails
5. **Undo Patterns** - Add examples explaining no undo available

## Summary Statistics

From our analysis of existing datasets:

### Current State (Before)
- **Very few** destructive tool examples
- Most datasets: 0 destructive examples
- contentManager/tools_v1.5.jsonl: 6 total (mixed agents)
- vaultManager/tools_v1.5.jsonl: 2 move operations (not deletes!)

### With This Generator
- Can add **50-100 examples per agent**
- Covers all destructive tools systematically
- Ensures proper HIGH RISK + HIGH CONFIDENCE pattern
- Uses correct tool schemas

## Quick Reference

```bash
# Test
python single_turn_destructive_gen.py --test

# Generate 50
python single_turn_destructive_gen.py -n 50 -o output.jsonl

# Add to datasets
python single_turn_destructive_gen.py -n 50 --add-to-datasets

# View example
head -1 output.jsonl | python -m json.tool
```

## Contact / Issues

For questions or issues:
1. Check the YAML config: `configs/destructive_single_turn.yaml`
2. Review tool schemas: `Tools/tool_schemas.json`
3. Test with: `python single_turn_destructive_gen.py --test`

---

**Generated**: 2024-12-09
**Author**: Claude Code Assistant
**Purpose**: Teaching models correct high-risk + high-confidence patterns for destructive operations
