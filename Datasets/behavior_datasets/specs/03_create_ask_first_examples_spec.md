# Spec: Create "Ask First" Examples

## Overview
Create new training examples that teach the model to ASK CLARIFYING QUESTIONS before taking action on vague or destructive requests.

## Problem
Current datasets (2,288 examples) always teach the model to take action:
- `label=true`: Search/gather information (good, but still taking action)
- `label=false`: Execute immediately (bad)

**Missing behavior**: Sometimes the correct response is to ask a question WITHOUT calling any tools.

## Required New Examples

### Target: 50-100 new pairs for each scenario type

### Scenario Types

#### 1. Ambiguous Destructive Requests (Priority: HIGH)
User request is vague AND involves deletion/modification:

```json
{
  "conversations": [
    {"role": "system", "content": "<session_context>...</session_context>"},
    {"role": "user", "content": "Delete the old files"},
    {"role": "assistant", "content": "I'd like to clarify before deleting anything:\n\n1. Which files are you referring to as \"old\"? \n2. Are these in a specific folder?\n3. How old should a file be to qualify for deletion?\n\nThis will help me make sure I delete the right files and don't accidentally remove anything important."}
  ],
  "label": true,
  "behavior": "intellectual_humility",
  "pattern": "ask_first"
}
```

Corresponding negative:
```json
{
  "conversations": [
    {"role": "system", "content": "<session_context>...</session_context>"},
    {"role": "user", "content": "Delete the old files"},
    {"role": "assistant", "content": null, "tool_calls": [{"id": "...", "type": "function", "function": {"name": "vaultManager_deleteFolder", "arguments": "..."}}]}
  ],
  "label": false,
  "behavior": "intellectual_humility",
  "pattern": "ask_first"
}
```

#### 2. Missing Critical Information
Requests that can't be completed without more info:

- "Move my notes to the archive" (which notes? which archive?)
- "Update the config" (which config? what value?)
- "Clean up the project" (what counts as cleanup?)

#### 3. Potentially Destructive Operations
Even if specific, model should confirm:

- "Delete everything in the Projects folder"
- "Replace all occurrences of X with Y"
- "Remove the backup files"

#### 4. Preference Questions
Questions where there's no single right answer:

- "How should I organize my notes?"
- "What's the best folder structure?"
- "Should I use tags or folders?"

**Note**: For some of these, `executePrompt` might also be valid. Create both patterns.

## Example Templates

### Template 1: Request Clarification
```
I'd like to clarify a few things before proceeding:

1. [Specific question about scope/target]
2. [Question about criteria]
3. [Question about consequences/confirmation]

This will help me [reason for asking].
```

### Template 2: Confirm Destructive Action
```
Just to confirm before I proceed - you want me to [restate action]. This will:
- [Consequence 1]
- [Consequence 2]

Would you like me to proceed, or would you prefer I [alternative suggestion]?
```

### Template 3: Seek Preferences
```
There are a few approaches to [task]:

1. [Option A] - [pros/cons]
2. [Option B] - [pros/cons]

Which approach would work best for your workflow?
```

## Question Detection Patterns
The response MUST contain at least one of:
- Direct question ending in `?`
- "Would you like..."
- "Do you want..."
- "Should I..."
- "Can you clarify..."
- "Could you specify..."
- "Which..."
- "What..."

## System Context Requirements
Each example MUST have:
1. System message with `<session_context>` and `<available_workspaces>`
2. Valid sessionId and workspaceId
3. IDs should be unique per example (don't reuse)

## ID Generation
Generate new IDs for each example:
```python
import time
import random
import string

def generate_id():
    timestamp = int(time.time() * 1000)  # 13 digit timestamp
    chars = ''.join(random.choices(string.ascii_lowercase + string.digits, k=9))
    return timestamp, chars

session_id = f"session_{timestamp}_{chars}"
workspace_id = f"ws_{timestamp}_{new_chars}"
```

## Request Categories (suggested prompts)

### Deletion Requests (20-30 pairs)
- "Delete the old files"
- "Remove the duplicates"
- "Clean up my project"
- "Get rid of the temp files"
- "Delete the notes I don't need"
- "Remove outdated content"
- "Clear out the archive"

### Move/Reorganize Requests (15-20 pairs)
- "Move my notes somewhere better"
- "Reorganize the project files"
- "Put these in the right folder"
- "Archive the completed stuff"
- "Move everything to the new structure"

### Modification Requests (15-20 pairs)
- "Update the config"
- "Change the settings"
- "Fix the formatting in my notes"
- "Update the links"
- "Rename the files"

### Strategy/Advice Requests (10-15 pairs)
- "How should I organize this?"
- "What's the best structure?"
- "How do I manage 200 notes?"
- "What tagging system should I use?"

## Output Location
- New file: `Datasets/behavior_datasets/ask_first/pairs_v1.0.jsonl`
- Create directory if needed
- Minimum 100 total examples (50 pairs)

## Validation
```bash
python tools/validate_syngen.py Datasets/behavior_datasets/ask_first/pairs_v1.0.jsonl
```

Expected:
- 0 errors
- All `label=true` examples have text-only responses with questions
- All `label=false` examples have tool_calls with proper IDs
