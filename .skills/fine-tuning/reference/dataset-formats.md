# Dataset Formats Reference

Dataset format requirements for each training method.

---

## SFT Dataset Format

**Positive examples only.** No labels needed.

```jsonl
{
  "conversations": [
    {"role": "user", "content": "Delete the file test.md"},
    {"role": "assistant", "content": "tool_call: vaultManager_deleteNote\narguments: {\"path\": \"test.md\"}\n\nResult: {\"success\": true}\n\nDeleted test.md successfully."}
  ]
}
```

**Key rules:**
- Starts with `user` role (NO system message in SFT datasets)
- Tool calls embedded in assistant response text
- `label` field is ignored (all examples treated as positive)
- Single-turn conversations preferred

---

## KTO Dataset Format

**Interleaved True/False examples.** Label is required.

```jsonl
{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "good response"}], "label": true}
{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "bad response"}], "label": false}
```

**Key rules:**
- `label: true` = desirable response
- `label: false` = undesirable response
- Must have BOTH True and False examples
- Data loader auto-interleaves and balances
- Imbalanced datasets get truncated to match minority count

**Internal conversion (automatic):**
```
conversations format → prompt/completion/label format (for TRL)
```

---

## GRPO Dataset Format

**Prompts with ground truth for reward scoring.**

```jsonl
{
  "prompt": [
    {"role": "system", "content": "You are a helpful assistant..."},
    {"role": "user", "content": "Open the file notes.md"}
  ],
  "ground_truth_tool": "vaultManager_openNote",
  "ground_truth_args_json": "{\"context\": {\"sessionId\": \"abc\"}, \"calls\": [{\"agent\": \"vaultManager\", \"tool\": \"openNote\", \"params\": {\"path\": \"notes.md\"}}]}"
}
```

**Key rules:**
- `prompt` is a list of message objects (can include system)
- `ground_truth_tool` — expected tool name
- `ground_truth_args_json` — expected arguments (useTools format)
- Ground truth used by reward rubrics for scoring

---

## Tool Call Format (in Datasets)

Tool calls can appear in several formats:

### Text-embedded (SFT/KTO)
```
tool_call: vaultManager_deleteNote
arguments: {"path": "test.md"}

Result: {"success": true}

Response text here.
```

### OpenAI format (in tool_calls field)
```json
{
  "tool_calls": [
    {
      "function": {
        "name": "useTools",
        "arguments": "{\"context\": {...}, \"calls\": [...]}"
      }
    }
  ]
}
```

### useTools wrapper (standard)
```json
{
  "name": "useTools",
  "arguments": {
    "context": {
      "sessionId": "abc",
      "workspaceId": "ws_123",
      "memory": "...",
      "goal": "..."
    },
    "calls": [
      {
        "agent": "vaultManager",
        "tool": "openNote",
        "params": {"path": "notes.md"}
      }
    ]
  }
}
```

---

## Where Datasets Live

```
Datasets/
├── behavior_datasets/          # Behavioral training
│   ├── thinking/               # With thinking blocks
│   └── non-thinking/           # Without thinking
├── tools_datasets/             # Tool-specific
│   ├── thinking/               # With thinking blocks
│   │   ├── agentManager/
│   │   ├── contentManager/
│   │   ├── storageManager/
│   │   └── ...
│   └── non-thinking/
├── synthchat/                  # SynthChat generated
└── *.jsonl                     # Combined/final datasets
```

---

## Creating KTO Datasets from Evaluator Results

Failed behavior tests make great KTO negatives:
1. Run evaluation: `python -m Evaluator.cli --backend lmstudio --model MODEL`
2. PASS results → `label: true` (positive examples)
3. WARN/FAIL results → `label: false` (negative examples)
4. Combine into interleaved dataset

---

## Combining Datasets

Use the combine script from the synthetic-data-generation skill:

```bash
.claude/skills/synethetic-data-generation/scripts/combine_datasets.sh \
  -o Datasets/combined_training.jsonl \
  Datasets/tools_datasets/thinking/
```

This shuffles and adds a `source` field from folder names.

---

## Validation

Always validate before training:

```bash
# Structural validation
python .skills/synethetic-data-generation/scripts/validate_syngen.py Datasets/my_dataset.jsonl

# Check KTO label balance
python -c "
import json
labels = [json.loads(l).get('label') for l in open('data.jsonl')]
print(f'True: {labels.count(True)}, False: {labels.count(False)}')
"
```
