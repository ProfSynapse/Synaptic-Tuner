# Dataset Conversion Plan: Strategic Thought Trace

## Objective
Convert the existing dataset (`Datasets/tools_sft_reasoning_v1.5.jsonl`) from the legacy thinking format to the new "Strategic Thought Trace" schema. This will improve the model's ability to verify actions and plan strategically.

## Input vs. Output

### Input (Legacy)
```json
<thinking>
{
  "sessionDescription": "...",
  "sessionMemory": "...",
  "toolContext": "...",
  "primaryGoal": "...",
  "subgoal": "..."
}
</thinking>
```

### Output (Strategic Thought Trace)
```json
<thinking>
{
  "goal": "...",
  "memory": "...",
  "requirements": ["..."],
  "assessment": { "complex": boolean, "risky": boolean },
  "confidence": float,
  "plan": ["..."]
}
</thinking>
```

## Conversion Strategy

Since the new schema requires generating new information (specifically `requirements`, `assessment`, and `plan`) that does not exist in the old format, we cannot use simple programmatic mapping. We must use an LLM (Teacher Model) to re-reason the examples.

### 1. The Conversion Script (`scripts/convert_dataset.py`)
We will create a Python script that processes the dataset line by line.

**Workflow:**
1.  **Read** `Datasets/tools_sft_reasoning_v1.5.jsonl`.
2.  **Iterate** through each conversation.
3.  **Extract Context:**
    *   `User Prompt`: The user's request.
    *   `Old Thinking`: The legacy JSON block.
    *   `Tool Call`: The actual tool the assistant called (ground truth).
4.  **Generate New Thinking (LLM Call):**
    *   Send the context to a high-quality model (e.g., GPT-4o, Claude 3.5 Sonnet).
    *   **Prompt:** "Analyze this user request and the tool that was called. The old thinking block is provided for context. Generate a new 'Strategic Thought Trace' JSON object that leads logically to this tool call. Ensure 'risky' is correctly assessed."
5.  **Update System Prompt:**
    *   Replace the verbose system prompt (which contains the old schema instructions) with a minimal one.
    *   *Old:* "...Output your reasoning in <thinking> tags..."
    *   *New:* "...(Keep session context)..." (Remove explicit schema instructions).
6.  **Save** to `Datasets/tools_sft_reasoning_v2.jsonl`.

### 2. System Prompt Refinement
The user requested that the training data *not* rely on a complex system prompt to trigger the behavior. The model should learn the pattern from the examples themselves.

**Action:** The conversion script will strip the specific `<thinking>` instructions from the `system` message in the dataset, leaving only the essential `session_context` (sessionId, workspaceId).

### 3. Validation
*   **Schema Check:** Ensure every generated JSON conforms strictly to the spec (types, required fields).
*   **Logic Check:** Randomly sample 50 examples to ensure the `plan` actually aligns with the `tool_calls`.

## Execution Steps
1.  **Develop Script:** Write `scripts/convert_dataset.py` (requires an API key for the Teacher Model).
2.  **Dry Run:** Process 10 examples and inspect the output.
3.  **Full Run:** Process the entire dataset (~6.8k lines).
4.  **Verify:** Run `Tools/validate_syngen.py` (or a new validator) on the new dataset.
