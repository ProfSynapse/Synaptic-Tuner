# Strategic Thought Trace Specification

## Overview
The Strategic Thought Trace is a structured reasoning schema designed to force the model to pause, verify, and plan before executing actions. This is particularly critical for "risky" operations (deletions, overwrites, complex modifications) where the model has previously shown a tendency to act recklessly.

## Schema Structure
The model must output a JSON object wrapped in `<thinking>` tags at the start of every turn where it intends to take action or provide a complex answer.

```json
<thinking>
{
  "goal": "The immediate objective of this specific turn",
  "memory": "Rolling summary of the conversation state (mini-compaction)",
  "requirements": [
    "List of conditions/info needed to safely achieve the goal",
    "e.g., 'Verify file existence', 'User confirmation'"
  ],
  "assessment": {
    "complex": boolean,  // TRUE = Delegate (use agentManager) or break down
    "risky": boolean     // TRUE = Verify (Read/List) before acting
  },
  "confidence": 0.0 to 1.0,
  "plan": [
    "Step 1",
    "Step 2",
    "...",
    "Step N"
  ]
}
</thinking>
```

## Field Definitions

### 1. goal (String)
The specific, immediate thing the model is trying to accomplish in *this* turn.
*   *Bad:* "Help the user."
*   *Good:* "Delete the 'temp' folder after verifying it contains no important files."

### 2. memory (String)
A concise summary of relevant context from previous turns. This acts as a "working memory" to prevent context loss in long context windows.
*   *Example:* "User asked to refactor `auth.py`. I have listed the files but haven't read `auth.py` yet. User emphasized not breaking the login flow."

### 3. requirements (Array of Strings)
A checklist of prerequisites that must be met *before* the plan can be executed. This is the primary mechanism for forcing verification.
*   *Example:* `["Confirm 'data.csv' exists", "Check if 'data.csv' is empty", "Verify user has backup"]`

### 4. assessment (Object)
*   **complex (Boolean):** Indicates if the task requires multiple steps, sub-agents, or complex reasoning. If `true`, the plan should reflect a breakdown or delegation.
*   **risky (Boolean):** The critical safety flag.
    *   **TRUE if:** Deleting files, overwriting files, making network calls, executing code, modifying system settings.
    *   **FALSE if:** Reading files, listing directories, searching, answering questions.
    *   **Constraint:** If `risky` is `true`, the `plan` MUST include a verification step (e.g., `read_file`, `list_dir`) as the first action, unless already verified.

### 5. confidence (Float)
A score (0.0 - 1.0) indicating how sure the model is that it can complete the task with the current information.
*   *Low (< 0.7):* Should probably ask for clarification or read more context.
*   *High (> 0.9):* Ready to execute.

### 6. plan (Array of Strings)
The step-by-step execution path.
*   Must be concrete and actionable.
*   Must align with the `requirements`.
*   If `risky` is true, Step 1 is almost always "Verify...".

## Example

**User:** "Delete the old logs folder."

**Assistant:**
```json
<thinking>
{
  "goal": "Delete the 'logs' directory",
  "memory": "User is cleaning up the project workspace. I previously listed the root directory. User has now specifically requested to delete the 'old logs folder'. I have not yet verified the folder's location or contents.",
  "requirements": [
    "Identify the specific path of the 'logs' folder",
    "Verify the folder contents to ensure it doesn't contain active data"
  ],
  "assessment": {
    "complex": false,
    "risky": true
  },
  "confidence": 0.8,
  "plan": [
    "List the current directory to find the 'logs' folder",
    "List the contents of the 'logs' folder to verify dates/content",
    "Delete the folder if confirmed safe"
  ]
}
</thinking>
```
