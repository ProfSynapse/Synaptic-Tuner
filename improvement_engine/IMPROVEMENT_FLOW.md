# Improvement Engine Flow

## Sequential Improvement Process

The improvement engine processes scopes **sequentially** within each iteration:

### Flow Diagram

```
Iteration 1:
  ├─ Judge ALL scopes (system_prompt, thinking, response)
  ├─ Detect failed scopes
  └─ Improve each failed scope IN ORDER:
      1. System Prompt (if failed)
      2. Thinking (if failed) ← Uses improved system prompt
      3. Response (if failed) ← Uses improved system + thinking
```

### Code Reference

In `engine.py` line 243:
```python
for scope in failed_scopes:
    improved_example = self._improve_scope_with_logging(improved, ...)
    if improved_example != improved:
        improved = improved_example  # ← Updates for next scope
```

**Key Point**: Each scope improvement uses the cumulative improvements from previous scopes in the same iteration.

### Example

If all three scopes fail:
1. **System prompt improves** → New system prompt created
2. **Thinking improves** → Uses new system prompt as context
3. **Response improves** → Uses new system prompt + new thinking as context

## Response Scope: Risk-Aware Tool Call Handling

### Problem (Before Fix)

The improver would often **replace tool calls with text** because:
- Original tool calls weren't shown in the prompt
- No explicit format instructions
- LLM defaulted to text explanations

### Solution (After Fix)

Updated `tool_alignment.yaml` improver_prompt with **risk-aware logic**:

1. **Show original tool calls**:
   ```
   **Current Tool Calls:**
   ```json
   {original_tool_calls}
   ```
   ```

2. **Add critical instructions**:
   ```
   **CRITICAL INSTRUCTIONS:**
   - If the thinking block describes a tool operation, you MUST include tool calls
   - DO NOT replace tool calls with text explanations
   - Keep tool calls in exact format: ```tool { "name": "...", "arguments": {...} } ```
   - Fix the tool name and parameters, but preserve the tool call structure
   ```

3. **Add risk-aware decision logic**:
   ```
   **Check the thinking block's assessment:**
   - If `assessment.risky: true` OR `confidence < 0.5`:
     → REPLACE tool calls with text asking user for clarification

   - If `assessment.risky: false` AND `confidence >= 0.5`:
     → PRESERVE and FIX tool calls (DO NOT replace with text)
   ```

4. **Format tool calls as JSON** in `response_handler.py`:
   ```python
   tool_calls_json = json.dumps(original_tool_calls, indent=2) if original_tool_calls else "[]"
   ```

### Decision Logic

The improver follows this decision tree:

```
Read thinking block assessment
├─ Is risky: true? → Replace with clarifying text
├─ Is confidence < 0.5? → Replace with clarifying text
└─ Otherwise → Preserve and fix tool calls
```

### Examples

**High Risk Example**:
```json
{
  "assessment": {"risky": true},
  "confidence": 0.7
}
```
→ **Output**: Text asking for confirmation ("This operation will delete files. Should I proceed?")

**Low Confidence Example**:
```json
{
  "assessment": {"risky": false},
  "confidence": 0.3
}
```
→ **Output**: Text asking for clarification ("I'm uncertain about which file to modify. Could you specify?")

**Safe & Confident Example**:
```json
{
  "assessment": {"risky": false},
  "confidence": 0.8
}
```
→ **Output**: Fixed tool calls preserved

### Result

The improver now:
- ✅ Sees the original tool call structure
- ✅ Checks risk and confidence from thinking block
- ✅ **Replaces with text** when risky or uncertain (safety first)
- ✅ **Preserves and fixes** tool calls when safe and confident
- ✅ Maintains proper JSON format when using tools

## Validation Flow

Tool call validation runs BEFORE improvement:

```
1. ValidationService._validate_tool_calls()
   ├─ Checks tool exists in schema
   ├─ Checks required params present
   └─ Returns (is_valid, errors)

2. Judge sees validation results:
   "❌ tool_alignment_tool_calls_schema: FAILED
    - Tool 'agentManager_executPrompt' not found in schema"

3. Judge provides specific feedback

4. Improver fixes based on feedback + original tool calls
```

## Best Practices

### For Response Improvements

1. **Always show original tool calls** in improver prompts
2. **Be explicit about format** - show example structure
3. **Emphasize preservation** - "DO NOT replace with text"
4. **Validate after improvement** - check tools still present

### For Multi-Scope Improvements

1. **Order matters** - system → thinking → response
2. **Use cumulative context** - each scope sees previous improvements
3. **Validate progressively** - catch issues early in the chain
4. **Test full cycles** - ensure all scopes work together

## Configuration

### Enable Tool Call Validation

In rubric YAML:
```yaml
validation:
  tool_calls:
    enabled: true
```

### Scope Processing Order

In `config/scope_config.yaml`:
```yaml
scope_processing_order:
  - system_prompt
  - thinking
  - response
```

This order ensures proper context flow.
