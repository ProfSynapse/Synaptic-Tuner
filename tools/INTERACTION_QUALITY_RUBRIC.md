# Interaction Quality Scoring Rubric

This rubric guides consistent quality assessment of synthetic training examples for the Claudesidian-MCP toolset. Score each dimension on a **1-5 scale** where 1 is poor and 5 is excellent.

## Scoring Dimensions

### 1. sessionMemory_quality

**What it measures**: How well the `sessionMemory` field captures prior context and state.

**1 - Poor**
- Empty string `""`
- Empty array `[]`
- Placeholder text like "None", "N/A", "User working"
- No actionable context

**2 - Below Average**
- Very generic: "User organizing files" with no specifics
- Extremely short (< 20 chars)
- States obvious facts without detail

**3 - Average**
- Generic but relevant: "User organizing API documentation"
- Some context but lacks concrete details
- Length 20-50 chars

**4 - Good**
- Specific actions: "Created project folder 'Q4-Reports', listed 12 files"
- References prior tool calls: "Searched 847 notes in archive/"
- Concrete details (numbers, paths, results)
- Length 50-100 chars

**5 - Excellent**
- Rich context with multiple prior actions: "Listed Research/ (23 files), searched 'quantum' (5 matches), now reading Research/Papers/Bell-Theorem.md"
- References specific tool calls and results: "Mapped Journals/Dreams/Dream Journal.md via vaultLibrarian_searchDirectory"
- Builds narrative continuity across conversation
- Length 100+ chars with high information density

---

### 2. toolContext_quality

**What it measures**: How well the `toolContext` field explains WHY this specific tool is being called now.

**1 - Poor**
- Empty or missing
- Generic: "Using tool", "Performing action"
- Just restates the tool name

**2 - Below Average**
- Slightly more specific than tool name: "Searching files"
- Describes WHAT but not WHY
- No workflow context

**3 - Average**
- Clear purpose: "Finding the dream journal"
- Explains immediate goal
- Missing workflow reasoning (why this step now?)

**4 - Good**
- Explains reasoning: "Confirming path before appending content"
- Shows decision-making: "Need to verify folder exists first"
- Connects to larger workflow

**5 - Excellent**
- Rich reasoning: "Append dated dream entry after confirming path via search and checking current content"
- Explains alternatives considered: "Using append instead of replace to preserve existing entries"
- Shows strategic thinking about tool choice

---

### 3. goal_coherence

**What it measures**: How well `primaryGoal` and `subgoal` form a clear, logical hierarchy.

**1 - Poor**
- Missing one or both goals
- Contradictory goals
- Both are identical
- Completely unrelated to user prompt

**2 - Below Average**
- Goals overlap significantly (90%+ similar)
- Unclear hierarchy (which is broader?)
- Vague: "Work on project" / "Do task"

**3 - Average**
- Clear distinction but generic
- Proper hierarchy: "Update documentation" → "Edit README.md"
- Accurately reflects user request

**4 - Good**
- Specific and actionable
- Clear decomposition: "Create Dream Journal Entry" → "Append Nov 7 dream notes"
- Both goals add unique information

**5 - Excellent**
- Strategic hierarchy showing multi-step planning
- Subgoal shows current step in larger workflow
- Goals show understanding of user's implicit intent
- Example: "Build project knowledge base" → "Extract and organize API endpoints from documentation"

---

### 4. prompt_naturalness

**What it measures**: How natural and realistic the user prompt is (human-like vs robotic).

**1 - Poor**
- Pure command syntax: "execute_tool_name --param value"
- No natural language
- Unrealistic precision: "Create file at /exact/path/file.md with exactly 500 words"

**2 - Below Average**
- Command-style with minimal context: "Search files for API"
- No pronouns or natural phrasing
- Overly terse or telegraphic

**3 - Average**
- Mix of natural and command: "Search the Resources folder for documentation"
- Some natural language but formal
- Explicit rather than implicit

**4 - Good**
- Natural phrasing: "Find all the documentation in Resources"
- Uses pronouns, contractions, casual language
- Some implicit context: "that folder we created yesterday"

**5 - Excellent**
- Highly natural and conversational: "Can you grab my notes about that quantum physics meeting? I think it's in Research somewhere"
- Ambiguity requiring inference
- References prior context implicitly: "Do the same for the Archives folder"
- Corrections/clarifications: "Actually, make that Friday not Thursday"
- Domain-appropriate language (creative, technical, personal)

---

### 5. response_realism

**What it measures**: How realistic and complete the tool response/result is.

**1 - Poor**
- Missing Result entirely
- Minimal structure: `{"success": true}`
- No metadata
- Unrealistic data (impossible scores, wrong types)

**2 - Below Average**
- Basic structure with minimal fields
- Success-only (never shows failures)
- Missing common metadata (timestamps, counts)

**3 - Average**
- Proper structure matching tool schema
- Includes basic metadata: `totalResults`, `success`
- Plausible but generic data
- Always successful results

**4 - Good**
- Rich metadata: `executionTime`, `searchCapabilities`, `totalResults`, `displayed`
- Realistic scores/confidence (0.73 not always 1.0)
- Occasionally shows edge cases
- Appropriate data types and structure

**5 - Excellent**
- Comprehensive metadata across multiple dimensions
- Realistic edge cases: `"warnings": ["2 files skipped due to permissions"]`
- Varied outcomes (success, partial success, informative failures)
- Domain-appropriate result structure
- Search results include scores with realistic distribution (not all perfect)
- Timestamps, execution times, capability flags all present and realistic

---

## overall_quality

**Calculation**: Average of the 5 dimension scores, rounded to 1 decimal place.

**Interpretation**:
- **4.0-5.0**: Excellent - Use as template examples
- **3.0-3.9**: Good - Minor improvements possible
- **2.0-2.9**: Fair - Needs enhancement in multiple areas
- **1.0-1.9**: Poor - Major rework needed

---

## Scoring Process

### For each example:

1. **Read the full example** (user prompt + assistant response including tool call)

2. **Write reasoning notes first** explaining your score for each dimension:
   - What's good?
   - What's missing or weak?
   - How does it compare to rubric levels?

3. **Assign scores** (1-5) for each dimension

4. **Calculate overall_quality** (average of 5 scores)

5. **Output format**:
```json
{
  "conversations": [...],
  "label": true,
  "quality_scores": {
    "notes": "sessionMemory is empty [] earning score=1. toolContext is specific 'Finding dream journal' but doesn't explain why this tool was chosen (score=3). Goals have clear hierarchy 'Create entry' → 'Find journal' (score=4). Prompt is natural 'Find my dream journal' (score=4). Response includes executionTime and proper structure but no edge cases (score=4). Overall: Solid except for missing sessionMemory.",
    "sessionMemory_quality": 1,
    "toolContext_quality": 3,
    "goal_coherence": 4,
    "prompt_naturalness": 4,
    "response_realism": 4,
    "overall_quality": 3.2
  }
}
```

---

## Common Patterns to Watch For

### Red Flags (Lower Scores)
- Empty sessionMemory: automatic 1
- Generic toolContext like "Using tool": 1-2
- Identical primaryGoal and subgoal: 2
- Command-style prompts with no natural language: 1-2
- Minimal Result structure with no metadata: 2

### Green Flags (Higher Scores)
- sessionMemory references specific prior tool calls: 4-5
- toolContext explains reasoning and workflow: 4-5
- Prompt includes ambiguity requiring inference: 4-5
- Result includes executionTime, scores, capabilities: 4-5
- Multiple conversation turns showing continuity: 4-5

---

## Edge Cases

### Multi-turn conversations
- Score based on the **full conversation arc**
- Does sessionMemory accumulate context across turns?
- Do goals evolve appropriately?

### First tool call in session
- sessionMemory can be shorter but shouldn't be empty
- "Starting new session" is better than ""
- Should still capture user's stated goal

### Error/failure scenarios
- These are VALUABLE for training
- Don't penalize realistic failures
- response_realism gets 5 if failure is well-structured

### Meta-tools (get_tools)
- Apply same rubric
- sessionMemory should explain why requesting tools
- Response should include complete schema structures

---

## Calibration Examples

### Example 1: Score 4.2 (Good)
```json
{
  "user": "Find my dream journal",
  "assistant": "tool_call: vaultLibrarian_searchContent\narguments: {\"context\": {\"sessionMemory\": \"User maintains dream journal in vault\", \"toolContext\": \"Locating dream journal for entry\", \"primaryGoal\": \"Access dream journal\", \"subgoal\": \"Search for journal file\"}, ...}"
}
```
- sessionMemory: 3 (relevant but generic)
- toolContext: 4 (explains purpose)
- goal_coherence: 4 (clear hierarchy)
- prompt_naturalness: 5 (natural language)
- response_realism: 4 (good metadata)

### Example 2: Score 2.4 (Fair)
```json
{
  "user": "Search files",
  "assistant": "tool_call: vaultLibrarian_searchContent\narguments: {\"context\": {\"sessionMemory\": \"\", \"toolContext\": \"Searching\", \"primaryGoal\": \"Search\", \"subgoal\": \"Find files\"}, ...}",
  "result": "{\"success\": true}"
}
```
- sessionMemory: 1 (empty)
- toolContext: 1 (generic)
- goal_coherence: 2 (weak, overlapping)
- prompt_naturalness: 2 (command-style)
- response_realism: 2 (minimal structure)

---

## Tips for Consistent Scoring

1. **Be strict with sessionMemory**: Empty = always 1, no exceptions
2. **Look for specifics**: Numbers, paths, tool names = higher scores
3. **Check workflow reasoning**: Does toolContext explain the "why"?
4. **Natural language matters**: Pronouns, ambiguity, corrections = higher naturalness
5. **Metadata is key for realism**: executionTime, scores, capabilities = 4-5 range
6. **Average honestly**: Don't round up, calculate true average

---

**Remember**: These scores help identify improvement opportunities. Be objective, use the rubric consistently, and document your reasoning in the notes field first.
