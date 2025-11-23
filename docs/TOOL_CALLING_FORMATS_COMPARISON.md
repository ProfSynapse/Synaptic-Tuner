# Tool Calling Formats: Complete Comparison

## Three Format Options

### 1. Your Current Format (Text-Based)

```json
{
  "role": "assistant",
  "content": "tool_call: contentManager_createContent\narguments: {\"context\": {...}, \"filePath\": \"note.md\", \"content\": \"...\"}"
}
```

**Characteristics:**
- Plain text format with markers
- `tool_call:` prefix followed by tool name
- `arguments:` prefix followed by JSON
- Simple string parsing

### 2. Anthropic Messages API Format (MCP)

```json
{
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "I'll create that file for you."
    },
    {
      "type": "tool_use",
      "id": "toolu_01A2B3C4D5E6F7G8H9I0J1K2",
      "name": "contentManager_createContent",
      "input": {
        "context": {...},
        "filePath": "note.md",
        "content": "..."
      }
    }
  ]
}
```

**Characteristics:**
- Content is an array of blocks
- Tool calls have `type: "tool_use"`
- IDs start with `toolu_`
- Arguments in `input` field (as dict)
- Text and tool calls interleaved in content array

### 3. OpenAI Function Calling Format ⭐ **BEST FOR MISTRAL**

```json
{
  "role": "assistant",
  "content": "I'll create that file for you.",
  "tool_calls": [
    {
      "id": "call_abc123xyz",
      "type": "function",
      "function": {
        "name": "contentManager_createContent",
        "arguments": "{\"context\": {...}, \"filePath\": \"note.md\", \"content\": \"...\"}"
      }
    }
  ]
}
```

**Characteristics:**
- Separate `tool_calls` field (not in content)
- `content` can be null or contain text
- Tool call IDs start with `call_`
- Type is always `"function"`
- Arguments are a **JSON string** (not a dict)
- Native format for Mistral models

## Why OpenAI Format for Mistral?

### ✅ Perfect Match for Your Stack

1. **Mistral Native Support**
   - Mistral models are trained on OpenAI-compatible format
   - Official Mistral API uses this format
   - Best performance with this structure

2. **Ollama/LM Studio Compatibility**
   - Both support OpenAI-compatible endpoints
   - Direct integration without parsing
   - Can use OpenAI client libraries

3. **vLLM Support**
   - vLLM (fast inference) uses OpenAI format
   - Better throughput with native format

4. **Industry Standard**
   - Most tools expect OpenAI format
   - Better ecosystem compatibility
   - More examples/documentation

### Detailed Comparison

| Feature | Current (Text) | Anthropic (MCP) | OpenAI (Mistral) |
|---------|---------------|-----------------|------------------|
| **For Mistral Models** | ⚠️ Custom | ❌ Wrong API | ✅ **Native** |
| **Ollama/LM Studio** | ⚠️ Manual Parse | ❌ Manual Parse | ✅ **Built-in** |
| **Learning Difficulty** | ✅ Easiest | ❌ Complex | ⚠️ Medium |
| **Tool Call Tracking** | ❌ No IDs | ✅ Has IDs | ✅ Has IDs |
| **API Compatibility** | ❌ None | ⚠️ Claude Only | ✅ **Universal** |
| **Ecosystem Support** | ❌ Custom | ⚠️ Anthropic | ✅ **Widespread** |
| **Training Data Size** | ✅ Compact | ❌ Verbose | ⚠️ Medium |
| **JSON Complexity** | ✅ Simple | ❌ Nested | ⚠️ String Args |
| **Multi-tool Support** | ⚠️ Manual | ✅ Native | ✅ **Native** |
| **vLLM Support** | ❌ No | ❌ No | ✅ **Yes** |

## Format Examples: Same Tool Call

### Your Current Format

```
tool_call: contentManager_createContent
arguments: {"context": {"sessionId": "session_1731398400000_a1b2c3d4e", "workspaceId": "ws_1731398400000_f5g6h7i8j", "sessionDescription": "Creating note", "sessionMemory": "User creating content...", "toolContext": "Create new file", "primaryGoal": "Create note", "subgoal": "Write content"}, "filePath": "notes/example.md", "content": "# Example\n\nThis is a test note."}
```

### Anthropic Format

```json
{
  "role": "assistant",
  "content": [
    {
      "type": "tool_use",
      "id": "toolu_01A2B3C4D5E6F7G8H9I0J1K2",
      "name": "contentManager_createContent",
      "input": {
        "context": {
          "sessionId": "session_1731398400000_a1b2c3d4e",
          "workspaceId": "ws_1731398400000_f5g6h7i8j",
          "sessionDescription": "Creating note",
          "sessionMemory": "User creating content...",
          "toolContext": "Create new file",
          "primaryGoal": "Create note",
          "subgoal": "Write content"
        },
        "filePath": "notes/example.md",
        "content": "# Example\n\nThis is a test note."
      }
    }
  ]
}
```

### OpenAI Format (for Mistral)

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_abc123xyz789",
      "type": "function",
      "function": {
        "name": "contentManager_createContent",
        "arguments": "{\"context\": {\"sessionId\": \"session_1731398400000_a1b2c3d4e\", \"workspaceId\": \"ws_1731398400000_f5g6h7i8j\", \"sessionDescription\": \"Creating note\", \"sessionMemory\": \"User creating content...\", \"toolContext\": \"Create new file\", \"primaryGoal\": \"Create note\", \"subgoal\": \"Write content\"}, \"filePath\": \"notes/example.md\", \"content\": \"# Example\\n\\nThis is a test note.\"}"
      }
    }
  ]
}
```

## Key Differences: OpenAI vs Anthropic

### Field Structure

**OpenAI:**
- `tool_calls` is a separate top-level field
- `content` and `tool_calls` are siblings
- Can have text + tool calls simultaneously

**Anthropic:**
- `content` is an array containing both text and tool_use blocks
- Everything is in content array
- More nested structure

### Arguments Format

**OpenAI:**
- Arguments are a **JSON string**: `"arguments": "{\"key\": \"value\"}"`
- Need to serialize dict to string
- Need to escape quotes

**Anthropic:**
- Arguments are a **dict**: `"input": {"key": "value"}`
- Direct JSON object
- No escaping needed

### ID Format

**OpenAI:**
- IDs start with `call_`
- Example: `call_abc123xyz789`

**Anthropic:**
- IDs start with `toolu_`
- Example: `toolu_01A2B3C4D5E6F7G8H9I0J1K2`

### Type Field

**OpenAI:**
- Always `"type": "function"`
- Fixed value

**Anthropic:**
- Always `"type": "tool_use"`
- Different semantic meaning

## Multiple Tool Calls

### OpenAI Format (Array of tool_calls)

```json
{
  "role": "assistant",
  "content": "I'll search for that information and create a note.",
  "tool_calls": [
    {
      "id": "call_001",
      "type": "function",
      "function": {
        "name": "vaultLibrarian_searchContent",
        "arguments": "{\"context\": {...}, \"query\": \"example\"}"
      }
    },
    {
      "id": "call_002",
      "type": "function",
      "function": {
        "name": "contentManager_createContent",
        "arguments": "{\"context\": {...}, \"filePath\": \"result.md\"}"
      }
    }
  ]
}
```

### Anthropic Format (Array of content blocks)

```json
{
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "I'll search for that information and create a note."
    },
    {
      "type": "tool_use",
      "id": "toolu_001",
      "name": "vaultLibrarian_searchContent",
      "input": {"context": {...}, "query": "example"}
    },
    {
      "type": "tool_use",
      "id": "toolu_002",
      "name": "contentManager_createContent",
      "input": {"context": {...}, "filePath": "result.md"}
    }
  ]
}
```

## Recommended Format: **OpenAI for Mistral**

### Why OpenAI Format is Best for You

1. ✅ **Mistral models expect this format**
   - Official Mistral API uses OpenAI-compatible format
   - Models are trained/fine-tuned on this structure
   - Best inference performance

2. ✅ **Direct Ollama/LM Studio support**
   ```python
   # Works out of the box
   from openai import OpenAI
   client = OpenAI(base_url="http://localhost:11434/v1")
   response = client.chat.completions.create(
       model="mistral",
       messages=[...],
       tools=[...]  # Native support!
   )
   ```

3. ✅ **Better ecosystem integration**
   - LangChain uses OpenAI format
   - LlamaIndex uses OpenAI format
   - Most inference servers expect this

4. ✅ **vLLM compatibility**
   - Fast inference with native tool calling
   - No manual parsing needed

### Migration Path

**Current → OpenAI is easier than Current → Anthropic:**

| Conversion | Difficulty | Reason |
|------------|-----------|--------|
| Text → OpenAI | ⚠️ Medium | Simple structure, JSON string args |
| Text → Anthropic | ❌ Hard | Complex nesting, content arrays |
| OpenAI → Anthropic | ⚠️ Medium | Different structure, different IDs |

## Training Data Considerations

### Token Count Comparison (Same Example)

| Format | Approximate Tokens | Relative Size |
|--------|-------------------|---------------|
| **Current (Text)** | ~220 | 1.0x (baseline) |
| **OpenAI** | ~260 | 1.18x |
| **Anthropic** | ~280 | 1.27x |

**Implication:** OpenAI format is only ~18% larger than your current format, while Anthropic is ~27% larger.

### Learning Difficulty

**Model's Perspective:**

1. **Current (Text):**
   - Learn: `tool_call:` marker, `arguments:` marker
   - Generate: Simple sequential text
   - Challenge: Must format JSON correctly inline

2. **OpenAI:**
   - Learn: JSON structure with `tool_calls` array
   - Generate: JSON with string-escaped arguments
   - Challenge: Must escape quotes in JSON string

3. **Anthropic:**
   - Learn: Nested content array structure
   - Generate: Mixed text/tool_use blocks
   - Challenge: Complex nesting, block types

**Verdict:** OpenAI is reasonable learning difficulty for structured output training.

## Recommendation Matrix

| Your Priority | Best Format | Second Best | Avoid |
|---------------|-------------|-------------|-------|
| **Mistral native compatibility** | OpenAI ✅ | Current ⚠️ | Anthropic ❌ |
| **Ollama/LM Studio integration** | OpenAI ✅ | Current ⚠️ | Anthropic ❌ |
| **Simplest training** | Current ✅ | OpenAI ⚠️ | Anthropic ❌ |
| **Tool ecosystem** | OpenAI ✅ | Anthropic ⚠️ | Current ❌ |
| **vLLM inference** | OpenAI ✅ | - | Both ❌ |
| **Smallest dataset** | Current ✅ | OpenAI ⚠️ | Anthropic ❌ |

## Final Recommendation

### For Mistral Models: **Use OpenAI Format** ✅

**Strong reasons:**
1. Native Mistral API compatibility
2. Ollama/LM Studio have built-in support
3. vLLM native tool calling
4. Widespread ecosystem support
5. Better than Anthropic for your use case

**Trade-offs:**
- Slightly more complex than current text format
- Need to serialize arguments to JSON strings
- ~18% more tokens per example

### Migration Strategy

1. **Convert dataset to OpenAI format** (script provided below)
2. **Update validator** to parse OpenAI tool_calls
3. **Update evaluator** to extract from tool_calls field
4. **Train new model** on OpenAI format
5. **Compare** with current text format
6. **Switch** if OpenAI format performs better

### Don't Use Anthropic Format

**Reasons:**
- ❌ Not native to Mistral
- ❌ No built-in support in Ollama/LM Studio
- ❌ More complex structure
- ❌ Larger token count
- ❌ Only useful if targeting Claude API

## Summary

| Format | Use When |
|--------|----------|
| **Current (Text)** | Prototyping, simplest training, you control everything |
| **OpenAI** | ✅ **Production with Mistral, ecosystem integration, vLLM** |
| **Anthropic** | Only if specifically targeting Claude API (not your case) |

**Bottom line:** For Mistral models with Ollama/LM Studio, **OpenAI format is the correct choice**. It's the industry standard for open-source models and gives you the best compatibility and ecosystem support.
