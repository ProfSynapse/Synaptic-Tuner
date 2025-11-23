# OpenAI Format Conversion - Quick Start

## TL;DR - For Mistral Models: Use OpenAI Format ✅

If you're training **Mistral models** for use with **Ollama/LM Studio**, the **OpenAI-compatible format is the right choice**.

## Why OpenAI Format?

✅ **Mistral native** - Official Mistral API uses OpenAI-compatible format
✅ **Ollama/LM Studio** - Built-in support, no manual parsing
✅ **vLLM compatible** - Fast inference with native tool calling
✅ **Industry standard** - Most open-source tools expect this
✅ **Better than Anthropic** - Anthropic format is for Claude, not Mistral

## Quick Decision Matrix

```bash
# Training Mistral models?
→ YES: Use OpenAI format ✅ (this guide)

# Need Claude API compatibility?
→ YES: Use Anthropic format (see convert_to_mcp_format.py)
→ NO:  Use OpenAI format ✅ (better for everything else)

# Want simplest training?
→ Keep current text format (but limited compatibility)
```

## Format Comparison

### Your Current Format (Text)
```json
{
  "role": "assistant",
  "content": "tool_call: toolName\narguments: {...}"
}
```

### OpenAI Format (Recommended for Mistral) ⭐
```json
{
  "role": "assistant",
  "content": "Optional text",
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "toolName",
        "arguments": "{...}"  // JSON string!
      }
    }
  ]
}
```

### Key Differences from Anthropic Format

| Feature | OpenAI | Anthropic |
|---------|--------|-----------|
| **Tool calls location** | Separate `tool_calls` field | Inside `content` array |
| **Arguments format** | JSON string | JSON object/dict |
| **ID prefix** | `call_` | `toolu_` |
| **Type value** | `"function"` | `"tool_use"` |
| **Best for** | Mistral, open models | Claude API only |

## Conversion Commands

### 1. Preview (Dry Run)
```bash
python tools/convert_to_openai_format.py \
  Datasets/syngen_tools_sft_11.22.25.jsonl \
  /tmp/preview_openai.jsonl \
  --dry-run
```

### 2. Convert SFT Dataset
```bash
python tools/convert_to_openai_format.py \
  Datasets/syngen_tools_sft_11.22.25.jsonl \
  Datasets/syngen_tools_sft_11.22.25_openai.jsonl \
  --validate
```

### 3. Convert KTO Dataset (if you have one)
```bash
python tools/convert_to_openai_format.py \
  Datasets/behavior_merged_kto_11.23.25.jsonl \
  Datasets/behavior_merged_kto_11.23.25_openai.jsonl \
  --validate
```

## Training with OpenAI Format

### SFT Training
```bash
cd Trainers/rtx3090_sft

# Train with converted dataset
./train.sh --model-size 7b \
  --local-file ../../Datasets/syngen_tools_sft_11.22.25_openai.jsonl
```

### KTO Training (if applicable)
```bash
cd Trainers/rtx3090_kto

# Train with converted dataset
./train.sh --model-size 7b \
  --local-file ../../Datasets/behavior_merged_kto_11.23.25_openai.jsonl
```

## Example: Before and After

### Before (Text Format)
```json
{
  "conversations": [
    {
      "role": "user",
      "content": "Create a note about the moon"
    },
    {
      "role": "assistant",
      "content": "I'll create that note for you.\n\ntool_call: contentManager_createContent\narguments: {\"context\": {...}, \"filePath\": \"moon.md\", \"content\": \"# Moon\\n\\nAbout the moon...\"}"
    }
  ]
}
```

### After (OpenAI Format)
```json
{
  "conversations": [
    {
      "role": "user",
      "content": "Create a note about the moon"
    },
    {
      "role": "assistant",
      "content": "I'll create that note for you.",
      "tool_calls": [
        {
          "id": "call_abc123xyz789",
          "type": "function",
          "function": {
            "name": "contentManager_createContent",
            "arguments": "{\"context\": {...}, \"filePath\": \"moon.md\", \"content\": \"# Moon\\n\\nAbout the moon...\"}"
          }
        }
      ]
    }
  ]
}
```

**Key changes:**
- Tool call moved to separate `tool_calls` field
- Text before tool call stays in `content`
- Arguments are a JSON **string** (note the escaped quotes)
- ID is `call_` prefix instead of `toolu_`
- Type is `"function"` instead of `"tool_use"`

## What You Need to Update

After converting your dataset, you'll need to update:

### 1. Validator (`tools/validate_syngen.py`)
```python
# Old: Parse text format
if "tool_call:" in content:
    extract_tool_calls(content)

# New: Parse OpenAI format
if "tool_calls" in message:
    for tool_call in message["tool_calls"]:
        name = tool_call["function"]["name"]
        args = json.loads(tool_call["function"]["arguments"])
```

### 2. Evaluator (`Evaluator/schema_validator.py`)
```python
# Old: Extract from text
tool_calls = extract_tool_calls(content)

# New: Extract from tool_calls field
if "tool_calls" in message:
    for tc in message["tool_calls"]:
        tool_name = tc["function"]["name"]
        arguments = json.loads(tc["function"]["arguments"])
```

### 3. Inference Parsing
When using the trained model with Ollama/LM Studio:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="dummy")

# Define your tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "contentManager_createContent",
            "description": "Create new content",
            "parameters": {
                "type": "object",
                "properties": {
                    "context": {...},
                    "filePath": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["context", "filePath", "content"]
            }
        }
    }
]

# Use with tool calling
response = client.chat.completions.create(
    model="your-finetuned-mistral",
    messages=[{"role": "user", "content": "Create a note"}],
    tools=tools,
    tool_choice="auto"
)

# Access tool calls (native support!)
if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        # Execute tool...
```

## Benefits of OpenAI Format

### 1. Native Ollama/LM Studio Support
```python
# Works out of the box with OpenAI client
response = client.chat.completions.create(
    model="mistral",
    tools=[...]  # Native tool support!
)
```

### 2. vLLM Integration
```bash
# vLLM server with tool calling
vllm serve your-model --enable-tool-calling

# Use with OpenAI client - it just works!
```

### 3. Ecosystem Compatibility
```python
# LangChain
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(base_url="http://localhost:11434/v1")
# Tool calling works automatically

# LlamaIndex
from llama_index.llms import OpenAI
llm = OpenAI(base_url="http://localhost:11434/v1")
# Function calling supported
```

## Conversion Stats

**Dataset:** `syngen_tools_sft_11.22.25.jsonl`
- **Total examples:** 5,515
- **Converted:** 5,514 (99.98%)
- **Warnings:** 1 (malformed JSON in source)
- **Time:** ~3 seconds

**Token increase:**
- Text format: ~220 tokens/example
- OpenAI format: ~260 tokens/example (+18%)
- Anthropic format: ~280 tokens/example (+27%)

**Verdict:** OpenAI format is only 18% larger, much better than Anthropic's 27%.

## Important Notes

### Arguments Must Be JSON Strings

**CRITICAL:** In OpenAI format, `arguments` is a **JSON string**, not a dict!

```python
# ❌ WRONG (this is Anthropic format)
"arguments": {"key": "value"}

# ✅ CORRECT (OpenAI format)
"arguments": "{\"key\": \"value\"}"
```

The converter handles this automatically.

### Multiple Tool Calls

OpenAI format supports multiple tool calls in one message:

```json
{
  "role": "assistant",
  "content": "I'll search and create a note.",
  "tool_calls": [
    {
      "id": "call_001",
      "type": "function",
      "function": {
        "name": "vaultLibrarian_searchContent",
        "arguments": "{\"query\": \"example\"}"
      }
    },
    {
      "id": "call_002",
      "type": "function",
      "function": {
        "name": "contentManager_createContent",
        "arguments": "{\"filePath\": \"result.md\"}"
      }
    }
  ]
}
```

## Comparison: All Three Formats

| Aspect | Current (Text) | OpenAI | Anthropic |
|--------|---------------|--------|-----------|
| **For Mistral** | ⚠️ Custom | ✅ **Native** | ❌ Wrong |
| **Ollama Support** | ❌ Manual | ✅ **Built-in** | ❌ Manual |
| **vLLM Support** | ❌ No | ✅ **Yes** | ❌ No |
| **Simplicity** | ✅ **Easiest** | ⚠️ Medium | ❌ Complex |
| **Token Size** | ✅ **Smallest** | ⚠️ +18% | ❌ +27% |
| **Ecosystem** | ❌ None | ✅ **Wide** | ⚠️ Claude |

## My Recommendation: OpenAI Format ✅

For your use case (Mistral models with Ollama/LM Studio):

**Convert to OpenAI format** because:
1. ✅ Mistral's native format
2. ✅ Ollama/LM Studio built-in support
3. ✅ vLLM compatibility
4. ✅ Industry standard for open models
5. ✅ Better than Anthropic (which is Claude-specific)

**Steps:**
1. Convert dataset (1 command)
2. Update validator and evaluator (~2 hours)
3. Train new model
4. Use with native tool calling support

**Alternative:** Keep current text format if you want absolute simplicity and don't need ecosystem compatibility.

**Don't use:** Anthropic format - it's designed for Claude API, not Mistral.

## Quick Start Checklist

- [ ] Convert dataset to OpenAI format
- [ ] Preview with `--dry-run` first
- [ ] Validate with `--validate` flag
- [ ] Update `tools/validate_syngen.py` for OpenAI format
- [ ] Update `Evaluator/schema_validator.py` for OpenAI format
- [ ] Train model on converted dataset
- [ ] Test with Ollama/LM Studio using OpenAI client
- [ ] Verify tool calling works natively

## Support

**Full comparison:** See `docs/TOOL_CALLING_FORMATS_COMPARISON.md`

**Scripts:**
- OpenAI format: `tools/convert_to_openai_format.py`
- Anthropic format: `tools/convert_to_mcp_format.py`

**Help:**
```bash
python tools/convert_to_openai_format.py --help
```

## Bottom Line

For Mistral models: **OpenAI format is the industry standard**. It gives you native tool calling support in Ollama, LM Studio, vLLM, and the entire open-source ecosystem. This is the format Mistral models expect and perform best with.

Only use a different format if you have very specific requirements:
- **Text format:** Absolute simplicity, prototype only
- **Anthropic format:** Claude API compatibility (not your use case)
