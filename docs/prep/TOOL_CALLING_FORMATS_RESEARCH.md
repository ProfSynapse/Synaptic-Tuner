# Tool Calling Formats and Best Practices for LLM Fine-Tuning

**Research Date:** November 18, 2025
**Purpose:** Understanding official tool/function calling formats across major LLM providers and best practices for training datasets

---

## Executive Summary

This document provides comprehensive research on how different LLM providers (OpenAI, Anthropic Claude, and Model Context Protocol) handle tool/function calling, with specific focus on:

1. **Can models output text AND tool calls together?**
   - **Anthropic Claude:** YES - Can output text blocks and tool use blocks in same response
   - **OpenAI:** NO (mostly) - Typically `content` is null when `tool_calls` present, though newer models may support both
   - **MCP:** Protocol-level - Not model-specific, handles structured request/response

2. **Training Dataset Implications:**
   - OpenAI format expects `content: null` with `tool_calls` in fine-tuning data
   - Claude format allows text reasoning before tool use
   - Real-world datasets use various approaches depending on target model

3. **Best Practice Recommendation:**
   - For OpenAI-style models: Separate text responses from tool calls
   - For Claude-style models: Can include reasoning text with tool calls
   - For general fine-tuning: Use established datasets like Glaive or xLAM as templates

---

## 1. OpenAI Function Calling Format

### 1.1 Official API Response Format

**Key Finding:** In OpenAI's tool calling system, the model typically outputs EITHER text OR tool calls, not both simultaneously.

#### Example Response with Tool Call

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "model": "gpt-4o",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\": \"San Francisco, CA\", \"unit\": \"celsius\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

**Critical Points:**
- `content` field is **null** when tool calls are made
- `finish_reason` is `"tool_calls"` (not `"stop"`)
- Each tool call has unique `id` for tracking responses
- Arguments are JSON-encoded strings

#### Parallel Tool Calls

OpenAI supports **parallel tool calling** where multiple tools can be invoked simultaneously:

```json
{
  "message": {
    "role": "assistant",
    "content": null,
    "tool_calls": [
      {
        "id": "call_1",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\": \"New York\"}"
        }
      },
      {
        "id": "call_2",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\": \"London\"}"
        }
      },
      {
        "id": "call_3",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\": \"Tokyo\"}"
        }
      }
    ]
  }
}
```

**Control Parameter:** Set `parallel_tool_calls: false` to limit to zero or one tool call at a time.

**Model Support:**
- GPT-4o, GPT-4o mini, GPT-4 Turbo, GPT-3.5-Turbo: ✅ Support parallel calls
- o1, o1-mini, o3-mini reasoning models: ❌ No parallel calls

### 1.2 Exception: Reasoning Models (o1, o3-mini)

OpenAI's reasoning models (o1 series) have a **different structure** that separates reasoning from tool calls:

```python
# Response contains multiple output types
response_types = ['reasoning', 'function_calls', 'message']

# Filter by type
reasoning = [rx for rx in response if rx.type == 'reasoning']
function_calls = [rx for rx in response if rx.type == 'function_call']
messages = [rx for rx in response if rx.type == 'message']
```

**Key Differences:**
- Reasoning tokens are **hidden** (not returned in content, but counted in `reasoning_tokens`)
- May produce **sequential** function calls (not parallel)
- Each step may depend on previous tool results

**Pattern for Reasoning Models:**
1. Initialize loop
2. If response contains function calls → feed results back for further inference
3. If response is message type (no tool calls) → break loop
4. Continue until task complete

### 1.3 Community Reports: Mixed Behavior

Some developers report **inconsistent behavior** where models occasionally return both:

> "A model response might include content like 'Perfect, please wait a moment while I forward the call...' along with a tool_call for a 'forward_call' function."

**However, this is NOT the standard/expected behavior for fine-tuning.**

### 1.4 Training Data Format for OpenAI

For fine-tuning with function calling, OpenAI expects this format:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "what's the weather in paris?"
    },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\": \"paris\"}"
        }
      }]
    },
    {
      "role": "tool",
      "tool_call_id": "call_123",
      "content": "{\"temperature\": 22, \"condition\": \"sunny\"}"
    },
    {
      "role": "assistant",
      "content": "The weather in Paris is currently 22°C and sunny."
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "City name"
            }
          },
          "required": ["location"]
        }
      }
    }
  ]
}
```

**Critical Rules:**
- ✅ `content: null` when tool_calls present (required for OpenAI)
- ✅ Include `tools` array in training data
- ✅ Use `tool` role for function results (with `tool_call_id`)
- ❌ Don't set both `content` and `tool_calls` in assistant message
- ⚠️ Azure OpenAI "gets fussy" about having both simultaneously

**Minimum Requirements:**
- At least 10 examples for fine-tuning
- Each training example must include the `tools` definition

### 1.5 Structured Outputs (2024 Update)

In June 2024, OpenAI introduced **Structured Outputs** feature:

- Set `strict: true` in function definition
- Guarantees arguments match JSON Schema exactly
- Simplifies function calling with SDK tools
- "Generate Anything" feature (October 2024) auto-generates schemas from code

**Migration Note:** `functions` and `function_call` parameters deprecated as of 2023-12-01. Use `tools` and `tool_choice` instead.

---

## 2. Anthropic Claude Tool Use Format

### 2.1 Official API Response Format

**Key Finding:** Claude CAN output both text and tool use blocks in the SAME response.

#### Example Response with Text + Tool Use

```json
{
  "id": "msg_01Aq9w938a90dw8q",
  "model": "claude-sonnet-4-5",
  "stop_reason": "tool_use",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "I'll check the current weather in San Francisco for you."
    },
    {
      "type": "tool_use",
      "id": "toolu_01A09q90qw90lq917835lq9",
      "name": "get_weather",
      "input": {
        "location": "San Francisco, CA",
        "unit": "celsius"
      }
    }
  ]
}
```

**Key Characteristics:**
- `content` is an **array** of blocks (not a single string)
- Can contain multiple block types: `text`, `tool_use`
- `stop_reason: "tool_use"` signals tool execution needed
- Tool inputs are **JSON objects** (not JSON strings like OpenAI)
- Each tool use has unique `id` for tracking

### 2.2 Content Block Types

Claude's `content` array can contain:

1. **Text blocks** - Natural language responses
   ```json
   {"type": "text", "text": "Let me help you with that."}
   ```

2. **Tool use blocks** - Function/tool invocations
   ```json
   {
     "type": "tool_use",
     "id": "toolu_123",
     "name": "search",
     "input": {"query": "weather"}
   }
   ```

**Workflow:**
1. User sends request with available tools
2. Claude responds with content blocks (may include text + tool use)
3. If `stop_reason == "tool_use"`, execute tools and return results
4. Continue conversation with tool results

### 2.3 The "Think" Tool

Anthropic introduced a special **"think" tool** for enhanced reasoning in tool use scenarios:

**Purpose:** Allow Claude to pause and reflect during tool calling chains

**When to Use:**
- Complex tools requiring careful parameter selection
- Long chains of tool calls with dependencies
- Policy-heavy environments with detailed guidelines
- Sequential decisions where mistakes are costly

**How It Works:**
```json
{
  "type": "tool_use",
  "id": "toolu_think_01",
  "name": "think",
  "input": {
    "reasoning": "I need to first check if the user has permission before accessing their data..."
  }
}
```

**Key Distinctions:**
- **Think tool:** Focused reasoning on new information discovered during tool use
- **Extended thinking:** Deep pre-response planning before any generation
- **Interleaved thinking:** Think between tool calls (Claude 4 models)

**Performance Impact:** Significantly enhances Claude 3.7 Sonnet's performance on complex multi-step tool calling tasks.

### 2.4 Chain of Thought Prompting

Claude supports explicit chain-of-thought reasoning before tool use:

**Recommended Pattern:**
```
Before calling any tools, think step-by-step:
1. What information do I need?
2. Which tool(s) can provide it?
3. What parameters are required?
4. Are there any dependencies?

Then make the appropriate tool call(s).
```

**Critical Rule:** Claude must OUTPUT its thinking for it to occur. If thinking is not in the response, no actual reasoning happened.

### 2.5 Legacy Format (Deprecated)

Older Claude versions used XML-based tool calling:

```xml
<function_calls>
  <invoke>
    <tool_name>get_weather</tool_name>
    <parameters>
      <location>New York</location>
    </parameters>
  </invoke>
</function_calls>
```

**Status:** DEPRECATED - Use modern Messages API with JSON format

### 2.6 Training Data Implications for Claude-style Models

For fine-tuning Claude-like models:

```json
{
  "conversations": [
    {
      "role": "user",
      "content": "What's the weather in Paris?"
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "I'll check the current weather in Paris for you."
        },
        {
          "type": "tool_use",
          "id": "toolu_01",
          "name": "get_weather",
          "input": {"location": "Paris", "unit": "celsius"}
        }
      ]
    }
  ]
}
```

**Key Points:**
- ✅ Text reasoning BEFORE tool use is encouraged
- ✅ Content is array of blocks, not single string
- ✅ Can include explanatory text with tool calls
- ✅ Tool inputs are objects, not JSON strings

---

## 3. Model Context Protocol (MCP)

### 3.1 Protocol Specification

MCP is a **standardized protocol** (not a model-specific format) for tool calling across different LLMs.

**Key Principle:** MCP uses JSON-RPC 2.0 for universal interoperability.

#### Tool Call Request Format

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "get_weather",
    "arguments": {
      "location": "New York"
    }
  }
}
```

**Structure:**
- `jsonrpc`: Always "2.0"
- `id`: Unique request identifier
- `method`: Always "tools/call" for tool invocation
- `params.name`: Tool identifier
- `params.arguments`: Parameter object matching tool's `inputSchema`

#### Tool Call Response Format

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Current weather in New York:\nTemperature: 72°F\nConditions: Partly cloudy"
      }
    ],
    "isError": false
  }
}
```

**Result Structure:**
- `content`: Array of result items (text, images, audio, resources)
- `isError`: Boolean flag for execution status
- `structuredContent` (optional): JSON conforming to `outputSchema`

### 3.2 Content Types

MCP supports rich content types in responses:

1. **Text** - Plain text results
2. **Images** - Binary image data with MIME type
3. **Audio** - Audio data
4. **Resource links** - References to external resources
5. **Embedded resources** - Inline resource data

### 3.3 Tool Definition Schema

```json
{
  "name": "get_weather",
  "description": "Get current weather for a location",
  "inputSchema": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "City name"
      },
      "unit": {
        "type": "string",
        "enum": ["celsius", "fahrenheit"],
        "description": "Temperature unit"
      }
    },
    "required": ["location"]
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "temperature": {"type": "number"},
      "conditions": {"type": "string"}
    }
  }
}
```

### 3.4 MCP Implementation Notes

**Server Responsibilities:**
- Expose tools via `tools/list` method
- Handle `tools/call` requests
- Return structured results with `content` array
- Set `isError` flag on failures

**Client Responsibilities:**
- Discover available tools via `tools/list`
- Format tool call requests correctly
- Handle responses (including errors)
- Present results to LLM for next steps

**Model Integration:**
MCP is transport/model-agnostic. The LLM (OpenAI, Claude, etc.) decides WHEN to call tools, then the client translates that to MCP format for execution.

---

## 4. Fine-Tuning Dataset Formats and Best Practices

### 4.1 Popular Training Datasets

#### Glaive Function Calling v2

**Source:** [glaiveai/glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)
**Size:** 113,000 examples
**License:** Apache 2.0

**Original Format:**
```
SYSTEM: <system message with function definitions>
USER: <user message>
ASSISTANT: <assistant response>
<functioncall> {"name": "get_random_quote", "arguments": {}}
FUNCTION RESPONSE: {"quote": "...", "author": "..."}
ASSISTANT: <response using function result>
```

**Formatted Versions Available:**
- ShareGPT format
- Llama format
- Standard chat format with roles

**Example (Formatted):**
```json
{
  "system_message": "You are a helpful assistant with access to functions.",
  "function_description": [
    {
      "name": "get_random_quote",
      "description": "Get a random inspirational quote",
      "parameters": {
        "type": "object",
        "properties": {}
      }
    }
  ],
  "conversations": [
    {"role": "user", "content": "Give me an inspirational quote"},
    {
      "role": "function-call",
      "content": "{\"name\": \"get_random_quote\", \"arguments\": {}}"
    },
    {
      "role": "function-response",
      "content": "{\"quote\": \"Success is not final\", \"author\": \"Churchill\"}"
    },
    {
      "role": "assistant",
      "content": "Here's an inspiring quote from Churchill: 'Success is not final'"
    }
  ]
}
```

**Coverage:**
- Simple single function calls
- Multiple function calls (choosing from available tools)
- Multi-turn conversations with sequential tool use
- Parallel function calling scenarios

#### Salesforce xLAM Function Calling Dataset

**Source:** [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
**Size:** 60,000 examples
**Generation:** APIGen automated pipeline (verifiable high-quality)

**Format:**
```json
{
  "query": "Find the sum of multiples of 3 and 5 between 1 and 1000, and find the product of the first five prime numbers",
  "tools": [
    {
      "name": "math_toolkit.sum_of_multiples",
      "description": "Calculate sum of multiples of given numbers in range",
      "parameters": {
        "multiples": {
          "type": "list",
          "description": "Numbers to find multiples of",
          "required": true
        },
        "start": {"type": "int", "required": true},
        "end": {"type": "int", "required": true}
      }
    },
    {
      "name": "math_toolkit.product_of_primes",
      "description": "Calculate product of first N prime numbers",
      "parameters": {
        "n": {"type": "int", "description": "Count of primes", "required": true}
      }
    }
  ],
  "answers": [
    {
      "name": "math_toolkit.sum_of_multiples",
      "arguments": {"multiples": [3, 5], "start": 1, "end": 1000}
    },
    {
      "name": "math_toolkit.product_of_primes",
      "arguments": {"n": 5}
    }
  ]
}
```

**Key Features:**
- Detailed parameter specifications (type, description, required)
- Multiple tool calls per query
- Verifiable automated generation
- Suitable for training Small Language Models

**Loading:**
```python
from datasets import load_dataset
dataset = load_dataset("Salesforce/xlam-function-calling-60k")
```

### 4.2 Dataset Structure Best Practices

#### Essential Components

1. **System Message (Optional but Recommended)**
   - Defines assistant's role
   - May include tool usage instructions
   - Should be consistent across examples

2. **Function/Tool Definitions**
   - Name, description, parameters
   - JSON Schema for parameters
   - Clear parameter descriptions
   - Required vs optional fields

3. **Conversation Flow**
   - User request
   - Assistant response (may include reasoning)
   - Tool call(s)
   - Tool result(s)
   - Assistant final response using results

#### Scenarios to Include

Based on research, a robust dataset should cover:

1. **Single Function Calling** - One tool, one call
2. **Multiple Function Selection** - Choosing correct tool from many
3. **Multi-turn Conversations** - Sequential tool calls with context
4. **Parallel Tool Calls** - Multiple simultaneous invocations
5. **Missing Information** - Asking user for required parameters
6. **No Tool Needed** - Responding without tools when appropriate
7. **Error Handling** - Invalid parameters, tool failures

### 4.3 OpenAI-Style Training Format

**For models targeting OpenAI compatibility:**

```jsonl
{"messages": [{"role": "user", "content": "what's the weather in paris?"}, {"role": "assistant", "content": null, "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\": \"paris\"}"}}]}, {"role": "tool", "tool_call_id": "call_123", "content": "{\"temp\": 22, \"condition\": \"sunny\"}"}, {"role": "assistant", "content": "The weather in Paris is 22°C and sunny."}], "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}]}
```

**Key Rules:**
- JSONL format (one example per line)
- `content: null` when `tool_calls` present
- Include `tools` array in each example
- Use `tool` role for function results
- Minimum 10 examples for fine-tuning

### 4.4 Claude-Style Training Format (ChatML)

**For models targeting Claude compatibility:**

```jsonl
{"conversations": [{"role": "user", "content": "What's the weather in Paris?"}, {"role": "assistant", "content": "I'll check the weather in Paris for you.\n\ntool_call: get_weather\narguments: {\"location\": \"Paris\", \"unit\": \"celsius\"}\n\nResult: {\"temperature\": 22, \"condition\": \"sunny\"}\n\nThe current weather in Paris is 22°C and sunny."}], "label": true}
```

**Alternative with Structured Blocks:**
```json
{
  "conversations": [
    {"role": "user", "content": "What's the weather in Paris?"},
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "I'll check the weather for you."},
        {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"location": "Paris"}}
      ]
    }
  ]
}
```

**Key Characteristics:**
- Can include reasoning text before/with tool calls
- ChatML format (role + content)
- Optional `label` field (true/false for preference learning)
- Content can be string or array of blocks

### 4.5 Llama 3 Function Calling Format

**System Prompt Pattern:**
```
You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose. If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.

If you decide to invoke any of the function(s), you MUST put it in the format of:
[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]

You SHOULD NOT include any other text in the response.

Here is a list of functions in JSON format that you can invoke:
[<function schemas as JSON>]
```

**Special Tokens:**
- `<|begin_of_text|>` - Start of sequence
- `<|start_header_id|>` - Role header start
- `<|end_header_id|>` - Role header end
- `<|eot_id|>` - End of turn (zero-shot calls)
- `<|eom_id|>` - End of message (builtin calls)

**Format Constraint:** Model must NOT generate both text and tool calls in same generation (implementation-specific limitation).

### 4.6 Training Data Quality Recommendations

**From Meta (2024):**
> "The rule of thumb is to ensure—as best as you can—that the training data reflects how the model should behave in the real world."

**Data Curation Best Practices:**

1. **Diversity through Deduplication**
   - Remove near-duplicate examples
   - Improves performance measures
   - Prevents overfitting to patterns

2. **Synthetic Data Generation**
   - Seed LLMs with high-quality examples
   - Prompt to generate similar variations
   - Validate generated data for correctness

3. **Annotation Cost Reduction**
   - Use LLMs to assist with labeling
   - Human verification of critical examples
   - Automated quality checks

**Dataset Size Recommendations:**

- **Minimum:** 1,000 samples with LoRA fine-tuning
- **Recommended:** 10,000-20,000 samples (from 60k datasets)
- **Comprehensive:** 50,000+ samples for broad coverage

**Quality over Quantity:**
> "The quality and amount [of data] largely reflect the end result of your fine-tune."

---

## 5. Real-World Implementation Patterns

### 5.1 Text + Tool Calls: When and How?

#### OpenAI Pattern (Mostly Separate)

**Standard Behavior:**
```json
// Tool call response - NO text content
{
  "role": "assistant",
  "content": null,
  "tool_calls": [{"id": "c1", "function": {...}}]
}
```

**After Tool Execution:**
```json
// Final response - text content, NO tool calls
{
  "role": "assistant",
  "content": "The weather in Paris is 22°C and sunny."
}
```

**Exception Cases (Newer Models):**
Some developers report occasional combined output:
```json
{
  "role": "assistant",
  "content": "Let me check that for you.",
  "tool_calls": [{"id": "c1", "function": {...}}]
}
```

**Recommendation:** DO NOT rely on this for training. Use standard `content: null` pattern.

#### Claude Pattern (Combined Explicitly)

**Standard Behavior:**
```json
{
  "role": "assistant",
  "content": [
    {"type": "text", "text": "I'll search for that information."},
    {"type": "tool_use", "id": "t1", "name": "search", "input": {...}}
  ]
}
```

**This is the EXPECTED pattern for Claude.**

### 5.2 Reasoning Before Tool Use

#### Chain of Thought Prompting

**Technique:** Instruct model to think step-by-step before calling tools

**Prompt Example:**
```
Before selecting a tool, please:
1. Analyze what the user is asking for
2. Identify which tool(s) could help
3. Determine the required parameters
4. Explain your reasoning
5. Make the tool call

User: What's the population of the capital of France?
```

**Expected Response (Claude-style):**
```json
{
  "content": [
    {
      "type": "text",
      "text": "To answer this, I need to:\n1. First identify the capital of France (Paris)\n2. Then get population data for Paris\nI'll use the get_city_info tool with Paris as the location."
    },
    {
      "type": "tool_use",
      "name": "get_city_info",
      "input": {"city": "Paris", "country": "France"}
    }
  ]
}
```

**OpenAI Alternative (Reasoning Models):**
Use separate reasoning phase, then tool calls:
```python
# Step 1: Reasoning (hidden tokens)
reasoning_output = model.generate(prompt, include_reasoning=True)

# Step 2: Tool calls based on reasoning
tool_calls = extract_tool_calls(reasoning_output)

# Step 3: Execute and continue
results = execute_tools(tool_calls)
```

#### The "Think" Tool Pattern (Claude)

**Definition:**
```json
{
  "name": "think",
  "description": "Use this to pause and think about your next steps before calling other tools",
  "parameters": {
    "type": "object",
    "properties": {
      "thoughts": {
        "type": "string",
        "description": "Your step-by-step reasoning about what to do next"
      }
    },
    "required": ["thoughts"]
  }
}
```

**Usage:**
```json
{
  "content": [
    {
      "type": "tool_use",
      "name": "think",
      "input": {
        "thoughts": "I need to verify the user has permission before accessing their data. I should call check_permissions first, then if authorized, call get_user_data."
      }
    }
  ]
}
```

**Benefits:**
- Explicit reasoning visible to developers
- Helps with complex multi-step workflows
- Improves performance on policy-heavy tasks

### 5.3 Multi-Step Tool Calling Workflows

#### Sequential Pattern (Dependent Steps)

**Scenario:** Each tool call depends on previous results

```json
// Step 1: User request
{"role": "user", "content": "What's the weather in the capital of France?"}

// Step 2: Assistant calls tool to find capital
{
  "role": "assistant",
  "content": null,
  "tool_calls": [{
    "function": {"name": "get_capital", "arguments": "{\"country\": \"France\"}"}
  }]
}

// Step 3: Tool result
{"role": "tool", "content": "{\"capital\": \"Paris\"}"}

// Step 4: Assistant calls weather tool with result
{
  "role": "assistant",
  "content": null,
  "tool_calls": [{
    "function": {"name": "get_weather", "arguments": "{\"location\": \"Paris\"}"}
  }]
}

// Step 5: Weather result
{"role": "tool", "content": "{\"temp\": 22, \"condition\": \"sunny\"}"}

// Step 6: Final response
{
  "role": "assistant",
  "content": "The capital of France is Paris, where it's currently 22°C and sunny."
}
```

**Training Consideration:** Include multi-turn examples to teach dependency handling.

#### Parallel Pattern (Independent Steps)

**Scenario:** Multiple tools can run simultaneously

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {"id": "c1", "function": {"name": "get_weather", "arguments": "{\"location\": \"Paris\"}"}},
    {"id": "c2", "function": {"name": "get_weather", "arguments": "{\"location\": \"London\"}"}},
    {"id": "c3", "function": {"name": "get_weather", "arguments": "{\"location\": \"Tokyo\"}"}}
  ]
}

// All results returned in single batch
{"role": "tool", "tool_call_id": "c1", "content": "{\"temp\": 22}"}
{"role": "tool", "tool_call_id": "c2", "content": "{\"temp\": 18}"}
{"role": "tool", "tool_call_id": "c3", "content": "{\"temp\": 26}"}
```

**Training Consideration:** Include parallel examples to teach when parallelization is appropriate.

### 5.4 Error Handling and Edge Cases

#### Missing Parameters

**Training Example:**
```json
{
  "messages": [
    {"role": "user", "content": "What's the weather?"},
    {
      "role": "assistant",
      "content": "I'd be happy to check the weather for you, but I need to know which location you're interested in. Could you please specify a city?"
    }
  ]
}
```

**Key:** Teach model to ASK for missing required parameters instead of guessing or calling with incomplete data.

#### No Suitable Tool

**Training Example:**
```json
{
  "messages": [
    {"role": "user", "content": "What's the meaning of life?"},
    {
      "role": "assistant",
      "content": "This is a philosophical question that doesn't require tool use. I can discuss this based on various perspectives..."
    }
  ]
}
```

**Key:** Include examples where NO tool should be called.

#### Tool Execution Failure

**Training Example:**
```json
{
  "messages": [
    {"role": "user", "content": "What's the weather in Atlantis?"},
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [{"function": {"name": "get_weather", "arguments": "{\"location\": \"Atlantis\"}"}}]
    },
    {
      "role": "tool",
      "content": "{\"error\": \"Location not found\", \"isError\": true}"
    },
    {
      "role": "assistant",
      "content": "I couldn't find weather information for 'Atlantis' as it's not a recognized location. Could you provide a valid city name?"
    }
  ]
}
```

**Key:** Teach model to handle errors gracefully and explain to user.

---

## 6. Recommendations for Your Training Dataset

Based on this research, here are specific recommendations for your Toolset-Training project:

### 6.1 Current Format Analysis

Your current format (from CLAUDE.md):

```jsonl
{
  "conversations": [
    {"role": "user", "content": "User request"},
    {
      "role": "assistant",
      "content": "tool_call: toolName\narguments: {...}\n\nResult: {...}\n\nResponse"
    }
  ],
  "label": true
}
```

**Assessment:**
- ✅ Single-turn format is good (you removed multi-turn)
- ✅ Includes tool call, result, and response in one example
- ⚠️ Custom format (not standard OpenAI or Claude format)
- ⚠️ Combines call + result + response in single assistant message

### 6.2 Considerations for Format Choice

#### Option 1: Keep Current Format (Custom)

**Pros:**
- Already have 2,676 SFT examples in this format
- Model learns complete workflow in single turn
- Simpler for model to learn pattern
- Works with your current training pipeline

**Cons:**
- Not compatible with standard OpenAI/Claude APIs
- Model won't generalize to standard tool calling formats
- May confuse model about when to wait for tool results vs generate them

**Recommendation:** If you're training for a CUSTOM tool calling system (not OpenAI/Claude API compatible), this is fine.

#### Option 2: Migrate to OpenAI-Compatible Format

**Format:**
```jsonl
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "toolName", "arguments": "{...}"}}]}, {"role": "tool", "tool_call_id": "call_1", "content": "{...}"}, {"role": "assistant", "content": "Response"}], "tools": [...]}
```

**Pros:**
- Compatible with OpenAI API format
- Can use existing datasets (Glaive, xLAM) for augmentation
- Model works with standard tooling/frameworks
- Industry standard

**Cons:**
- Need to convert existing 2,676 examples
- More complex format (multiple messages per example)
- Requires tool definitions in each example

**Recommendation:** If you want model compatible with standard APIs.

#### Option 3: Hybrid - Claude-Style with Reasoning

**Format:**
```json
{
  "conversations": [
    {"role": "user", "content": "..."},
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "I'll use the toolName to..."},
        {
          "type": "tool_use",
          "id": "t1",
          "name": "toolName",
          "input": {"context": {...}, "param": "value"}
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "tool_result",
          "tool_use_id": "t1",
          "content": "{...}"
        }
      ]
    },
    {
      "role": "assistant",
      "content": "Response using result"
    }
  ]
}
```

**Pros:**
- Supports reasoning before tool calls
- Compatible with Claude API
- Clearer separation of steps
- Allows "think" tool pattern

**Cons:**
- Multi-turn (but with tool results as separate user message)
- More complex than current format
- Need to convert existing data

**Recommendation:** If you want Claude-style reasoning capabilities.

### 6.3 Specific Recommendation for Toolset-Training

Given your project specifics:

1. **Keep current format for now** - You have working SFT pipeline with 2,676 examples
2. **Add reasoning text** - Enhance examples with explanatory text before tool calls:
   ```json
   {
     "conversations": [
       {"role": "user", "content": "Create a note about machine learning"},
       {
         "role": "assistant",
         "content": "I'll create a new note in your vault using the contentManager.createNote tool.\n\ntool_call: contentManager.createNote\narguments: {\"context\": {...}, \"notePath\": \"ML Notes\", \"content\": \"...\"}\n\nResult: {\"success\": true, \"notePath\": \"ML Notes.md\"}\n\nI've created a new note about machine learning at 'ML Notes.md' in your vault."
       }
     ],
     "label": true
   }
   ```

3. **Ensure context object is always first** - You already do this (critical for your system)

4. **Add edge case examples:**
   - Missing sessionMemory → Error
   - Invalid tool name → Explanation to user
   - Missing required parameters → Ask user for info
   - No tool needed → Direct response

5. **Consider "reasoning" examples:**
   ```json
   {
     "conversations": [
       {"role": "user", "content": "Find all notes about Python and create a summary"},
       {
         "role": "assistant",
         "content": "I need to break this into steps:\n1. First, search for notes about Python\n2. Then, create a new note with the summary\n\nLet me start by searching:\n\ntool_call: vaultLibrarian.advancedSearch\narguments: {\"context\": {...}, \"query\": \"Python\", \"contentPattern\": \"*.md\"}\n\nResult: {\"matches\": [\"intro-to-python.md\", \"python-tips.md\"]}\n\nNow I'll create a summary note:\n\ntool_call: contentManager.createNote\narguments: {\"context\": {...}, \"notePath\": \"Python Summary\", \"content\": \"Summary of found notes...\"}\n\nResult: {\"success\": true}\n\nI've found 2 Python-related notes and created a summary in 'Python Summary.md'."
       }
     ],
     "label": true
   }
   ```

### 6.4 Format Evolution Path

**Phase 1 (Current):** Custom format with reasoning text added
- Keep working SFT pipeline
- Enhance with reasoning examples
- Add edge cases

**Phase 2 (Future):** Add parallel OpenAI-compatible format
- Convert subset of data to OpenAI format
- Test with standard APIs
- Maintain both formats

**Phase 3 (Optional):** Full migration to standard format
- Convert all data to OpenAI or Claude format
- Leverage community datasets (Glaive, xLAM)
- Full API compatibility

---

## 7. Key Takeaways

### 7.1 Format Compatibility Matrix

| Feature | OpenAI | Claude | MCP | Your Current |
|---------|--------|--------|-----|--------------|
| Text + Tool Call Same Response | ❌ (No) | ✅ (Yes) | N/A | ✅ (Yes) |
| Parallel Tool Calls | ✅ (Yes) | ✅ (Yes) | ✅ (Yes) | ❓ (Unknown) |
| Reasoning Before Call | ⚠️ (Hidden in o1) | ✅ (Explicit) | N/A | ✅ (Can add) |
| Tool Result Format | `tool` role | `tool_result` block | JSON-RPC | Inline |
| Arguments Format | JSON string | JSON object | JSON object | JSON object |
| Content When Tool Called | `null` | Array with blocks | N/A | String |

### 7.2 Training Data Best Practices Summary

1. **Choose format based on target deployment:**
   - OpenAI API compatibility → Use `content: null` + `tool_calls`
   - Claude API compatibility → Use content blocks array
   - Custom system → Design for your needs

2. **Include diverse scenarios:**
   - Single tool calls
   - Multiple tool selection
   - Parallel calls (if supported)
   - Sequential dependencies
   - Missing parameters
   - No tool needed
   - Error handling

3. **Reasoning text:**
   - Claude-style: Include explanatory text WITH tool calls
   - OpenAI-style: Separate tool calling from text responses
   - Custom: Your choice based on desired behavior

4. **Dataset size:**
   - Minimum: 1,000 examples (LoRA)
   - Recommended: 10,000-20,000 examples
   - Quality > Quantity

5. **Validation:**
   - Test with validator scripts
   - Check context object requirements
   - Verify JSON schema compliance
   - Ensure label consistency (if using KTO)

### 7.3 Critical Don'ts

1. ❌ **Don't mix formats** - Pick one format and be consistent
2. ❌ **Don't assume text + tool_calls works** - Only reliable in Claude
3. ❌ **Don't skip edge cases** - Include error handling and "no tool" examples
4. ❌ **Don't ignore model-specific quirks:**
   - OpenAI: `content: null` required
   - Claude: Content is array
   - Llama: Specific token patterns
5. ❌ **Don't forget tool definitions** - Include in training data (OpenAI requires it)

---

## 8. References and Resources

### 8.1 Official Documentation

**OpenAI:**
- Function Calling Guide: https://platform.openai.com/docs/guides/function-calling
- Fine-tuning Cookbook: https://cookbook.openai.com/examples/fine_tuning_for_function_calling
- Reasoning Models: https://cookbook.openai.com/examples/reasoning_function_calls

**Anthropic Claude:**
- Tool Use Overview: https://docs.claude.com/en/docs/agents-and-tools/tool-use/overview
- Chain of Thought: https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/chain-of-thought
- Think Tool: https://www.anthropic.com/engineering/claude-think-tool

**Model Context Protocol:**
- MCP Specification: https://modelcontextprotocol.io/specification/2025-06-18/server/tools
- GitHub Repository: https://github.com/modelcontextprotocol/modelcontextprotocol

### 8.2 Training Datasets

**Glaive Function Calling v2:**
- Hugging Face: https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2
- Size: 113,000 examples
- License: Apache 2.0

**Salesforce xLAM:**
- Hugging Face: https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k
- Size: 60,000 examples
- Blog: https://www.salesforce.com/blog/large-action-model-ai-agent/

**Other Notable Datasets:**
- Llama 3 Function Calling: https://github.com/michaelnny/Llama3-FunctionCalling
- Trelis Llama 3 Fine-tuned: https://huggingface.co/Trelis/Meta-Llama-3-8B-Instruct-function-calling

### 8.3 Tools and Frameworks

**Fine-tuning Tools:**
- Unsloth: https://docs.unsloth.ai/
- Axolotl: https://github.com/axolotl-ai-cloud/axolotl
- TRL (Transformers Reinforcement Learning): https://github.com/huggingface/trl

**Evaluation:**
- vLLM: https://docs.vllm.ai/en/stable/features/tool_calling.html
- LlamaIndex: https://docs.llamaindex.ai/en/stable/examples/finetuning/openai_fine_tuning_functions/

### 8.4 Research Papers and Articles

- Meta AI: "How to Fine-tune: Focus on Effective Datasets" (2024)
- Salesforce: "APIGen - Automated Pipeline for Generating Verifiable Datasets" (2024)
- Microsoft: "Fine-Tuning Small Language Models for Function-Calling" (2024)
- Anthropic: "Reasoning Models Don't Always Say What They Think" (2024)

---

## 9. Conclusion

The landscape of tool calling in LLMs is **not standardized**, with different providers taking different approaches:

- **OpenAI:** Strict separation of text and tool calls (mostly)
- **Claude:** Flexible combination of text and tool use blocks
- **MCP:** Protocol-level abstraction (model-agnostic)

For **fine-tuning**, the key decision is:

1. **Target deployment environment** - Which API/format will model use in production?
2. **Desired behavior** - Should model reason before calling tools?
3. **Dataset availability** - Can you leverage existing datasets?

**For your Toolset-Training project:**
- Current custom format is FINE for isolated deployment
- Consider adding reasoning text for better interpretability
- Future migration to OpenAI format would enable broader compatibility
- Focus on edge cases and error handling in training data

**Remember:** Models learn what they see in training data. If you want reasoning before tool calls, SHOW it in examples. If you want clean separation, ENFORCE it in format. Quality and consistency matter more than format choice.

---

**Document Version:** 1.0
**Last Updated:** November 18, 2025
**Next Review:** When new major LLM updates release or before next training run
