## Shared LLM Client System

Unified interface to multiple LLM providers used throughout the codebase.

### **Supported Providers**

| Provider | Type | Use Case |
|----------|------|----------|
| **OpenRouter** | Cloud | Access to GPT-5-mini, Claude, etc. via API |
| **OpenAI Responses** | Cloud | Direct OpenAI Responses API for text and structured output |
| **LM Studio** | Local | Run models locally on your machine |
| **Ollama** | Local | Run models locally via Ollama |

### **Quick Start**

```python
from shared.llm import create_client

# Auto-detect from environment variables
client = create_client()

# Simple chat
response = client.chat([
    {"role": "user", "content": "Hello!"}
])

# Structured output with JSON schema
schema = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"}
    },
    "required": ["answer", "confidence"]
}
result = client.structured_output(messages, schema)
```

### **Configuration**

Set environment variables in `.env`:

```bash
# Default provider for improvement engine
IMPROVEMENT_BACKEND=openrouter  # or openai_responses, lmstudio, ollama
IMPROVEMENT_MODEL=openai/gpt-5-mini

# OpenRouter (cloud)
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# OpenAI Responses (cloud)
OPENAI_API_KEY=sk-your-key-here
OPENAI_RESPONSES_BASE_URL=https://api.openai.com/v1
OPENAI_RESPONSES_TIMEOUT_SECONDS=60
OPENAI_RESPONSES_STORE=false
OPENAI_RESPONSES_STRUCTURED_OUTPUT_STRICT=false

# LM Studio (local)
LMSTUDIO_HOST=localhost
LMSTUDIO_PORT=1234

# Ollama (local)
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
```

### **Usage Examples**

#### **1. OpenRouter (Cloud)**

```bash
# .env
IMPROVEMENT_BACKEND=openrouter
IMPROVEMENT_MODEL=openai/gpt-5-mini
OPENROUTER_API_KEY=sk-or-v1-...
```

```python
from shared.llm import create_client

client = create_client()
response = client.chat([{"role": "user", "content": "Hello!"}])
```

#### **2. OpenAI Responses (Cloud)**

```bash
# .env
IMPROVEMENT_BACKEND=openai_responses
IMPROVEMENT_MODEL=gpt-5-mini
OPENAI_API_KEY=sk-...
```

```python
from shared.llm import create_client

client = create_client()
response = client.chat([{"role": "user", "content": "Hello!"}])
```

The provider calls `POST /v1/responses`, sends `store: false` by default, maps the shared `max_tokens` argument to `max_output_tokens`, and uses `text.format` for JSON Schema structured output. Structured-output strict mode defaults to false for schema compatibility and can be enabled with `OPENAI_RESPONSES_STRUCTURED_OUTPUT_STRICT=true` or explicit config.

#### **3. LM Studio (Local)**

```bash
# .env
IMPROVEMENT_BACKEND=lmstudio
IMPROVEMENT_MODEL=local-model
LMSTUDIO_HOST=localhost  # Or IP if on Windows from WSL
LMSTUDIO_PORT=1234
```

```python
from shared.llm import create_client

client = create_client()
response = client.chat([{"role": "user", "content": "Hello!"}])
```

#### **4. Ollama (Local)**

```bash
# .env
IMPROVEMENT_BACKEND=ollama
IMPROVEMENT_MODEL=llama2
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
```

```python
from shared.llm import create_client

client = create_client()
response = client.chat([{"role": "user", "content": "Hello!"}])
```

#### **5. Explicit Configuration**

```python
from shared.llm import create_client

# Override environment variables
client = create_client(
    provider="ollama",
    model="mistral"
)

# Or use different env prefix
client = create_client(env_prefix="EVAL")  # Uses EVAL_BACKEND, EVAL_MODEL
```

### **Adding New Providers**

Super easy! Just implement the `BaseLLMClient` interface:

```python
# shared/llm/providers/new_provider.py
from ..base import BaseLLMClient
from ..exceptions import LLMConnectionError, LLMResponseError

class NewProviderClient(BaseLLMClient):
    @property
    def provider_name(self) -> str:
        return "newprovider"

    @property
    def model_name(self) -> str:
        return self.model

    def chat(self, messages, temperature=0.7, max_tokens=1024, **kwargs) -> str:
        # Implement chat logic
        pass

    def structured_output(self, messages, schema, temperature=0.3, max_tokens=2048, **kwargs) -> dict:
        # Implement structured output logic
        pass

    def test_connection(self) -> bool:
        # Implement connection test
        pass
```

Then register in `factory.py`:

```python
from .providers.new_provider import NewProviderClient

def create_client(...):
    # ...
    elif config.provider == "newprovider":
        return NewProviderClient(...)
```

### **Architecture**

```
shared/llm/
├── __init__.py          # Public API
├── base.py              # BaseLLMClient interface
├── config.py            # Configuration classes
├── exceptions.py        # Custom exceptions
├── factory.py           # Client factory
└── providers/
    ├── openrouter.py
    ├── lmstudio.py
    └── ollama.py
```

### **API Reference**

#### **create_client()**

```python
def create_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    config: Optional[LLMConfig] = None,
    env_prefix: str = "IMPROVEMENT"
) -> BaseLLMClient
```

Create an LLM client based on configuration.

#### **BaseLLMClient Methods**

```python
# Simple chat
response: str = client.chat(
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 1024,
    **kwargs
)

# Structured output
result: Dict = client.structured_output(
    messages: List[Dict[str, str]],
    schema: Dict[str, Any],
    temperature: float = 0.3,
    max_tokens: int = 2048,
    **kwargs
)

# Test connection
is_connected: bool = client.test_connection()

# Properties
provider_name: str = client.provider_name
model_name: str = client.model_name
```

### **Notes**

- **Structured output support varies by provider**:
  - OpenRouter: Full JSON Schema support via Chat Completions response format
  - OpenAI Responses: JSON Schema support via Responses `text.format`
  - LM Studio: Basic JSON object format
  - Ollama: JSON format hint

- **WSL + LM Studio**: If running LM Studio on Windows, enable "Serve on Local Network" and set `LMSTUDIO_HOST` to the Windows IP address.

- **Error handling**: All methods raise `LLMError` subclasses:
  - `LLMConnectionError` - Connection/network issues
  - `LLMResponseError` - Invalid/malformed responses
  - `LLMConfigError` - Configuration problems
