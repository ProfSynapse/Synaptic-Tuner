#!/usr/bin/env python3
"""Demo of the shared LLM client system."""

from pathlib import Path
from dotenv import load_dotenv

# Load .env from improvement_engine
env_path = Path("Datasets/improvement_engine/.env")
load_dotenv(env_path)

from shared.llm import create_client, list_providers

print("="*60)
print("Shared LLM Client Demo")
print("="*60)

# Show available providers
print(f"\nAvailable providers: {', '.join(list_providers())}")

# Create client (auto-detects from env: IMPROVEMENT_BACKEND, IMPROVEMENT_MODEL)
print("\nCreating client from environment variables...")
client = create_client()

print(f"  Provider: {client.provider_name}")
print(f"  Model: {client.model_name}")
if hasattr(client, 'base_url'):
    print(f"  URL: {client.base_url}")

# Test connection
print("\nTesting connection...")
if client.test_connection():
    print("  ✓ Connection successful!")
else:
    print("  ✗ Connection test failed (continuing anyway...)")

# Example 1: Simple chat
print("\n" + "="*60)
print("Example 1: Simple Chat")
print("="*60)

messages = [
    {"role": "user", "content": "Say 'Hello!' in exactly one word."}
]

response = client.chat(messages, temperature=0.3, max_tokens=10)
print(f"Response: {response}")

# Example 2: Structured output
print("\n" + "="*60)
print("Example 2: Structured Output (JSON Schema)")
print("="*60)

# Define a simple schema
schema = {
    "type": "object",
    "properties": {
        "greeting": {
            "type": "string",
            "description": "A friendly greeting"
        },
        "language": {
            "type": "string",
            "description": "The language of the greeting"
        },
        "enthusiasm": {
            "type": "integer",
            "description": "Enthusiasm level from 1-10"
        }
    },
    "required": ["greeting", "language", "enthusiasm"],
    "additionalProperties": False
}

messages = [
    {"role": "system", "content": "You are a friendly greeter. Return structured JSON matching the schema."},
    {"role": "user", "content": "Give me an enthusiastic greeting in Spanish!"}
]

try:
    result = client.structured_output(messages, schema, temperature=0.7)
    print(f"Structured result:")
    print(f"  Greeting: {result['greeting']}")
    print(f"  Language: {result['language']}")
    print(f"  Enthusiasm: {result['enthusiasm']}/10")
except Exception as e:
    print(f"  Note: Structured output may not work with all providers")
    print(f"  Error: {e}")

print("\n" + "="*60)
print("Demo Complete!")
print("="*60)
print("\nTo switch providers, set environment variables:")
print("  IMPROVEMENT_BACKEND=lmstudio")
print("  IMPROVEMENT_MODEL=local-model")
print("\nOr create client explicitly:")
print("  client = create_client(provider='ollama', model='llama2')")
