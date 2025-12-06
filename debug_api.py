#!/usr/bin/env python3
"""Debug OpenRouter API response."""

import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path("Datasets/improvement_engine/.env")
load_dotenv(env_path)

api_key = os.getenv("OPENROUTER_API_KEY")
model = os.getenv("OPENROUTER_MODEL", "openai/gpt-5-mini")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/ProfSynapse/Toolset-Training",
    "X-Title": "Dataset Improvement Engine"
}

# Simple test thinking block
thinking_block = {
    "goal": "Create summary of meeting",
    "memory": "Previously reviewed similar documents to understand content structure.",
    "requirements": [
        "Read meeting notes",
        "Reading file content to gather necessary information and context."
    ],
    "assessment": {
        "complex": False,
        "risky": False
    },
    "confidence": 0.97,
    "plan": [
        "Read meeting notes",
        "Reading file content to gather necessary information and context."
    ]
}

messages = [
    {
        "role": "system",
        "content": "You are a dataset quality expert. Improve the thinking block to make requirements distinct from plan."
    },
    {
        "role": "user",
        "content": f"Improve this thinking block:\n\n{json.dumps(thinking_block, indent=2)}"
    }
]

payload = {
    "model": model,
    "messages": messages,
    "temperature": 0.3,
    "max_tokens": 800,
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "thinking_block",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string"},
                    "memory": {"type": "string"},
                    "requirements": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "assessment": {
                        "type": "object",
                        "properties": {
                            "complex": {"type": "boolean"},
                            "risky": {"type": "boolean"}
                        },
                        "required": ["complex", "risky"],
                        "additionalProperties": False
                    },
                    "confidence": {"type": "number"},
                    "plan": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["goal", "memory", "requirements", "assessment", "confidence", "plan"],
                "additionalProperties": False
            }
        }
    }
}

print("Sending request to OpenRouter...")
print(f"Model: {model}")
print()

response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers=headers,
    json=payload,
    timeout=60
)

print(f"Status code: {response.status_code}")
print()

data = response.json()

# Print full response for debugging
print("Full response:")
print(json.dumps(data, indent=2))
print()

# Try to extract content
try:
    content = data["choices"][0]["message"]["content"]
    print(f"Content: {content}")
    print(f"Content type: {type(content)}")
    print(f"Content length: {len(content) if content else 0}")

    if content:
        parsed = json.loads(content)
        print("\nParsed JSON:")
        print(json.dumps(parsed, indent=2))
except Exception as e:
    print(f"Error: {e}")
