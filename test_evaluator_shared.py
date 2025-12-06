#!/usr/bin/env python3
"""Test Evaluator with shared LLM adapters."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv(Path(".env"))

from Evaluator.client_factory import create_client_from_args
from Evaluator.enums import BackendType

print("\n" + "="*60)
print("Testing Evaluator with Shared LLM Adapters")
print("="*60 + "\n")

# Test with LM Studio
print("1. Testing LM Studio adapter...")
print(f"   Host: {os.getenv('LMSTUDIO_HOST', 'localhost')}")

try:
    client = create_client_from_args(
        backend=BackendType.LMSTUDIO,
        model="local-model",
        temperature=0.7,
        max_tokens=100
    )

    # Test connection
    if client.is_server_running():
        print("   ✅ LM Studio server is running")

        # Test chat
        response = client.chat([
            {"role": "user", "content": "Say 'Hello from shared adapter' and nothing else"}
        ])

        print(f"   ✅ Chat response: {response.message[:80]}...")
        print(f"   ⏱  Latency: {response.latency_s:.2f}s")
    else:
        print("   ⚠  LM Studio server not accessible (this is OK if not running)")

except Exception as e:
    print(f"   ❌ Error: {e}")

print()

# Test with Ollama (if available)
print("2. Testing Ollama adapter...")
print(f"   Host: {os.getenv('OLLAMA_HOST', 'localhost')}")

try:
    client = create_client_from_args(
        backend=BackendType.OLLAMA,
        model="llama2",
        temperature=0.7,
        max_tokens=100
    )

    # Test connection
    if client.is_server_running():
        print("   ✅ Ollama server is running")

        # Test chat
        response = client.chat([
            {"role": "user", "content": "Say 'Hello from shared adapter' and nothing else"}
        ])

        print(f"   ✅ Chat response: {response.message[:80]}...")
        print(f"   ⏱  Latency: {response.latency_s:.2f}s")
    else:
        print("   ⚠  Ollama server not accessible (this is OK if not running)")

except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "="*60)
print("Testing complete!")
print("="*60 + "\n")
