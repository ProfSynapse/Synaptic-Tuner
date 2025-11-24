#!/usr/bin/env python3
"""
Standalone script to test dataset formatting with proper tool call ID handling.
This script fixes the TemplateError by converting long OpenAI-style tool call IDs
to 9-character alphanumeric IDs that Mistral expects.

Usage:
    python test_dataset_formatting.py
"""

import hashlib
import json
from datasets import load_dataset
from transformers import AutoTokenizer

# Configuration
DATASET_NAME = "professorsynapse/claudesidian-synthetic-dataset"
DATASET_FILE = "syngen_tools_sft_11.23.25_workspace_aware.jsonl"
MODEL_NAME = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"

def shorten_tool_call_id(original_id: str) -> str:
    """
    Convert long OpenAI-style tool call IDs to 9-character alphanumeric strings.

    Uses hash of original ID to ensure consistency and uniqueness.
    Example: "call_adPXqyEdL1B2SsEaQ7PndqUm" -> "a3f9d2c1e"
    """
    # Hash the original ID and take first 9 characters (alphanumeric)
    hash_hex = hashlib.md5(original_id.encode()).hexdigest()
    return hash_hex[:9]

def fix_tool_call_ids(example):
    """
    Fix tool call IDs in a single example to be 9-character alphanumeric strings.

    Modifies the example in place, converting any long tool call IDs in
    assistant messages to the format Mistral expects.
    """
    for message in example.get("conversations", []):
        if message.get("role") == "assistant" and "tool_calls" in message:
            tool_calls = message.get("tool_calls")
            if tool_calls is not None:
                for tool_call in tool_calls:
                    if "id" in tool_call and len(tool_call["id"]) != 9:
                        # Convert long ID to 9-char format
                        original_id = tool_call["id"]
                        tool_call["id"] = shorten_tool_call_id(original_id)
                        print(f"  Fixed ID: {original_id} -> {tool_call['id']}")

    return example

def format_chat_template(example, tokenizer):
    """
    Convert conversations to tokenizer's chat template.

    Input: {"conversations": [{"role": "user", "content": "..."}, ...]}
    Output: {"text": "[INST] user [/INST] assistant"}
    """
    text = tokenizer.apply_chat_template(
        example["conversations"],
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

def main():
    print("=" * 70)
    print("DATASET FORMATTING TEST")
    print("=" * 70)
    print()

    # Step 1: Load dataset
    print("📁 Loading dataset...")
    print(f"   Repository: {DATASET_NAME}")
    print(f"   File: {DATASET_FILE}")

    dataset = load_dataset(
        DATASET_NAME,
        data_files=DATASET_FILE,
        split="train"
    )

    print(f"✓ Loaded {len(dataset)} examples")
    print()

    # Step 2: Show original example
    print("📋 Original example (first one):")
    print("-" * 70)
    original_example = dataset[0]
    print(json.dumps(original_example, indent=2)[:500])
    print("...")
    print()

    # Check for tool call IDs
    has_tool_calls = False
    for msg in original_example.get("conversations", []):
        if "tool_calls" in msg and msg["tool_calls"] is not None:
            has_tool_calls = True
            tool_calls = msg["tool_calls"]
            print(f"🔍 Found {len(tool_calls)} tool call(s)")
            for tc in tool_calls:
                original_id = tc.get("id", "N/A")
                print(f"   Tool Call ID: {original_id} (length: {len(original_id)})")
            break

    if not has_tool_calls:
        print("⚠️  No tool calls found in first example")
    print()

    # Step 3: Fix tool call IDs
    print("🔧 Fixing tool call IDs...")
    dataset = dataset.map(
        fix_tool_call_ids,
        desc="Fixing tool call IDs"
    )
    print("✓ Tool call IDs fixed")
    print()

    # Step 4: Load tokenizer
    print("🤖 Loading tokenizer...")
    print(f"   Model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Set Mistral chat template if needed
    MISTRAL_CHAT_TEMPLATE = """{{ bos_token }}{% for message in messages %}{% if message['role'] == 'system' %}{% if loop.index == 1 %}{{ message['content'] + ' ' }}{% endif %}{% elif message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + eos_token }}{% endif %}{% endfor %}"""

    if tokenizer.chat_template is None:
        print("   Setting Mistral chat template")
        tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE
    else:
        print("   Using existing chat template")

    print("✓ Tokenizer ready")
    print()

    # Step 5: Format dataset
    print("📝 Formatting dataset...")

    try:
        formatted_dataset = dataset.map(
            lambda ex: format_chat_template(ex, tokenizer),
            remove_columns=dataset.column_names,
            desc="Formatting with chat template"
        )

        print("✓ Dataset formatted successfully!")
        print()

        # Step 6: Show formatted example
        print("📄 Formatted example (first 500 chars):")
        print("-" * 70)
        print(formatted_dataset[0]["text"][:500])
        print("...")
        print()

        # Step 7: Summary
        print("=" * 70)
        print("✅ SUCCESS!")
        print("=" * 70)
        print(f"Total examples: {len(formatted_dataset)}")
        print(f"Columns: {formatted_dataset.column_names}")
        print()
        print("💡 The dataset is now ready for training!")
        print()
        print("Next steps:")
        print("1. Copy the fix_tool_call_ids() function to your notebook")
        print("2. Apply it BEFORE format_chat_template():")
        print()
        print("   dataset = dataset.map(fix_tool_call_ids)")
        print("   dataset = dataset.map(format_chat_template, ...)")
        print()

        return True

    except Exception as e:
        print("❌ ERROR during formatting:")
        print(f"   {type(e).__name__}: {e}")
        print()
        print("This indicates the tool call ID fix didn't work as expected.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
