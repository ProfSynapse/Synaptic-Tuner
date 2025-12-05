#!/usr/bin/env python3
"""
Test script for SelfPlay generation with LM Studio.

Tests the 3-prompt pipeline to verify LM Studio integration works.
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from Evaluator.lmstudio_client import LMStudioClient
from Evaluator.config import LMStudioSettings
from SelfPlay.generator import SelfPlayGenerator
from validate_syngen import validate_example


def test_lmstudio_connection():
    """Test if LM Studio is accessible."""
    print("=" * 70)
    print("TESTING LM STUDIO CONNECTION")
    print("=" * 70)

    settings = LMStudioSettings(model="local-model")
    client = LMStudioClient(settings=settings)

    print(f"\nConnecting to: {settings.base_url()}")

    # Check if server is running
    if not client.is_server_running():
        print("\n❌ LM Studio server is not accessible!")
        print("\nMake sure:")
        print("  1. LM Studio is running")
        print("  2. A model is loaded")
        print("  3. Server is started (click 'Start Server' in LM Studio)")
        return None

    print("✅ LM Studio server is accessible!")

    # List available models
    print("\n📋 Available models:")
    try:
        models = client.list_models()
        if models:
            for i, model in enumerate(models, 1):
                print(f"  {i}. {model}")
        else:
            print("  (No models listed - using default)")
    except Exception as e:
        print(f"  Could not list models: {e}")

    # Test simple chat
    print("\n🧪 Testing simple chat...")
    try:
        # Update settings for this test
        client.settings.temperature = 0.7
        client.settings.max_tokens = 50

        messages = [{"role": "user", "content": "Say 'Hello, SelfPlay!' and nothing else."}]
        response = client.chat(messages)
        print(f"  Response: {response.message[:100]}")
        print("  ✅ Chat works!")
        return client
    except Exception as e:
        print(f"  ❌ Chat failed: {e}")
        return None


def test_prompt_1(client):
    """Test Prompt 1: Generate environment."""
    print("\n" + "=" * 70)
    print("TESTING PROMPT 1: GENERATE ENVIRONMENT")
    print("=" * 70)

    generator = SelfPlayGenerator(model_client=client)

    # Select a variation
    variation = generator.select_environment_variation()
    print(f"\n📝 Selected variation: {variation['name']}")
    print(f"   Description: {variation['config']['description']}")

    # Get the prompt
    prompt = variation['config']['prompt']
    print(f"\n📤 Sending prompt ({len(prompt)} chars)...")
    print(f"   First 200 chars: {prompt[:200]}...")

    try:
        # Call LLM
        client.settings.temperature = 0.9
        client.settings.max_tokens = 500

        messages = [{"role": "user", "content": prompt}]
        response = client.chat(messages)

        print("\n📥 Response received!")
        print("-" * 70)
        print(response.message)
        print("-" * 70)

        return response.message
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        return None


def test_prompt_2_tool(client, session_context):
    """Test Prompt 2: Generate user request (tool-based)."""
    print("\n" + "=" * 70)
    print("TESTING PROMPT 2: GENERATE USER REQUEST (TOOL-BASED)")
    print("=" * 70)

    generator = SelfPlayGenerator(model_client=client)

    # Select random agent
    import random
    agent_name = random.choice(list(generator.agents_config["agents"].keys()))
    agent_config = generator.agents_config["agents"][agent_name]

    # Select random tool from that agent
    tool = random.choice(agent_config["tools"])

    print(f"\n🎯 Selected agent: {agent_name}")
    print(f"   Tool: {tool}")

    # Build prompt
    base_template = generator.user_tool_config["base_template"]
    prompt = base_template.format(
        session_context=session_context,
        agent_name=agent_name,
        tool_list=tool
    )

    print(f"\n📤 Sending prompt...")
    print(f"   Prompt length: {len(prompt)} chars")

    try:
        client.settings.temperature = 0.7
        client.settings.max_tokens = 150

        messages = [{"role": "user", "content": prompt}]
        response = client.chat(messages)

        print("\n📥 Generated user request:")
        print("-" * 70)
        print(response.message)
        print("-" * 70)

        return response.message
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        return None


def test_prompt_3(client, session_context, user_request):
    """Test Prompt 3: Generate assistant response."""
    print("\n" + "=" * 70)
    print("TESTING PROMPT 3: GENERATE ASSISTANT RESPONSE")
    print("=" * 70)

    print(f"\n📝 User request: {user_request[:100]}...")
    print(f"\n📤 Sending to model WITH session context as system prompt...")

    try:
        client.settings.temperature = 0.5
        client.settings.max_tokens = 500

        # INCLUDE session context as system prompt!
        # This gives the model the workspace structure and context
        messages = [
            {"role": "system", "content": session_context},
            {"role": "user", "content": user_request}
        ]
        response = client.chat(messages)

        print("\n📥 Assistant response:")
        print("-" * 70)
        print(response.message)
        print("-" * 70)

        return response.message
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        return None


def test_full_pipeline():
    """Test the complete 3-prompt pipeline."""
    print("\n" + "=" * 70)
    print("FULL PIPELINE TEST")
    print("=" * 70)

    # Step 1: Connect to LM Studio
    client = test_lmstudio_connection()
    if not client:
        print("\n❌ Cannot proceed without LM Studio connection")
        return

    # Step 2: Generate environment
    workspace_env = test_prompt_1(client)
    if not workspace_env:
        print("\n❌ Prompt 1 failed, cannot continue")
        return

    # Build session context
    from SelfPlay.id_utils import generate_ids
    session_id, workspace_id = generate_ids()

    session_context = f"""<session_context>
IMPORTANT: When using tools, include these values in your tool call parameters:

- sessionId: "{session_id}"
- workspaceId: "{workspace_id}" (current workspace)

Include these in the "context" parameter of your tool calls.
</session_context>

<current_workspace>
{workspace_env}
</current_workspace>"""

    print("\n" + "=" * 70)
    print("SESSION CONTEXT BUILT")
    print("=" * 70)
    print(f"Session ID:  {session_id}")
    print(f"Workspace ID: {workspace_id}")

    # Step 3: Generate user request
    user_request = test_prompt_2_tool(client, session_context)
    if not user_request:
        print("\n❌ Prompt 2 failed, cannot continue")
        return

    # Step 4: Generate assistant response
    assistant_response = test_prompt_3(client, session_context, user_request)
    if not assistant_response:
        print("\n❌ Prompt 3 failed")
        return

    # Final summary
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 70)

    final_example = {
        "conversations": [
            {"role": "system", "content": session_context},
            {"role": "user", "content": user_request},
            {"role": "assistant", "content": assistant_response}
        ],
        "metadata": {
            "session_id": session_id,
            "workspace_id": workspace_id,
            "test": True
        }
    }

    print("\n📦 Final JSONL example:")
    print(json.dumps(final_example, indent=2))

    # Validate the example
    print("\n" + "=" * 70)
    print("VALIDATING EXAMPLE")
    print("=" * 70)

    report = validate_example(0, final_example)

    if report.is_valid:
        print("\n✅ VALIDATION PASSED!")
    else:
        print("\n❌ VALIDATION FAILED!")

    # Show all issues
    if report.issues:
        print(f"\n📋 Validation issues ({len(report.issues)}):")
        for issue in report.issues:
            icon = "❌" if issue.level == "ERROR" else "⚠️ "
            print(f"  {icon} [{issue.level}] {issue.message}")
    else:
        print("\n✨ No validation issues found!")

    # Save to file
    output_file = Path(__file__).parent / "test_output.jsonl"
    with open(output_file, 'w') as f:
        f.write(json.dumps(final_example) + '\n')

    print(f"\n💾 Saved to: {output_file}")

    # Return validation status
    return report.is_valid


if __name__ == "__main__":
    try:
        test_full_pipeline()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
