#!/usr/bin/env python3
"""
Test Mistral chat template on the entire dataset.
This simulates what the notebook does to catch any formatting errors.
"""
import json
import sys
from jinja2 import Template, TemplateError

# Mistral chat template (same as notebook)
MISTRAL_CHAT_TEMPLATE = """{{ bos_token }}{% for message in messages %}{% if message['role'] == 'system' %}{% if loop.index == 1 %}{{ message['content'] + ' ' }}{% endif %}{% elif message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + eos_token }}{% endif %}{% endfor %}"""

# Compile the template
template = Template(MISTRAL_CHAT_TEMPLATE)

# Mock tokens for testing
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"

def test_example(line_num, conversations):
    """Test a single example with the Mistral template."""
    try:
        # Check role alternation first
        roles = [msg['role'] for msg in conversations]

        # Validation 1: Must start with user (or system+user)
        if roles and roles[0] not in ['user', 'system']:
            return f"❌ First message must be 'user' or 'system', got '{roles[0]}'"

        # Validation 2: No consecutive same roles (except system at start)
        for i in range(len(roles) - 1):
            if roles[i] == roles[i+1] and roles[i] != 'system':
                return f"❌ Consecutive '{roles[i]}' at positions {i} and {i+1}"
            # System can only be first
            if roles[i] == 'system' and i > 0:
                return f"❌ System message at position {i} (must be first)"

        # Validation 3: After optional system, must alternate user/assistant
        start_idx = 1 if roles and roles[0] == 'system' else 0
        for i in range(start_idx, len(roles)):
            expected = 'user' if (i - start_idx) % 2 == 0 else 'assistant'
            if roles[i] != expected:
                return f"❌ Expected '{expected}' at position {i}, got '{roles[i]}'"

        # Validation 4: Apply the template
        result = template.render(
            messages=conversations,
            bos_token=BOS_TOKEN,
            eos_token=EOS_TOKEN
        )

        # Validation 5: Check result is not empty
        if not result or result == BOS_TOKEN:
            return f"❌ Template produced empty/invalid output"

        return "✅ OK"

    except TemplateError as e:
        return f"❌ Template error: {str(e)}"
    except KeyError as e:
        return f"❌ Missing key: {str(e)}"
    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"

def main():
    filename = 'syngen_tools_sft_11.22.25.jsonl'

    print(f"Testing Mistral chat template on {filename}")
    print("=" * 70)
    print()

    errors = []
    warnings = []
    total = 0

    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                conversations = data.get('conversations', [])

                if not conversations:
                    warnings.append(f"Line {line_num}: No conversations found")
                    continue

                result = test_example(line_num, conversations)

                if result.startswith("❌"):
                    errors.append(f"Line {line_num}: {result}")
                    # Print errors immediately for visibility
                    print(f"Line {line_num}: {result}")

                total += 1

                # Progress indicator every 1000 lines
                if line_num % 1000 == 0:
                    print(f"✓ Processed {line_num} lines...", file=sys.stderr)

            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: JSON parse error - {e}")
                print(f"Line {line_num}: ❌ JSON parse error - {e}")
            except Exception as e:
                errors.append(f"Line {line_num}: Unexpected error - {e}")
                print(f"Line {line_num}: ❌ Unexpected error - {e}")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total examples: {total}")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")
    print()

    if errors:
        print("❌ FAILED - Found errors:")
        print()
        for error in errors[:20]:  # Show first 20 errors
            print(f"  {error}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors")
        sys.exit(1)
    else:
        print("✅ SUCCESS - All examples pass Mistral template validation!")
        print()
        print("Your dataset is ready for training with the SFT notebook.")
        sys.exit(0)

if __name__ == '__main__':
    main()
