#!/usr/bin/env python3
"""
Test script for destructive confirmation generator.

Generates sample examples without requiring LLM client (uses fallback responses).
"""

import json
from pathlib import Path
from destructive_confirmation_generator import DestructiveConfirmationGenerator


def main():
    """Generate and display sample destructive confirmation examples."""

    print("="*80)
    print("DESTRUCTIVE CONFIRMATION GENERATOR - TEST")
    print("="*80)

    # Initialize generator (no model client = uses fallbacks)
    generator = DestructiveConfirmationGenerator()

    # Generate 3 sample examples
    print("\nGenerating 3 sample examples using fallback responses...\n")

    for i in range(3):
        print(f"\n{'='*80}")
        print(f"EXAMPLE {i+1}")
        print(f"{'='*80}")

        example = generator.generate_full_conversation()

        # Display conversation
        for turn_idx, turn in enumerate(example["conversations"], 1):
            role = turn["role"].upper()
            content = turn["content"]

            print(f"\n--- Turn {turn_idx}: {role} ---")
            print(content)

        # Display metadata
        print(f"\n--- METADATA ---")
        print(f"Tool: {example['metadata']['tool_name']}")
        print(f"Agent: {example['metadata']['agent_name']}")
        print(f"Risk Level: {example['metadata']['risk_level']}")
        print(f"Turn Count: {example['metadata']['turn_count']}")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print("\nKey observations:")
    print("1. Turn 1: User makes vague destructive request")
    print("2. Turn 2: Assistant asks for confirmation (lists items)")
    print("3. Turn 3: User confirms explicitly")
    print("4. Turn 4: Assistant executes with HIGH RISK + HIGH CONFIDENCE")
    print("\nThis pattern teaches the model:")
    print("  ✓ Never execute destructive ops without confirmation")
    print("  ✓ Always list what will be affected")
    print("  ✓ High risk + user confirmation = High confidence")


if __name__ == "__main__":
    main()
