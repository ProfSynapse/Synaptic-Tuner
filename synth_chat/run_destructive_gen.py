#!/usr/bin/env python3
"""
Runner for generating destructive confirmation examples.

This script generates multi-turn training examples showing proper handling
of destructive operations with user confirmation.

Usage:
    # Generate 50 examples
    python run_destructive_gen.py --count 50 --output destructive_confirmations.jsonl

    # With LM Studio
    python run_destructive_gen.py --count 100 --backend lmstudio --model your-model

    # Just test (10 examples, fallback mode)
    python run_destructive_gen.py --test
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from destructive_confirmation_generator import DestructiveConfirmationGenerator

# Try to import LLM client (optional)
try:
    from shared.llm import create_client
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("Warning: shared.llm not available. Using fallback mode.")


def setup_llm_client(backend: Optional[str], model: Optional[str]):
    """
    Setup LLM client if requested.

    Args:
        backend: LLM backend (lmstudio, ollama, openrouter)
        model: Model name

    Returns:
        LLM client or None (uses fallbacks)
    """
    if not backend or not LLM_AVAILABLE:
        return None

    try:
        print(f"Connecting to {backend} with model {model}...")
        client = create_client(provider=backend, model=model)

        # Test connection
        if not client.test_connection():
            print(f"Warning: Could not connect to {backend}. Using fallback mode.")
            return None

        print(f"✓ Connected to {backend}")
        return client

    except Exception as e:
        print(f"Error setting up LLM client: {e}")
        print("Using fallback mode.")
        return None


def add_to_agent_datasets(examples: list, base_path: Path):
    """
    Add generated examples to the appropriate agent datasets.

    Args:
        examples: List of generated examples
        base_path: Base path to tools_datasets (e.g., Datasets/tools_datasets/thinking/)
    """
    print("\n" + "="*80)
    print("ADDING TO AGENT DATASETS")
    print("="*80)

    # Group by agent
    by_agent = {}
    for ex in examples:
        agent = ex["metadata"]["agent_name"]
        if agent not in by_agent:
            by_agent[agent] = []
        by_agent[agent].append(ex)

    # Add to each agent's dataset
    for agent, agent_examples in by_agent.items():
        agent_dir = base_path / agent

        if not agent_dir.exists():
            print(f"Warning: {agent_dir} does not exist. Skipping.")
            continue

        # Find latest version file
        latest_file = None
        for version_file in sorted(agent_dir.glob("tools_v*.jsonl"), reverse=True):
            if "backup" not in version_file.name and "automated" not in version_file.name:
                latest_file = version_file
                break

        if not latest_file:
            print(f"Warning: No dataset file found for {agent}. Skipping.")
            continue

        # Append to dataset
        with open(latest_file, 'a', encoding='utf-8') as f:
            for example in agent_examples:
                # Remove metadata before saving
                clean_example = {
                    "conversations": example["conversations"]
                }
                f.write(json.dumps(clean_example) + '\n')

        print(f"✓ Added {len(agent_examples)} examples to {latest_file.name}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate destructive confirmation training examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 50 examples with fallback mode (no LLM)
  python run_destructive_gen.py --count 50 --output destructive_confirmations.jsonl

  # Generate 100 examples using LM Studio
  python run_destructive_gen.py --count 100 --backend lmstudio --model your-model

  # Quick test (10 examples, fallback mode)
  python run_destructive_gen.py --test

  # Add to agent datasets
  python run_destructive_gen.py --count 50 --add-to-datasets
        """
    )

    parser.add_argument(
        "--count", "-n",
        type=int,
        default=10,
        help="Number of examples to generate (default: 10)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output JSONL file path (optional)"
    )

    parser.add_argument(
        "--backend",
        choices=["lmstudio", "ollama", "openrouter"],
        help="LLM backend to use (optional, uses fallback if not specified)"
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Model name for LLM backend"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick test mode (generate 3 examples, no save)"
    )

    parser.add_argument(
        "--add-to-datasets",
        action="store_true",
        help="Add generated examples to agent datasets in Datasets/tools_datasets/thinking/"
    )

    args = parser.parse_args()

    # Test mode overrides
    if args.test:
        args.count = 3
        args.output = None
        args.backend = None

    # Setup LLM client
    llm_client = None
    if args.backend:
        llm_client = setup_llm_client(args.backend, args.model)

    # Initialize generator
    generator = DestructiveConfirmationGenerator(
        model_client=llm_client,
    )

    # Generate examples
    print("\n" + "="*80)
    print(f"GENERATING {args.count} DESTRUCTIVE CONFIRMATION EXAMPLES")
    print("="*80)

    if llm_client:
        print(f"Using: {args.backend} with model {args.model}")
    else:
        print("Using: Fallback mode (template responses)")

    print()

    # Determine output path
    output_path = None
    if args.output:
        output_path = Path(args.output)
    elif not args.test:
        # Default output path
        output_path = Path(__file__).parent / "output" / "destructive_confirmations.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate
    examples = generator.generate_batch(
        count=args.count,
        output_path=output_path,
    )

    # Print summary
    print("\n" + "="*80)
    print("GENERATION SUMMARY")
    print("="*80)
    print(f"Total examples: {len(examples)}")

    # Count by tool
    tool_counts = {}
    for ex in examples:
        tool = ex["metadata"]["tool_name"]
        tool_counts[tool] = tool_counts.get(tool, 0) + 1

    print("\nBy tool:")
    for tool, count in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {tool}: {count}")

    # Add to datasets if requested
    if args.add_to_datasets and not args.test:
        base_path = Path(__file__).parent.parent / "Datasets" / "tools_datasets" / "thinking"
        add_to_agent_datasets(examples, base_path)

    print("\n" + "="*80)
    print("DONE")
    print("="*80)

    if args.test:
        print("\n🧪 Test mode completed successfully!")
        print("To generate real data, run:")
        print("  python run_destructive_gen.py --count 50 --backend lmstudio --model your-model")
    else:
        print(f"\n✓ Generated {len(examples)} examples")
        if output_path:
            print(f"✓ Saved to: {output_path}")
        if args.add_to_datasets:
            print(f"✓ Added to agent datasets")


if __name__ == "__main__":
    main()
