#!/usr/bin/env python3
"""
Run SelfPlay generation to create synthetic training data.

This script connects to LM Studio and generates a batch of training examples
using the 3-prompt pipeline. Valid and invalid examples are collected separately
for KTO training.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from SelfPlay.generator import SelfPlayGenerator
from Evaluator.lmstudio_client import LMStudioClient
from Evaluator.config import LMStudioSettings


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data using SelfPlay pipeline"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=100,
        help="Number of examples to generate (default: 100)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="SelfPlay/selfplay_output.jsonl",
        help="Output file path (default: SelfPlay/selfplay_output.jsonl)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="LM Studio host (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=1234,
        help="LM Studio port (default: 1234)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="local-model",
        help="Model name (default: local-model)"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation of generated examples"
    )
    parser.add_argument(
        "--no-save-invalid",
        action="store_true",
        help="Don't save invalid examples to separate file"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("SELFPLAY GENERATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  LM Studio: {args.host}:{args.port}")
    print(f"  Model: {args.model}")
    print(f"  Examples: {args.num_examples}")
    print(f"  Output: {args.output}")
    print(f"  Validation: {'Disabled' if args.no_validate else 'Enabled'}")
    print(f"  Save Invalid: {'No' if args.no_save_invalid else 'Yes'}")

    # Initialize LM Studio client
    print("\n" + "=" * 70)
    print("CONNECTING TO LM STUDIO")
    print("=" * 70)

    settings = LMStudioSettings(
        host=args.host,
        port=args.port,
        model=args.model
    )
    client = LMStudioClient(settings=settings)

    # Check connection
    if not client.is_server_running():
        print("\n❌ LM Studio server is not accessible!")
        print("\nMake sure:")
        print("  1. LM Studio is running")
        print("  2. A model is loaded")
        print("  3. Server is started (click 'Start Server' in LM Studio)")
        print(f"  4. Server is accessible at {settings.base_url()}")
        sys.exit(1)

    print("✅ LM Studio connected!")

    # List available models
    try:
        models = client.list_models()
        if models:
            print(f"\n📋 Available models: {', '.join(models[:3])}")
    except Exception as e:
        print(f"\nNote: Could not list models ({e})")

    # Initialize generator
    print("\n" + "=" * 70)
    print("INITIALIZING GENERATOR")
    print("=" * 70)

    generator = SelfPlayGenerator(model_client=client)
    print("✅ Generator initialized!")

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Clear output file if it exists
    if output_path.exists():
        print(f"\n⚠️  Output file exists, will append to: {output_path}")
    else:
        print(f"\n📁 Output file: {output_path}")

    # Generate batch
    print("\n" + "=" * 70)
    print(f"GENERATING {args.num_examples} EXAMPLES")
    print("=" * 70)

    try:
        results = generator.generate_batch(
            num_examples=args.num_examples,
            output_file=output_path,
            validate=not args.no_validate,
            save_invalid=not args.no_save_invalid
        )

        # Final summary
        print("\n" + "=" * 70)
        print("✅ GENERATION COMPLETE!")
        print("=" * 70)

        print(f"\n📊 Results:")
        print(f"  Valid examples: {len(results['valid'])}")
        print(f"  Invalid examples: {len(results['invalid'])}")
        print(f"  Success rate: {len(results['valid']) / args.num_examples * 100:.1f}%")

        print(f"\n💾 Files created:")
        print(f"  Valid: {output_path}")
        if results['invalid'] and not args.no_save_invalid:
            invalid_file = output_path.parent / f"{output_path.stem}_invalid.jsonl"
            print(f"  Invalid: {invalid_file}")

        # Show example
        if results['valid']:
            print("\n📝 Sample valid example:")
            example = results['valid'][0]
            print(f"  Type: {example['metadata']['type']}")
            print(f"  Agent: {example['metadata'].get('agent', 'N/A')}")
            print(f"  Environment: {example['metadata']['environment_variation']}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
