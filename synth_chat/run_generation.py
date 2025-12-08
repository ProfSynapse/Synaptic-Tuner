#!/usr/bin/env python3
"""
Run Synthetic Chat generation to create synthetic training data.

This script connects to LM Studio and generates a batch of training examples
using the 3-prompt pipeline. Valid and invalid examples are collected separately
for KTO training.
"""

import os
import sys
import argparse
from pathlib import Path
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from synth_chat.generator import SynthChatGenerator
from Evaluator.shared_llm_adapters import SharedLMStudioAdapter
from Evaluator.config import LMStudioSettings


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data using Synthetic Chat pipeline"
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
        default="synth_chat/synth_chat_output.jsonl",
        help="Output file path (default: synth_chat/synth_chat_output.jsonl)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("LMSTUDIO_HOST", "localhost"),
        help="LM Studio host (default: env LMSTUDIO_HOST or localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("LMSTUDIO_PORT", "1234")),
        help="LM Studio port (default: env LMSTUDIO_PORT or 1234)"
    )
    # Model default comes from synth_chat/config/config.yaml (cloud default) or env override
    cfg = {}
    cfg_path = Path(__file__).parent / "config" / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
    llm_defaults = cfg.get("llm", {}) if isinstance(cfg, dict) else {}
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("SYNTHCHAT_MODEL", llm_defaults.get("model", "local-model")),
        help="Model name (default: SYNTHCHAT_MODEL env or config.yaml llm.model or 'local-model')"
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
    print("SYNTHETIC CHAT GENERATION")
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
    # Use shared LLM adapter for LM Studio
    client = SharedLMStudioAdapter(settings=settings)

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

    generator = SynthChatGenerator(model_client=client)
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
