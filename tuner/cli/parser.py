"""
CLI argument parser.

Location: tuner/cli/parser.py
Purpose: Parse command-line arguments
Used by: Main entry point (cli/main.py)
"""

import argparse


def create_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for Synaptic Tuner CLI.

    Returns:
        argparse.ArgumentParser: Configured parser

    Commands:
        (none)    Interactive menu
        train     Training submenu
        upload    Upload submenu
        eval      Evaluation submenu
        pipeline  Full pipeline (train -> upload -> eval)

    Example:
        >>> parser = create_parser()
        >>> args = parser.parse_args(['train'])
        >>> args.command
        'train'
    """
    parser = argparse.ArgumentParser(
        description="Synaptic Tuner - Fine-tuning CLI for Nexus MCP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  (none)    Interactive menu
  train     Training submenu
  upload    Upload submenu
  eval      Evaluation submenu
  run       Run model inference (chat with your model)
  generate  Generate synthetic data (SelfPlay)
  improve   Improve dataset quality (LLM-based)
  pipeline  Full pipeline (train -> upload -> eval)
  gguf      Convert model to GGUF format
  webllm    Convert model for WebLLM/browser deployment

Examples:
  python tuner.py           # Interactive mode
  python tuner.py train     # Go directly to training
  python tuner.py upload    # Go directly to upload
  python tuner.py eval      # Go directly to evaluation
  python tuner.py run       # Run/chat with a trained model
  python tuner.py generate  # Generate synthetic data
  python tuner.py improve   # Improve dataset quality
  python tuner.py pipeline  # Run full pipeline
  python tuner.py gguf      # Convert to GGUF format
  python tuner.py webllm    # Convert for WebLLM/MLC
"""
    )

    parser.add_argument(
        "command",
        nargs="?",
        choices=["train", "upload", "eval", "run", "generate", "improve", "pipeline", "gguf", "webllm"],
        help="Command to run (optional, defaults to interactive menu)"
    )

    return parser
