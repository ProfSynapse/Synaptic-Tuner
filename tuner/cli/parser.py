"""
CLI argument parser.

Location: tuner/cli/parser.py
Purpose: Parse command-line arguments for the Synaptic Tuner CLI
Used by: Main entry point (cli/main.py)

The parser defines the top-level command structure:
  - train: Training workflows (SFT, KTO, GRPO)
  - eval: Model evaluation
  - synthchat: Synthetic data generation and improvement
  - modelops: Model operations (run, merge, convert, upload)
"""

import argparse


def create_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for Synaptic Tuner CLI.

    Returns:
        argparse.ArgumentParser: Configured parser

    Commands:
        (none)      Interactive menu
        train       Training workflow (SFT, KTO, GRPO)
        eval        Evaluate a model
        synthchat   Synthetic data generation and improvement
        modelops    Model operations (run, merge, convert, upload)

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
  (none)      Interactive menu
  train       Training workflow (SFT, KTO, GRPO)
  eval        Evaluate a model
  synthchat   Synthetic data generation and improvement
  modelops    Model operations (run, merge, convert, upload)

Examples:
  python tuner.py           # Interactive mode
  python tuner.py train     # Go directly to training
  python tuner.py eval      # Go directly to evaluation
  python tuner.py synthchat # Generate or improve data
  python tuner.py modelops  # Model operations submenu
"""
    )

    parser.add_argument(
        "command",
        nargs="?",
        choices=["train", "eval", "synthchat", "modelops"],
        help="Command to run (optional, defaults to interactive menu)"
    )

    return parser
