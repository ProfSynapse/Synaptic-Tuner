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
  pipeline  Full pipeline (train -> upload -> eval)

Examples:
  python tuner.py           # Interactive mode
  python tuner.py train     # Go directly to training
  python tuner.py upload    # Go directly to upload
  python tuner.py eval      # Go directly to evaluation
  python tuner.py pipeline  # Run full pipeline
"""
    )

    parser.add_argument(
        "command",
        nargs="?",
        choices=["train", "upload", "eval", "pipeline"],
        help="Command to run (optional, defaults to interactive menu)"
    )

    return parser
