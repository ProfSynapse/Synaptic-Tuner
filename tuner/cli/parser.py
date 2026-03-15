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
  - status: System status overview
  - doctor: System diagnostics with recommendations and auto-fix
  - list: Resource discovery (datasets, models, runs, rubrics, scenarios)
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
        status      System status overview (environment, GPU, services)
        doctor      System diagnostics with recommendations and auto-fix
        list        Discover available resources

    Example:
        >>> parser = create_parser()
        >>> args = parser.parse_args(['train'])
        >>> args.command
        'train'

        >>> args = parser.parse_args(['status', '--json'])
        >>> args.command
        'status'
        >>> args.json
        True

        >>> args = parser.parse_args(['doctor', '--fix'])
        >>> args.command
        'doctor'
        >>> args.doctor_fix
        True

        >>> args = parser.parse_args(['list', 'datasets'])
        >>> args.command
        'list'
        >>> args.list_subcommand
        'datasets'
    """
    parser = argparse.ArgumentParser(
        description="Synaptic Tuner - Fine-tuning CLI for Nexus MCP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  (none)      Interactive menu
  train       Training workflow (SFT, KTO, GRPO)
  cloud       Cloud training (HF Jobs, Modal, RunPod)
  cloud-pipeline Train on HF Jobs, then evaluate on HF Jobs
  cloud-eval  Cloud evaluation on HF Jobs using vLLM
  cloud-inspect Inspect saved HF cloud evaluation results
  eval        Evaluate a model
  synthchat   Synthetic data generation and improvement
  modelops    Model operations (run, merge, convert, upload)
  status      System status overview (use --json for structured output)
  doctor      System diagnostics (use --fix to auto-fix issues)
  list        Discover available resources

List Subcommands:
  list datasets   List available JSONL datasets
  list models     List base models and fine-tuned models
  list runs       List training runs with status
  list rubrics    List available rubrics for improvement
  list scenarios  List evaluation scenarios

Examples:
  python tuner.py              # Interactive mode
  python tuner.py train        # Go directly to training
  python tuner.py cloud        # Cloud training submenu
  python tuner.py eval         # Go directly to evaluation
  python tuner.py synthchat    # Generate or improve data
  python tuner.py modelops     # Model operations submenu
  python tuner.py status       # Show system status
  python tuner.py status --json    # JSON output for AI parsing
  python tuner.py doctor       # Run diagnostics
  python tuner.py doctor --fix     # Auto-fix simple issues
  python tuner.py list datasets    # List datasets
  python tuner.py list models --json   # List models as JSON
"""
    )

    parser.add_argument(
        "command",
        nargs="?",
        choices=["train", "cloud", "cloud-pipeline", "cloud-eval", "cloud-inspect", "eval", "synthchat", "modelops", "status", "doctor", "list"],
        help="Command to run (optional, defaults to interactive menu)"
    )

    # List subcommand argument (only used when command is 'list')
    parser.add_argument(
        "list_subcommand",
        nargs="?",
        choices=["datasets", "models", "runs", "rubrics", "scenarios"],
        default=None,
        help="Resource type to list (only for 'list' command)"
    )

    # Global flags
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format for AI-parseable output (disables interactive menus)"
    )

    # Doctor-specific flags
    parser.add_argument(
        "--fix",
        action="store_true",
        dest="doctor_fix",
        help="Auto-fix simple issues (only used with 'doctor' command)"
    )

    parser.add_argument("--run", help="Cloud run slug or prefix to use (cloud-eval only). Use 'latest' for newest.")
    parser.add_argument("--method", choices=["sft", "kto"], help="Training method for cloud-pipeline, or training method filter for cloud-eval.")
    parser.add_argument("--bucket", help="Override HF bucket identifier for cloud-eval.")
    parser.add_argument("--preset", help="Evaluation preset from Evaluator/config/eval_run.yaml (cloud-eval, cloud-pipeline).")
    parser.add_argument(
        "--scenario",
        action="append",
        help="Evaluation scenario file(s) to run (cloud-eval, cloud-pipeline; can specify multiple).",
    )
    parser.add_argument("--tags", help="Comma-separated evaluation tag filter (cloud-eval, cloud-pipeline).")
    parser.add_argument("--upload-to-hf", help="Optional HF model repo to receive evaluation lineage.")
    parser.add_argument(
        "--update-model-card",
        action="store_true",
        help="Update README.md when using --upload-to-hf (cloud-eval, cloud-pipeline).",
    )
    parser.add_argument("--gpu", help="Override HF Jobs hardware flavor for cloud-eval.")
    parser.add_argument("--timeout-hours", type=float, help="Override timeout in hours for cloud-eval.")
    parser.add_argument("--eval-run", help="Cloud evaluation run slug or prefix to inspect (cloud-inspect only). Use 'latest' for newest.")

    return parser
