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
  - ml: Traditional ML training (LightGBM, XGBoost, sklearn)
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
  cloud-run   Config-driven HF cloud job
  cloud-pipeline Train on HF Jobs, then evaluate on HF Jobs
  cloud-eval  Cloud evaluation on HF Jobs using vLLM
  cloud-gym   Run the vault gym against a trained cloud run on HF Jobs
  cloud-inspect Inspect saved HF cloud evaluation results
  eval        Evaluate a model
  synthchat   Synthetic data generation and improvement
  modelops    Model operations (run, merge, convert, upload)
  ml          Traditional ML training (LightGBM, XGBoost, sklearn)
  status      System status overview (use --json for structured output)
  doctor      System diagnostics (use --fix to auto-fix issues)
  flywheel    Data flywheel (self-improving training pipeline)
  experiment-loop  Autonomous hyperparameter search (LLM + surrogate)
  surgery     LoRA weight surgery (eval-guided post-training optimization)
  list        Discover available resources
  list-runs   Query unified experiment tracking registry

Flywheel Subcommands:
  flywheel status       Show flywheel system status
  flywheel run-cycle    Execute flywheel cycle (--skip-retrain, --retrain-mode, --dry-run)
  flywheel configure    Show flywheel configuration
  flywheel readiness    Check retrain readiness
  flywheel stage        Stage dataset version (no retrain)
  flywheel logs         Show inference log statistics
  flywheel versions     List staged dataset versions

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
  python tuner.py cloud-run --job-config Trainers/cloud/jobs/job.yaml --yes
  python tuner.py doctor       # Run diagnostics
  python tuner.py doctor --fix     # Auto-fix simple issues
  python tuner.py list datasets    # List datasets
  python tuner.py ml                   # Interactive ML training
  python tuner.py ml train --config path/to/config.yaml
  python tuner.py ml list-configs      # Show available configs
  python tuner.py list models --json   # List models as JSON
"""
    )

    parser.add_argument(
        "command",
        nargs="?",
        choices=["train", "cloud", "cloud-run", "cloud-pipeline", "cloud-eval", "cloud-gym", "cloud-inspect", "eval", "synthchat", "modelops", "ml", "flywheel", "experiment-loop", "surgery", "status", "doctor", "list", "list-runs", "compute-losses", "compare-runs", "judge-sample", "create-experiment", "cloud-compare", "download-experiment"],
        help="Command to run (optional, defaults to interactive menu)"
    )

    # Subcommand argument (shared positional for list and ml sub-commands)
    # When command is 'list': accepts datasets, models, runs, rubrics, scenarios
    # When command is 'ml': accepts train, list-configs
    parser.add_argument(
        "subcommand",
        nargs="?",
        default=None,
        help="Sub-command (e.g., 'datasets' for list, 'train' for ml)"
    )

    # Global flags
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format for AI-parseable output (disables interactive menus)"
    )
    parser.add_argument(
        "--yes",
        "--auto-confirm",
        action="store_true",
        dest="auto_confirm",
        help="Skip confirmation prompts for non-interactive command execution",
    )

    # Doctor-specific flags
    parser.add_argument(
        "--fix",
        action="store_true",
        dest="doctor_fix",
        help="Auto-fix simple issues (only used with 'doctor' command)"
    )

    # ML-specific flags
    parser.add_argument(
        "--config",
        dest="ml_config",
        help="Path to ML training config YAML (only used with 'ml train')"
    )

    # Flywheel-specific flags
    parser.add_argument(
        "--skip-retrain",
        action="store_true",
        dest="skip_retrain",
        help="Stop after staging (flywheel run-cycle only)"
    )
    parser.add_argument(
        "--retrain-mode",
        choices=["gpu_mutex", "hot_swap", "cloud"],
        dest="retrain_mode",
        help="Override retrain mode (flywheel run-cycle only)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Show what would happen without executing (flywheel run-cycle only)"
    )
    parser.add_argument(
        "--flywheel-config",
        dest="flywheel_config",
        help="Path to flywheel config YAML (flywheel commands only)"
    )

    # Surgery-specific flags
    parser.add_argument(
        "--surgery-config",
        dest="surgery_config",
        help="Path to LoRA surgery config YAML (surgery command only)"
    )

    # Cloud-specific flags
    parser.add_argument("--run", help="Cloud run slug or prefix to use (cloud-eval, cloud-gym only). Use 'latest' for newest.")
    parser.add_argument("--method", choices=["sft", "kto"], help="Training method for cloud-pipeline, or training method filter for cloud-eval/cloud-gym.")
    parser.add_argument("--bucket", help="Override HF bucket identifier for cloud-eval/cloud-gym.")
    parser.add_argument("--preset", help="Evaluation preset from Evaluator/config/eval_run.yaml (cloud-eval, cloud-pipeline).")
    parser.add_argument(
        "--scenario",
        action="append",
        help="Evaluation scenario file(s) to run (cloud-eval, cloud-gym, cloud-pipeline; can specify multiple).",
    )
    parser.add_argument("--tags", help="Comma-separated evaluation tag filter (cloud-eval, cloud-gym, cloud-pipeline).")
    parser.add_argument("--upload-to-hf", help="Optional HF model repo to receive evaluation lineage.")
    parser.add_argument(
        "--update-model-card",
        action="store_true",
        help="Update README.md when using --upload-to-hf (cloud-eval, cloud-gym, cloud-pipeline).",
    )
    parser.add_argument("--gpu", help="Override HF Jobs hardware flavor for cloud-eval/cloud-gym.")
    parser.add_argument("--timeout-hours", type=float, help="Override timeout in hours for cloud-eval/cloud-gym.")
    parser.add_argument("--env-backend", choices=["none", "local", "e2b"], help="Remote evaluator environment backend for cloud-eval/cloud-gym.")
    parser.add_argument("--env-template", help="E2B template ID for cloud-eval/cloud-gym when --env-backend e2b.")
    parser.add_argument("--env-tool-schema", help="Custom tool schema YAML for cloud-eval/cloud-gym.")
    parser.add_argument("--env-exec-config", help="Custom environment execution YAML for cloud-eval/cloud-gym.")
    parser.add_argument("--job-config", help="Config-driven cloud job YAML (cloud-run workflow).")
    parser.add_argument("--eval-run", help="Cloud evaluation run slug or prefix to inspect (cloud-inspect only). Use 'latest' for newest.")

    # Experiment loop flags
    parser.add_argument(
        "--experiment-config",
        dest="experiment_loop_config",
        help="Path to experiment loop config YAML (experiment-loop only)"
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        dest="max_experiments",
        help="Override max number of experiments (experiment-loop only)"
    )

    # list-runs filters (unified tracking registry)
    parser.add_argument("--run-type", help="Filter by run type: sft, kto, grpo, ml, evaluation, cloud_sft, cloud_kto, cloud_grpo (list-runs only)")
    parser.add_argument("--since", help="Filter runs after this ISO 8601 date (list-runs only)")
    parser.add_argument("--until-date", help="Filter runs before this ISO 8601 date (list-runs only)")

    # Experiment pipeline flags
    parser.add_argument("--experiment-id", help="Experiment ID for tracking")
    parser.add_argument("--base-dir", default=".tracking", help="Tracking base directory")
    parser.add_argument("--model", help="Model path for inference")
    parser.add_argument("--dataset-path", help="Path to jsonl dataset")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--no-completion-only", action="store_true", help="Disable completion-only masking")
    parser.add_argument("--base-model-name", help="Base model name for experiment")
    parser.add_argument("--dataset-hash", help="Dataset hash for experiment")

    return parser
