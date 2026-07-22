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

from shared.utilities.paths import TRAINING_METHODS


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
  cloud-jobs  Inspect or manage live HF Jobs
  plan-hardware Blind hardware planning for experiment specs
  cloud-pipeline Train on HF Jobs, then evaluate on HF Jobs
  cloud-eval  Cloud evaluation on HF Jobs
  cloud-gym   Run the vault gym against a trained cloud run on HF Jobs
  cloud-inspect Inspect saved HF cloud evaluation results
  cloud-extract Forward-only hidden-state extraction on HF Jobs (publish-by-id)
  local-run   Config-driven local Docker training/eval job
  bucket      Read, list, pull, push, or analyze local / HF bucket artifacts
  run-experiment  Run train -> eval -> loss from one experiment config
  analyze-experiment Inspect a finished experiment bundle and recommendations
  eval        Evaluate a model
  synthchat   Synthetic data generation and improvement
  modelops    Model operations (run, merge, convert, upload)
  ml          Traditional ML training (LightGBM, XGBoost, sklearn)
  status      System status overview (use --json for structured output)
  doctor      System diagnostics (use --fix to auto-fix issues)
  flywheel    Data flywheel (self-improving training pipeline)
  experiment-loop  Autonomous hyperparameter search (LLM + surrogate)
  prompt-optimize Deterministic config-first prompt optimization
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
  flywheel export-fixtures --export-config <yaml> --output <yaml>

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
  python tuner.py cloud-run --job-config Trainers/recipes/job.yaml --yes
  python tuner.py local-run --job-config Trainers/recipes/job.yaml --yes
  python tuner.py mechinterp run --config MechInterp/configs/my_pipeline.yaml --provider local --yes
  python tuner.py cloud-jobs list
  python tuner.py bucket analyze --path runs/hf_jobs/sft/<run-prefix>/
  python tuner.py bucket read --path runs/hf_jobs/sft/<run-prefix>/logs/training_latest.jsonl --jsonl-latest --pretty
  python tuner.py bucket pull --path runs/hf_jobs/sft/<run-prefix>/ --dest .
  python tuner.py bucket push --path local/results.json --dest runs/manual_uploads/
  python tuner.py plan-hardware --experiment-spec Trainers/cloud/experiments/smollm2_full_cycle_smoke.yaml
  python tuner.py cloud-jobs logs --job professorsynapse/<job-id> --tail 200
  python tuner.py run-experiment --experiment-spec Trainers/cloud/experiments/smollm2_full_cycle_smoke.yaml --yes
  python tuner.py analyze-experiment --experiment-id latest
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
        choices=["train", "cloud", "cloud-run", "local-run", "cloud-jobs", "plan-hardware", "cloud-pipeline", "cloud-eval", "cloud-gym", "cloud-inspect", "cloud-extract", "batch-generate", "batch-capture", "bucket", "run-experiment", "analyze-experiment", "eval", "synthchat", "modelops", "ml", "mechinterp", "flywheel", "experiment-loop", "prompt-optimize", "surgery", "status", "doctor", "list", "list-runs", "compute-losses", "compare-runs", "judge-sample", "create-experiment", "cloud-compare", "download-experiment"],
        help="Command to run (optional, defaults to interactive menu)"
    )

    # Subcommand argument (shared positional for list, ml, and cloud-jobs sub-commands)
    # When command is 'list': accepts datasets, models, runs, rubrics, scenarios
    # When command is 'ml': accepts train, list-configs
    # When command is 'cloud-jobs': accepts list, show, logs, cancel
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
        help="Path to config YAML (ml train or mechinterp run)."
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
        help="Show what would happen without executing (flywheel run-cycle/export-fixtures only)"
    )
    parser.add_argument(
        "--flywheel-config",
        dest="flywheel_config",
        help="Path to flywheel config YAML (flywheel commands only)"
    )
    parser.add_argument(
        "--export-config",
        dest="export_config",
        help="Path to flywheel Evaluator fixture export config YAML (flywheel export-fixtures only)"
    )
    parser.add_argument(
        "--output",
        dest="output",
        help="Output file path (flywheel export-fixtures only)"
    )

    # Surgery-specific flags
    parser.add_argument(
        "--surgery-config",
        dest="surgery_config",
        help="Path to LoRA surgery config YAML (surgery command only)"
    )

    # Cloud-extract-specific flags (forward-only hidden-state extraction on HF
    # Jobs, publish-by-id). Reuses the shared --base-model-name, --gpu,
    # --timeout-hours, and --dry-run flags defined elsewhere in this parser.
    parser.add_argument(
        "--extraction-config",
        help="Extraction config (hub id or repo-relative path) for cloud-extract."
    )
    parser.add_argument(
        "--slice-dataset-name",
        help="Hub dataset id for the matched extraction slice (cloud-extract)."
    )
    parser.add_argument(
        "--base-model-revision",
        help="Base model commit SHA to pin for cloud-extract (required for cloud)."
    )
    parser.add_argument(
        "--adapter-repo-id",
        help="Hub model repo id for the contrast adapter (cloud-extract)."
    )
    parser.add_argument(
        "--adapter-revision",
        help="Contrast adapter commit SHA to pin (cloud-extract)."
    )
    parser.add_argument(
        "--output-dataset-name",
        help="Hub dataset id to receive the extraction outputs (cloud-extract)."
    )
    parser.add_argument(
        "--cloud-image",
        help="Override the Docker image for the cloud-extract HF Job."
    )
    parser.add_argument(
        "--repo-url",
        help="Override the tuner repo clone URL for cloud-extract (default: git origin)."
    )
    parser.add_argument(
        "--repo-branch",
        help="Override the tuner repo branch for cloud-extract (default: current branch)."
    )
    parser.add_argument(
        "--repo-commit",
        help="Override the tuner repo commit SHA for cloud-extract (default: HEAD)."
    )

    # Cloud-specific flags
    parser.add_argument("--run", help="Cloud run slug or prefix to use (cloud-eval, cloud-gym only). Use 'latest' for newest.")
    parser.add_argument("--method", choices=list(TRAINING_METHODS), help="Training method for cloud-pipeline, or training method filter for cloud-eval/cloud-gym.")
    parser.add_argument("--bucket", help="Override HF bucket identifier for cloud-eval/cloud-gym/bucket.")
    parser.add_argument("--path", help="Bucket-relative, hf://, or local path for bucket commands.")
    parser.add_argument("--dest", help="Destination for bucket pull/push. Local dir for pull, remote bucket path for push.")
    parser.add_argument("--eval-path", help="Explicit evaluation prefix or evaluation_lineage.json path for bucket analyze.")
    parser.add_argument("--loss-path", help="Explicit loss prefix or loss_lineage.json path for bucket analyze.")
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
    parser.add_argument("--eval-timeout-hours", type=float, help="Override evaluation timeout in hours for cloud-eval/cloud-pipeline.")
    parser.add_argument(
        "--eval-runtime",
        choices=["unsloth", "vllm"],
        help="Select the cloud evaluation runtime. Default comes from Trainers/cloud/cloud_config.yaml.",
    )
    parser.add_argument(
        "--eval-image-profile",
        help="Override the cloud evaluation image profile for cloud-eval/cloud-pipeline (for example: stable_unsloth, latest_unsloth, fast_vllm).",
    )
    parser.add_argument(
        "--eval-cloud-image",
        help="Override the exact cloud evaluation Docker image for cloud-eval/cloud-pipeline.",
    )
    parser.add_argument(
        "--with-loss",
        action="store_true",
        help="For cloud-eval, also compute per-example dataset loss in the same remote job when supported.",
    )
    parser.add_argument(
        "--loss-dataset-name",
        help="Override the Hugging Face dataset repo used for same-job loss computation during cloud-eval.",
    )
    parser.add_argument(
        "--loss-dataset-file",
        help="Override the dataset file used for same-job loss computation during cloud-eval.",
    )
    parser.add_argument(
        "--loss-max-seq-length",
        type=int,
        help="Override max sequence length for same-job loss computation during cloud-eval.",
    )
    parser.add_argument(
        "--loss-no-completion-only",
        action="store_true",
        help="Disable completion-only masking for same-job loss computation during cloud-eval.",
    )
    parser.add_argument("--train-gpu", help="Override training GPU flavor for cloud/cloud-pipeline.")
    parser.add_argument("--train-timeout-hours", type=float, help="Override training timeout in hours for cloud/cloud-pipeline.")
    parser.add_argument("--train-image-profile", help="Override the cloud training image profile for cloud/cloud-pipeline (for example: stable, next).")
    parser.add_argument("--train-cloud-image", help="Override the exact cloud training Docker image for cloud/cloud-pipeline.")
    parser.add_argument("--train-model-name", help="Override the base model for cloud/cloud-pipeline training.")
    parser.add_argument("--train-dataset-name", help="Override the Hugging Face dataset repo for cloud/cloud-pipeline training.")
    parser.add_argument("--train-dataset-file", help="Override the dataset file within the Hugging Face dataset repo.")
    parser.add_argument("--train-batch-size", type=int, help="Override per-device batch size for cloud/cloud-pipeline training.")
    parser.add_argument("--train-save-steps", type=int, help="Override checkpoint save frequency (steps) for cloud/cloud-pipeline training.")
    parser.add_argument("--train-save-total-limit", type=int, help="Override max checkpoints kept for cloud/cloud-pipeline training.")
    parser.add_argument("--train-gradient-accumulation", type=int, help="Override gradient accumulation steps for cloud/cloud-pipeline training.")
    parser.add_argument("--train-learning-rate", type=float, help="Override learning rate for cloud/cloud-pipeline training.")
    parser.add_argument("--train-seed", type=int, help="Override the training random seed for cloud/cloud-pipeline training.")
    parser.add_argument("--train-beta", type=float, help="Override the DPO/KTO beta parameter for cloud/cloud-pipeline training.")
    parser.add_argument("--train-num-epochs", type=int, help="Override epochs for cloud/cloud-pipeline training.")
    parser.add_argument("--train-max-steps", type=int, help="Override max training steps for cloud/cloud-pipeline training.")
    parser.add_argument("--train-max-seq-length", type=int, help="Override max sequence length for cloud/cloud-pipeline training.")
    parser.add_argument("--train-lora-r", type=int, help="Override LoRA rank for cloud/cloud-pipeline SFT training.")
    parser.add_argument("--train-lora-alpha", type=int, help="Override LoRA alpha for cloud/cloud-pipeline SFT training.")
    parser.add_argument("--train-lora-dropout", type=float, help="Override LoRA dropout for cloud/cloud-pipeline SFT training.")
    parser.add_argument("--train-use-dora", action="store_true", help="Enable DoRA for cloud/cloud-pipeline SFT training.")
    parser.add_argument("--train-use-rslora", action="store_true", help="Enable rsLoRA for cloud/cloud-pipeline SFT training.")
    parser.add_argument("--train-init-lora-weights", help="Override init_lora_weights for cloud/cloud-pipeline SFT training.")
    parser.add_argument("--train-lora-target-modules", help="Comma-separated LoRA target modules for cloud/cloud-pipeline SFT training.")
    parser.add_argument("--train-evolutionary-enabled", action="store_true", help="Enable experimental evolutionary gradient selection for cloud/cloud-pipeline SFT training.")
    parser.add_argument("--train-evolutionary-candidates", type=int, help="Override evolutionary candidate count for cloud/cloud-pipeline SFT training.")
    parser.add_argument("--train-evolutionary-eval-batch-size", type=int, help="Override evolutionary fitness eval batch size for cloud/cloud-pipeline SFT training.")
    parser.add_argument("--train-evolutionary-validation-config", help="Override evolutionary validation config path for cloud/cloud-pipeline SFT training.")
    parser.add_argument("--train-evolutionary-strategy", choices=["gradient_noise", "antithetic_noise", "scale_variation", "combined"], help="Override evolutionary candidate generation strategy for cloud/cloud-pipeline SFT training.")
    parser.add_argument("--train-evolutionary-noise-scale", type=float, help="Override evolutionary gradient noise scale for cloud/cloud-pipeline SFT training.")
    parser.add_argument("--train-evolutionary-max-grad-norm", type=float, help="Override evolutionary gradient clipping / max grad norm for cloud/cloud-pipeline SFT training.")
    parser.add_argument("--train-evolutionary-scale-factors", help="Comma-separated evolutionary scale factors for cloud/cloud-pipeline SFT training.")
    parser.add_argument("--train-evolutionary-selection-method", choices=["best", "tournament", "proportional"], help="Override evolutionary selection method for cloud/cloud-pipeline SFT training.")
    parser.add_argument("--train-evolutionary-min-improvement", type=float, help="Override evolutionary minimum fitness improvement for cloud/cloud-pipeline SFT training.")
    parser.add_argument("--train-evolutionary-min-relative-improvement", type=float, help="Override evolutionary minimum relative fitness improvement for cloud/cloud-pipeline SFT training.")
    parser.add_argument("--train-evolutionary-noise-floor-epsilon", type=float, help="Override evolutionary minimum absolute acceptance floor for cloud/cloud-pipeline SFT training.")
    parser.add_argument("--train-evolutionary-eval-frequency", type=int, help="Override evolutionary eval frequency for cloud/cloud-pipeline SFT training.")
    parser.add_argument("--train-evolutionary-warmup-steps", type=int, help="Override evolutionary warmup steps for cloud/cloud-pipeline SFT training.")
    parser.add_argument("--train-evolutionary-no-cache-baseline", action="store_false", dest="train_evolutionary_cache_baseline", help="Disable evolutionary baseline caching for cloud/cloud-pipeline SFT training.")
    parser.add_argument("--train-evolutionary-cache-baseline", action="store_true", dest="train_evolutionary_cache_baseline", help="Enable evolutionary baseline caching for cloud/cloud-pipeline SFT training.")
    parser.add_argument("--train-evolutionary-no-log-candidates", action="store_false", dest="train_evolutionary_log_candidates", help="Disable evolutionary candidate logging for cloud/cloud-pipeline SFT training.")
    parser.add_argument("--train-evolutionary-log-candidates", action="store_true", dest="train_evolutionary_log_candidates", help="Enable evolutionary candidate logging for cloud/cloud-pipeline SFT training.")
    parser.add_argument("--train-evolutionary-no-log-selected", action="store_false", dest="train_evolutionary_log_selected", help="Disable evolutionary selection logging for cloud/cloud-pipeline SFT training.")
    parser.add_argument("--train-evolutionary-log-selected", action="store_true", dest="train_evolutionary_log_selected", help="Enable evolutionary selection logging for cloud/cloud-pipeline SFT training.")
    parser.set_defaults(
        train_evolutionary_cache_baseline=None,
        train_evolutionary_log_candidates=None,
        train_evolutionary_log_selected=None,
    )
    parser.add_argument("--train-load-in-4bit", action="store_true", dest="train_load_in_4bit", help="Force 4-bit model loading for cloud/cloud-pipeline SFT training.")
    parser.add_argument("--train-no-load-in-4bit", action="store_false", dest="train_load_in_4bit", help="Disable 4-bit model loading for cloud/cloud-pipeline SFT training.")
    parser.set_defaults(train_load_in_4bit=None)
    parser.add_argument("--env-backend", choices=["none", "local", "e2b"], help="Remote evaluator environment backend for cloud-eval/cloud-gym.")
    parser.add_argument("--env-template", help="E2B template ID for cloud-eval/cloud-gym when --env-backend e2b.")
    parser.add_argument("--env-tool-schema", help="Custom tool schema YAML for cloud-eval/cloud-gym.")
    parser.add_argument("--env-exec-config", help="Custom environment execution YAML for cloud-eval/cloud-gym.")
    parser.add_argument("--job-config", help="Config-driven job YAML (cloud-run or local-run workflow).")
    parser.add_argument(
        "--provider",
        choices=["local", "modal"],
        help="Execution provider for config-driven mechinterp run.",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop (but keep) the persistent container for the given --job-config (local-run only).",
    )
    parser.add_argument(
        "--rm-persistent",
        action="store_true",
        help="Stop and remove the persistent container for the given --job-config (local-run only).",
    )
    parser.add_argument(
        "--container-status",
        action="store_true",
        help="Print the persistent container's state for the given --job-config (local-run only).",
    )
    parser.add_argument("--eval-run", help="Cloud evaluation run slug or prefix to inspect (cloud-inspect only). Use 'latest' for newest.")
    parser.add_argument("--job", help="HF job reference for cloud-jobs show/logs/cancel. Accepts either <job-id> or <namespace>/<job-id>.")
    parser.add_argument("--namespace", help="HF namespace override for cloud-jobs list/show/logs/cancel.")
    parser.add_argument("--tail", type=int, default=200, help="Number of log lines to show for cloud-jobs logs or bucket read tail (default: 200).")
    parser.add_argument("--limit", type=int, default=20, help="Maximum jobs or bucket entries to show (default: 20).")
    parser.add_argument("--jsonl-latest", action="store_true", help="For bucket read, parse JSONL and print the latest record.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output for bucket read.")
    parser.add_argument("--recursive", action="store_true", help="For bucket list, recurse through directories.")
    parser.add_argument("--files-only", action="store_true", help="For bucket list, show files only.")
    parser.add_argument("--dirs-only", action="store_true", help="For bucket list, show directories only.")
    parser.add_argument("--follow", action="store_true", help="Stream live logs for cloud-jobs logs.")

    # MechInterp flags (extract / probe-fit / steer / dose-calibrate / score-gates sub-commands)
    parser.add_argument(
        "--mi-config",
        dest="mechinterp_config",
        help="Path to a MechInterp recipe YAML (mechinterp extract/probe-fit/steer/dose-calibrate)",
    )
    parser.add_argument(
        "--pipeline-config",
        dest="pipeline_config",
        help="Path to a multi-stage MechInterp pipeline YAML (mechinterp run).",
    )
    parser.add_argument(
        "--render-fn",
        dest="render_fn",
        help="Prompt render callable 'module.path:callable' (mechinterp steer/extract/dose-calibrate)",
    )
    parser.add_argument(
        "--adapter",
        dest="adapter",
        help="Optional PEFT adapter path/id to load on the base model (mechinterp).",
    )
    parser.add_argument(
        "--gates-config",
        dest="gates_config",
        help="Path to a gates.yaml (mechinterp score-gates only).",
    )
    parser.add_argument(
        "--rows-path",
        dest="rows_path",
        help="Path to a per-row output JSONL to score (mechinterp score-gates only).",
    )
    parser.add_argument(
        "--arm-field",
        dest="arm_field",
        default="arm",
        help="Row field naming the arm for gate grouping (mechinterp score-gates).",
    )
    parser.add_argument(
        "--i-know-this-runs-on-gpu",
        dest="i_know_this_runs_on_gpu",
        action="store_true",
        help="Acknowledge that mechinterp extract/steer/dose-calibrate load a model and use a GPU.",
    )
    parser.add_argument(
        "--force-full-run",
        dest="force_full_run",
        action="store_true",
        help="Skip the smoke gate and run the full mechinterp steer arms directly.",
    )
    parser.add_argument(
        "--only-step",
        dest="only_step",
        help="Run only one named stage from a mechinterp pipeline.",
    )
    parser.add_argument(
        "--from-step",
        dest="from_step",
        help="Start a mechinterp pipeline at this named stage.",
    )
    parser.add_argument(
        "--skip-step",
        dest="skip_step",
        action="append",
        help="Skip a named stage from a mechinterp pipeline. May be repeated.",
    )

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

    # Prompt optimization flags
    parser.add_argument(
        "--prompt-opt-config",
        dest="prompt_opt_config",
        help="Path to prompt optimization config YAML (prompt-optimize only)",
    )
    parser.add_argument(
        "--prompt-opt-output-dir",
        dest="prompt_opt_output_dir",
        help="Override output directory for prompt optimization artifacts.",
    )

    # list-runs filters (unified tracking registry)
    parser.add_argument("--run-type", help="Filter by run type: sft, kto, grpo, ml, evaluation, cloud_sft, cloud_kto, cloud_grpo (list-runs only)")
    parser.add_argument("--since", help="Filter runs after this ISO 8601 date (list-runs only)")
    parser.add_argument("--until-date", help="Filter runs before this ISO 8601 date (list-runs only)")

    # Experiment pipeline flags
    parser.add_argument("--experiment-id", help="Experiment ID for tracking")
    parser.add_argument("--experiment-spec", help="Path to experiment orchestration YAML (run-experiment only)")
    parser.add_argument(
        "--auto-hardware",
        action="store_true",
        help="Use the blind hardware planner to choose stage hardware when experiment specs omit explicit GPU flavors.",
    )
    parser.add_argument(
        "--optimize-for",
        choices=["balanced", "cost", "speed"],
        default="balanced",
        help="Optimization objective for hardware planning (plan-hardware / run-experiment auto-hardware).",
    )
    parser.add_argument(
        "--max-hourly-price",
        type=float,
        help="Optional hourly price cap for hardware planning (plan-hardware / run-experiment auto-hardware).",
    )
    parser.add_argument(
        "--only-stage",
        choices=["training", "evaluation", "loss", "analysis", "recommendation"],
        help="Run only the selected stage family for run-experiment.",
    )
    parser.add_argument(
        "--from-stage",
        choices=["training", "evaluation", "loss", "analysis", "recommendation"],
        help="Run run-experiment starting from this stage and continue forward.",
    )
    parser.add_argument(
        "--skip-stage",
        action="append",
        choices=["training", "evaluation", "loss", "analysis", "recommendation"],
        help="Skip a stage in run-experiment. May be repeated.",
    )
    parser.add_argument("--base-dir", default=".tracking", help="Tracking base directory")
    parser.add_argument("--model", help="Model path for inference")
    parser.add_argument("--model-revision", dest="model_revision", help="Model commit SHA/revision for reproducible inference loads")
    parser.add_argument("--tokenizer-revision", dest="tokenizer_revision", help="Tokenizer commit SHA/revision for reproducible inference loads")
    parser.add_argument("--dataset-path", help="Path to jsonl dataset")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--no-completion-only", action="store_true", help="Disable completion-only masking")
    parser.add_argument("--base-model-name", help="Base model name for experiment")
    parser.add_argument("--dataset-hash", help="Dataset hash for experiment")

    # Batch inference flags (batch-generate / batch-capture).
    # These verbs are generic: prompts/sequences in, completions/hidden-states
    # out, with crash-safe incremental persistence and resume. --model is reused
    # from above for the model id/path.
    parser.add_argument("--prompts", help="Input JSONL of {id, prompt} rows (batch-generate).")
    parser.add_argument("--rows", help="Input JSONL of {id, text|token_ids, positions} rows (batch-capture).")
    parser.add_argument("--out-dir", help="Output directory for batch artifacts (batch-generate / batch-capture).")
    parser.add_argument(
        "--engine",
        choices=["hf-batched", "vllm"],
        default="hf-batched",
        help="Inference engine for batch-generate / batch-capture (default: hf-batched).",
    )
    parser.add_argument("--batch-size", type=int, default=16, dest="batch_size", help="Micro-batch size for batch verbs; auto-halves on CUDA OOM (default: 16).")
    parser.add_argument("--max-new-tokens", type=int, default=48, dest="max_new_tokens", help="Max new tokens for batch-generate (default: 48).")
    parser.add_argument("--min-new-tokens", type=int, default=0, dest="min_new_tokens", help="Min new tokens for batch-generate (default: 0).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for batch verbs.")
    parser.add_argument("--do-sample", action="store_true", dest="do_sample", help="Sample instead of greedy decode (batch-generate).")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for batch-generate --do-sample.")
    parser.add_argument("--top-p", type=float, default=1.0, dest="top_p", help="Nucleus top-p for batch-generate --do-sample.")
    parser.add_argument("--extra-eos-token", action="append", dest="extra_eos_tokens", help="Additional tokenizer token to treat as EOS for batch-generate; may be repeated.")
    parser.add_argument("--stop-string", action="append", dest="stop_strings", help="Stop string for batch-generate; may be repeated.")
    parser.add_argument("--json-schema", dest="json_schema", help="Path to a JSON Schema enforced by vLLM structured outputs (batch-generate).")
    parser.add_argument(
        "--structured-output-backend",
        choices=["auto", "xgrammar"],
        default="auto",
        dest="structured_output_backend",
        help="Pinned vLLM structured-output backend (default: auto).",
    )
    parser.add_argument(
        "--structured-output-disable-any-whitespace",
        action="store_true",
        dest="structured_output_disable_any_whitespace",
        help="Disallow arbitrary JSON whitespace in supported vLLM backends.",
    )
    parser.add_argument("--expected-vllm-version", dest="expected_vllm_version", help="Exact installed vLLM version required by --engine vllm.")
    parser.add_argument(
        "--vllm-model-runner",
        choices=["v1", "v2"],
        default=None,
        dest="vllm_model_runner",
        help="Pinned vLLM GPU model-runner implementation (v1 or v2).",
    )
    parser.add_argument("--min-compute-capability", dest="min_compute_capability", help="Minimum CUDA compute capability required for the pinned vLLM batch-invariance runtime, for example 8.0.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, dest="tensor_parallel_size", help="vLLM tensor parallel size (default: 1).")
    parser.add_argument("--max-num-seqs", type=int, default=None, dest="max_num_seqs", help="Pinned vLLM scheduler maximum concurrent sequences.")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None, dest="max_num_batched_tokens", help="Pinned vLLM scheduler maximum batched tokens.")
    parser.add_argument("--max-model-len", type=int, default=None, dest="max_model_len", help="Pinned vLLM maximum model context length.")
    parser.add_argument(
        "--limit-mm-per-prompt",
        dest="limit_mm_per_prompt",
        help='JSON modality limit mapping for vLLM, for example {"image":0,"audio":0}.',
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=None, dest="gpu_memory_utilization", help="Pinned vLLM fraction of GPU memory available to the executor.")
    parser.add_argument("--trust-remote-code", action="store_true", dest="trust_remote_code", help="Allow model repository remote code for batch inference (default: false).")
    parser.add_argument("--layers", default="all", help="Layers to capture: 'all' or a comma list of hidden_states indices (batch-capture).")
    parser.add_argument(
        "--persist-dtype",
        choices=["float32", "bfloat16"],
        default="float32",
        dest="persist_dtype",
        help="On-disk dtype for captured tensors (batch-capture; default: float32).",
    )
    parser.add_argument(
        "--compute-dtype",
        choices=["float32", "bfloat16", "float16"],
        default=None,
        dest="compute_dtype",
        help="Override model compute dtype for batch verbs (default: bf16 on supported GPU, else fp32).",
    )
    parser.add_argument("--resume", action="store_true", help="Resume a batch run in --out-dir, skipping already-done ids (config hash must match).")
    parser.add_argument("--sync-every", type=int, default=0, dest="sync_every", help="Run --sync-cmd after every N newly persisted rows for batch verbs (0 = never; a final sync still fires).")
    parser.add_argument("--sync-cmd", dest="sync_cmd", help="Shell command run after every --sync-every rows (and once at the end) with TUNER_SYNC_DIR / TUNER_SYNC_REASON env vars; failures warn and continue.")

    return parser
