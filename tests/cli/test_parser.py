from tuner.cli.parser import create_parser


def test_parser_supports_yes_alias_for_auto_confirm():
    parser = create_parser()

    args = parser.parse_args(["cloud-run", "--job-config", "Trainers/recipes/example.yaml", "--yes"])
    assert args.auto_confirm is True

    args = parser.parse_args(["cloud-run", "--job-config", "Trainers/recipes/example.yaml", "--auto-confirm"])
    assert args.auto_confirm is True

    args = parser.parse_args(["local-run", "--job-config", "Trainers/recipes/example.yaml", "--yes"])
    assert args.command == "local-run"
    assert args.auto_confirm is True


def test_run_experiment_stage_selection_flags_parse():
    parser = create_parser()

    args = parser.parse_args(
        [
            "run-experiment",
            "--experiment-spec",
            "Trainers/cloud/experiments/example.yaml",
            "--only-stage",
            "evaluation",
            "--skip-stage",
            "loss",
            "--skip-stage",
            "analysis",
        ]
    )

    assert args.command == "run-experiment"
    assert args.experiment_spec == "Trainers/cloud/experiments/example.yaml"
    assert args.only_stage == "evaluation"
    assert args.skip_stage == ["loss", "analysis"]


def test_analyze_experiment_command_parses():
    parser = create_parser()

    args = parser.parse_args(["analyze-experiment", "--experiment-id", "latest", "--json"])

    assert args.command == "analyze-experiment"
    assert args.experiment_id == "latest"
    assert args.json is True


def test_plan_hardware_command_parses():
    parser = create_parser()

    args = parser.parse_args(
        [
            "plan-hardware",
            "--experiment-spec",
            "Trainers/cloud/experiments/example.yaml",
            "--optimize-for",
            "cost",
            "--max-hourly-price",
            "1.50",
        ]
    )

    assert args.command == "plan-hardware"
    assert args.experiment_spec == "Trainers/cloud/experiments/example.yaml"
    assert args.optimize_for == "cost"
    assert args.max_hourly_price == 1.50


def test_run_experiment_auto_hardware_flags_parse():
    parser = create_parser()

    args = parser.parse_args(
        [
            "run-experiment",
            "--experiment-spec",
            "Trainers/cloud/experiments/example.yaml",
            "--auto-hardware",
            "--optimize-for",
            "balanced",
        ]
    )

    assert args.command == "run-experiment"
    assert args.auto_hardware is True
    assert args.optimize_for == "balanced"


def test_cloud_training_lora_flags_parse():
    parser = create_parser()

    args = parser.parse_args(
        [
            "cloud-pipeline",
            "--method",
            "sft",
            "--train-lora-r",
            "128",
            "--train-lora-alpha",
            "256",
            "--train-lora-dropout",
            "0.05",
            "--train-use-dora",
            "--train-use-rslora",
            "--train-init-lora-weights",
            "loftq",
            "--train-lora-target-modules",
            "all-linear",
        ]
    )

    assert args.train_lora_r == 128
    assert args.train_lora_alpha == 256
    assert args.train_lora_dropout == 0.05
    assert args.train_use_dora is True
    assert args.train_use_rslora is True
    assert args.train_init_lora_weights == "loftq"
    assert args.train_lora_target_modules == "all-linear"


def test_cloud_training_evolutionary_flags_parse():
    parser = create_parser()

    args = parser.parse_args(
        [
            "cloud-pipeline",
            "--method",
            "sft",
            "--train-evolutionary-enabled",
            "--train-evolutionary-candidates",
            "4",
            "--train-evolutionary-eval-batch-size",
            "2",
            "--train-evolutionary-validation-config",
            "configs/fitness/tool_calling.yaml",
            "--train-evolutionary-strategy",
            "antithetic_noise",
            "--train-evolutionary-noise-scale",
            "0.03",
            "--train-evolutionary-max-grad-norm",
            "1.0",
            "--train-evolutionary-scale-factors",
            "0.5,1.0,1.5",
            "--train-evolutionary-selection-method",
            "best",
            "--train-evolutionary-min-improvement",
            "0.01",
            "--train-evolutionary-min-relative-improvement",
            "0.0001",
            "--train-evolutionary-noise-floor-epsilon",
            "0.000001",
            "--train-evolutionary-eval-frequency",
            "5",
            "--train-evolutionary-warmup-steps",
            "200",
            "--train-evolutionary-no-log-candidates",
        ]
    )

    assert args.train_evolutionary_enabled is True
    assert args.train_evolutionary_candidates == 4
    assert args.train_evolutionary_eval_batch_size == 2
    assert args.train_evolutionary_validation_config == "configs/fitness/tool_calling.yaml"
    assert args.train_evolutionary_strategy == "antithetic_noise"
    assert args.train_evolutionary_noise_scale == 0.03
    assert args.train_evolutionary_max_grad_norm == 1.0
    assert args.train_evolutionary_scale_factors == "0.5,1.0,1.5"
    assert args.train_evolutionary_selection_method == "best"
    assert args.train_evolutionary_min_improvement == 0.01
    assert args.train_evolutionary_min_relative_improvement == 0.0001
    assert args.train_evolutionary_noise_floor_epsilon == 0.000001
    assert args.train_evolutionary_eval_frequency == 5
    assert args.train_evolutionary_warmup_steps == 200
    assert args.train_evolutionary_log_candidates is False


def test_cloud_method_flag_accepts_grpo():
    parser = create_parser()

    args = parser.parse_args(["cloud-pipeline", "--method", "grpo"])

    assert args.command == "cloud-pipeline"
    assert args.method == "grpo"


def test_cloud_eval_timeout_flag_parses():
    parser = create_parser()

    args = parser.parse_args(["cloud-pipeline", "--eval-timeout-hours", "7.5"])

    assert args.command == "cloud-pipeline"
    assert args.eval_timeout_hours == 7.5


def test_bucket_command_parses():
    parser = create_parser()

    args = parser.parse_args(
        [
            "bucket",
            "read",
            "--path",
            "runs/hf_jobs/sft/example/logs/training_latest.jsonl",
            "--jsonl-latest",
            "--pretty",
        ]
    )

    assert args.command == "bucket"
    assert args.subcommand == "read"
    assert args.path == "runs/hf_jobs/sft/example/logs/training_latest.jsonl"
    assert args.jsonl_latest is True
    assert args.pretty is True


def test_bucket_pull_command_parses():
    parser = create_parser()

    args = parser.parse_args(
        [
            "bucket",
            "pull",
            "--path",
            "runs/hf_jobs/sft/example/analysis/loss",
            "--dest",
            ".",
        ]
    )

    assert args.command == "bucket"
    assert args.subcommand == "pull"
    assert args.path == "runs/hf_jobs/sft/example/analysis/loss"
    assert args.dest == "."


def test_bucket_push_command_parses():
    parser = create_parser()

    args = parser.parse_args(
        [
            "bucket",
            "push",
            "--path",
            "local/results.json",
            "--dest",
            "runs/manual_uploads/",
        ]
    )

    assert args.command == "bucket"
    assert args.subcommand == "push"
    assert args.path == "local/results.json"
    assert args.dest == "runs/manual_uploads/"
