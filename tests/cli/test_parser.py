from tuner.cli.parser import create_parser


def test_parser_supports_yes_alias_for_auto_confirm():
    parser = create_parser()

    args = parser.parse_args(["cloud-run", "--job-config", "Trainers/cloud/jobs/example.yaml", "--yes"])
    assert args.auto_confirm is True

    args = parser.parse_args(["cloud-run", "--job-config", "Trainers/cloud/jobs/example.yaml", "--auto-confirm"])
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


def test_cloud_method_flag_accepts_grpo():
    parser = create_parser()

    args = parser.parse_args(["cloud-pipeline", "--method", "grpo"])

    assert args.command == "cloud-pipeline"
    assert args.method == "grpo"


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
