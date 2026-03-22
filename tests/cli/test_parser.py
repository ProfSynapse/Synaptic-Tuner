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
