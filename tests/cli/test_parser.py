from tuner.cli.parser import create_parser


def test_parser_supports_yes_alias_for_auto_confirm():
    parser = create_parser()

    args = parser.parse_args(["cloud-run", "--job-config", "Trainers/cloud/jobs/example.yaml", "--yes"])
    assert args.auto_confirm is True

    args = parser.parse_args(["cloud-run", "--job-config", "Trainers/cloud/jobs/example.yaml", "--auto-confirm"])
    assert args.auto_confirm is True
