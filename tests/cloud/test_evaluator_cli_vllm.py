from Evaluator.cli import parse_args


def test_evaluator_cli_accepts_vllm_backend():
    args = parse_args(["--backend", "vllm", "--model", "finetuned", "--scenario", "behavior_prompts.yaml"])
    assert args.backend == "vllm"
