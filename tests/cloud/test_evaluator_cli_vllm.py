from Evaluator.cli import parse_args


def test_evaluator_cli_accepts_vllm_backend():
    args = parse_args(["--backend", "vllm", "--model", "finetuned", "--scenario", "behavior_prompts.yaml"])
    assert args.backend == "vllm"


def test_evaluator_cli_accepts_optional_loss_flags():
    args = parse_args(
        [
            "--backend",
            "unsloth",
            "--model",
            "/tmp/final_model",
            "--preset",
            "full",
            "--with-loss",
            "--loss-dataset-name",
            "professorsynapse/claudesidian-synthetic-dataset",
            "--loss-dataset-file",
            "train.jsonl",
            "--loss-output-jsonl",
            "/tmp/per_example_losses.jsonl",
        ]
    )
    assert args.with_loss is True
    assert args.loss_dataset_name == "professorsynapse/claudesidian-synthetic-dataset"
    assert args.loss_dataset_file == "train.jsonl"
