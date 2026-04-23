from argparse import Namespace

import yaml

from tuner.handlers.local_run_handler import LocalRunHandler


def test_local_run_sft_config_compiles_repo_relative_dataset(tmp_path):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"conversations":[]}\n', encoding="utf-8")
    config = tmp_path / "job.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "name": "unit-local-sft",
                "provider": "local_docker",
                "job": {"transfer": "copy"},
                "run": {"method": "sft"},
                "model": {"name": "Qwen/Qwen3.5-2B", "load_in_4bit": False},
                "dataset": {"local_file": str(dataset)},
                "training": {"max_steps": 1},
                "artifacts": {
                    "output_root": "toolset-training-artifacts/runs/local_docker/sft/unit-local-sft",
                    "run_timestamp": "unit",
                },
            }
        ),
        encoding="utf-8",
    )

    handler = LocalRunHandler(args=Namespace(json=True, job_config=str(config)))
    plan = handler._compile(config, handler._load_yaml(config))

    assert plan["transfer"] == "copy"
    assert plan["command"][:3] == ["python", "train_sft.py", "--model-name"]
    local_file_index = plan["command"].index("--local-file") + 1
    assert plan["command"][local_file_index].startswith("../../")
    assert plan["host_artifact_path"].name == "unit"
