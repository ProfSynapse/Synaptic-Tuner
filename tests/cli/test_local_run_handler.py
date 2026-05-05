from argparse import Namespace
from pathlib import Path

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


def test_local_run_eval_vllm_config_compiles_base_plus_lora_overlay(tmp_path):
    run_dir = tmp_path / "runs" / "eval-run"
    adapter_dir = run_dir / "final_model"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text(
        '{"base_model_name_or_path":"Qwen/Qwen3.5-2B","r":64}\n',
        encoding="utf-8",
    )

    config = tmp_path / "eval-job.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "name": "unit-local-eval",
                "provider": "local_docker",
                "job": {"transfer": "copy"},
                "run": {"method": "eval"},
                "model": {"adapter_path": str(adapter_dir)},
                "evaluation": {
                    "runtime": "vllm",
                    "scenario": "tool_prompts.yaml",
                    "served_model_name": "finetuned",
                    "port": 8011,
                },
                "artifacts": {"host_path": "Evaluator/results/unit_local_eval"},
            }
        ),
        encoding="utf-8",
    )

    handler = LocalRunHandler(args=Namespace(json=True, job_config=str(config)))
    plan = handler._compile(config, handler._load_yaml(config))

    assert plan["method"] == "eval"
    assert plan["image"] == "vllm/vllm-openai:latest"
    assert plan["copy_artifacts_on_failure"] is True
    assert Path("Evaluator") in plan["copy_paths"]
    assert Path("SynthChat") in plan["copy_paths"]
    assert Path("shared") in plan["copy_paths"]
    assert Path("tuner") in plan["copy_paths"]
    assert Path(".skills/synethetic-data-generation/scripts") in plan["copy_paths"]
    assert Path("runs/eval-run/final_model") in plan["copy_paths"]
    assert plan["host_artifact_path"].name == "unit_local_eval"
    command_text = " ".join(plan["command"])
    assert "vllm.entrypoints.openai.api_server" in command_text
    assert "Qwen/Qwen3.5-2B" in command_text
    assert "finetuned=/workspace/repo/runs/eval-run/final_model" in command_text
    assert "--backend" in command_text and "vllm" in command_text
