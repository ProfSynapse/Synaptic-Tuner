from pathlib import Path

from Evaluator.cloud_hf_job_vllm import _load_base_model_name, _parse_args


def test_load_base_model_name_reads_adapter_config(tmp_path: Path):
    model_dir = tmp_path / "final_model"
    model_dir.mkdir(parents=True)
    (model_dir / "adapter_config.json").write_text(
        '{"base_model_name_or_path":"Qwen/Qwen3-4B"}',
        encoding="utf-8",
    )

    assert _load_base_model_name(model_dir) == "Qwen/Qwen3-4B"


def test_parse_args_supports_tensor_parallel_and_loss_workers(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "cloud_hf_job_vllm.py",
            "--bucket-id",
            "bucket",
            "--run-prefix",
            "runs/hf_jobs/sft/demo",
            "--eval-prefix",
            "runs/hf_jobs/sft/demo/evaluations/vllm/ts",
            "--vllm-tensor-parallel-size",
            "4",
            "--loss-workers",
            "2",
        ],
    )

    args = _parse_args()

    assert args.vllm_tensor_parallel_size == 4
    assert args.loss_workers == 2
