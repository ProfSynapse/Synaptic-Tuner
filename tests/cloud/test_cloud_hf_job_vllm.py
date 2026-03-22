from pathlib import Path

from Evaluator.cloud_hf_job_vllm import _load_base_model_name


def test_load_base_model_name_reads_adapter_config(tmp_path: Path):
    model_dir = tmp_path / "final_model"
    model_dir.mkdir(parents=True)
    (model_dir / "adapter_config.json").write_text(
        '{"base_model_name_or_path":"Qwen/Qwen3-4B"}',
        encoding="utf-8",
    )

    assert _load_base_model_name(model_dir) == "Qwen/Qwen3-4B"
